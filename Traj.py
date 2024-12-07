"""
Trajectory planners for the drone and arm
"""

import numpy as np
from scipy.linalg import null_space

from pydrake.systems.framework import LeafSystem
from pydrake.multibody.plant import MultibodyPlant
from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    AbstractValue,
    SpatialVelocity,
    JacobianWrtVariable,
    AngleAxis,
    RollPitchYaw,
    MathematicalProgram,
    SnoptSolver,
)

import utils

'''
Drone is differentially flat: can follow any traj given in [x,y,z,yaw].
This converts [x,y,z] -> trajectory (picking yaw = 0)

Adapted from GCS quadrotor examples:
https://github.com/RobotLocomotion/gcs-science-robotics/blob/main/reproduction/uav/helpers.py
'''
class FlatnessInverter(LeafSystem):
    def __init__(self, traj, animator, t_offset=0):
        LeafSystem.__init__(self)
        self.traj = traj
        # output port: [xyz, rpy, v, omega]
        self.output_port = self.DeclareVectorOutputPort("drone.state_des", 15, self.DoCalcState, {self.time_ticket()})
        self.t_offset = t_offset
        self.animator = animator

    def DoCalcState(self, context, output):
        t = context.get_time() + self.t_offset - 1e-4

        q = np.squeeze(self.traj.value(t))
        q_dot = np.squeeze(self.traj.EvalDerivative(t))
        q_ddot = np.squeeze(self.traj.EvalDerivative(t, 2))

        fz = np.sqrt(q_ddot[0]**2 + q_ddot[1]**2 + (q_ddot[2] + 9.81)**2)
        r = np.arcsin(-q_ddot[1]/fz)
        p = np.arcsin(q_ddot[0]/fz)
        # yaw doesn't really matter

        # set to hover
        # q = np.array([0.25,0.,0.8])
        # r = 0
        # p = 0
        # q_dot = np.zeros_like(q_dot)
        # q_ddot = np.zeros_like(q_ddot)

        output.set_value(np.concatenate((q, [r, p, 0], q_dot, np.zeros(3), q_ddot)))

        if self.animator is not None:
            frame = self.animator.frame(context.get_time())
            self.animator.SetProperty(frame, "/Cameras/default/rotated/<object>", "position", [-2.5, 4, 2.5])
            self.animator.SetTransform(frame, "/drake", RigidTransform(-q))


class ArmTrajectory(LeafSystem):
    def __init__(self, q_final, start, end):
        LeafSystem.__init__(self)

        self.q_final = q_final
        self.start = start
        self.end = end

        # input: current state
        self.input_state_port = self.DeclareVectorInputPort("arm.state_cur", 27)

        # build trajectory
        self.traj_idx = 0
        self.traj = None

        # output joint port: [q_cmd, qd_cmd, qdd_cmd]
        self.output_joint_port = self.DeclareVectorOutputPort("arm.state_des", 21, self.CalcJointState, {self.time_ticket()})

    def CalcJointState(self, context, output):
        t = context.get_time() - 1e-4
        if self.traj == None:
            self.create_new_traj(context)

        if t > self.traj.end_time():
            self.traj_idx += 1
            if (self.traj_idx < len(self.q_final)):
                self.create_new_traj(context)

        q = np.squeeze(self.traj.value(t))
        q_dot = np.squeeze(self.traj.EvalDerivative(t))
        q_ddot = np.squeeze(self.traj.EvalDerivative(t, 2))

        output.set_value(np.concatenate((q, q_dot, q_ddot)))

    def create_new_traj(self, context):
        state = self.input_state_port.Eval(context)
        start = state[7:14].reshape([7,1])
        end = self.q_final[self.traj_idx].reshape([7,1])
        self.traj = utils.make_bspline(start, end, (start+end)/2.,
            [self.start[self.traj_idx],self.start[self.traj_idx] + 1e-3, (self.start[self.traj_idx] + self.end[self.traj_idx])/2., self.end[self.traj_idx]])
        

'''
Does high-level reasoning for arm trajectory tracking:
1. Plan trajectory to match box position and velocity
2. Adjust trajectory when close using true drone position?
3. Close gripper when contact is made

Compared to the drone trajectory planner, this should 
generate smooth trajectories given a desired final pose,
final velocity, and time at which to reach these.
'''
class ArmTrajectoryPlanner(LeafSystem):
    def __init__(self, plant: MultibodyPlant, meshcat, pose_desired, vel_desired = SpatialVelocity([0]*6)):
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.meshcat = meshcat
        self.pose_desired = pose_desired
        self.vel_desired = vel_desired

        # setup for fkin
        self.virtual_context = plant.CreateDefaultContext()
        self.drone_model = plant.GetModelInstanceByName("drone")
        self.end_effector_frame = plant.GetBodyByName("arm_link_fngr").body_frame()
        self.drone_frame = plant.GetBodyByName("quadrotor_link").body_frame()

        # build trajectory
        self.start_time = 0.
        self.end_time = 1.5
        self.position_traj = None
        self.rotation_traj = None

        # input: current (joint) state
        self.state_input_port = self.DeclareVectorInputPort("arm.state_cur", 14) # [q, qdot]

        # output joint port: [q_cmd, qd_cmd, qdd_cmd]
        self.posevel_output_port = self.DeclareAbstractOutputPort("arm.state_des",
            lambda: AbstractValue.Make([utils.PoseVelocity()]), self.CalcJointState, {self.time_ticket()})

    def CalcJointState(self, context, output):
        t = context.get_time() - 1e-4
        if self.position_traj == None:
            self.start_time = t
            self.make_traj_to_grasp(context)
            utils.plot_ref_frame(self.meshcat, f"visualizer/drone/quadrotor_link/should", self.pose_desired)

        if t > (self.start_time + self.position_traj.duration()):
            # TODO: need some reasoning
            pass

        if t < self.start_time:
            return

        # eval position trajectory
        p = self.position_traj.evaluate(t - self.start_time)
        p_dot = self.position_traj.EvalDerivative(t - self.start_time)

        # send command
        pose_des = RigidTransform(RotationMatrix(), p)
        vel_des = SpatialVelocity(np.zeros(3), p_dot)

        out = utils.PoseVelocity(pose_des, vel_des)

        # TODO
        # might want to do the ikin here so we can weight rotations differently
        # for now, just send des pose/vel to controller.
        # TODO: also may want to run RRT or some kind of simple planner

        output.set_value(out)

    def make_traj_to_grasp(self, context):
        q_cur = self.state_input_port.Eval(context)
        # fkin
        self.plant.SetPositions(self.virtual_context, self.drone_model, q_cur[:7])
        self.plant.SetVelocities(self.virtual_context, self.drone_model, q_cur[7:])
        pose_cur = self.plant.CalcRelativeTransform(self.virtual_context, self.drone_frame, self.end_effector_frame)
        test_context = self.virtual_context #self.GetMyContextFromRoot(context)
        vel_cur  = self.end_effector_frame.CalcSpatialVelocity(test_context, self.drone_frame, self.drone_frame)

        # spline translation/linear velocity
        # TODO: convert to polynomial spline for velocity constraints
        self.position_traj = utils.CubicSpline(pose_cur.translation(), vel_cur.translational(),
                                      self.pose_desired.translation(), self.vel_desired.translational(),
                                      self.end_time - self.start_time)
        
        # spline rotation/angular velocity
        # TODO: later
        self.rotation_traj = None
        
    def compute_relative_vel(self, context, X_DC):
        # frame D: drone
        # frame C: end effector
        # frame W: world
        # TODO: CHECK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        V_D_CD = self.end_effector_frame.CalcSpatialVelocity(context, self.drone_frame, self.drone_frame)

        return V_D_CD



'''
State machine for trajectory planner.
Issues commands in joint space.
'''
class ArmTrajectoryPlanner2(LeafSystem):
    def __init__(self, plant: MultibodyPlant, meshcat, drone_traj):
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.meshcat = meshcat
        self.drone_traj = drone_traj

        # setup for fkin
        self.virtual_context = plant.CreateDefaultContext()
        self.drone_model = plant.GetModelInstanceByName("drone")
        self.sugar_model = plant.GetModelInstanceByName("sugar_box")
        self.end_effector_frame = plant.GetBodyByName("arm_link_fngr").body_frame()
        self.drone_frame = plant.GetBodyByName("quadrotor_link").body_frame()
        self.sugar_frame = plant.GetBodyByName("base_link_sugar").body_frame()

        # build trajectory
        self.start_time = 0.
        self.end_time = 1.5
        self.segments = []

        # input: current (joint) state
        self.state_input_port = self.DeclareVectorInputPort("arm.state_cur", 14 + (7+6)) # [drone quat/xyz, q, drone twist, qdot]
        # input: current sugar state
        self.sugar_input_port = self.DeclareVectorInputPort("sugar.state_cur", 7+6)

        # output joint port: [q_cmd, qd_cmd, qdd_cmd]
        self.joint_output_port = self.DeclareVectorOutputPort("arm.state_des", 21, self.EvalStateMachine, {self.time_ticket()})
        # output: is grasp complete?
        self.grasped_output_port = self.DeclareVectorOutputPort("drone.grasped", 1, self.CheckGrasped)
        self.grasped = False

    def CheckGrasped(self, context, output):
        output.set_value(np.array([2*self.grasped]))

    def EvalStateMachine(self, context, output):
        t = context.get_time() - 1e-4

        # get current state
        state = self.state_input_port.Eval(context)
        self.q_cur = state[7:14-1] # no gripper
        self.qdot_cur = state[-7:-1]
        self.q_gripper = state[13]
        self.qdot_gripper = state[-1]
        self.state_cur = state

        # first time setup
        if len(self.segments) == 0:
            # creates basic segments
            self.make_initial_traj()
            self.start_time = t
            print(f"Executing {self.segments[0].name}...")
            self.draw_target(self.segments[0])

            # draw grasp
            side_grasp = RigidTransform(RotationMatrix(AngleAxis(np.pi,np.array([1.,0.,0.]))), np.array([-0.05,-0.03,0.]))
            top_grasp  = RigidTransform(RotationMatrix(np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]]).T), np.array([0.,-0.09,0.]))
            self.grasp_pose_sugar_frame = side_grasp
            utils.plot_ref_frame(self.meshcat, "visualizer/sugar_box/base_link_sugar/grasp", self.grasp_pose_sugar_frame)
            utils.plot_ref_frame(self.meshcat, "visualizer/drone/quadrotor_link/home", RigidTransform())
            # TODO:
            # convert grasp frame to drone frame AT TIME OF GRASP
            # get velocity of grasp frame in drone frame AT TIME OF GRASP
            self.not_drawn = True
            self.do_first_calc = True
            self.gripper_opened_time = 1000

        # allow 1 second for world to settle then make (static) plan
        if t > 1. and self.do_first_calc:
            self.do_first_calc = False
            t_grasp = 3.2
            self.make_static_plan(context, t_grasp)

        # TEMP: draw grasp
        if t > 3.510 and self.not_drawn:
            self.not_drawn = False
            state_sugar = self.sugar_input_port.Eval(context)
            self.plant.SetPositionsAndVelocities(self.virtual_context, self.sugar_model, state_sugar)
            self.plant.SetPositions (self.virtual_context, self.drone_model, np.concatenate([self.state_cur[:7], self.q_cur, np.array([self.q_gripper])]))
            self.plant.SetVelocities(self.virtual_context, self.drone_model, np.concatenate([self.state_cur[14:20], self.qdot_cur, np.array([self.qdot_gripper])]))
            X_DS = self.plant.CalcRelativeTransform(self.virtual_context, self.drone_frame, self.sugar_frame)
            # pose of grasp in quadrotor frame
            X_DG = X_DS @ self.grasp_pose_sugar_frame
            # velocity of grasp in quadrotor frame
            vel = self.sugar_frame.CalcSpatialVelocity(self.virtual_context, self.drone_frame, self.drone_frame)

            utils.plot_ref_frame(self.meshcat, "visualizer/drone/quadrotor_link/true_grasp", X_DG)

        current_segment = self.segments[0]
        # move to next segment if needed
        if t > (self.start_time + current_segment.duration()):
            self.segments.pop(0)
            self.start_time = t
            if len(self.segments) == 0:
                self.terminate()

            if type(self.segments[0]) == utils.SegmentConstructor:
                self.segments[0] = self.make_segment(self.segments[0])
            current_segment = self.segments[0]
            print(f"Executing {current_segment.name}...")

            # draw target position
            self.draw_target(current_segment)
        
        # execute current segment
        q = current_segment.evaluate(t - self.start_time)
        q_dot = current_segment.EvalDerivative(t - self.start_time)
        q_ddot = current_segment.EvalDerivative(t - self.start_time, 2)

        # [x, y, z position, theta angle]
        if q.shape[0] == 3:
            p = q
            p_dot = q_dot
            p_ddot = q_ddot

            # (differential) inverse kinematics
            q, q_dot, q_ddot = self.ikin(p, p_dot, p_ddot)

        # [x, y, z position, theta angle]
        elif q.shape[0] == 4:
            p = q[:3]
            p_dot = q_dot[:3]
            p_ddot = q_ddot[:3]

            theta = q[-1] * current_segment.tot_angle
            R = utils.R_from_axisangle(current_segment.axis, theta) @ current_segment.R0
            theta_dot = current_segment.axis * q_dot[-1]
            theta_ddot = current_segment.axis * q_ddot[-1] # check

            # (differential) inverse kinematics
            q, q_dot, q_ddot = self.ikin(p, p_dot, p_ddot, R, theta_dot, theta_ddot, rot_axes=current_segment.rot_axes)
        elif q.shape[0] != 6:
            print("Unrecognized spline shape:", q.shape)
            input()

        # add gripper command
        gripper_q = 0.
        gripper_qdot = 0.
        gripper_qddot = 0.

        # distance threshold to open gripper
        state_sugar = self.sugar_input_port.Eval(context)
        self.plant.SetPositionsAndVelocities(self.virtual_context, self.sugar_model, state_sugar)
        self.fkin(np.append(self.q_cur, self.q_gripper))
        X_ES = self.plant.CalcRelativeTransform(self.virtual_context, self.end_effector_frame, self.sugar_frame) # world frame
        X_SG = self.grasp_pose_sugar_frame
        X_EG = X_ES @ X_SG

            
        if (np.linalg.norm(X_EG.translation()) < 0.5) and self.gripper_opened_time > t:
            gripper_q = -1.5
            gripper_qdot = -15
        if (np.linalg.norm(X_EG.translation()) < 0.12):
            gripper_qdot = 50 * np.linalg.norm(X_EG.translation())/ 0.12
            if self.gripper_opened_time > t:
                self.gripper_opened_time = t
        if ((t - self.gripper_opened_time) > 0.25) and not self.grasped:
            self.grasped = True



        # append gripper
        q = np.append(q, gripper_q)
        q_dot = np.append(q_dot, gripper_qdot)
        q_ddot = np.append(q_ddot, gripper_qddot)

        # send command to joint controller
        output.set_value(np.concatenate((q, q_dot, q_ddot)))

    def make_initial_traj(self):
        """
        Defines a list of segments to grasp object,
        can mix joint space and task space commands (ikin is handled in state machine)
        TODO
        """
        # get current joint position
        q0 = self.q_cur
        q0_dot = self.qdot_cur
        
        # inital joint motions
        # qd0 = np.array([0., -2.63, 1.18, 0., 0., 0.])
        qd0 = np.array([0., -1.84, 1.46, 0, 0.5, 1.41])
        # qd0 = np.array([0.,-1.68,0.57,-1.47,0,-0.15])
        duration = 1.0
        s = utils.QuinticSpline(q0, q0_dot, np.zeros_like(q0),
                                qd0, np.zeros_like(qd0), np.zeros_like(qd0),
                                duration, name="initial")
        self.segments.append(s)
        # qd1 = np.array([0., -2.63, 1.18, 0., 0., 0.])
        qd1 = np.array([0., -1.84, 1.46, 0, 0.5, 1.41])
        # qd1 = np.array([0.,-1.68,0.57,-1.47,0,-0.15])
        duration = 0.5
        s = utils.QuinticSpline(qd0, np.zeros_like(qd0), np.zeros_like(qd0),
                                qd1, np.zeros_like(qd1), np.zeros_like(qd1),
                                duration, name="initial hold")
        self.segments.append(s)

    def make_demo_traj(self):
        # get current joint position
        q0 = self.q_cur
        q0_dot = self.qdot_cur
        
        # inital joint motions
        qd0 = np.array([0., -2.63, 2.89, 0., 0., 0.])
        duration = 1.0
        s = utils.QuinticSpline(q0, q0_dot, np.zeros_like(q0),
                                qd0, np.zeros_like(qd0), np.zeros_like(qd0),
                                duration, name="initial")
        self.segments.append(s)
        qd1 = np.array([0., -2.63, 2.89, 0., 0., 0.])
        duration = 0.5
        s = utils.QuinticSpline(qd0, np.zeros_like(qd0), np.zeros_like(qd0),
                                qd1, np.zeros_like(qd1), np.zeros_like(qd1),
                                duration, name="initial hold")
        self.segments.append(s)
        
        # a position-only task space motion
        pose_d1 = self.fkin(np.append(qd1, 0.))
        p_d1 = pose_d1.translation()
        p_d2 = p_d1 + np.array([0.,0.,-0.3])
        duration = 1.0
        s = utils.QuinticSpline(p_d1, np.zeros_like(p_d1), np.zeros_like(p_d1),
                                p_d2, np.array([0.,0.,-0.5]), np.zeros_like(p_d2),
                                duration, name="fun")
        self.segments.append(s)

        # a whole pose task-space motion
        delta_p_d3 = np.array([-0.1, 0.,-0.1])
        Rf3 = np.eye(3)
        duration = 1.0
        c = utils.SegmentConstructor("pose", duration,
                delta_p_d3, np.zeros_like(delta_p_d3), np.zeros_like(delta_p_d3), delta=[True, False],
                Rf=Rf3, delta_R=True, name="whole")
        self.segments.append(c)
        
    def ikin2(self, p, p_dot, p_ddot, R=None, theta_dot=None, theta_ddot=None, rot_axes=[0,1,2]):
        """
        returns joint space commands for arm
        TODO
        """
        # forward kinematics
        pose_cur = self.fkin(np.append(self.q_cur, self.q_gripper)) # must call before calc jacobian to set positions
        J = self.plant.CalcJacobianSpatialVelocity(self.virtual_context, JacobianWrtVariable.kQDot,
                self.end_effector_frame, [0, 0, 0], self.drone_frame, self.drone_frame)
        # ignore gripper and drone terms
        J = J[:,7:14-1]
        # ignore specified rot axes
        J = J[rot_axes + [3,4,5],:]

        # velocity only (TODO: use current pose)
        if R is None:
            J = J[-3:,:]
            target_vel = p_dot
        else:
            target_vel = np.concatenate([theta_dot[rot_axes], p_dot])

        # Jacobian pinv (TODO: try optimization approach)
        gam = 0.05
        J_inv = J.T @ np.linalg.inv(J @ J.T + gam**2 * np.eye(J.shape[0]))
        q_dot = J_inv @ target_vel

        # discretely integrate to get q
        q = self.q_cur + q_dot*self.plant.time_step()

        q_ddot = np.zeros_like(q)

        return q, q_dot, q_ddot
    
    def ikin(self, p, p_dot, p_ddot, R=None, theta_dot=None, theta_ddot=None, rot_axes=[0,1,2]):
        # forward kinematics
        pose_cur = self.fkin(np.append(self.q_cur, self.q_gripper)) # must call before calc jacobian to set positions
        J = self.plant.CalcJacobianSpatialVelocity(self.virtual_context, JacobianWrtVariable.kQDot,
                self.end_effector_frame, [0, 0, 0], self.drone_frame, self.drone_frame)
        # ignore gripper and drone terms
        J = J[:,7:14-1]
        # ignore specified rot axes
        J = J[rot_axes + [3,4,5],:]

        # velocity only (TODO: use current pose)
        if R is None:
            J = J[-3:,:]
            target_vel = p_dot
        else:
            target_vel = np.concatenate([theta_dot[rot_axes], p_dot])

        ## QP approach
        prog = MathematicalProgram()
        # joint velocities
        v = prog.NewContinuousVariables(6, "v")
        prog.SetInitialGuess(v, self.qdot_cur)
        # velocity scale
        a = prog.NewContinuousVariables(1, "alpha")
        prog.SetInitialGuess(a[0], 1.)
        prog.AddBoundingBoxConstraint(0.,1.,a)

        # cost: velocity in desired direction
        prog.AddLinearCost(-np.array([1]), 0., a)

        # cost: joint centering (in null space)
        # we can't do joint centering since the arm is 6 dof
        # # P*(v - N*K_center*(q_centered - q_cur))
        # N = 2.
        # K_center = 10.
        # P = null_space(J)
        # lower = self.plant.GetPositionLowerLimits()[7:13]
        # upper = self.plant.GetPositionUpperLimits()[7:13]
        # q_center = (lower+upper)/2.
        # prog.Add2NormSquaredCost(P, P*N*K_center*(q_center - self.q_cur))

        # constraint: J * v = a * target_vel
        A = np.hstack([J, -target_vel.reshape(target_vel.shape[0],1)])
        prog.AddLinearEqualityConstraint(A, np.zeros(A.shape[0]), np.concatenate([v,a]))

        # constraint: joint limits
        # min <= q_cur + N*v*dt <= max
        lower = self.plant.GetPositionLowerLimits()[7:13]
        upper = self.plant.GetPositionUpperLimits()[7:13]
        N = 2.
        prog.AddBoundingBoxConstraint((lower - self.q_cur)/(N*self.plant.time_step()),
                                      (upper - self.q_cur)/(N*self.plant.time_step()), v)

        # # constraint: joint velocities
        v_max = 5.
        prog.AddBoundingBoxConstraint(-v_max, v_max, v)



        # solve
        solver = SnoptSolver()
        result = solver.Solve(prog)

        if not (result.is_success()):
            raise ValueError("Could not find the optimal solution.")

        q_dot = result.GetSolution(v)

        # discretely integrate to get q
        q = self.q_cur + q_dot*self.plant.time_step()
        q_ddot = np.zeros_like(q)

        return q, q_dot, q_ddot
    
    def fkin(self, q, q_dot=None):
        q = np.concatenate([self.state_cur[:7], q])
        self.plant.SetPositions(self.virtual_context, self.drone_model, q)
        pose = self.plant.CalcRelativeTransform(self.virtual_context, self.drone_frame, self.end_effector_frame)
        if q_dot is None:
            return pose

        q_dot = np.concatenate([self.state_cur[14:20], q_dot])
        self.plant.SetVelocities(self.virtual_context, self.drone_model, q_dot)
        vel = self.end_effector_frame.CalcSpatialVelocity(self.virtual_context, self.drone_frame, self.drone_frame)
        return pose, vel
    
    def make_segment(self, constructor: utils.SegmentConstructor):
        if constructor.space == "joint":
            q_f = self.q_cur*constructor.delta[0] + constructor.qf
            qdot_f = self.q_cur*constructor.delta[1] + constructor.qf_dot
            qddot_f = constructor.qf_ddot

            return utils.QuinticSpline(self.q_cur, self.qdot_cur, np.zeros_like(self.q_cur),
                        q_f, qdot_f, qddot_f, constructor.T, name=constructor.name)
        elif constructor.space == "position":
            pose, vel = self.fkin(np.append(self.q_cur, 0.),np.append(self.qdot_cur, 0.))
            p0 = pose.translation()
            v0 = vel.translational()
            pf = p0*constructor.delta[0] + constructor.qf
            vf = v0*constructor.delta[1] + constructor.qf_dot
            af = constructor.qf_ddot
            
            return utils.QuinticSpline(p0, v0, np.zeros_like(p0),
                        pf, vf, af, constructor.T, name=constructor.name)
        elif constructor.space == "pose":
            pose, vel = self.fkin(np.append(self.q_cur, 0.),np.append(self.qdot_cur, 0.))
            R0 = pose.rotation().matrix()
            p0 = np.append(pose.translation(), 0.)
            v0 = np.append(vel.translational(), 0.)
            pf = p0[:3]*constructor.delta[0] + constructor.qf
            pf = np.append(pf, 1.)
            vf = v0[:3]*constructor.delta[1] + constructor.qf_dot
            vf = np.append(vf, 0.)
            af = constructor.qf_ddot
            af = np.append(af, 0.)

            Rf = constructor.Rf
            if constructor.delta_R:
                Rf = Rf @ R0
            
            return utils.QuinticSplineR(p0, v0, np.zeros_like(p0), R0,
                        pf, vf, af, Rf, constructor.T, rot_axes=constructor.rot_axes, name=constructor.name)
        else:
            print(f"Unknown space: {constructor.space}")
    
    def terminate(self):
        """
        returns terminating segments to stop and hold
        """
        q0 = self.q_cur
        q0_dot = self.qdot_cur

        # slow to stop
        qd = q0 + q0_dot*0.05
        s = utils.QuinticSpline(q0, q0_dot, np.zeros_like(q0),
                                qd, np.zeros_like(qd), np.zeros_like(qd),
                                0.1, name="slow")
        self.segments.append(s)

        # hold indefinitely
        s = utils.QuinticSpline(qd, np.zeros_like(qd), np.zeros_like(qd),
                                qd, np.zeros_like(qd), np.zeros_like(qd),
                                100., name="stop")
        self.segments.append(s)

    def draw_target(self, segment: utils.Segment):
        q = segment.get_pf()
        if q.shape[0] == 3:
            # position space
            pose = RigidTransform(RotationMatrix(), q)

        # [x, y, z position, theta angle]
        elif q.shape[0] == 4:
            # se(3)
            pose = RigidTransform(RotationMatrix(segment.Rf), q[:3])
        else:
            pose = self.fkin(np.append(q, 0.))

        utils.plot_ref_frame(self.meshcat, f"visualizer/drone/quadrotor_link/{segment.name}", pose)

    def make_static_plan(self, context, t_grasp):
        # simulate world at time of grasp
        # assume sugar box is static
        state_sugar = self.sugar_input_port.Eval(context)
        self.plant.SetPositionsAndVelocities(self.virtual_context, self.sugar_model, state_sugar)

        # figure out drone state at time of grasp using diff flatness
        q = np.squeeze(self.drone_traj.value(t_grasp)) # x y z
        q_dot = np.squeeze(self.drone_traj.EvalDerivative(t_grasp))
        q_ddot = np.squeeze(self.drone_traj.EvalDerivative(t_grasp, 2))
        fz = np.sqrt(q_ddot[0]**2 + q_ddot[1]**2 + (q_ddot[2] + 9.81)**2)
        r = np.arcsin(-q_ddot[1]/fz)
        p = np.arcsin(q_ddot[0]/fz)
        # assume yaw = 0
        quat = RollPitchYaw(np.array([r, p, 0.])).ToQuaternion()
        quat = quat.wxyz()
        # for now, w = 0
        state_drone = np.concatenate([quat, q, np.zeros(7), np.zeros(3), q_dot, np.zeros(7)])
        self.plant.SetPositionsAndVelocities(self.virtual_context, self.drone_model, state_drone)

        # get grasp pose and velocity
        X_DS = self.plant.CalcRelativeTransform(self.virtual_context, self.drone_frame, self.sugar_frame)
        # pose of grasp in quadrotor frame
        X_DG = X_DS @ self.grasp_pose_sugar_frame
        # velocity of grasp in quadrotor frame
        vel = self.sugar_frame.CalcSpatialVelocity(self.virtual_context, self.drone_frame, self.drone_frame)

        # add segment to reach pose and velocity at t_grasp
        t_start = 0.
        for s in self.segments:
            t_start += s.T

        # TODO: probably should remove this eventually
        final_accel = 3.0 * vel.translational() / np.linalg.norm(vel.translational())
        # TODO: DEFINITELY should remove this eventually
        p_grasp = X_DG.translation() + np.array([0.,0.010,0])
        
        duration = t_grasp - t_start - self.start_time - 1e-1
        c = utils.SegmentConstructor("pose", duration,
                p_grasp, vel.translational(), final_accel, delta=[False, False],
                Rf=X_DG.rotation().matrix(), delta_R=False, rot_axes=[1,2], name="grasp")
        # ignore x-axis rotation
        self.segments.append(c)

        # duration = 0.25
        # s = utils.QuinticSpline(p_grasp, vel.translational(), final_accel,
        #                         p_grasp, np.zeros(3), np.zeros(3),
        #                         duration, name="post grasp hold")
        # self.segments.append(s)
        # duration = 0.75
        # s = utils.QuinticSpline(p_grasp, np.zeros(3), np.zeros(3),
        #                         p_grasp + np.array([0.,0.,0.1]), np.zeros(3), np.zeros(3),
        #                         duration, name="raise")
        # self.segments.append(s)

        duration = 0.5
        s = utils.QuinticSpline(p_grasp, vel.translational(), final_accel,
                                p_grasp + np.array([0.,0.,0.05]), np.zeros(3), np.zeros(3),
                                duration, name="post grasp raise")
        self.segments.append(s)
        # duration = 0.75
        # s = utils.QuinticSpline(p_grasp, np.zeros(3), np.zeros(3),
        #                         p_grasp + np.array([0.,0.,0.1]), np.zeros(3), np.zeros(3),
        #                         duration, name="raise")
        # self.segments.append(s)
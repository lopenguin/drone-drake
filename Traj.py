"""
Trajectory planners for the drone and arm
"""

import numpy as np

from pydrake.systems.framework import LeafSystem
from pydrake.multibody.plant import MultibodyPlant
from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    AbstractValue,
    SpatialVelocity,
    JacobianWrtVariable,
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
    def __init__(self, plant: MultibodyPlant, meshcat):
        LeafSystem.__init__(self)
        
        self.plant = plant
        self.meshcat = meshcat

        # setup for fkin
        self.virtual_context = plant.CreateDefaultContext()
        self.drone_model = plant.GetModelInstanceByName("drone")
        self.end_effector_frame = plant.GetBodyByName("arm_link_fngr").body_frame()
        self.drone_frame = plant.GetBodyByName("quadrotor_link").body_frame()

        # build trajectory
        self.start_time = 0.
        self.end_time = 1.5
        self.segments = []

        # input: current (joint) state
        self.state_input_port = self.DeclareVectorInputPort("arm.state_cur", 14) # [q, qdot]

        # output joint port: [q_cmd, qd_cmd, qdd_cmd]
        self.joint_output_port = self.DeclareVectorOutputPort("arm.state_des", 21, self.EvalStateMachine, {self.time_ticket()})

    def EvalStateMachine(self, context, output):
        t = context.get_time() - 1e-4

        # get current state
        state = self.state_input_port.Eval(context)
        self.q_cur = state[:6] # no gripper
        self.qdot_cur = state[-7:-1]
        self.q_gripper = state[6]
        self.qdot_gripper = state[-1]

        # first time setup
        if len(self.segments) == 0:
            # creates basic segments
            self.make_traj_to_grasp(context)
            self.start_time = t
            print(f"Executing {self.segments[0].name}...")
            self.draw_target(self.segments[0])

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
            q, q_dot, q_ddot = self.ikin(p, p_dot, p_ddot, R, theta_dot, theta_ddot)
        elif q.shape[0] != 6:
            print("Unrecognized spline shape:", q.shape)
            input()

        # add gripper command
        q = np.append(q, 0.)
        q_dot = np.append(q_dot, 0.)
        q_ddot = np.append(q_ddot, 0.)

        # send command to joint controller
        output.set_value(np.concatenate((q, q_dot, q_ddot)))

    def make_traj_to_grasp(self, context):
        """
        Defines a list of segments to grasp object,
        can mix joint space and task space commands (ikin is handled in state machine)
        TODO
        """
        # get current joint position
        q0 = self.q_cur
        q0_dot = self.qdot_cur
        
        # inital joint motion
        qd1 = np.array([0., -1.16, 1.18, 1.37, 0, 0])
        duration = 1.0
        s = utils.QuinticSpline(q0, q0_dot, np.zeros_like(q0),
                                qd1, np.zeros_like(qd1), np.zeros_like(qd1),
                                duration, name="initial")
        self.segments.append(s)
        
        # a position-only task space motion
        pose_d1 = self.fkin(np.append(qd1, 0.))
        p_d1 = pose_d1.translation()
        p_d2 = p_d1 + np.array([0.,0.,0.4])
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
                Rf=np.eye(3), delta_R=True, name="whole")
        self.segments.append(c)
        
    def ikin(self, p, p_dot, p_ddot, R=None, theta_dot=None, theta_ddot=None):
        """
        returns joint space commands for arm
        TODO
        """
        # forward kinematics
        pose_cur = self.fkin(np.append(self.q_cur, self.q_gripper)) # must call before calc jacobian to set positions
        J = self.plant.CalcJacobianSpatialVelocity(self.virtual_context, JacobianWrtVariable.kQDot,
                self.end_effector_frame, [0, 0, 0], self.drone_frame, self.drone_frame)
        # ignore gripper and drone terms
        J = J[:,:6]

        # velocity only (TODO: use current pose)
        if R is None:
            J = J[-3:,:]
            target_vel = p_dot
        else:
            target_vel = np.concatenate([theta_dot, p_dot])

        # Jacobian pinv (TODO: try optimization approach)
        gam = 0.05
        J_inv = J.T @ np.linalg.inv(J @ J.T + gam**2 * np.eye(J.shape[0]))
        q_dot = J_inv @ target_vel

        # discretely integrate to get q
        q = self.q_cur + q_dot*self.plant.time_step()

        q_ddot = np.zeros_like(q)

        return q, q_dot, q_ddot
    
    def fkin(self, q, q_dot=None):
        self.plant.SetPositions(self.virtual_context, self.drone_model, q)
        pose = self.plant.CalcRelativeTransform(self.virtual_context, self.drone_frame, self.end_effector_frame)
        if q_dot is None:
            return pose

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
                        pf, vf, af, Rf, constructor.T, name=constructor.name)
        else:
            print(f"Unknown space: {constructor.space}")
    
    def terminate(self):
        """
        returns terminating segments to stop and hold
        """
        q0 = self.q_cur
        q0_dot = self.qdot_cur

        # slow to stop
        qd = q0 + q0_dot*0.2
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
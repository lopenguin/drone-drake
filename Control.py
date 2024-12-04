"""
Controllers for the drone and arm
"""

import numpy as np

from pydrake.systems.framework import LeafSystem
from pydrake.math import RollPitchYaw
from pydrake.multibody.plant import MultibodyPlant
from pydrake.all import (
    RigidTransform,
    AbstractValue,
    RotationMatrix,
    ExternallyAppliedSpatialForce,
    SpatialForce,
    SpatialVelocity,
    JacobianWrtVariable,
    MathematicalProgram,
    SnoptSolver,
)

import utils

"""
Geometric controller for the drone.
"""
class DroneRotorController(LeafSystem):
    def __init__(self, plant: MultibodyPlant, meshcat, plot_traj=True):
        LeafSystem.__init__(self)

        self.plant = plant
        self.meshcat = meshcat
        self.plot_traj = plot_traj
        self.last_time = 0.0

        # input/output ports
        self.input_poses_port = self.DeclareAbstractInputPort("drone.pose_cur",
            AbstractValue.Make([RigidTransform()]))
        self.input_vels_port = self.DeclareAbstractInputPort("drone.vel_cur",
            AbstractValue.Make([SpatialVelocity()]))
        self.input_state_d_port = self.DeclareVectorInputPort("drone.state_des", 15)
        self.output_port = self.DeclareAbstractOutputPort("drone.rotor_force",
            lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]), self.CalcRotorForces)
        
        # make a context for the controller
        self.plant_context = plant.CreateDefaultContext()
        
    def CalcRotorForces(self, context, output):
        drone_instance = self.plant.GetModelInstanceByName("drone")
        drone_bodies = self.plant.GetBodyIndices(drone_instance)
        quadrotor_body = drone_bodies[0]

        # pull state and command from inputs
        pose_input = self.get_input_port(self.input_poses_port.get_index()).Eval(context)
        X_DW = pose_input[int(quadrotor_body)]
        vel_input = self.get_input_port(self.input_vels_port.get_index()).Eval(context)
        vel_spatial: SpatialVelocity = vel_input[int(quadrotor_body)]
        vel_cur = vel_spatial.translational()
        omega_cur = vel_spatial.rotational()
        state_d = self.get_input_port(self.input_state_d_port.get_index()).Eval(context)
        X_DW_desired = RigidTransform(
            RotationMatrix(RollPitchYaw(state_d[3], state_d[4], state_d[5])), 
            np.array(state_d[:3]))
        yaw_desired = state_d[5]
        vel_desired = np.array(state_d[6:9])
        acc_desired = np.array(state_d[-3:])

        # Rotor positions
        rotor_pos = np.zeros([3,4])
        rotor_pos[:,0] = np.array([0.0676,-0.12,0.0])
        rotor_pos[:,1] = np.array([0.0676, 0.12,0.0])
        rotor_pos[:,2] = np.array([-0.1076,-0.11,0.0])
        rotor_pos[:,3] = np.array([-0.1076,0.11,0.0])

        # # plot rotor positions via meshcat
        # if (context.get_time() - self.last_time > 0.1):
        #     X_R1D = RigidTransform(RotationMatrix(),rotor_pos[:,0])
        #     X_R2D = RigidTransform(RotationMatrix(),rotor_pos[:,1])
        #     X_R3D = RigidTransform(RotationMatrix(),rotor_pos[:,2])
        #     X_R4D = RigidTransform(RotationMatrix(),rotor_pos[:,3])
        #     utils.plot_ref_frame(self.meshcat, f"rotor1_{context.get_time()}", X_DW @ X_R1D)
        #     utils.plot_ref_frame(self.meshcat, f"rotor2_{context.get_time()}", X_DW @ X_R2D)
        #     utils.plot_ref_frame(self.meshcat, f"rotor3_{context.get_time()}", X_DW @ X_R3D)
        #     utils.plot_ref_frame(self.meshcat, f"rotor4_{context.get_time()}", X_DW @ X_R4D)
        #     self.last_time = 99999.

        # plot trajectory via meshcat
        if self.plot_traj:
            if (context.get_time() - self.last_time > 0.1):
                utils.plot_ref_frame(self.meshcat, f"traj_{context.get_time()}", X_DW_desired)
                self.last_time = context.get_time()

        ## geometric controller
        # based on https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5717652
        # gains
        # original: 16, 5.6, 9, 2.5
        Kp = 50.
        Kv = 5.6
        Kr = 91.
        Kw = 1.5
        # drone properties
        mass = self.plant.CalcTotalMass(self.plant_context, [drone_instance])
        J = self.plant.CalcSpatialInertia(self.plant_context,self.plant.GetFrameByName("quadrotor_link"),drone_bodies).CalcRotationalInertia().CopyToFullMatrix3()
        g = 9.807 # TODO: get from drake?
        
        # linear errors
        ep = X_DW.translation() - X_DW_desired.translation()
        ev = vel_cur - vel_desired

        # force frame
        f_eff = -Kp*ep - Kv*ev + mass*g*np.array([0,0,1.]) + mass*acc_desired
        Rd = np.zeros([3,3])
        Rd[:,2] = f_eff# direction of effective thrust
        Rd[:,1] = np.cross(Rd[:,2], np.array([np.cos(yaw_desired), np.sin(yaw_desired), 0.])) # arbitrary direction aligned with yaw
        Rd[:,0] = np.cross(Rd[:,1],Rd[:,2])
        Rd[:,0] /= np.linalg.norm(Rd[:,0])
        Rd[:,1] /= np.linalg.norm(Rd[:,1])
        Rd[:,2] /= np.linalg.norm(Rd[:,2])

        # rotation errors
        R = X_DW.rotation().matrix()
        eR = 0.5*utils.skew_to_vec(Rd.T @ R - R.T @ Rd)
        ew = omega_cur # just drive this to 0 (slightly lazy but should work)

        # compute force command
        f = f_eff.dot(R[:,2])
        t = -Kr * eR - Kw * ew + np.cross(omega_cur, J @ omega_cur)

        # force command per motor
        mass_center = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context, [drone_instance])

        # send forces to simulator
        rotor_forces = []
        # for i in range(rotor_pos.shape[1]):
        #     force_drone_frame = SpatialForce(np.array([0,0,0]), np.array([0,0,-1]))
        #     force_world = force_drone_frame # TODO: convert frames

        #     extforce = ExternallyAppliedSpatialForce()
        #     extforce.body_index = quadrotor_body
        #     extforce.p_BoBq_B = rotor_pos[:,i]
        #     extforce.F_Bq_W = force_world
        #     rotor_forces.append(extforce)

        # temp: one effective force/torque vector
        f_world = X_DW.rotation().matrix() @ np.array([0,0.,f])
        force_world = SpatialForce(t, f_world)
        extforce = ExternallyAppliedSpatialForce()
        extforce.body_index = quadrotor_body
        extforce.p_BoBq_B = X_DW.inverse() @ mass_center
        extforce.F_Bq_W = force_world
        rotor_forces.append(extforce)

        output.set_value(rotor_forces)
        return output
    

"""
Joint command controller for the arm
"""
class JointController(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)

        # inputs/outputs
        # drone quat, drone xyz, q, quatdot, xyzdot, qdot
        self.state_input_port = self.DeclareVectorInputPort("arm.state_cur", 14)
        self.state_d_input_port = self.DeclareVectorInputPort("arm.state_des", 21) # q, qdot, qddot
        # we only control velocity
        state_index = self.DeclareContinuousState(7)
        self.output_port = self.DeclareStateOutputPort("arm.qd_cmd", state_index)

    def DoCalcTimeDerivatives(self, context, derivatives):
        # current state
        state = self.state_input_port.Eval(context)
        q_cur = state[7:14]
        qdot_cur = state[-7:]

        # desired state
        state_d = self.state_d_input_port.Eval(context)
        q_d = state_d[:7]
        qdot_d = state_d[7:14]
        qddot_d = state_d[-7:]

        # PI control
        qdot = 100*(q_d - q_cur) + 1000.*(qdot_d - qdot_cur)
        derivatives.get_mutable_vector().SetFromVector(qdot)


"""
Task space command controller for the arm (commands in drone frame).
Currently designed with fixed drone in mind.
"""
class TaskController(LeafSystem):
    def __init__(self, plant: MultibodyPlant, meshcat):
        LeafSystem.__init__(self)

        self.plant = plant
        self.meshcat = meshcat

        # for fkin
        self.vcontext = plant.CreateDefaultContext()
        self.drone_model = plant.GetModelInstanceByName("drone")
        self.end_effector_frame = plant.GetBodyByName("arm_link_fngr").body_frame()
        self.drone_frame = plant.GetBodyByName("quadrotor_link").body_frame()

        # input: current state
        self.state_input_port = self.DeclareVectorInputPort("arm.state_cur", 14) # [q, qdot]
        # input: target pose & velocity
        self.state_d_input_port = self.DeclareAbstractInputPort("arm.state_des", 
            AbstractValue.Make([utils.PoseVelocity()]))

        # ouput: joint velocity command
        state_idx = self.DeclareContinuousState(7)
        self.cmd_output_port = self.DeclareStateOutputPort("arm.qd_cmd", state_idx)

        # FOR TESTING
        self.trigger_only_once = True

        # pre-define target pose
        # self.target_position = np.array([0.3, 0., -0.52])

    def DoCalcTimeDerivatives(self, context, derivatives):
        # get current state
        state = self.state_input_port.Eval(context)

        # forward kinematics
        self.plant.SetPositions(self.vcontext, self.drone_model, state[:7])
        pose_cur = self.plant.CalcRelativeTransform(self.vcontext, self.drone_frame, self.end_effector_frame)
        J = self.plant.CalcJacobianSpatialVelocity(self.vcontext, JacobianWrtVariable.kQDot,
                self.end_effector_frame, [0, 0, 0], self.drone_frame, self.drone_frame)
        # ignore gripper and drone terms
        J = J[:,:6]
        
        # set last q first time only
        if self.trigger_only_once:
            self.trigger_only_once = False
            self.last_q_des = state[:6]
            # utils.plot_ref_frame(self.meshcat, f"visualizer/drone/quadrotor_link/should", RigidTransform(RotationMatrix(), target_position))
            # self.target_position = pose_cur.translation()
            
        
        # update target position
        target_posevel = self.state_d_input_port.Eval(context)
        target_position = target_posevel.pose.translation()
        target_velocity = target_posevel.vel.translational() # TODO: incorporate!
        # TODO: plot!

        # pick spatial velocity to drive towards target position
        target_vel = 3*(target_position - pose_cur.translation())

        # inverse kinematics TODO: replace with opt prob with joint limits
        qd_des = self.DiffIK_WeightPinv(J, target_vel, translation_only=True)
        q_des = self.last_q_des + qd_des*self.plant.time_step()
        self.last_q_des = q_des

        qdot = 1000*(qd_des - state[-7:-1]) #+ 10*(q_des - self.last_q_des)
        # TODO: this is not a great controller, but need to move on

        # second command
        qdot_cmd = np.concatenate([qdot, np.zeros(1)])
        derivatives.get_mutable_vector().SetFromVector(qdot_cmd)


    def DiffIK_WeightPinv(self, J, target_vel, translation_only=False):
        """
        Solves for qdot to give a target velocity
        """
        if translation_only:
            J = J[-3:,:]
            target_vel = target_vel[-3:]
        
        gam = 0.05
        J_inv = J.T @ np.linalg.inv(J @ J.T + gam*gam*np.eye(J.shape[0]))
        qd_cmd = J_inv @ target_vel
        return qd_cmd


    # TODO: try this out!
    def DiffIK_JointLimits(self, J, target_vel):
        prog = MathematicalProgram()
        v = prog.NewContinuousVariables(6, "v")
        v_max = 3.0 # TODO: reconsider?

        # Add cost and constraints to prog here.
        prog.AddCost((J @ v - target_vel).T @ (J @ v - target_vel)) # l2 norm
        prog.AddBoundingBoxConstraint(-v_max, v_max, v)
        # TODO: add joint limits

        solver = SnoptSolver()
        result = solver.Solve(prog)

        if not (result.is_success()):
            raise ValueError("Could not find the optimal solution.")

        v_solution = result.GetSolution(v)

        return v_solution
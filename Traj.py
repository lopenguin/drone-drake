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
        self.segments = []

        # input: current (joint) state
        self.state_input_port = self.DeclareVectorInputPort("arm.state_cur", 14) # [q, qdot]

        # output joint port: [q_cmd, qd_cmd, qdd_cmd]
        self.joint_output_port = self.DeclareVectorOutputPort("arm.state_des", 21, self.CalcJointState, {self.time_ticket()})

    def EvalStateMachine(self, context, output):
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
        
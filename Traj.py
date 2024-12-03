"""
Trajectory planners for the drone and arm
"""

import numpy as np

from pydrake.systems.framework import LeafSystem
from pydrake.all import (
    RigidTransform,
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
'''
class ArmTrajectoryPlanner(LeafSystem):
    def __init__(self, grasp_pose):
        LeafSystem.__init__(self)

        # build trajectory
        self.traj_idx = 0
        self.traj = None

        # input: current (joint) state
        self.input_state_port = self.DeclareVectorInputPort("arm.state_cur", 27)

        # input: 

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
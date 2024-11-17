'''
Utilities for manipulation drone grasping project.
'''

import numpy as np
from scipy.optimize import root_scalar

from pydrake.systems.framework import LeafSystem
from pydrake.math import BsplineBasis, RollPitchYaw
from pydrake.trajectories import BsplineTrajectory
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.all import (
    RigidTransform,
    AbstractValue,
    FramePoseVector,
    RotationMatrix,
)

'''
Converts a trajectory in xyz space to a set of poses the drone can follow.

Adapted from GCS quadrotor examples:
https://github.com/RobotLocomotion/gcs-science-robotics/blob/main/reproduction/uav/helpers.py
'''
class FlatnessInverter(LeafSystem):
    def __init__(self, traj, animator, t_offset=0):
        LeafSystem.__init__(self)
        self.traj = traj
        # output port: [xyz, rpy, v, omega]
        self.port = self.DeclareVectorOutputPort("state", 12, self.DoCalcState, {self.time_ticket()})
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

        output.set_value(np.concatenate((q, [r, p, 0], q_dot, np.zeros(3))))

        if self.animator is not None:
            frame = self.animator.frame(context.get_time())
            self.animator.SetProperty(frame, "/Cameras/default/rotated/<object>", "position", [-2.5, 4, 2.5])
            self.animator.SetTransform(frame, "/drake", RigidTransform(-q))


class SimpleController(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)

        state_index = self.DeclareContinuousState(1)  # One state variable.
        self.output_port = self.DeclareStateOutputPort("y", state_index)  # One output: y=x.
        
        self.goal = 1.

    def DoCalcTimeDerivatives(self, context, derivatives):
        x = context.get_continuous_state_vector().GetAtIndex(0)
        if (x > 1.):
            xdot = -0.1
        else:
            xdot = 0.1
        derivatives.get_mutable_vector().SetAtIndex(0, xdot)


'''
Builds a trajectory from B-splines for the drone.

Adapted from GCS quadrotor examples:
https://github.com/RobotLocomotion/gcs-science-robotics/blob/main/gcs/bezier.py
'''
class BezierTrajectory:
    def __init__(self, path_traj, time_traj):
        assert path_traj.start_time() == time_traj.start_time()
        assert path_traj.end_time() == time_traj.end_time()
        self.path_traj = path_traj
        self.time_traj = time_traj
        self.start_s = path_traj.start_time()
        self.end_s = path_traj.end_time()

    def invert_time_traj(self, t):
        if t <= self.start_time():
            return self.start_s
        if t >= self.end_time():
            return self.end_s
        error = lambda s: self.time_traj.value(s)[0, 0] - t
        res = root_scalar(error, bracket=[self.start_s, self.end_s])
        return np.min([np.max([res.root, self.start_s]), self.end_s])

    def value(self, t):
        return self.path_traj.value(self.invert_time_traj(np.squeeze(t)))

    def EvalDerivative(self, t, derivative_order=1):
        if derivative_order == 0:
            return self.value(t)
        elif derivative_order == 1:
            s = self.invert_time_traj(np.squeeze(t))
            s_dot = 1./self.time_traj.EvalDerivative(s, 1)[0, 0]
            r_dot = self.path_traj.EvalDerivative(s, 1)
            return r_dot * s_dot
        elif derivative_order == 2:
            s = self.invert_time_traj(np.squeeze(t))
            s_dot = 1./self.time_traj.EvalDerivative(s, 1)[0, 0]
            h_ddot = self.time_traj.EvalDerivative(s, 2)[0, 0]
            s_ddot = -h_ddot*(s_dot**3)
            r_dot = self.path_traj.EvalDerivative(s, 1)
            r_ddot = self.path_traj.EvalDerivative(s, 2)
            return r_ddot * s_dot * s_dot + r_dot * s_ddot
        else:
            raise ValueError()


    def start_time(self):
        return self.time_traj.value(self.start_s)[0, 0]

    def end_time(self):
        return self.time_traj.value(self.end_s)[0, 0]

    def rows(self):
        return self.path_traj.rows()

    def cols(self):
        return self.path_traj.cols()
    

'''
Make a b-spline of specified order from start to end,
with intermediate points and travel times specified.

Assumes uniform interpolation, mostly out of laziness.
Does not currently support odd orders

start/end: 3 x 1 array
intermediate: 3 x n array
times: list/array of size n + 3 specifying start time, start end time, intermediate times, end time

Good reference: https://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node4.html
'''
def make_bspline(start, end, intermediate, times, order=8):
    n = intermediate.shape[1]
    num_control_points = order + n
    knots = np.zeros([order + num_control_points])
    # open uniform knots: repeat `order` times at start and end
    knots[:order] = 0
    knots[-order:] = n + 1

    # start and end
    oo2 = int(order/2)
    path_control_points = np.zeros([3,num_control_points])
    path_control_points[:,:oo2] = start
    path_control_points[:,-oo2:] = end
    time_control_points = np.zeros([1,num_control_points])
    time_control_points[:,:oo2] = np.linspace(times[0], times[1], oo2)
    time_control_points[:,-oo2:] = np.linspace(times[-2], times[-1], oo2)

    # intermediate control points and knots
    for i in range(n):
        knots[order + i] = 1 + i
        path_control_points[:,oo2+i] = intermediate[:,i]
        time_control_points[:,oo2+i] = times[2 + i]

    path = BsplineTrajectory(BsplineBasis(order, knots), path_control_points) # list of 3 x 1 array
    time_traj = BsplineTrajectory(BsplineBasis(order, knots), time_control_points) # list of 1 x 1 array
    trajectory = BezierTrajectory(path, time_traj)
    return trajectory



'''
Quadrotor connection class

Adapted from Drake examples:
https://github.com/RobotLocomotion/drake/blob/master/examples/quadrotor/quadrotor_geometry.cc
'''
class Quadrotor(LeafSystem):
    def __init__(self, scene_graph, plant):
        LeafSystem.__init__(self)
        # create temporary plant to set everything up
        plant = MultibodyPlant(0.0)
        parser = Parser(plant, scene_graph)
        parser.package_map().PopulateFromFolder("aerial_grasping")
        # self.model_idxs = parser.AddModelsFromUrl("package://aerial_grasping/assets/skydio_2/quadrotor_arm.urdf")[0]
        self.model_idxs = parser.AddModelsFromUrl("package://drake_models/skydio_2/quadrotor.urdf")[0]
        plant.Finalize()

        # self.model_idxs = plant.GetModelInstanceByName("drone")

        # connections
        self.DeclareVectorInputPort("quadrotor_state", 12)
        self.DeclareAbstractOutputPort("quadrotor_pose",
            lambda: AbstractValue.Make(FramePoseVector()), self.OutputGeometryPose)

        # save frame
        body_idxs = plant.GetBodyIndices(self.model_idxs)
        drone_body_idx = body_idxs[0]
        self.source_id = plant.get_source_id()
        self.frame_id = plant.GetBodyFrameIdOrThrow(drone_body_idx)
        self.arm_frames = []
        for idx in body_idxs:
            if idx == drone_body_idx:
                continue
            self.arm_frames.append(plant.GetBodyFrameIdOrThrow(idx))

    def get_frame_id(self):
        return self.frame_id
    
    def OutputGeometryPose(self, context, output):
        pose_out = self.EvalAbstractInput(context, 0).get_value()
        position = np.array([pose_out[0], pose_out[1], pose_out[2]])
        rotation = RotationMatrix(RollPitchYaw(pose_out[3], pose_out[4], pose_out[5]))
        pose = RigidTransform(rotation, position)
        
        # set drone frame
        output.get_mutable_value().set_value(self.frame_id, pose)

        # set arm frame
        for frame in self.arm_frames:
            output.get_mutable_value().set_value(frame, RigidTransform())
        return output

    @staticmethod
    def AddToBuilder(builder, state_port, scene_graph, plant):
        quadrotor = builder.AddSystem(Quadrotor(scene_graph, plant))
        # connect drone ports
        builder.Connect(state_port, quadrotor.get_input_port(0))
        builder.Connect(quadrotor.get_output_port(0), scene_graph.get_source_pose_port(quadrotor.source_id))
        # builder.Connect(quadrotor.get_output_port(0), plant.get_desired_state_input_port(quadrotor.model_idxs))
        # problem: not set up to control quadrotor ports
        # potential solution: add motors?

        return quadrotor

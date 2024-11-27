'''
Utilities for manipulation drone grasping project.
'''

import numpy as np
from scipy.optimize import root_scalar

from pydrake.math import BsplineBasis
from pydrake.trajectories import BsplineTrajectory
from pydrake.all import (
    RigidTransform,
    RotationMatrix,
    Cylinder,
    Rgba,
    Meshcat,
)

'''
Builds a trajectory from B-splines.

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
    dim = start.shape[0]
    num_control_points = order + n
    knots = np.zeros([order + num_control_points])
    # open uniform knots: repeat `order` times at start and end
    knots[:order] = 0
    knots[-order:] = n + 1

    # start and end
    oo2 = int(order/2)
    path_control_points = np.zeros([dim,num_control_points])
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


def plot_ref_frame(meshcat: Meshcat, path, X_PT, length=0.15, radius=0.005, opacity=1.0):
    '''
    Make coordinate frame on meshcat with name `path` at transform X_PT
    Modified from:
    https://github.com/RussTedrake/manipulation/blob/master/manipulation/meshcat_utils.py
    '''
    meshcat.SetTransform(path, X_PT)
    # x-axis
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0])
    meshcat.SetTransform(path + "/x-axis", X_TG)
    meshcat.SetObject(
        path + "/x-axis", Cylinder(radius, length), Rgba(1, 0, 0, opacity)
    )
    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0])
    meshcat.SetTransform(path + "/y-axis", X_TG)
    meshcat.SetObject(
        path + "/y-axis", Cylinder(radius, length), Rgba(0, 1, 0, opacity)
    )
    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    meshcat.SetTransform(path + "/z-axis", X_TG)
    meshcat.SetObject(
        path + "/z-axis", Cylinder(radius, length), Rgba(0, 0, 1, opacity)
    )

def skew_to_vec(S):
    return np.array([S[2,1], S[0,2], S[1,0]])
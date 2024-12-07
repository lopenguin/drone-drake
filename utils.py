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
    SpatialVelocity,
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

class PoseVelocity:
    def __init__(self, T: RigidTransform = RigidTransform(), v: SpatialVelocity = SpatialVelocity()):
        self.pose = T
        self.vel = v


## Standard splines
class Segment:
    def __init__(self, T, name=""):
        self.T = T
        self.name = name
        pass

    def evaluate(self, t):
        pass

    def EvalDerivative(self, t, order=1):
        pass

    def get_p0(self):
        return self.evaluate(0.0)

    def get_v0(self):
        return self.EvalDerivative(0.0)

    def get_pf(self):
        return self.evaluate(self.T)

    def get_vf(self):
        return self.EvalDerivative(self.T)

    def duration(self):
        return self.T
    
class CubicSpline(Segment):
    # Initialize.
    def __init__(self, p0, v0, pf, vf, T, name=""):
        Segment.__init__(self, T, name)
        # Precompute the spline parameters.
        self.a = p0
        self.b = v0
        self.c = 3 * (pf - p0) / T ** 2 - vf / T - 2 * v0 / T
        self.d = -2 * (pf - p0) / T ** 3 + vf / T ** 2 + v0 / T ** 2

    # Compute the position/velocity for a given time (w.r.t. t=0 start).
    def evaluate(self, t):
        # Compute and return the position and velocity.
        p = self.a + self.b * t + self.c * t ** 2 + self.d * t ** 3
        return p
    
    def EvalDerivative(self, t, order=1):
        if order == 1:
            v = self.b + 2 * self.c * t + 3 * self.d * t ** 2
            return v
        elif order == 2:
            a = 2 * self.c + 6 * self.d * t
            return a

    
class QuinticSpline(Segment):
    # Initialize.
    def __init__(self, p0, v0, a0, pf, vf, af, T, name=""):
        Segment.__init__(self, T, name)
        # Precompute the six spline parameters.
        self.a = p0
        self.b = v0
        self.c = a0
        self.d = -10 * p0 / T ** 3 - 6 * v0 / T ** 2 - 3 * a0 / T + 10 * pf / T ** 3 - 4 * vf / T ** 2 + 0.5 * af / T
        self.e = 15 * p0 / T ** 4 + 8 * v0 / T ** 3 + 3 * a0 / T ** 2 - 15 * pf / T ** 4 + 7 * vf / T ** 3 - 1 * af / T ** 2
        self.f = -6 * p0 / T ** 5 - 3 * v0 / T ** 4 - 1 * a0 / T ** 3 + 6 * pf / T ** 5 - 3 * vf / T ** 4 + 0.5 * af / T ** 3

    # Compute the position/velocity for a given time (w.r.t. t=0 start).
    def evaluate(self, t):
        # Compute and return the position and velocity.
        p = self.a + self.b * t + self.c * t ** 2 + self.d * t ** 3 + self.e * t ** 4 + self.f * t ** 5
        return p
    
    def EvalDerivative(self, t, order=1):
        if order == 1:
            v = self.b + 2 * self.c * t + 3 * self.d * t ** 2 + 4 * self.e * t ** 3 + 5 * self.f * t ** 4
            return v
        elif order == 2:
            a = 2 * self.c + 6 * self.d * t + 12 * self.e * t ** 2 + 20 * self.f * t ** 3
            return a
        
class QuinticSplineR(QuinticSpline):
    # Initialize.
    def __init__(self, p0, v0, a0, R0, pf, vf, af, Rf, T, rot_axes=[0,1,2], name=""):
        QuinticSpline.__init__(self,  p0, v0, a0, pf, vf, af, T, name)
        self.R0 = R0
        self.Rf = Rf
        (self.axis, self.tot_angle) = axisangle_from_R(R0.T @ Rf)
        self.rot_axes=rot_axes

class SegmentConstructor():
    '''
    Parameters to construct segment from where previous segment ended.
    joint space:
        - give qf (final joint config) or deltaqf (change in joint angles)
    position space:
        - give qf (final position) or deltaqf (change in position)
    pose space:
        - qf is positions, Rf is rotations
    '''
    def __init__(self, space, T, qf, qf_dot, qf_ddot, delta=[False,False], Rf=None, delta_R=False, rot_axes=[0,1,2], name=""):
        self.space = space
        self.T = T
        self.qf = qf
        self.qf_dot = qf_dot
        self.qf_ddot = qf_ddot
        self.delta = delta

        self.Rf = Rf
        self.delta_R = delta_R
        self.rot_axes = rot_axes
        self.name = name

# reference: http://www.farinhansford.com/gerald/classes/cse570/additions/orientation.pdf
def axisangle_from_R(R):
    axis = (R + R.T - (np.trace(R) - 1)*np.eye(3))[0:3, 0]
    axis = axis / np.linalg.norm(axis)
    if (np.trace(R) > 2 or np.trace(R) < -2):
        angle = 0;
    else:
        angle = np.arccos((np.trace(R) - 1)/2)
    return (axis, angle)

def R_from_axisangle(axis, theta):
    ex = np.array([[     0.0, -float(axis[2]),  axis[1]],
                   [ axis[2],      0.0, -axis[0]],
                   [-axis[1],  axis[0],     0.0]])
    return np.eye(3) + np.sin(theta) * ex + (1.0-np.cos(theta)) * ex @ ex
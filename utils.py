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
    ExternallyAppliedSpatialForce,
    SpatialForce,
    RigidTransform,
    RotationMatrix,
    Cylinder,
    Rgba,
    Meshcat,
    SpatialVelocity,
)

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
        self.output_port = self.DeclareVectorOutputPort("state", 15, self.DoCalcState, {self.time_ticket()})
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


class DroneRotorController(LeafSystem):
    def __init__(self, plant: MultibodyPlant, meshcat):
        LeafSystem.__init__(self)

        self.plant = plant
        self.meshcat = meshcat
        self.last_time = 0.0

        # input/output ports
        self.input_poses_port = self.DeclareAbstractInputPort("drone_pose_current",
            AbstractValue.Make([RigidTransform()]))
        self.input_vels_port = self.DeclareAbstractInputPort("drone_vel_current",
            AbstractValue.Make([SpatialVelocity()]))
        self.input_state_d_port = self.DeclareVectorInputPort("drone_state_desired", 15)
        self.output_port = self.DeclareAbstractOutputPort("rotor_force",
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
        #     plot_ref_frame(self.meshcat, f"rotor1_{context.get_time()}", X_DW @ X_R1D)
        #     plot_ref_frame(self.meshcat, f"rotor2_{context.get_time()}", X_DW @ X_R2D)
        #     plot_ref_frame(self.meshcat, f"rotor3_{context.get_time()}", X_DW @ X_R3D)
        #     plot_ref_frame(self.meshcat, f"rotor4_{context.get_time()}", X_DW @ X_R4D)
        #     self.last_time = 99999.

        # plot trajectory via meshcat
        if (context.get_time() - self.last_time > 0.1):
            plot_ref_frame(self.meshcat, f"traj_{context.get_time()}", X_DW_desired)
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
        eR = 0.5*skew_to_vec(Rd.T @ R - R.T @ Rd)
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

        # TODO: debug
        # - freeze arm
        # - change to hover trajectory

        output.set_value(rotor_forces)
        return output
    

# accepts joint commands
class JointController(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)

        # inputs/outputs
        # drone quat, drone xyz, q, quatdot, xyzdot, qdot?
        self.input_state_port = self.DeclareVectorInputPort("cur_state", 27)
        self.input_state_d_port = self.DeclareVectorInputPort("q_des", 21)
        # we only control velocity
        state_index = self.DeclareContinuousState(7)
        self.output_port = self.DeclareStateOutputPort("qd_cmd", state_index)

    def DoCalcTimeDerivatives(self, context, derivatives):
        # current state
        state = self.input_state_port.Eval(context)
        q_cur = state[7:14]
        qdot_cur = state[-7:]

        # desired state
        state_d = self.input_state_d_port.Eval(context)
        q_d = state_d[:7]
        qdot_d = state_d[7:14]
        qddot_d = state_d[-7:]

        # PI control
        qdot = 1000*(q_d - q_cur) + 1000.*(qdot_d - qdot_cur)
        derivatives.get_mutable_vector().SetFromVector(qdot)


class ArmTrajectory(LeafSystem):
    def __init__(self, q_final, duration, plant: MultibodyPlant):
        LeafSystem.__init__(self)

        self.q_final = q_final
        self.duration = duration

        # input: current state
        self.input_state_port = self.DeclareVectorInputPort("cur_state", 27)

        # build trajectory
        self.traj = None

        # output joint port: [q_cmd, qd_cmd, qdd_cmd]
        self.output_joint_port = self.DeclareVectorOutputPort("arm_des", 21, self.CalcJointState, {self.time_ticket()})

    def CalcJointState(self, context, output):
        t = context.get_time() - 1e-4
        if self.traj == None:
            state = self.input_state_port.Eval(context)
            start = state[7:14].reshape([7,1])
            end = self.q_final.reshape([7,1])
            self.traj = make_bspline(start, end, (start+end)/2.,
                [t,t + 1e-3,t + self.duration/2.,t + self.duration])

        q = np.squeeze(self.traj.value(t))
        q_dot = np.squeeze(self.traj.EvalDerivative(t))
        q_ddot = np.squeeze(self.traj.EvalDerivative(t, 2))

        output.set_value(np.concatenate((q, q_dot, q_ddot)))

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
    Make reference frame on meshcat with name `path` at transform X_PT
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
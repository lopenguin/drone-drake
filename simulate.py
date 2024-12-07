"""
Simulate the drone and its environment.
"""

import numpy as np

from pydrake.geometry import StartMeshcat
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, ApplyMultibodyPlantConfig, MultibodyPlantConfig
from pydrake.multibody.parsing import Parser
from pydrake.all import (
    DiagramBuilder,
    LoadModelDirectives,
    ProcessModelDirectives,
    MeshcatVisualizer,
    Simulator,
    VisualizationConfig,
    ApplyVisualizationConfig,
)

from utils import make_bspline
import Control
import Traj

import pydot
def save_diagram(diagram):
    pngfile = pydot.graph_from_dot_data(
                  diagram.GetGraphvizString(max_depth=2))[0].create_svg()
    
    with open('diagram.svg','wb') as png_file:
        png_file.write(pngfile)

if __name__ == '__main__':
    ## Basic drone trajectory
    # in poses
    start = np.array([-1.5,0,1.]).reshape([3,1])
    end = np.array([1.5,0,1.]).reshape([3,1])
    intermediate = np.array([0.,0,-0.6]).reshape([3,1])
    trajectory = make_bspline(start, end, intermediate,[1.,2,4,5.])

    # start = np.array([0,0,1.]).reshape([3,1])
    # end = np.array([0,0,1.]).reshape([3,1])
    # intermediate = np.array([0.,0,1]).reshape([3,1])
    # trajectory = make_bspline(start, end, intermediate,[1.,1.1,1.5,3.])

    ## Simulation
    # Start meshcat: URL will appear in command line
    meshcat = StartMeshcat()

    # Simulation parameters
    sim_time_step = 0.0001
    model_directive_file = "default_directive.yaml"
    plant_config = MultibodyPlantConfig(
        time_step = sim_time_step,
        contact_model = "hydroelastic", # TODO: switch to hydroelastic (much slower)
        discrete_contact_approximation = "sap"
    )

    # Create plant
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)
    parser = Parser(plant, scene_graph)
    parser.package_map().PopulateFromFolder("aerial_grasping")
    directives = LoadModelDirectives(model_directive_file)
    models = ProcessModelDirectives(directives, plant, parser)
    ApplyMultibodyPlantConfig(plant_config, plant)

    plant.Finalize()

    # Add visualizer
    meshcat_vis = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    visualization = VisualizationConfig()
    ApplyVisualizationConfig(visualization, builder, meshcat=meshcat)

    animator = meshcat_vis.StartRecording()
    animator = None # stop auto-tracking of drone movement

    ## Drone
    # low-level controller
    drone_controller = builder.AddNamedSystem("drone_controller", Control.DroneRotorController(plant, meshcat, plot_traj=True))
    builder.Connect(plant.get_body_poses_output_port(), drone_controller.input_poses_port)
    builder.Connect(plant.get_body_spatial_velocities_output_port(), drone_controller.input_vels_port)
    builder.Connect(drone_controller.output_port, plant.get_applied_spatial_force_input_port())

    # trajectory generation
    traj_system = builder.AddSystem(Traj.FlatnessInverter(trajectory, animator))
    builder.Connect(traj_system.output_port, drone_controller.input_state_d_port)

    ## Arm
    drone_instance = plant.GetModelInstanceByName("drone")
    sugar_instance = plant.GetModelInstanceByName("sugar_box")
    # plant.GetJointByName("arm_sh0").set_position_limits([-np.inf],[np.inf])
    # plant.GetJointByName("arm_sh1").set_position_limits([-np.inf],[np.inf])
    # plant.GetJointByName("arm_el0").set_position_limits([-np.inf],[np.inf])
    # plant.GetJointByName("arm_el1").set_position_limits([-np.inf],[np.inf])
    # plant.GetJointByName("arm_wr0").set_position_limits([-np.inf],[np.inf])
    # plant.GetJointByName("arm_wr1").set_position_limits([-np.inf],[np.inf])
    # plant.GetJointByName("arm_f1x").set_position_limits([-0.],[0.])
    # low-level controller
    arm_controller = builder.AddNamedSystem("arm_controller", Control.JointController())
    builder.Connect(plant.get_state_output_port(drone_instance), arm_controller.state_input_port)
    builder.Connect(arm_controller.output_port, plant.get_actuation_input_port(drone_instance))

    # trajectory generation
    arm_traj_system = builder.AddSystem(Traj.ArmTrajectoryPlanner2(plant, meshcat, trajectory))
    builder.Connect(plant.get_state_output_port(drone_instance), arm_traj_system.state_input_port)
    builder.Connect(plant.get_state_output_port(sugar_instance), arm_traj_system.sugar_input_port)
    builder.Connect(arm_traj_system.joint_output_port, arm_controller.state_d_input_port)
    builder.Connect(arm_traj_system.grasped_output_port, drone_controller.input_grasped_port)

    # Add visualizer and build
    diagram = builder.Build()

    # Show the simulation
    end_time = trajectory.end_time()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    meshcat.Delete()

    simulator.AdvanceTo(end_time+0.05)
    meshcat_vis.PublishRecording()

# Current state:
# - Predefined drone trajectory & arm trajectory
# - Joint commands for arm

# Vision:
# a) DRONE TRAJECTORY: given the position (and orientation?) of a box, 
#       drone plans a trajectory to approach just above.
#   - low priority for now
# b) ARM TRAJECTORY: given the drone trajectory and a desired grasp time,
#       plan to match position and velocity, and close gripper.
#   - inverse kinematics for position/velocity commands
#   - compute object position/velocity in (moving) drone frame
#   - first, assume drone perfectly tracks trajectory
#   - migrate to dynamically redoing trajectory based on current drone pose

# Plan:
# 1. task space control for arm
# 2. task space planning for arm: match position and velocity of object
# Ideally finish these by Wednesday.
# 3. Grasp selection (should be easy)
# 4. Compare with Hessian
"""
Simulate the drone and its environment.
"""

import numpy as np

from pydrake.geometry import StartMeshcat
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, ApplyMultibodyPlantConfig, MultibodyPlantConfig, MultibodyPlant
from pydrake.multibody.parsing import Parser
from pydrake.all import (
    DiagramBuilder,
    LoadModelDirectives,
    ProcessModelDirectives,
    MeshcatVisualizer,
    Simulator,
)

from utils import FlatnessInverter, make_bspline, JointController, DroneRotorController, ArmTrajectory

import pydot
def save_diagram(diagram):
    pngfile = pydot.graph_from_dot_data(
                  diagram.GetGraphvizString(max_depth=2))[0].create_svg()
    
    with open('diagram.svg','wb') as png_file:
        png_file.write(pngfile)

if __name__ == '__main__':
    ## Basic drone trajectory
    # in poses
    # start = np.array([-1.5,0,1.]).reshape([3,1])
    # end = np.array([1.5,0,1.]).reshape([3,1])
    # intermediate = np.array([0.,0,-0.5]).reshape([3,1])
    # trajectory = make_bspline(start, end, intermediate,[1.,3,4,5.])

    start = np.array([0,0,1.]).reshape([3,1])
    end = np.array([0,0,1.]).reshape([3,1])
    intermediate = np.array([0.,0,1]).reshape([3,1])
    trajectory = make_bspline(start, end, intermediate,[1.,3,4,5.])

    ## Simulation
    # Start meshcat: URL will appear in command line
    meshcat = StartMeshcat()

    # Simulation parameters
    sim_time_step = 0.0001
    model_directive_file = "default_directive.yaml"
    plant_config = MultibodyPlantConfig(
        time_step = sim_time_step,
        contact_model = "point", # TODO: switch to hydroelastic once drone is fixed
        discrete_contact_approximation = "sap"
    )

    # Create plant (objects we will not control)
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)
    parser = Parser(plant, scene_graph)
    parser.package_map().PopulateFromFolder("aerial_grasping")
    directives = LoadModelDirectives(model_directive_file)
    models = ProcessModelDirectives(directives, plant, parser)
    ApplyMultibodyPlantConfig(plant_config, plant)

    # drone_instance = plant.GetModelInstanceByName("drone")
    # plant.set_gravity_enabled(drone_instance, False) # gravity compensation.
    plant.Finalize()

    # Add visualizer
    meshcat_vis = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    animator = meshcat_vis.StartRecording()
    animator = None # stop auto-tracking of drone movement

    # Drone position -> poses
    traj_system = builder.AddSystem(FlatnessInverter(trajectory, animator))

    # simple controller for drone arm
    drone_instance = plant.GetModelInstanceByName("drone")
    # TEMP: FIX THE ARM
    # plant.GetJointByName("arm_sh0").set_position_limits([-0.],[0.])
    # plant.GetJointByName("arm_sh1").set_position_limits([-0.],[0.])
    # plant.GetJointByName("arm_el0").set_position_limits([-0.],[0.])
    # plant.GetJointByName("arm_el1").set_position_limits([-0.],[0.])
    # plant.GetJointByName("arm_wr0").set_position_limits([-0.],[0.])
    # plant.GetJointByName("arm_wr1").set_position_limits([-0.],[0.])
    # plant.GetJointByName("arm_f1x").set_position_limits([-0.],[0.])
    # plant.GetJointByName("arm_sh1").set_position_limits(
    #         [-np.inf], [np.inf]
    #     )
    q_desired = np.array([0., -1.16, 1.18, 1.37, 0, 0, -0.72])
    drone_traj = builder.AddSystem(ArmTrajectory(q_desired, 1., plant))
    builder.Connect(plant.get_state_output_port(drone_instance), drone_traj.input_state_port)

    arm_controller = builder.AddNamedSystem("arm_controller", JointController())
    builder.Connect(plant.get_state_output_port(drone_instance), arm_controller.input_state_port)
    builder.Connect(arm_controller.output_port, plant.get_actuation_input_port(drone_instance))
    builder.Connect(drone_traj.output_joint_port, arm_controller.input_state_d_port)

    # simple controller for drone
    drone_controller = builder.AddNamedSystem("drone controller", DroneRotorController(plant, meshcat))
    builder.Connect(plant.get_body_poses_output_port(), drone_controller.input_poses_port)
    builder.Connect(plant.get_body_spatial_velocities_output_port(), drone_controller.input_vels_port)
    builder.Connect(traj_system.output_port, drone_controller.input_state_d_port)
    builder.Connect(drone_controller.output_port, plant.get_applied_spatial_force_input_port())

    # Add visualizer and build
    diagram = builder.Build()

    # Show the simulation
    end_time = trajectory.end_time()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    meshcat.Delete()

    simulator.AdvanceTo(end_time+0.05)
    meshcat_vis.PublishRecording()
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
    ModelInstanceIndex
)

from utils import FlatnessInverter, make_bspline, SimpleController

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
    intermediate = np.array([0.,0,-1.5]).reshape([3,1])
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

    drone_instance = plant.GetModelInstanceByName("drone")
    plant.set_gravity_enabled(drone_instance, False) # gravity compensation.
    plant.Finalize()

    # Add visualizer
    meshcat_vis = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    animator = meshcat_vis.StartRecording()
    animator = None # stop auto-tracking of drone movement

    # Drone position -> poses
    traj_system = builder.AddSystem(FlatnessInverter(trajectory, animator))

    # simple controller for drone
    drone_instance = plant.GetModelInstanceByName("drone")
    plant.GetJointByName("arm_sh0").set_position_limits(
            [-np.inf], [np.inf]
        )
    # TODO: add 
    controller = builder.AddNamedSystem("controller", SimpleController())
    builder.Connect(controller.output_port, plant.get_actuation_input_port(drone_instance))

    # Add visualizer and build
    diagram = builder.Build()

    # Show the simulation
    end_time = trajectory.end_time()
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    meshcat.Delete()

    simulator.AdvanceTo(end_time+0.05)
    meshcat_vis.PublishRecording()
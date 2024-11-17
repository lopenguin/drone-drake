"""
Simulate the drone and its environment.
"""

import numpy as np

from pydrake.geometry import StartMeshcat
from pydrake.visualization import AddDefaultVisualization
from pydrake.multibody.parsing import Parser
from pydrake.all import (
    RobotDiagramBuilder,
    LoadModelDirectives,
    ProcessModelDirectives,
    ModelInstanceIndex,
    Simulator
)

import pydot
def save_diagram(diagram):
    pngfile = pydot.graph_from_dot_data(
                  diagram.GetGraphvizString(max_depth=2))[0].create_svg()
    
    with open('diagram.svg','wb') as png_file:
        png_file.write(pngfile)

def make_diagram(directive_file, sim_time_step=0.01):
    # build robot
    robot_builder = RobotDiagramBuilder(time_step=sim_time_step)
    builder = robot_builder.builder()
    sim_plant = robot_builder.plant()
    scene_graph = robot_builder.scene_graph()

    parser = Parser(sim_plant)
    parser.package_map().PopulateFromFolder("aerial_grasping")

    # load setup from file
    directives = LoadModelDirectives(directive_file)
    models = ProcessModelDirectives(directives, sim_plant, parser)

    # controller setup for drone
    drone_instance = sim_plant.GetModelInstanceByName("drone")
    sim_plant.set_gravity_enabled(drone_instance, False) # gravity compensation.
    # for name, gains in driver_config.gains.items():
    #     actuator = sim_plant.GetJointActuatorByName(name, drone_instance)
    #     actuator.set_controller_gains(PdControllerGains(p=gains.kp, d=gains.kd))

    sim_plant.Finalize()

    # Ports for the drone
    builder.ExportInput(sim_plant.get_desired_state_input_port(drone_instance),
                        "drone.desired_state")
    builder.ExportInput(sim_plant.get_actuation_input_port(drone_instance),
                        "drone.tau_feedforward")
    builder.ExportOutput(sim_plant.get_state_output_port(drone_instance),
                        "drone.state_estimated")

    ## Cheat ports
    # TODO: necessary?
    builder.ExportInput(
        sim_plant.get_applied_generalized_force_input_port(),
        "applied_generalized_force",
    )
    builder.ExportInput(
        sim_plant.get_applied_spatial_force_input_port(),
        "applied_spatial_force",
    )

    # Export any actuation (non-empty) input ports that are not already
    # connected (e.g. by a driver).
    for i in range(sim_plant.num_model_instances()):
        port = sim_plant.get_actuation_input_port(ModelInstanceIndex(i))
        if port.size() > 0 and not builder.IsConnectedOrExported(port):
            builder.ExportInput(port, port.get_name())
    # Export all MultibodyPlant output ports.
    for i in range(sim_plant.num_output_ports()):
        builder.ExportOutput(
            sim_plant.get_output_port(i),
            sim_plant.get_output_port(i).get_name(),
        )
    # Export the only SceneGraph output port.
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    ## End Cheat Ports

    # add default visualization so you see things
    AddDefaultVisualization(builder=builder, meshcat=meshcat)

    # build!
    diagram = builder.Build()
    diagram.set_name("station")
    return diagram

def start_simulation(finish_time=5.):
    print("Starting simulation...")
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(1.)
    meshcat.StartRecording()
    simulator.AdvanceTo(finish_time)
    meshcat.PublishRecording()

if __name__ == '__main__':
    ## Start meshcat: URL will appear in command line
    meshcat = StartMeshcat()

    ## Simulation parameters
    sim_time_step = 0.0001
    model_directive_file = "default_directive.yaml"

    ## Set up a simulation
    diagram = make_diagram(model_directive_file, sim_time_step)

    ## Show a simulation
    finish_time = 5.
    start_simulation(finish_time)
"""
Visualize the initial scene
"""

import numpy as np

from pydrake.geometry import StartMeshcat
from pydrake.visualization import ModelVisualizer

if __name__ == '__main__':
    # Start meshcat: URL will appear in command line
    meshcat = StartMeshcat()

    # Visualizer
    visualizer = ModelVisualizer(meshcat=meshcat)
    visualizer.parser().package_map().PopulateFromFolder("aerial_grasping")
    visualizer.parser().AddModels(
        url="package://aerial_grasping/assets/skydio_2/quadrotor_arm.urdf"
    )
    visualizer.Run()
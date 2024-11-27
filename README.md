# 6.4212 Robotic Manipulation Final Project
Drone-based dynamic grasping of objects.

## Quick Start
This project was tested with Python 3.10.12 on Ubuntu 22.04. To install, create a virtual environment and run:
```
TODO
```

You may now run the project by calling:
```
ipython -i simulate.py
```
Remember to click on the meshcat link!

## Viewing the drone
In the debugging process I also created a ModelVisualizer script. You can use this to play with robot joints directly.
```
python visualize.py
```

## Structure
TODO: put in diagram.

We split the drone and arm into two pieces which are effectively controlled independently. Each piece is composed of a *controller* and *trajectory generator*. In the case of the drone:

`DroneRotorController` is the low-level controller. It implements a geometric controller based on [this paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5717652) which essentially cancels the drone dynamics and uses proportional gains of position, velocity, rotation, and angular velocity to achieve a desired trajectory. Actuation is force commands.

`FlatnessInverter` is the trajectory generator. It takes position commands (instances of `BezierTrajectory`) and converts them into full pose commands using the differential flatness property of the drone platform.

The arm is a little different. Instead of pre-planning a trajectory using `FlatnessInverter`, the trajectory generator takes only a desired position and time. I pass a list of trajectory generators to the controller and it cycles through them.

`JointController` uses PI control to achieve a desired joint position with joint velocity commands.

`ArmTrajectory` is the trajectory generator, which takes position and time commands and plans a smooth trajectory to reach them.
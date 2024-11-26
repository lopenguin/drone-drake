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
In this project we take full advantage of Drake's dynamical system modeling functionality.
The drone and arm are modeled separately.
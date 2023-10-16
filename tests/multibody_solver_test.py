#!/usr/bin/env python

from multibody_dynamics_solver.multibody_solver import *
from multibody_dynamics_solver.spatial_transformations import *

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.init()

@ti.kernel
def setup():
    w.setup()

@ti.kernel
def run_simulation_step():
    w.update_kinematics_dynamics()

# l0: stationary base
l0 = StaticLink(
    State(pose = pose(orientation = [0., 0., 0., 1.])),
    "link_0"
)
# l1: pendulum block pulled to the same height as the hinge
l1 = Link(1.0, [1./3., 0., 0., 0., 1./3., 0., 0., 0., 0.01],
    #State(pose = pose(position = [0., 0.866, 0.5], orientation = [0.866, 0., 0., 0.5])),
    State(pose = pose(position = [0., 0.866, -0.5], orientation = [0.5, 0., 0., 0.866])),
    #State(pose = pose(position = [0., 1., 0.], orientation = [0.707, 0., 0., 0.707])),
    "link_1"
)
# the pendulum hinge
j1 = Joint(
    l0, l1,
    transform(rotation=[0., 0., 0., 1.]),
    transform(translation=[0., 0., 1.], rotation=[0., 0., 0., 1.]),
    #State(pose = pose(position=[0., 0., 0.], orientation=[0.866, 0., 0., 0.5])),
    State(pose = pose(position=[0., 0., 0.], orientation=[0.5, 0., 0., 0.866])),
    #State(pose = pose(position=[0., 0., 0.], orientation=[0.707, 0., 0., 0.707])),
    joint_type=JointType.REVOLUTE
)
l0.assemble(l1, j1)
w = World([l0, l1], [0], dt=0.1)

result = np.zeros([19, 1000])
setup()
for i in range(result.shape[1]):
    result[0, i] = w.links[1].state_com_in_world[None].pose.position.x
    result[1, i] = w.links[1].state_com_in_world[None].pose.position.y
    result[2, i] = w.links[1].state_com_in_world[None].pose.position.z
    result[3, i] = w.links[1].state_com_in_world[None].pose.orientation.x
    result[4, i] = w.links[1].state_com_in_world[None].pose.orientation.y
    result[5, i] = w.links[1].state_com_in_world[None].pose.orientation.z
    result[6, i] = w.links[1].state_com_in_world[None].pose.orientation.w
    result[7, i] = w.links[1].state_com_in_world[None].twist.linear.x
    result[8, i] = w.links[1].state_com_in_world[None].twist.linear.y
    result[9, i] = w.links[1].state_com_in_world[None].twist.linear.z
    result[10, i] = w.links[1].state_com_in_world[None].twist.angular.x
    result[11, i] = w.links[1].state_com_in_world[None].twist.angular.y
    result[12, i] = w.links[1].state_com_in_world[None].twist.angular.z
    result[13, i] = w.links[1].state_com_in_world[None].acceleration.linear.x
    result[14, i] = w.links[1].state_com_in_world[None].acceleration.linear.y
    result[15, i] = w.links[1].state_com_in_world[None].acceleration.linear.z
    result[16, i] = w.links[1].state_com_in_world[None].acceleration.angular.x
    result[17, i] = w.links[1].state_com_in_world[None].acceleration.angular.y
    result[18, i] = w.links[1].state_com_in_world[None].acceleration.angular.z
    run_simulation_step()
print(result)
#plt.plot(result[2,:])

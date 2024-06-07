"""
This file is used to generate the maps which the DRL agent will be trained and
evaluated in. Such a map constitutes e.g. the initial robot position, the goal
position and the locations of obstacles and boundaries.
"""

import math
import random

import numpy as np
from math import pi, radians, cos, sin

from ..environment import MobileRobot, Obstacle, Boundary, Goal, MapDescription, MapGenerator
from .map_multi_robot import generate_map_multi_robot3
from .map import generate_map_corridor
from typing import Union, List, Tuple

def generate_eval_map1() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """

    init_state = np.array([5, 10, 0.0, 0, 0])
    atr = MobileRobot(init_state)
    boundary = Boundary([(0, 0), (40, 0), (40, 20), (0, 20)])
    obstacles = []
    unexpected_obstacles = []
    obstacles.append(Obstacle.create_mpc_static([(0, 0), (0, 9), (40, 9), (40, 0)]))
    obstacles.append(Obstacle.create_mpc_static([(0, 20), (0, 14), (40, 14), (40, 20)]))
    goal = Goal((35, 10))

    
    for i in [10,15,20,25,30]:
        unexpected_obstacles.append(Obstacle.create_mpc_static([(i, 11), (i, 9), (i+2, 9), (i+2, 11)]))
    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    # unexpected_obstacle = Obstacle.create_mpc_dynamic_old(p1=(35, 10), p2=(5, 10), freq=0.1, rx=0.8, ry=0.8, angle=0.0, corners=20)
    # # unexpected_obstacle.visible_on_reference_path = False
    # unexpected_obstacles.append(unexpected_obstacle)
    obstacles.extend(unexpected_obstacles)
    return atr, boundary, obstacles, goal
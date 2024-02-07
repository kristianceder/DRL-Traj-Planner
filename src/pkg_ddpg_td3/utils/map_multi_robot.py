import math
import random

import numpy as np
from math import pi, radians, cos, sin

from ..environment import MobileRobot, Obstacle, Boundary, Goal, MapDescription, MapGenerator

from typing import Union, List, Tuple

def generate_map_multi_robot1() -> MapDescription:
    """
    Generates a randomized map with one static obstacle
    """

    nodes = np.array([(0,0),(0,3),(3,3),(3,2.8),(0.2,2.8),(0.2,0.2),(3,0.2),(3,0)])

    init_state = np.array([8, 8, random.uniform(-pi, pi), 0, 0])
    robot = MobileRobot(init_state)
    size = 30
    boundary = Boundary([(0, 0), (size, 0), (size, size), (0, size)])

    obstacles = []
    for i in range(size//3):
        obstacles.append(Obstacle.create_mpc_static(list(map(tuple,nodes+np.array([0,i*3])))))
    goal = Goal((1.5, 1.5))
    if random.random() < 0.2:
        obstacles[0].visible_on_reference_path=False
    return robot, boundary, obstacles, goal
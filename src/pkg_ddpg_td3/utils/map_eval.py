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

def generate_eval_map111() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """

    init_state = np.array([0.6, 3.5, 0.0, 0, 0])
    atr = MobileRobot(init_state)
    boundary = Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)])
    obstacles = []
    unexpected_obstacles = []
    scene_1_obstacles_list = [[(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                            [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                            [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                            [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in scene_1_obstacles_list]
    goal = Goal((15.4, 3.5))

    unexpected_obstacle = Obstacle.create_mpc_static([(7.5, 3.0), (7.5, 4.0), (8.5, 4.0), (8.5, 3.0)]) # small
    unexpected_obstacles.append(unexpected_obstacle)

    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    obstacles.extend(unexpected_obstacles)
    return atr, boundary, obstacles, goal

def generate_eval_map112() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """

    init_state = np.array([0.6, 3.5, 0.0, 0, 0])
    atr = MobileRobot(init_state)
    boundary = Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)])
    obstacles = []
    unexpected_obstacles = []
    scene_1_obstacles_list = [[(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                            [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                            [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                            [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in scene_1_obstacles_list]
    goal = Goal((15.4, 3.5))

    unexpected_obstacle = Obstacle.create_mpc_static([(7.2, 2.8), (7.2, 4.2), (8.8, 4.2), (8.8, 2.8)]) # medium
    unexpected_obstacles.append(unexpected_obstacle)

    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    obstacles.extend(unexpected_obstacles)
    return atr, boundary, obstacles, goal

def generate_eval_map113() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """

    init_state = np.array([0.6, 3.5, 0.0, 0, 0])
    atr = MobileRobot(init_state)
    boundary = Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)])
    obstacles = []
    unexpected_obstacles = []
    scene_1_obstacles_list = [[(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                            [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                            [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                            [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in scene_1_obstacles_list]
    goal = Goal((15.4, 3.5))

    unexpected_obstacle = Obstacle.create_mpc_static([(7.0, 2.5), (7.0, 4.5), (9.0, 4.5), (9.0, 2.5)]) # large
    unexpected_obstacles.append(unexpected_obstacle)

    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    obstacles.extend(unexpected_obstacles)
    return atr, boundary, obstacles, goal

def generate_eval_map121() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """

    obstacles_list = [  [(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                        [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                        [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                        [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in obstacles_list]

    unexpected_obstacle_list = [[(5,1.5), (5,4), (6,4), (6,1.5)],
                                [(8.5, 3.5), (8.5, 8.0), (9.5, 8.0), (9.5, 3.5)]]
    unexpected_obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in unexpected_obstacle_list]

    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    obstacles.extend(unexpected_obstacles)
    return MobileRobot(np.array([0.6, 3.5, 0.0, 0, 0])), Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)]), obstacles, Goal((15.4, 3.5))

def generate_eval_map122() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """

    obstacles_list = [  [(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                        [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                        [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                        [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in obstacles_list]

    unexpected_obstacle_list = [[(5,1.5), (5,5), (6,5), (6,1.5)],
                                [(8.5, 3.5), (8.5, 8.0), (9.5, 8.0), (9.5, 3.5)]]
    unexpected_obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in unexpected_obstacle_list]

    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    obstacles.extend(unexpected_obstacles)
    return MobileRobot(np.array([0.6, 3.5, 0.0, 0, 0])), Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)]), obstacles, Goal((15.4, 3.5))

def generate_eval_map123() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """

    obstacles_list = [  [(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                        [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                        [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                        [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in obstacles_list]

    unexpected_obstacle_list = [[(4.2, 2.8), (4.2, 4.2), (5.8, 4.2), (5.8, 2.8)],
                                [(6.2, 2.8), (6.2, 4.2), (7.8, 4.2), (7.8, 2.8)]]
    unexpected_obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in unexpected_obstacle_list]

    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    obstacles.extend(unexpected_obstacles)
    return MobileRobot(np.array([0.6, 3.5, 0.0, 0, 0])), Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)]), obstacles, Goal((15.4, 3.5))

def generate_eval_map124() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """

    obstacles_list = [  [(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                        [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                        [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                        [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in obstacles_list]

    unexpected_obstacle_list = [[(4.2, 2.8), (4.2, 4.2), (5.8, 4.2), (5.8, 2.8)],
                                [(8.2, 2.8), (8.2, 4.2), (9.8, 4.2), (9.8, 2.8)]]
    unexpected_obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in unexpected_obstacle_list]

    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    obstacles.extend(unexpected_obstacles)
    return MobileRobot(np.array([0.6, 3.5, 0.0, 0, 0])), Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)]), obstacles, Goal((15.4, 3.5))

def generate_eval_map131() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """

    obstacles_list = [  [(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                        [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                        [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                        [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in obstacles_list]

    unexpected_obstacle_list = [[(6.0, 4.5), (6.0, 5.0), (8.5, 5.0), (8.5, 4.5)],
                                [(8.5, 5.0), (8.5, 2.0), (8.0, 2.0), (8.0, 5.0)],
                                [(8.5, 2.0), (6.0, 2.0), (6.0, 2.5), (8.5, 2.5)]]
    unexpected_obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in unexpected_obstacle_list]

    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    obstacles.extend(unexpected_obstacles)
    return MobileRobot(np.array([0.6, 3.5, 0.0, 0, 0])),Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)]), obstacles, Goal((15.4, 3.5))

def generate_eval_map132() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """

    obstacles_list = [  [(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                        [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                        [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                        [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in obstacles_list]

    unexpected_obstacle_list = [[(6.0, 4.0), (6.0, 4.5), (7.5, 4.5), (7.5, 4.0)],
                                [(7.5, 4.5), (7.5, 2.0), (7.0, 2.0), (7.0, 4.5)],
                                [(7.5, 2.0), (6.0, 2.0), (6.0, 2.5), (7.5, 2.5)]]
    unexpected_obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in unexpected_obstacle_list]

    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    obstacles.extend(unexpected_obstacles)
    return MobileRobot(np.array([0.6, 3.5, 0.0, 0, 0])),Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)]), obstacles, Goal((15.4, 3.5))

def generate_eval_map133() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """

    obstacles_list = [  [(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                        [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                        [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                        [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in obstacles_list]

    unexpected_obstacle_list = [[(6.0, 5.0), (9.5, 5.0), (9.5, 3.5), (9.0, 3.5)],
                                [(9.5, 3.5), (9.5, 2.0), (6.0, 2.0), (9.0 ,3.5)]]
    unexpected_obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in unexpected_obstacle_list]

    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    obstacles.extend(unexpected_obstacles)
    return MobileRobot(np.array([0.6, 3.5, 0.0, 0, 0])),Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)]), obstacles, Goal((15.4, 3.5))

def generate_eval_map134() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """

    obstacles_list = [  [(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                        [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                        [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                        [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in obstacles_list]

    unexpected_obstacle_list = [[(6.5, 4.5), (8.5, 4.5), (8.5, 3.5), (8.0, 3.5)],
                                [(8.5, 3.5), (8.5, 2.5), (6.5, 2.5), (8.0, 3.5)]]
    unexpected_obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in unexpected_obstacle_list]

    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    obstacles.extend(unexpected_obstacles)
    return MobileRobot(np.array([0.6, 3.5, 0.0, 0, 0])),Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)]), obstacles, Goal((15.4, 3.5))

def generate_eval_map141() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """

    obstacles_list = [  [(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                        [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                        [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                        [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in obstacles_list]

    unexpected_obstacles = []
    unexpected_obstacle = Obstacle.create_mpc_dynamic_old(p1=(15.4, 3.5), p2=(0.6, 3.5), freq=0.15, rx=0.8, ry=0.8, angle=0.0, corners=20)
    unexpected_obstacles.append(unexpected_obstacle)

    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    obstacles.extend(unexpected_obstacles)
    return MobileRobot(np.array([0.6, 3.5, 0.0, 0, 0])),Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)]), obstacles, Goal((15.4, 3.5))

def generate_eval_map142() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """

    obstacles_list = [  [(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                        [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                        [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                        [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in obstacles_list]

    unexpected_obstacles = []
    unexpected_obstacle = Obstacle.create_mpc_dynamic_old(p1=(10.0, 1.0), p2=(10.0, 9.0), freq=0.2, rx=0.8, ry=0.8, angle=0.0, corners=20)
    unexpected_obstacles.append(unexpected_obstacle)
    
    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    obstacles.extend(unexpected_obstacles)
    return MobileRobot(np.array([0.6, 3.5, 0.0, 0, 0])),Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)]), obstacles, Goal((15.4, 3.5))

def generate_eval_map143() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """

    obstacles_list = [  [(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
                        [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
                        [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
                        [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
    obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in obstacles_list]

    unexpected_obstacles = []
    unexpected_obstacle = Obstacle.create_mpc_static([(7.2, 2.8), (7.2, 4.2), (8.8, 4.2), (8.8, 2.8)]) # medium
    unexpected_obstacles.append(unexpected_obstacle)
    unexpected_obstacle = Obstacle.create_mpc_dynamic_old(p1=(10.0, 1.0), p2=(10.0, 9.0), freq=0.2, rx=0.8, ry=0.8, angle=0.0, corners=20)
    unexpected_obstacles.append(unexpected_obstacle)
    
    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    obstacles.extend(unexpected_obstacles)
    return MobileRobot(np.array([0.6, 3.5, 0.0, 0, 0])),Boundary([(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)]), obstacles, Goal((15.4, 3.5))


def generate_eval_maps() -> MapDescription:
    return random.choice([generate_eval_map111,generate_eval_map112,generate_eval_map113,generate_eval_map121,generate_eval_map122,generate_eval_map123,generate_eval_map124,generate_eval_map131,generate_eval_map132,generate_eval_map133,generate_eval_map134,generate_eval_map141,generate_eval_map142,generate_eval_map143])()
"""
This file is used to generate the maps which the DRL agent will be trained and
evaluated in. Such a map constitutes e.g. the initial robot position, the goal
position and the locations of obstacles and boundaries.
"""

import math
import random

import shapely.ops
from shapely.geometry import LineString, Polygon, JOIN_STYLE, Point
import numpy as np
from math import pi, radians, cos, sin

from ..environment import MobileRobot, Obstacle, Boundary, Goal, MapDescription, MapGenerator
from .map_multi_robot import generate_map_multi_robot3
from .map import generate_map_corridor
from typing import Union, List, Tuple

# def generate_eval_map111() -> MapDescription:
#     """
#     Generates a randomized map with many dynamic obstacles
#     """

#     ob_list = [ #[-1, -1],
#                 #[0, 2],
#                 #[4.0, 2.0],
#                 [5.0, 4.0],
#                 #[5.0, 5.0],
#                 [5.0, 6.0],
#                 #[5.0, 9.0],
#                 [8.0, 9.0],
#                 #[7.0, 9.0],
#                 #[8.0, 10.0],
#                 #[9.0, 11.0],
#                 [12.0, 13.0],
#                 #[12.0, 12.0],
#                 [15.0, 15.0],
#                 #[13.0, 13.0]
#                 ]
#     init_state = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])
#     atr = MobileRobot(init_state)
#     boundary = Boundary([(-3.0, -3.0), (16.0, -3.0), (16.0, 16.0), (-3.0, 16.0)])
#     obstacles = []
#     unexpected_obstacles = []
#     #scene_1_obstacles_list = [[(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
#     #                        [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
#     #                        [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
#     #                        [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
#     #obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in scene_1_obstacles_list]
#     goal = Goal((10, 10))
#     # unexpected_obstacle = Obstacle.create_mpc_dynamic_old(p1=(15.4, 3.5), p2=(0.6, 3.5), freq=0.0, rx=0.5, ry=0.5, angle=0.0, corners=20, is_static=True)
#     # unexpected_obstacles.append(unexpected_obstacle)
#     unexpected_obstacles = [Obstacle.create_mpc_static(obstacle,is_circle=True) for obstacle in ob_list]
#     # unexpected_obstacle = Obstacle.create_mpc_static([7.5, 3.0],is_circle=True) # small
#     # unexpected_obstacles.append(unexpected_obstacle)

#     for o in unexpected_obstacles:
#         o.visible_on_reference_path = False

#     obstacles.extend(unexpected_obstacles)
#     return atr, boundary, obstacles, goal

# def generate_eval_map111x() -> MapDescription:
#     """
#     Generates a randomized map with many dynamic obstacles
#     """
#     offset = 0
#     ob_list = [ [5.0+offset, -1.0],
#                 [6.0+offset, -1.0],
#                 [7.0+offset, -1.0],
#                 [7.0+offset, 0.0],
#                 [7.0+offset, 1.0],
#                 [6.0+offset, 1.0],
#                 [5.0+offset, 1.0],
#                 ]
#     init_state = np.array([-3.0, 0.0, 0.0, 0.0, 0.0])
#     atr = MobileRobot(init_state)
#     boundary = Boundary([(-5.0, -7.0), (16.0, -7.0), (16.0, 7.0), (-5.0, 7.0)])
#     obstacles = []
#     unexpected_obstacles = []
#     #scene_1_obstacles_list = [[(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
#     #                        [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
#     #                        [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
#     #                        [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
#     #obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in scene_1_obstacles_list]
#     goal = Goal((14, 0))
#     # unexpected_obstacle = Obstacle.create_mpc_dynamic_old(p1=(15.4, 3.5), p2=(0.6, 3.5), freq=0.0, rx=0.5, ry=0.5, angle=0.0, corners=20, is_static=True)
#     # unexpected_obstacles.append(unexpected_obstacle)
#     unexpected_obstacles = [Obstacle.create_mpc_static(obstacle,is_circle=True) for obstacle in ob_list]
#     # unexpected_obstacle = Obstacle.create_mpc_static([7.5, 3.0],is_circle=True) # small
#     # unexpected_obstacles.append(unexpected_obstacle)

#     for o in unexpected_obstacles:
#         o.visible_on_reference_path = False

#     obstacles.extend(unexpected_obstacles)
#     return atr, boundary, obstacles, goal

# def generate_eval_map111y() -> MapDescription:
#     """
#     Generates a randomized map with many dynamic obstacles
#     """
#     offset = 12
#     ob_list_vert = [ #[0.0, -6.0],
#                 #[2.4, -6.0],
#                 #[4.8, -6.0],
#                 [7.2, -6.0],
#                 [9.6, -6.0],
#                 ]
    
#     ob_list_hor = [ [0.3, -1.0],
#                     [-0.3, 1.0]]
    
#     init_state = np.array([-3.0, 0.0, 0.0, 0.0, 0.0])
#     atr = MobileRobot(init_state)
#     boundary = Boundary([(-5.0, -7.0), (16.0, -7.0), (16.0, 7.0), (-5.0, 7.0)])
#     obstacles = []
#     unexpected_obstacles = []
#     #scene_1_obstacles_list = [[(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
#     #                        [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
#     #                        [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
#     #                        [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],]
#     #obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in scene_1_obstacles_list]
#     goal = Goal((14, 0))
#     unexpected_obstacles_vert = [Obstacle.create_mpc_dynamic_old(p1=(o[0], o[1]), p2=(o[0], o[1]+offset), freq=0.05+np.random.random()*0.1, rx=0.5, ry=0.5, angle=0.0, corners=20,offset_val=o[0]) for o in ob_list_vert]
#     unexpected_obstacles.extend(unexpected_obstacles_vert)
#     unexpected_obstacles_hor = [Obstacle.create_mpc_dynamic_old(p1=(o[0], o[1]), p2=(o[0]+offset, o[1]), freq=0.07+np.random.random()*0.1, rx=0.5, ry=0.5, angle=0.0, corners=20,offset_val=o[0]+1) for o in ob_list_hor]
#     unexpected_obstacles.extend(unexpected_obstacles_hor)
#     # unexpected_obstacles = [Obstacle.create_mpc_static(obstacle,is_circle=True) for obstacle in ob_list]
#     # unexpected_obstacle = Obstacle.create_mpc_static([7.5, 3.0],is_circle=True) # small
#     # unexpected_obstacles.append(unexpected_obstacle)

#     for o in unexpected_obstacles:
#         o.visible_on_reference_path = False

#     obstacles.extend(unexpected_obstacles)
#     return atr, boundary, obstacles, goal

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
    unexpected_obstacle = Obstacle.create_mpc_dynamic_old(p1=(15.4, 3.5), p2=(0.6, 3.5), freq=0.0, rx=0.5, ry=0.5, angle=0.0, corners=20)
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


def generate_eval_map151() -> MapDescription:
    """
    Generates a map with a narrow corridor and dynamic obstacles, 
    used for evaluating which model is best.
    """

    max_angle = math.pi / 2

    wall_padding = 5
    corridor_padding = 1.5

    coords = np.asarray([(0, 0), (wall_padding, 0)])
    angle = 0

    dangle = np.array([-44.2292,56.5713,-46.1655,77.2675,-27.0029])*pi/180
    length = [7.7585,4.0423,5.5116,3.3429,6.5076]

    for i in range(5):
        lo = -max_angle - angle
        hi = max_angle - angle
        dangle[i] = dangle[i]**2 / (hi if dangle[i] > 0 else lo)
        angle += dangle[i]
        coords = np.vstack((coords, coords[i + 1, :] + length[i]*np.asarray((cos(angle), sin(angle)))))
    coords = np.vstack((coords, coords[-1, :] + (wall_padding, 0)))
    
    corridor = LineString(coords)
    
    minx, miny, maxx, maxy = corridor.bounds

    wall_padding = 5
    pminx = minx - wall_padding
    pminy = miny - wall_padding-5
    pmaxx = maxx + wall_padding
    pmaxy = maxy + wall_padding

    boundary = Boundary([(pminx-20, pminy), (pmaxx+25, pminy), (25+pmaxx, pmaxy), (pminx-20, pmaxy)])

    init_state = np.array([pminx-15, 3, -pi/3, 0, 0])
    robot = MobileRobot(init_state)
    goal = Goal((18.5+pmaxx, -10))

    pminx = minx + wall_padding
    pmaxx = maxx - wall_padding

    obstacles = []

    obstacles.append(Obstacle.create_mpc_static([(-14, -5), (-8, -5), (-8, 7), (-14, 7)]))
    
    
    # obstacles.append(Obstacle.create_mpc_dynamic((-18, -4), (-15, 0), 0.4, 0.7, 0.5, pi/4, random = False))
    obstacles.append(Obstacle.create_mpc_dynamic((-10, -10), (-10, -7), 0.4, 0.7, 1, 0, random = False))
    obstacles.append(Obstacle.create_mpc_dynamic((1, -7), (-4, 1), 0.2, 1, 0.5, -pi/4, random = False))
    # obstacles.append(Obstacle.create_mpc_dynamic((30,2), (40,5), 0.2, 1, 1, pi/3, random = False))
    
    obstacles.append(Obstacle.create_non_convex_u_shape((54.25,-2.5), (54.25,-2.5), 0.3, pi+pi/8, use_random=False))
    obstacles.append(Obstacle.create_mpc_static([(42, -14), (50, -14), (50, 5), (42, 5)]))

    if pminx < pmaxx:
        box = Polygon([(pminx, pminy), (pmaxx, pminy), (pmaxx, pmaxy), (pminx, pmaxy)])
        left = corridor.parallel_offset(corridor_padding, 'left', join_style=JOIN_STYLE.mitre, mitre_limit=1)
        right = corridor.parallel_offset(corridor_padding, 'right', join_style=JOIN_STYLE.mitre, mitre_limit=1)

        eps = 1e-3

        split = shapely.ops.split(box, right)
        test = Point((pminx + eps, pminy + eps))
        for geom in split.geoms:
            if geom.contains(test):
                obstacles.append(Obstacle.create_mpc_static(geom.exterior.coords[:-1]))
                break
        
        split = shapely.ops.split(box, left)
        test = Point((pminx + eps, pmaxy - eps))
        for geom in split.geoms:
            if geom.contains(test):
                obstacles.append(Obstacle.create_mpc_static(geom.exterior.coords[:-1]))
                break
    
    return robot, boundary, obstacles, goal


def generate_eval_map152() -> MapDescription:
    """
    Generates a simplified map with a corridor and fewer obstacles.
    """

    max_angle = math.pi / 2

    wall_padding = 5
    corridor_padding = 1.5

    coords = np.asarray([(0, 0), (wall_padding, 0)])
    angle = 0

    dangle = np.array([-44.2292,56.5713,-46.1655,77.2675,-27.0029])*pi/180
    length = [7.7585,4.0423,5.5116,3.3429,6.5076]

    for i in range(5):
        lo = -max_angle - angle
        hi = max_angle - angle
        dangle[i] = dangle[i]**2 / (hi if dangle[i] > 0 else lo)
        angle += dangle[i]
        coords = np.vstack((coords, coords[i + 1, :] + length[i]*np.asarray((cos(angle), sin(angle)))))
    coords = np.vstack((coords, coords[-1, :] + (wall_padding, 0)))
    
    corridor = LineString(coords)
    
    minx, miny, maxx, maxy = corridor.bounds

    wall_padding = 5
    pminx = minx
    pminy = miny - wall_padding-5
    pmaxx = maxx
    pmaxy = maxy + wall_padding

    boundary = Boundary([(pminx-10, pminy), (pmaxx+15, pminy), (15+pmaxx, pmaxy), (pminx-10, pmaxy)])

    init_state = np.array([pminx-5, 3, -pi/3, 0, 0])
    robot = MobileRobot(init_state)
    goal = Goal((5+pmaxx, -10))

    pminx = minx + wall_padding
    pmaxx = maxx - wall_padding

    obstacles = []

    obstacles.append(Obstacle.create_mpc_dynamic((5,-0.5), (5,3), 0.1, 1, 1, pi/3, random = False))    
    obstacles.append(Obstacle.create_non_convex_u_shape((36,-5.5), (36,-5.5), 0.3, pi+pi/5, use_random=False))

    if pminx < pmaxx:
        box = Polygon([(pminx, pminy), (pmaxx, pminy), (pmaxx, pmaxy), (pminx, pmaxy)])
        left = corridor.parallel_offset(corridor_padding, 'left', join_style=JOIN_STYLE.mitre, mitre_limit=1)
        right = corridor.parallel_offset(corridor_padding, 'right', join_style=JOIN_STYLE.mitre, mitre_limit=1)

        eps = 1e-3

        split = shapely.ops.split(box, right)
        test = Point((pminx + eps, pminy + eps))
        for geom in split.geoms:
            if geom.contains(test):
                obstacles.append(Obstacle.create_mpc_static(geom.exterior.coords[:-1]))
                break
        
        split = shapely.ops.split(box, left)
        test = Point((pminx + eps, pmaxy - eps))
        for geom in split.geoms:
            if geom.contains(test):
                obstacles.append(Obstacle.create_mpc_static(geom.exterior.coords[:-1]))
                break

    return robot, boundary, obstacles, goal


def generate_eval_maps() -> MapDescription:
    return random.choice([generate_eval_map111,generate_eval_map112,generate_eval_map113,generate_eval_map121,generate_eval_map122,generate_eval_map123,generate_eval_map124,generate_eval_map131,generate_eval_map132,generate_eval_map133,generate_eval_map134,generate_eval_map141,generate_eval_map142,generate_eval_map143])()
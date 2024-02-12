import math
import random
from typing import Union, List, Tuple

import numpy as np

from ..environment import MobileRobot, Obstacle, Boundary, Goal, MapDescription, MapGenerator


def generate_map_scene_0() -> List[MapDescription]:
    """Empty scene + 4 robots
    """
    robot_1 = MobileRobot(np.array([1.0, 1.0, math.radians(45), 0, 0]))
    goal_1 = Goal((9.0, 9.0))

    robot_2 = MobileRobot(np.array([9.0, 9.0, math.radians(-135), 0, 0]))
    goal_2 = Goal((1.0, 1.0))

    robot_3 = MobileRobot(np.array([1.0, 9.0, math.radians(-45), 0, 0]))
    goal_3 = Goal((9.0, 1.0))

    robot_4 = MobileRobot(np.array([9.0, 1.0, math.radians(135), 0, 0]))
    goal_4 = Goal((1.0, 9.0))

    scene_boundary = Boundary([(-5.0, -5.0), (-5.0, 15.0), (15.0, 15.0), (15.0, -5.0)])
    scene_obstacles_list = [[(-1.0, -1.0), (-1.1, -1.0), (-1.1, -1.1), (-1.0, -1.1)]]
    scene_obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in scene_obstacles_list]

    scene_for_1 = (robot_1, scene_boundary, scene_obstacles, goal_1)
    scene_for_2 = (robot_2, scene_boundary, scene_obstacles, goal_2)
    scene_for_3 = (robot_3, scene_boundary, scene_obstacles, goal_3)
    scene_for_4 = (robot_4, scene_boundary, scene_obstacles, goal_4)

    return [scene_for_1, scene_for_2, scene_for_3, scene_for_4]


def generate_map_scene_1() -> List[MapDescription]:
    """1 obstacle in the middle of the scene + 4 robots
    """
    robot_1 = MobileRobot(np.array([1.0, 1.0, math.radians(45), 0, 0]))
    goal_1 = Goal((9.0, 9.0))

    robot_2 = MobileRobot(np.array([9.0, 9.0, math.radians(-135), 0, 0]))
    goal_2 = Goal((1.0, 1.0))

    robot_3 = MobileRobot(np.array([1.0, 9.0, math.radians(-45), 0, 0]))
    goal_3 = Goal((9.0, 1.0))

    robot_4 = MobileRobot(np.array([9.0, 1.0, math.radians(135), 0, 0]))
    goal_4 = Goal((1.0, 9.0))

    scene_boundary = Boundary([(-5.0, -5.0), (-5.0, 15.0), (15.0, 15.0), (15.0, -5.0)])
    scene_obstacles_list = [[(4.0, 4.0), (4.0, 6.0), (6.0, 6.0), (6.0, 4.0)]]
    scene_obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in scene_obstacles_list]

    unexpected_obstacles = scene_obstacles
    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    scene_for_1 = (robot_1, scene_boundary, scene_obstacles, goal_1)
    scene_for_2 = (robot_2, scene_boundary, scene_obstacles, goal_2)
    scene_for_3 = (robot_3, scene_boundary, scene_obstacles, goal_3)
    scene_for_4 = (robot_4, scene_boundary, scene_obstacles, goal_4)

    return [scene_for_1, scene_for_2, scene_for_3, scene_for_4]


def generate_map_scene_2() -> List[MapDescription]:
    """Plus-shaped non-convex obstacle + 4 robots
    """
    robot_1 = MobileRobot(np.array([1.0, 1.0, math.radians(45), 0, 0]))
    goal_1 = Goal((9.0, 9.0))

    robot_2 = MobileRobot(np.array([9.0, 9.0, math.radians(-135), 0, 0]))
    goal_2 = Goal((1.0, 1.0))

    robot_3 = MobileRobot(np.array([1.0, 9.0, math.radians(-45), 0, 0]))
    goal_3 = Goal((9.0, 1.0))

    robot_4 = MobileRobot(np.array([9.0, 1.0, math.radians(135), 0, 0]))
    goal_4 = Goal((1.0, 9.0))

    scene_boundary = Boundary([(-5.0, -5.0), (-5.0, 15.0), (15.0, 15.0), (15.0, -5.0)])
    scene_obstacles_list = [[(4.5, 3.0), (4.5, 7.0), (5.5, 7.0), (5.5, 3.0)],
                            [(3.0, 4.5), (3.0, 5.5), (7.0, 5.5), (7.0, 4.5)]]
    scene_obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in scene_obstacles_list]

    unexpected_obstacles = scene_obstacles
    for o in unexpected_obstacles:
        o.visible_on_reference_path = False

    scene_for_1 = (robot_1, scene_boundary, scene_obstacles, goal_1)
    scene_for_2 = (robot_2, scene_boundary, scene_obstacles, goal_2)
    scene_for_3 = (robot_3, scene_boundary, scene_obstacles, goal_3)
    scene_for_4 = (robot_4, scene_boundary, scene_obstacles, goal_4)

    return [scene_for_1, scene_for_2, scene_for_3, scene_for_4]
    

def generate_map_scene_3() -> List[MapDescription]:
    """Empty scene + 8 robots
    """
    num_robots = 8
    base_degree = 360 / num_robots

    scene_boundary = Boundary([(-5.0, -5.0), (-5.0, 15.0), (15.0, 15.0), (15.0, -5.0)])
    scene_obstacles_list = [[(-4.0, -4.0), (-4.1, -4.0), (-4.1, -4.1), (-4.0, -4.1)]]
    scene_obstacles = [Obstacle.create_mpc_static(obstacle) for obstacle in scene_obstacles_list]

    scene_list = []
    for i in range(num_robots):
        x = 5.0 + 8.0 * math.cos(math.radians(base_degree * i))
        y = 5.0 + 8.0 * math.sin(math.radians(base_degree * i))
        robot = MobileRobot(np.array([x, y, 90+math.radians(base_degree * i), 0, 0]))
        goal = Goal((5.0 - 8.0 * math.cos(math.radians(base_degree * i)), 
                     5.0 - 8.0 * math.sin(math.radians(base_degree * i))))
        scene_list.append((robot, scene_boundary, scene_obstacles, goal))

    return scene_list
    




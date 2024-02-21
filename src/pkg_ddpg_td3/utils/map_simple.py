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

### Training maps ###

def generate_simple_map_easy() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstacles
    """
    direction = np.random.choice(['N','E','S','W'])
    if direction == 'N':
        x_init =        random.uniform(1,19)
        y_init = 2 +    random.uniform(-1,1)
        x_goal =        random.uniform(1,19)
        y_goal = 18 +   random.uniform(-1,1)
    elif direction == 'E':
        x_init = 2 +    random.uniform(-1,1)
        y_init =        random.uniform(1,19)
        x_goal = 18 +   random.uniform(-1,1)
        y_goal =        random.uniform(1,19)
    elif direction == 'S':
        x_init =        random.uniform(1,19)
        y_init = 18 +   random.uniform(-1,1)
        x_goal =        random.uniform(1,19)
        y_goal = 2 +    random.uniform(-1,1)
    else:
        x_init = 18 +   random.uniform(-1,1)
        y_init =        random.uniform(1,19)
        x_goal = 2 +    random.uniform(-1,1)
        y_goal =        random.uniform(1,19)
    
    init_state = np.array([x_init, y_init, random.uniform(-pi, pi), 0, 0])
    atr = MobileRobot(init_state)
    boundary = Boundary([(0, 0), (20, 0), (20, 20), (0, 20)])
    obstacles = []

    goal = Goal((x_goal, y_goal))

    return atr, boundary, obstacles, goal

def generate_simple_map_static1() -> MapDescription:
    """
    Generates a randomized map with one static obstacle
    """

    init_state = np.array([2.6 + random.uniform(-2,2), random.uniform(0.6,19.4), random.uniform(-pi, pi), 0, 0])
    robot = MobileRobot(init_state)
    boundary = Boundary([(0, 0), (40, 0), (40, 20), (0, 20)])
    obstacles = []
    x_top = random.uniform(15.5,17)
    x_bot = random.uniform(13,14.5)
    y_top = random.uniform(12,18)
    y_bot = random.uniform(2,8)
    obstacles.append(Obstacle.create_mpc_static([(x_bot, y_top), 
                                                 (x_bot, y_bot), 
                                                 (x_top, y_bot),
                                                 (x_top, y_top)]))
    goal = Goal((38.5 + random.uniform(-1,1), random.uniform(0.6,19.4)))
    if random.random() < 0.5:
        obstacles[0].visible_on_reference_path=False
    return robot, boundary, obstacles, goal

def generate_simple_map_static2() -> MapDescription:
    """
    Generates a randomized map with one static obstacle
    """
    start_x = 2.6 + random.uniform(-2,1)
    start_y = random.uniform(3,17)
    goal_x = 37.5 + random.uniform(-1,2)
    goal_y = random.uniform(3,17)

    init_state = np.array([start_x, start_y, random.uniform(-pi, pi), 0, 0])
    robot = MobileRobot(init_state)
    boundary = Boundary([(0, 0), (40, 0), (40, 20), (0, 20)])
    obstacles = []

    wall_center = random.uniform(15, 30)
    r = random.uniform(0,3)
    port_center = random.uniform(3,17)
    tunnel = random.uniform(0.7,1.2)

    obstacles.append(Obstacle.create_mpc_static([   (wall_center - r, 20), 
                                                    (wall_center - r, port_center + 1), 
                                                    (wall_center + r, port_center + 1),
                                                    (wall_center + r, 20)]))
    
    obstacles.append(Obstacle.create_mpc_static([   (wall_center - r, 0), 
                                                    (wall_center - r, port_center - 1), 
                                                    (wall_center + r, port_center - 1),
                                                    (wall_center + r, 0)]))
    offset = random.uniform(1.5,4.5)
    thickness = random.uniform(1,3)
    height = random.uniform(0.5,2)
    if random.random() < 0.5:
        obstacles.append(Obstacle.create_mpc_static([   (wall_center - r - offset, 0), 
                                                        (wall_center - r - offset, port_center + 1), 
                                                        (wall_center - r - offset - thickness, port_center + 1),
                                                        (wall_center - r - offset - thickness, 0)]))
    else:
        obstacles.append(Obstacle.create_mpc_static([   (wall_center - r - offset, 20), 
                                                        (wall_center - r - offset, port_center - 1), 
                                                        (wall_center - r - offset - thickness, port_center - 1),
                                                        (wall_center - r - offset - thickness, 20)]))

    goal = Goal((goal_x, goal_y))
    obstacles[-1].visible_on_reference_path=random.choice([True,False])

    rx = random.uniform(0.2, 1.2)
    ry = random.uniform(0.2, 1.2)
    freq = 1/math.sqrt(((goal_x-(wall_center+r))**2+(goal_y-port_center)**2))*random.uniform(1,2)
    angle = math.atan2(port_center-goal_y,wall_center + r - goal_x)
    obstacles.append(Obstacle.create_mpc_dynamic((goal_x, goal_y), (wall_center+r, port_center), freq, rx, ry, angle, random = False))

    return robot, boundary, obstacles, goal


def generate_simple_map_static3() -> MapDescription:
    """
    Generates a randomized map with one dynamic obstacle
    """

    init_state = np.array([2 + random.uniform(-1,0), random.uniform(1,29), random.uniform(-pi, pi), 0, 0])
    robot = MobileRobot(init_state)
    boundary = Boundary([(0, 0), (30, 0), (30, 30), (0, 30)])
    obstacles = []
    r=0.5
    for i in np.arange(4,26,2):
        for j in np.arange(4,26,2):
            if random.random() < 0.15:
                obstacles.append(Obstacle.create_mpc_static([(i-r, j+r), 
                                                            (i-r, j-r), 
                                                            (i+r, j-r),
                                                            (i+r, j+r)]))
    goal = Goal((28 + random.uniform(-1,1), random.uniform(1,29)))
    for o in obstacles:
        if random.random() < 0.5:
            o.visible_on_reference_path=False
    return robot, boundary, obstacles, goal

def generate_simple_map_static4() -> MapDescription:
    """
    Generates a randomized maze-like map
    """

    init_state = np.array([random.uniform(0.6,2), random.uniform(0.6,19.4), random.uniform(-pi, pi), 0, 0])
    robot = MobileRobot(init_state)
    boundary = Boundary([(0, 0), (25, 0), (25, 20), (0, 20)])
    obstacles = []
    walls = [5,10,15,20]

    last_right = False

    for wall in walls:
        center = random.uniform(2,18)
        offset = random.uniform(0,2)
        obstacles.append(Obstacle.create_mpc_static([   ( wall - 1, 20), 
                                                        ( wall + 1, 20), 
                                                        ( wall + 1, center + 1),
                                                        ( wall - 1, center + 1)]))
        obstacles.append(Obstacle.create_mpc_static([   ( wall - 1, 0), 
                                                        ( wall + 1, 0), 
                                                        ( wall + 1, center - 1),
                                                        ( wall - 1, center - 1)]))
        
        if not last_right and random.choice([True,False]):
            if center < 10:
                obstacles.append(Obstacle.create_mpc_static([( wall - 2, center + 2 + offset), 
                                                            ( wall - 1, center + 2 + offset), 
                                                            ( wall - 1, center + 1 + offset),
                                                            ( wall - 2, center + 1 + offset)]))
            else:
                obstacles.append(Obstacle.create_mpc_static([( wall - 2, center - 2 - offset), 
                                                            ( wall - 1, center - 2 - offset), 
                                                            ( wall - 1, center - 1 - offset),
                                                            ( wall - 2, center - 1 - offset)]))
            obstacles[-1].visible_on_reference_path=False

        if random.choice([True,False]):
            if center < 10:
                obstacles.append(Obstacle.create_mpc_static([( wall + 2, center + 2 + offset), 
                                                            ( wall + 1, center + 2 + offset), 
                                                            ( wall + 1, center + 1 + offset),
                                                            ( wall + 2, center + 1 + offset)]))
            else:
                obstacles.append(Obstacle.create_mpc_static([( wall + 2, center - 2 - offset), 
                                                            ( wall + 1, center - 2 - offset), 
                                                            ( wall + 1, center - 1 - offset),
                                                            ( wall + 2, center - 1 - offset)]))
            obstacles[-1].visible_on_reference_path=False
            last_right = True
        else:
            last_right = False
    
    goal = Goal((23.4 + random.uniform(0,1), random.uniform(0.6,19.4)))
    return robot, boundary, obstacles, goal



def generate_simple_map_nonconvex_U() -> MapDescription:
    """
    Generates a randomized map with with one U-shape obstacle
    """
    start_x = 5 + random.uniform(-4.4,2)
    start_y = random.uniform(0.6,19.4)
    goal_x = 35 + random.uniform(-2,4.4)
    goal_y = random.uniform(0.6,19.4)

    init_state = np.array([start_x, start_y, random.uniform(-pi, pi), 0, 0])
    robot = MobileRobot(init_state)
    boundary = Boundary([(0, 0), (40, 0), (40, 20), (0, 20)])
    obstacles = []

    angle = math.atan2(start_y-goal_y,start_x-goal_x) + pi/2

    obstacles.append(Obstacle.create_non_convex_u_shape(((goal_x+start_x)/2,(goal_y+start_y)/2),((goal_x+start_x)/2,(goal_y+start_y)/2),0.0,angle))


    goal = Goal((goal_x, goal_y))

    return robot, boundary, obstacles, goal

def generate_simple_map_nonconvex_L() -> MapDescription:
    """
    Generates a randomized map with with one L-shape obstacle
    """
    start_x = 5 + random.uniform(-4.4,2)
    start_y = random.uniform(0.6,19.4)
    goal_x = 35 + random.uniform(-2,4.4)
    goal_y = random.uniform(0.6,19.4)

    init_state = np.array([start_x, start_y, random.uniform(-pi, pi), 0, 0])
    robot = MobileRobot(init_state)
    boundary = Boundary([(0, 0), (40, 0), (40, 20), (0, 20)])
    obstacles = []

    angle = math.atan2(start_y-goal_y,start_x-goal_x) - random.uniform(0,pi/2)

    obstacles.append(Obstacle.create_non_convex_l_shape(((goal_x+start_x)/2,(goal_y+start_y)/2),((goal_x+start_x)/2,(goal_y+start_y)/2),0.0,angle))


    goal = Goal((goal_x, goal_y))

    return robot, boundary, obstacles, goal

def generate_simple_map_nonconvex_static() -> MapDescription:
    """
    Generates a randomized map with with one L-shape obstacle
    """
    start_x = 5 + random.uniform(-4.4,2)
    start_y = random.uniform(0.6,19.4)
    goal_x = 35 + random.uniform(-2,4.4)
    goal_y = random.uniform(0.6,19.4)

    init_state = np.array([start_x, start_y, random.uniform(-pi, pi), 0, 0])
    robot = MobileRobot(init_state)
    boundary = Boundary([(0, 0), (40, 0), (40, 20), (0, 20)])
    obstacles = []

    wall_center = random.uniform(start_x + 7, goal_x - 11)
    r = random.uniform(0,3)
    port_center = random.uniform(3,17)

    obstacles.append(Obstacle.create_mpc_static([   (wall_center - r, 20), 
                                                    (wall_center - r, port_center + 1), 
                                                    (wall_center + r, port_center + 1),
                                                    (wall_center + r, 20)]))
    
    obstacles.append(Obstacle.create_mpc_static([   (wall_center - r, 0), 
                                                    (wall_center - r, port_center - 1), 
                                                    (wall_center + r, port_center - 1),
                                                    (wall_center + r, 0)]))

    if np.random.choice([True,False]):
        angle = math.atan2(port_center-goal_y,wall_center + r - goal_x) - random.uniform(0,pi/2)
        obstacles.append(Obstacle.create_non_convex_l_shape(((goal_x+wall_center+r)/2,(goal_y+port_center)/2),((goal_x+wall_center+r)/2,(goal_y+port_center)/2),0.0,angle))
    else:
        angle = math.atan2(port_center-goal_y,wall_center + r - goal_x) + pi/2
        obstacles.append(Obstacle.create_non_convex_u_shape(((goal_x+wall_center+r)/2,(goal_y+port_center)/2),((goal_x+wall_center+r)/2,(goal_y+port_center)/2),0.0,angle))


    goal = Goal((goal_x, goal_y))

    return robot, boundary, obstacles, goal

def generate_simple_map_dynamic1() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstalces
    """

    init_state = np.array([2 + random.uniform(-1,1), random.uniform(2,18), random.uniform(-pi, pi), 0, 0])
    robot = MobileRobot(init_state)
    boundary = Boundary([(0, 0), (20, 0), (20, 20), (0, 20)])
    obstacles = []
    goal = Goal((18 + random.uniform(-1,1), random.uniform(2,18)))
    num_obstacle = random.randint(3,6)
    for i in range(num_obstacle):
        x1 = random.uniform(4,16)
        y1 = random.uniform(4,16)

        x2 = random.uniform(4,16)
        y2 = random.uniform(4,16)
        rx = random.uniform(0.2, 1.2)
        ry = random.uniform(0.2, 1.2)
        freq = 1/(math.sqrt((x1-x2)**2+(y1-y2)**2))*random.uniform(1,2)
        # freq = random.uniform(0.2, 0.4)
        angle = random.uniform(0, 2 * pi)
        obstacles.append(Obstacle.create_mpc_dynamic((x1, y1), (x2, y2), freq, rx, ry, angle, random = False))
        # obstacles.append(Obstacle.create_dynamic_obstacle((x1, y1), (x2, y2), freq, rx, ry, angle))


    return robot, boundary, obstacles, goal

def generate_simple_map_dynamic2() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstalces
    """
    x_min = 0
    x_max = 40
    y_min = 0
    y_max = 15

    init_state = np.array([x_min + random.uniform(0.5,2), random.uniform(y_min + 2, y_max - 2), random.uniform(-pi, pi), 0, 0])
    robot = MobileRobot(init_state)
    boundary = Boundary([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    obstacles = []
    goal = Goal((x_max - random.uniform(0.5,2), random.uniform(y_min + 2,y_max - 2)))
    num_obstacle = random.randint(4,10)
    delta = x_max//num_obstacle
    for i in range(num_obstacle):
        x1 =  (i+1)*delta + random.uniform(-2,2)
        # x1 = random.uniform(x_min + 5, x_max - 5)
        y1 = y_min + random.uniform(1,3)

        x2 = x1 + random.uniform(-2,2)
        y2 = random.uniform(12,14)
        rx = random.uniform(0.2, 1.2)
        ry = random.uniform(0.2, 1.2)
        freq = 1/(math.sqrt((x1-x2)**2+(y1-y2)**2))*random.uniform(1,2)
        angle = random.uniform(0, 2 * pi)
        obstacles.append(Obstacle.create_mpc_dynamic((x1, y1), (x2, y2), freq, rx, ry, angle, random = False))
        # obstacles.append(Obstacle.create_dynamic_obstacle((x1, y1), (x2, y2), freq, rx, ry, angle))


    return robot, boundary, obstacles, goal

def generate_simple_map_dynamic3() -> MapDescription:
    """
    Generates a randomized map with many dynamic obstalces
    """
    x_min = 0
    x_max = 40
    y_min = 0
    y_max = 20

    init_state = np.array([x_min + random.uniform(0.5,2), random.uniform(y_min + 2, y_max - 2), random.uniform(-pi, pi), 0, 0])
    robot = MobileRobot(init_state)
    boundary = Boundary([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])
    obstacles = []
    goal = Goal((x_max - random.uniform(1,3), random.uniform(y_min + 2,y_max - 2)))
    num_obstacle = random.randint(4,8)
    delta = y_max//num_obstacle
    for i in range(num_obstacle):
        x1 =  x_min + 5
        # x1 = random.uniform(x_min + 5, x_max - 5)
        y1 = i*delta + random.uniform(1,3)

        x2 = x_max - 5
        y2 = y1 + random.uniform(-2,2)
        rx = random.uniform(0.2, 1.2)
        ry = random.uniform(0.2, 1.2)
        freq = 1/(math.sqrt((x1-x2)**2+(y1-y2)**2))*random.uniform(1,3)
        angle = random.uniform(0, 2 * pi)
        obstacles.append(Obstacle.create_mpc_dynamic((x1, y1), (x2, y2), freq, rx, ry, angle, random = False))
        # obstacles.append(Obstacle.create_dynamic_obstacle((x1, y1), (x2, y2), freq, rx, ry, angle))

    return robot, boundary, obstacles, goal

def generate_simple_map_dynamic4() -> MapDescription:
    """
    Generates a randomized map with with one L-shape obstacle
    """
    start_x = 5 + random.uniform(-2,2)
    start_y = random.uniform(3,17)
    goal_x = 37 + random.uniform(-2,2)
    goal_y = random.uniform(3,17)

    init_state = np.array([start_x, start_y, random.uniform(-pi, pi), 0, 0])
    robot = MobileRobot(init_state)
    boundary = Boundary([(0, 0), (40, 0), (40, 20), (0, 20)])
    obstacles = []

    wall_center = random.uniform(start_x + 7, goal_x - 11)
    r = random.uniform(0,3)
    port_center = random.uniform(3,17)

    obstacles.append(Obstacle.create_mpc_static([   (wall_center - r, 20), 
                                                    (wall_center - r, port_center + 1), 
                                                    (wall_center + r, port_center + 1),
                                                    (wall_center + r, 20)]))
    
    obstacles.append(Obstacle.create_mpc_static([   (wall_center - r, 0), 
                                                    (wall_center - r, port_center - 1), 
                                                    (wall_center + r, port_center - 1),
                                                    (wall_center + r, 0)]))



    rx = random.uniform(0.2, 1.2)
    ry = random.uniform(0.2, 1.2)
    freq = 1/math.sqrt(((goal_x-(wall_center+r))**2+(goal_y-port_center)**2))*random.uniform(1,2)
    angle = math.atan2(port_center-goal_y,wall_center + r - goal_x)
    obstacles.append(Obstacle.create_mpc_dynamic((goal_x, goal_y), (wall_center+r, port_center), freq, rx, ry, angle, random = False))



    goal = Goal((goal_x, goal_y))

    return robot, boundary, obstacles, goal

def generate_simple_map_dynamic() -> MapDescription:
    return random.choice([generate_simple_map_dynamic1, generate_simple_map_dynamic2, generate_simple_map_dynamic3, generate_simple_map_dynamic4])()

def generate_simple_map_static() -> MapDescription:
    return random.choice([generate_simple_map_static2, generate_simple_map_static4, generate_map_multi_robot3,generate_map_corridor])()

def generate_simple_map_nonconvex() -> MapDescription:
    return random.choice([generate_simple_map_nonconvex_U, generate_simple_map_nonconvex_L, generate_simple_map_nonconvex_static])()
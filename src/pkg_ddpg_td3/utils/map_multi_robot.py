import math
import random

import numpy as np
from math import pi, radians, cos, sin

from ..environment import MobileRobot, Obstacle, Boundary, Goal, MapDescription, MapGenerator

from typing import Union, List, Tuple


rot_90 = np.array([[0,-1],[1,0]])
rot_180 = np.array([[-1,0],[0,-1]])
rot_270 = np.array([[0,1],[-1,0]])

def generate_map_multi_robot1() -> MapDescription:
    """
    Generates a randomized map with one static obstacle
    """
    
    box_resolution = 9
    nest_size = 3
    
    

    map_size = nest_size*box_resolution

    nodes = np.array([(0,0),(0,nest_size),(nest_size,nest_size),(nest_size,nest_size - 0.2),(0.2,nest_size - 0.2),(0.2,0.2),(nest_size,0.2),(nest_size,0)]).T


    boundary = Boundary([(0, 0), (map_size, 0), (map_size, map_size), (0, map_size)])
    positions = []

    obstacles = []
    for i in range(map_size//nest_size):
        if i != 0 and i != map_size//nest_size-1:
            obstacles.append(Obstacle.create_mpc_static(list(map(tuple,(nodes.T+np.array([0,i*nest_size]))))))
            positions.append((nest_size/2,i*nest_size+nest_size/2))

            obstacles.append(Obstacle.create_mpc_static(list(map(tuple,(rot_90@nodes+np.array([(i+1)*nest_size,0]).reshape(2,1)).T))))
            positions.append((i*nest_size+nest_size/2,nest_size/2))

            obstacles.append(Obstacle.create_mpc_static(list(map(tuple,(rot_180@nodes+np.array([map_size,(i+1)*nest_size]).reshape(2,1)).T))))
            positions.append((map_size-nest_size/2,i*nest_size + nest_size/2))

            obstacles.append(Obstacle.create_mpc_static(list(map(tuple,(rot_270@nodes+np.array([i*nest_size,map_size]).reshape(2,1)).T))))
            positions.append((i*nest_size+nest_size/2,map_size - nest_size/2))

            if i>2 and i < map_size//nest_size - 3 and box_resolution > 6:
                obstacles.append(Obstacle.create_mpc_static(list(map(tuple,(nodes.T+np.array([map_size - 3*nest_size,i*nest_size]))))))
                positions.append((map_size - 2.5*nest_size,i*nest_size + nest_size/2))

                obstacles.append(Obstacle.create_mpc_static(list(map(tuple,(rot_90@nodes+np.array([(i+1)*nest_size,map_size - 3*nest_size]).reshape(2,1)).T))))
                positions.append((i*nest_size + nest_size/2, map_size - 2.5*nest_size))

                obstacles.append(Obstacle.create_mpc_static(list(map(tuple,(rot_180@nodes+np.array([3*nest_size,(i+1)*nest_size]).reshape(2,1)).T))))
                positions.append((2.5*nest_size,i*nest_size + nest_size/2))

                obstacles.append(Obstacle.create_mpc_static(list(map(tuple,(rot_270@nodes+np.array([i*nest_size,3*nest_size]).reshape(2,1)).T))))
                positions.append((i*nest_size + nest_size/2, 2.5*nest_size))


    random.shuffle(positions)
    start_position = positions.pop()

    init_state = np.array([start_position[0], start_position[1], random.uniform(-pi, pi), 0, 0])
    robot = MobileRobot(init_state)
    
    goal = Goal(positions.pop())
    # print(positions)
    return robot, boundary, obstacles, goal


def generate_map_multi_robot2() -> MapDescription:
    """
    Generates a randomized map with one static obstacle
    """
    nest_size = 3
    box_resolution = 7

    nest = np.array([(nest_size,0.1),(0.1,0.1),(0.1,nest_size - 0.1),(nest_size,nest_size - 0.1)]).T
    # nest = np.array([(nest_size,0.1),(nest_size,0.5),(nest_size-0.2,0.5),(nest_size-0.2,0.1),(0.1,0.1),(0.1,nest_size - 0.1),(nest_size-0.2,nest_size - 0.1),(nest_size-0.2,nest_size - 0.5),(nest_size,nest_size - 0.5),(nest_size,nest_size - 0.1)]).T
    
    map_size = nest_size*box_resolution
    nodes = []#[(nest_size,nest_size)]
    positions = []

    obstacles = []
    for i in range(map_size//nest_size): #LEFTSECTION
        if i < map_size//nest_size - 1 and i > 0:
            nodes += list(map(tuple,(nest+np.array([0,i*nest_size]).reshape(2,1)).T))
            positions.append((nest_size/2,i*nest_size+nest_size/2,random.choice([0,pi])))
    # nodes += [(nest_size,size-nest_size)]

    for i in range(map_size//nest_size): #TOPSECTION
        if i < map_size//nest_size - 1 and i > 0:
            nodes += list(map(tuple,(rot_270@nest+np.array([i*nest_size,map_size]).reshape(2,1)).T))
            positions.append((i*nest_size+nest_size/2,map_size - nest_size/2,random.choice([pi/2,-pi/2])))
    # nodes += [(size-nest_size,size-nest_size)]

    for i in range(map_size//nest_size): #RIGHTSECTION
        if i < map_size//nest_size - 1 and i > 0:
            nodes += list(map(tuple,(rot_180@nest+np.array([map_size,(map_size//nest_size-i)*nest_size]).reshape(2,1)).T))
            positions.append((map_size-nest_size/2,(map_size//nest_size-i-1)*nest_size + nest_size/2,random.choice([0,pi])))
    # nodes += [(size-nest_size,nest_size)]

    for i in range(map_size//nest_size): #BOTTOMSECTION
        if i < map_size//nest_size - 1 and i > 0:
            nodes += list(map(tuple,(rot_90@nest+np.array([(map_size//nest_size-i)*nest_size,0]).reshape(2,1)).T))
            positions.append(((map_size//nest_size-i-1)*nest_size+nest_size/2,nest_size/2,random.choice([pi/2,-pi/2])))

    random.shuffle(positions)
    start_position = positions.pop()
    
    init_state = np.array([start_position[0], start_position[1], start_position[2], 0, 0])
    goal_position = positions.pop()

    return MobileRobot(init_state), Boundary(nodes), obstacles, Goal((goal_position[0],goal_position[1]))

def generate_map_multi_robot3() -> MapDescription:
    """
    Generates a randomized map with one static obstacle
    """
    nest_size = 4
    door_size = 2
    box_resolution = 8

    wall = (nest_size-door_size)/2

    # nest = np.array([(nest_size,0.1),(0.1,0.1),(0.1,nest_size - 0.1),(nest_size,nest_size - 0.1)]).T
    nest = np.array([(nest_size,0.1),(nest_size,wall),(nest_size-0.2,wall),(nest_size-0.2,0.1),(0.1,0.1),(0.1,nest_size - 0.1),(nest_size-0.2,nest_size - 0.1),(nest_size-0.2,nest_size - wall),(nest_size,nest_size - wall),(nest_size,nest_size - 0.1)]).T
    map_size = nest_size*box_resolution
    
    nodes = []
    positions = []
    obstacles = []
    
    for i in range(map_size//nest_size): #LEFTSECTION
        if i < map_size//nest_size - 1 and i > 0:
            nodes += list(map(tuple,(nest+np.array([0,i*nest_size]).reshape(2,1)).T))
            positions.append((nest_size/2,i*nest_size+nest_size/2,random.choice([0,pi])))

    for i in range(map_size//nest_size): #TOPSECTION 1
        if i < map_size//(2*nest_size) and i > 0:
            nodes += list(map(tuple,(rot_270@nest+np.array([i*nest_size,map_size]).reshape(2,1)).T))
            positions.append((i*nest_size+nest_size/2,map_size - nest_size/2,random.choice([pi/2,-pi/2])))

    for i in range(map_size//nest_size): #MIDDLE UP 1
        if i < map_size//nest_size - 3 and i > map_size//(2*nest_size) - 1:
            nodes += list(map(tuple,(rot_90@nest+np.array([(map_size//nest_size-i)*nest_size,map_size-3*nest_size]).reshape(2,1)).T))
            positions.append(((map_size//nest_size-i-1)*nest_size+nest_size/2,map_size-2.5*nest_size,random.choice([pi/2,-pi/2])))
    
    nodes += [(3*nest_size-0.1,map_size-2*nest_size),(3*nest_size-0.1,map_size-3*nest_size+0.1),(2*nest_size,map_size-3*nest_size+0.1)]

    for i in range(map_size//nest_size): #MIDDLE LEFT SECTION
        if i < map_size//nest_size - 3 and i > 2:
            nodes += list(map(tuple,(rot_180@nest+np.array([3*nest_size,(map_size//nest_size-i)*nest_size]).reshape(2,1)).T))
            positions.append((2.5*nest_size,(map_size//nest_size-i-1)*nest_size + nest_size/2,random.choice([0,pi])))

    nodes += [(2*nest_size,3*nest_size-0.1),(3*nest_size-0.1,3*nest_size-0.1),(3*nest_size-0.1,2*nest_size)]

    for i in range(map_size//nest_size): #MIDDLE BOTTOM SECTION
        if i < map_size//nest_size - 3 and i > 2:
            nodes += list(map(tuple,(rot_270@nest+np.array([i*nest_size,3*nest_size]).reshape(2,1)).T))
            positions.append((i*nest_size+nest_size/2,2.5*nest_size,random.choice([pi/2,-pi/2])))

    nodes += [(map_size - 3*nest_size + 0.1, 2*nest_size),(map_size - 3*nest_size + 0.1, 3*nest_size - 0.1),(map_size - 2*nest_size, 3*nest_size - 0.1)]

    for i in range(map_size//nest_size): #MIDDLE RIGHT SECTION
        if i < map_size//nest_size - 3 and i > 2:
            nodes += list(map(tuple,(nest+np.array([map_size-3*nest_size,i*nest_size]).reshape(2,1)).T))
            positions.append((map_size-2.5*nest_size,i*nest_size+nest_size/2,random.choice([0,pi])))

    nodes += [(map_size-2*nest_size,map_size-3*nest_size+0.1),(map_size-3*nest_size+0.1,map_size-3*nest_size+0.1),(map_size-3*nest_size+0.1,map_size-2*nest_size)]

    for i in range(map_size//nest_size): #MIDDLE UP 2
        if i > 2 and i < map_size//(2*nest_size):
            nodes += list(map(tuple,(rot_90@nest+np.array([(map_size//nest_size-i)*nest_size,map_size-3*nest_size]).reshape(2,1)).T))
            positions.append(((map_size//nest_size-i-1)*nest_size+nest_size/2,map_size-2.5*nest_size,random.choice([pi/2,-pi/2])))

    for i in range(map_size//nest_size): #TOPSECTION 2
        if i < map_size//nest_size - 1 and i > map_size//(2*nest_size)-1:
            nodes += list(map(tuple,(rot_270@nest+np.array([i*nest_size,map_size]).reshape(2,1)).T))
            positions.append((i*nest_size+nest_size/2,map_size - nest_size/2,random.choice([pi/2,-pi/2])))

    for i in range(map_size//nest_size): #RIGHTSECTION
        if i < map_size//nest_size - 1 and i > 0:
            nodes += list(map(tuple,(rot_180@nest+np.array([map_size,(map_size//nest_size-i)*nest_size]).reshape(2,1)).T))
            positions.append((map_size-nest_size/2,(map_size//nest_size-i-1)*nest_size + nest_size/2,random.choice([0,pi])))

    for i in range(map_size//nest_size): #BOTTOMSECTION
        if i < map_size//nest_size - 1 and i > 0:
            nodes += list(map(tuple,(rot_90@nest+np.array([(map_size//nest_size-i)*nest_size,0]).reshape(2,1)).T))
            positions.append(((map_size//nest_size-i-1)*nest_size+nest_size/2,nest_size/2,random.choice([pi/2,-pi/2])))

    random.shuffle(positions)
    
    start_position = positions.pop()
    goal_position = positions.pop()

    init_state = np.array([start_position[0], start_position[1], start_position[2], 0, 0])
    
    speed = 0.1
    obstacles.append(Obstacle.create_mpc_dynamic((1.5*nest_size, 1.5*nest_size), (1.5*nest_size, map_size - 1.5*nest_size), speed, 1, 0.5, pi/2, random = False))
    obstacles.append(Obstacle.create_mpc_dynamic((1.5*nest_size, 1.5*nest_size), (map_size-1.5*nest_size, 1.5*nest_size), speed, 1, 0.5, 0, random = False))
    obstacles.append(Obstacle.create_mpc_dynamic((map_size-1.5*nest_size, 1.5*nest_size), (map_size-1.5*nest_size, map_size-1.5*nest_size), speed, 1, 0.5, pi/2, random = False))
    return MobileRobot(init_state), Boundary(nodes), obstacles, Goal((goal_position[0],goal_position[1]))


def generate_map_multi_robot3_eval() -> MapDescription:

    """
    Generates a randomized map with one static obstacle
    """
    nest_size = 4
    door_size = 2
    box_resolution = 8

    wall = (nest_size-door_size)/2

    # nest = np.array([(nest_size,0.1),(0.1,0.1),(0.1,nest_size - 0.1),(nest_size,nest_size - 0.1)]).T
    nest = np.array([(nest_size,0.1),(nest_size,wall),(nest_size-0.2,wall),(nest_size-0.2,0.1),(0.1,0.1),(0.1,nest_size - 0.1),(nest_size-0.2,nest_size - 0.1),(nest_size-0.2,nest_size - wall),(nest_size,nest_size - wall),(nest_size,nest_size - 0.1)]).T
    map_size = nest_size*box_resolution
    
    nodes = []
    # positions = []
    obstacles = []
    
    for i in range(map_size//nest_size): #LEFTSECTION
        if i < map_size//nest_size - 1 and i > 0:
            nodes += list(map(tuple,(nest+np.array([0,i*nest_size]).reshape(2,1)).T))
            # positions.append((nest_size/2,i*nest_size+nest_size/2,random.choice([0,pi])))

    for i in range(map_size//nest_size): #TOPSECTION 1
        if i < map_size//(2*nest_size) and i > 0:
            nodes += list(map(tuple,(rot_270@nest+np.array([i*nest_size,map_size]).reshape(2,1)).T))
            # positions.append((i*nest_size+nest_size/2,map_size - nest_size/2,random.choice([pi/2,-pi/2])))

    for i in range(map_size//nest_size): #MIDDLE UP 1
        if i < map_size//nest_size - 3 and i > map_size//(2*nest_size) - 1:
            nodes += list(map(tuple,(rot_90@nest+np.array([(map_size//nest_size-i)*nest_size,map_size-3*nest_size]).reshape(2,1)).T))
            # positions.append(((map_size//nest_size-i-1)*nest_size+nest_size/2,map_size-2.5*nest_size,random.choice([pi/2,-pi/2])))
    
    nodes += [(3*nest_size-0.1,map_size-2*nest_size),(3*nest_size-0.1,map_size-3*nest_size+0.1),(2*nest_size,map_size-3*nest_size+0.1)]

    for i in range(map_size//nest_size): #MIDDLE LEFT SECTION
        if i < map_size//nest_size - 3 and i > 2:
            nodes += list(map(tuple,(rot_180@nest+np.array([3*nest_size,(map_size//nest_size-i)*nest_size]).reshape(2,1)).T))
            # positions.append((2.5*nest_size,(map_size//nest_size-i-1)*nest_size + nest_size/2,random.choice([0,pi])))

    nodes += [(2*nest_size,3*nest_size-0.1),(3*nest_size-0.1,3*nest_size-0.1),(3*nest_size-0.1,2*nest_size)]

    for i in range(map_size//nest_size): #MIDDLE BOTTOM SECTION
        if i < map_size//nest_size - 3 and i > 2:
            nodes += list(map(tuple,(rot_270@nest+np.array([i*nest_size,3*nest_size]).reshape(2,1)).T))
            # positions.append((i*nest_size+nest_size/2,2.5*nest_size,random.choice([pi/2,-pi/2])))

    nodes += [(map_size - 3*nest_size + 0.1, 2*nest_size),(map_size - 3*nest_size + 0.1, 3*nest_size - 0.1),(map_size - 2*nest_size, 3*nest_size - 0.1)]

    for i in range(map_size//nest_size): #MIDDLE RIGHT SECTION
        if i < map_size//nest_size - 3 and i > 2:
            nodes += list(map(tuple,(nest+np.array([map_size-3*nest_size,i*nest_size]).reshape(2,1)).T))
            # positions.append((map_size-2.5*nest_size,i*nest_size+nest_size/2,random.choice([0,pi])))

    nodes += [(map_size-2*nest_size,map_size-3*nest_size+0.1),(map_size-3*nest_size+0.1,map_size-3*nest_size+0.1),(map_size-3*nest_size+0.1,map_size-2*nest_size)]

    for i in range(map_size//nest_size): #MIDDLE UP 2
        if i > 2 and i < map_size//(2*nest_size):
            nodes += list(map(tuple,(rot_90@nest+np.array([(map_size//nest_size-i)*nest_size,map_size-3*nest_size]).reshape(2,1)).T))
            # positions.append(((map_size//nest_size-i-1)*nest_size+nest_size/2,map_size-2.5*nest_size,random.choice([pi/2,-pi/2])))

    for i in range(map_size//nest_size): #TOPSECTION 2
        if i < map_size//nest_size - 1 and i > map_size//(2*nest_size)-1:
            nodes += list(map(tuple,(rot_270@nest+np.array([i*nest_size,map_size]).reshape(2,1)).T))
            # positions.append((i*nest_size+nest_size/2,map_size - nest_size/2,random.choice([pi/2,-pi/2])))

    for i in range(map_size//nest_size): #RIGHTSECTION
        if i < map_size//nest_size - 1 and i > 0:
            nodes += list(map(tuple,(rot_180@nest+np.array([map_size,(map_size//nest_size-i)*nest_size]).reshape(2,1)).T))
            # positions.append((map_size-nest_size/2,(map_size//nest_size-i-1)*nest_size + nest_size/2,random.choice([0,pi])))

    for i in range(map_size//nest_size): #BOTTOMSECTION
        if i < map_size//nest_size - 1 and i > 0:
            nodes += list(map(tuple,(rot_90@nest+np.array([(map_size//nest_size-i)*nest_size,0]).reshape(2,1)).T))
            # positions.append(((map_size//nest_size-i-1)*nest_size+nest_size/2,nest_size/2,random.choice([pi/2,-pi/2])))

    # random.shuffle(positions)
    
    start_position = (0.5*nest_size, 4.5*nest_size,0)
    goal_position = (map_size-3.5*nest_size, map_size-2.5*nest_size)

    init_state = np.array([start_position[0], start_position[1], start_position[2], 0, 0])
    
    speed = 0.1
    obstacles.append(Obstacle.create_mpc_dynamic((1.5*nest_size, 1.5*nest_size), (1.5*nest_size, map_size - 1.5*nest_size), speed, 1, 0.5, pi/2, random = False))
    obstacles.append(Obstacle.create_mpc_dynamic((1.5*nest_size, 1.5*nest_size), (map_size-1.5*nest_size, 1.5*nest_size), speed, 1, 0.5, 0, random = False))
    obstacles.append(Obstacle.create_mpc_dynamic((map_size-1.5*nest_size, 1.5*nest_size), (map_size-1.5*nest_size, map_size-1.5*nest_size), speed, 1, 0.5, pi/2, random = False))
    return MobileRobot(init_state), Boundary(nodes), obstacles, Goal((goal_position[0],goal_position[1]))
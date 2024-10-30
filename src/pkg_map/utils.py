import random
import logging

from pkg_ddpg_td3.utils.map import (
    generate_map_corridor,
    generate_map_dynamic,
    generate_map_mpc,
    generate_map_dynamic_convex_obstacle,
    generate_map_eval,
    generate_map_static_nonconvex_obstacle,
    generate_map_train_1,
)
from pkg_ddpg_td3.environment import MapDescription

def generate_map_random() -> MapDescription:
    return random.choice([generate_map_dynamic, generate_map_corridor, generate_map_dynamic_convex_obstacle, generate_map_mpc()])()


def get_map(map_key):
    if map_key == 'corridor':
        generate_map = generate_map_corridor
    elif map_key == 'dynamic_convex_obstacle':
        generate_map = generate_map_dynamic_convex_obstacle
    elif map_key == 'static_nonconvex_obstacle':
        generate_map = generate_map_static_nonconvex_obstacle
    elif map_key == 'eval':
        generate_map = generate_map_eval
    elif map_key == 'train_1':
        generate_map = generate_map_train_1
    elif map_key == 'random':
        generate_map = generate_map_random
    else:
        logging.error(f'Could not find map key {map_key}')
    return generate_map
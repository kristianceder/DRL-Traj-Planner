"""
Code used to train the continous DRL agents, DDPG and TD3.

Eight different example agent variants are present, the first four of which 
corresponds to the DDPG algorithm, while the second four are TD3. You can 
select which example agent to train and evaluate by setting the ``index`` 
varaible as the first argument from the command line. 
This is generally done by the slurm array function as seen in ``SLURM_jobscript.sh``.
"""

import random
import logging
from datetime import datetime
from pathlib import Path

import wandb
import torch
import numpy as np
from pkg_ddpg_td3.utils.map import (
    generate_map_corridor,
    generate_map_dynamic_convex_obstacle,
    generate_map_eval,
    generate_map_static_nonconvex_obstacle,
)
from pkg_torchrl.env import make_env
from pkg_torchrl.sac import SAC
from pkg_torchrl.ppo import PPO
from pkg_torchrl.td3 import TD3

from configs import BaseConfig

logging.basicConfig(level=logging.ERROR)

# TODO (kilian)
# make speed of obstacles a parameter
# increasing discount factor
# n-step schedule


def run():
    tags = ["testing"]
    _ = wandb.init(
        project="DRL-Traj-Planner",
        tags=tags,
    )
    config = BaseConfig(**wandb.config)
    config.wg = 1 - config.alpha
    config.wc = config.alpha

    config.w1 = config.wc / 3  # speed
    config.w2 = config.wc / 3  # acceleration
    config.w3 = config.wg  # goal distance
    config.w4 = config.wc / 3  # cross track

    wandb.config.update(config.model_dump())

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    map_key = config.map_key
    if map_key == 'corridor':
        generate_map = generate_map_corridor
    elif map_key == 'dynamic_convex_obstacle':
        generate_map = generate_map_dynamic_convex_obstacle
    elif map_key == 'static_nonconvex_obstacle':
        generate_map = generate_map_static_nonconvex_obstacle
    else:
        logging.error(f'Could not find map key {map_key}')

    train_env = make_env(config, generate_map=generate_map, use_wandb=True)
    eval_env = make_env(config, generate_map=generate_map)
    # env_maker = lambda: make_env(config, generate_map=generate_map)

    algo_config = getattr(config, config.algo.lower())
    model = eval(config.algo.upper())(algo_config, train_env, eval_env)
    models_path = Path('../Model/testing')

    timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    path = models_path / f"{timestamp}_{config.algo.upper()}"
    path.mkdir(exist_ok=True, parents=True)
    wandb.config["path"] = path
    wandb.config["map"] = generate_map.__name__

    model.train()
    model.save(f"{path}/final_model.pth")
    logging.info(f"Final model saved to {path}")


if __name__ == "__main__":
    run()

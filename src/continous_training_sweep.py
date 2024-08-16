"""
Code used to train the continous DRL agents, DDPG and TD3.

Eight different example agent variants are present, the first four of which 
corresponds to the DDPG algorithm, while the second four are TD3. You can 
select which example agent to train and evaluate by setting the ``index`` 
varaible as the first argument from the command line. 
This is generally done by the slurm array function as seen in ``SLURM_jobscript.sh``.
"""

import random
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


# TODO (kilian)
# make speed of obstacles a parameter
# increasing discount factor
# n-step schedule


def run():
    tags = ["exploration_sweep_1"]
    _ = wandb.init(
        project="DRL-Traj-Planner",
        tags=tags,
    )
    config = BaseConfig(**wandb.config)
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
        print(f'Could not find map key {map_key}')

    train_env = make_env(config, generate_map=generate_map, use_wandb=True)
    eval_env = make_env(config, generate_map=generate_map)
    env_maker = lambda: make_env(config, generate_map=generate_map)

    algo_config = getattr(config, config.algo.lower())
    model = eval(config.algo.upper())(algo_config, train_env, eval_env)
    models_path = Path('../Model/testing')

    timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    path = models_path / f"{timestamp}_{config.algo.upper()}"
    path.mkdir(exist_ok=True, parents=True)
    wandb.config["path"] = path
    wandb.config["map"] = generate_map.__name__

    model.train(env_maker=env_maker)
    model.save(f"{path}/final_model.pth")
    print(f"Final model saved to {path}")


if __name__ == "__main__":
    run()

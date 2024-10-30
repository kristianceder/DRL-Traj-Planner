"""
Code used to train the continous DRL agents, DDPG and TD3.

Eight different example agent variants are present, the first four of which 
corresponds to the DDPG algorithm, while the second four are TD3. You can 
select which example agent to train and evaluate by setting the ``index`` 
varaible as the first argument from the command line. 
This is generally done by the slurm array function as seen in ``SLURM_jobscript.sh``.
"""

import os
import logging
import random
import argparse
from pathlib import Path

import torch
import numpy as np
from pkg_map.utils import get_map
from pkg_ddpg_td3.environment import MapDescription
from pkg_torchrl.env import make_env, render_rollout
from pkg_torchrl.sac import SAC
from pkg_torchrl.ppo import PPO
from pkg_torchrl.td3 import TD3
from pkg_torchrl.ddpg import DDPG

from pkg_ddpg_td3.utils.map_eval import *

from configs import BaseConfig


def process_args():
    parser = argparse.ArgumentParser(
        prog='DRL-Traj-Planner',
        description='Mobile robot navigation', )
    parser.add_argument('-p', '--path', default=None, type=str)

    return parser.parse_args()


def run():
    args = process_args()
    config = BaseConfig()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    generate_map = get_map(config.map_key)
    eval_env = make_env(config, generate_map=generate_map)

    algo_config = getattr(config, config.algo.lower())
    model = eval(config.algo.upper())(algo_config, eval_env, eval_env)
    models_path = Path('../Model')

    if args.path is None:
        # get latest path
        path = max(models_path.glob('*/'), key=os.path.getmtime) / "final_model.pth"
        model.load(path)
    else:
        path = models_path / args.path / "final_model.pth"
        model.load(path)

    print(f"Loaded {path}")
    render_rollout(eval_env, model, config, n_steps=3_000)


if __name__ == "__main__":
    run()

"""
Code used to train the continous DRL agents, DDPG and TD3.

Eight different example agent variants are present, the first four of which 
corresponds to the DDPG algorithm, while the second four are TD3. You can 
select which example agent to train and evaluate by setting the ``index`` 
varaible as the first argument from the command line. 
This is generally done by the slurm array function as seen in ``SLURM_jobscript.sh``.
"""

import os
import sys
import numpy as np
import random
import argparse
from datetime import datetime
from pathlib import Path

import wandb
import torch
from wandb.integration.sb3 import WandbCallback
from utils.plotresults import plot_training_results
from torch import no_grad
from pkg_ddpg_td3.utils.map import generate_map_dynamic, generate_map_corridor, generate_map_mpc, generate_map_eval
from pkg_ddpg_td3.environment import MapDescription
from typing import Callable

from utils.torchrl.env import make_env
from utils.torchrl.sac import SAC

from configs import BaseConfig


def generate_map() -> MapDescription:
    # return random.choice([generate_map_dynamic, generate_map_corridor, generate_map_mpc()])()
    return generate_map_dynamic()

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def run():
    parser = argparse.ArgumentParser(
                    prog='DRL-Traj-Planner',
                    description='Mobile robot navigation',)
    parser.add_argument('-lc', '--load-checkpoint',
                    action='store_true')
    parser.add_argument('-p', '--path', default=None, type=str)

    args = parser.parse_args()
    
    # Load a pre-trained model
    load_checkpoint = args.load_checkpoint

    config = BaseConfig()
    train_env = make_env(generate_map_dynamic, config)
    eval_env = make_env(generate_map_dynamic, config)

    model = SAC(config.sac, train_env, eval_env)

    if load_checkpoint:
        # get latest path
        if args.path is None:
            path = max(Path('Model/testing').glob('*/'), key=os.path.getmtime)
            path += "/final_model.pth"
        else:
            path = args.path
        # model = Algorithm.load(f"{path}/best_model", env=eval_env)
        # path = "../Model/torchrl_testing/24_07_10_12_07_58_SAC/final_model.pth"
        model.load(path)
        # plot_training_results(path)

        with no_grad():
            state = eval_env.reset()
            steps = 0
            ep_rwd = torch.zeros(1)
            for i in range(0, 2_000):
                action = state.copy()
                action['action'] = model.sample_action(state, sample_mean=True)
                # action = eval_env.rand_action(state)
                next_state = eval_env.step(action)

                steps += 1
                ep_rwd += next_state['next']['reward']

                # Only render every third frame for performance (matplotlib is slow)
                if i % 3 == 0 and i > 0:
                    eval_env.render()

                if next_state['next']['done'] or steps > config.sac.max_eps_steps:
                    print('reset')
                    state = eval_env.reset()
                    steps = 0
                else:
                    state = next_state
    
    else:

        timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        path = f"./Model/torch_rl_testing/{timestamp}_sac"

        run = wandb.init(
            project="DRL-Traj-Planner",
            config=config.model_dump(),
            tags="sac_exploration",
        )

        model.train()
        model.save(f"{path}/final_model.pth")


if __name__ == "__main__":
    run()

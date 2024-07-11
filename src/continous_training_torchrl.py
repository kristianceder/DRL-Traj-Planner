"""
Code used to train the continous DRL agents, DDPG and TD3.

Eight different example agent variants are present, the first four of which 
corresponds to the DDPG algorithm, while the second four are TD3. You can 
select which example agent to train and evaluate by setting the ``index`` 
varaible as the first argument from the command line. 
This is generally done by the slurm array function as seen in ``SLURM_jobscript.sh``.
"""

import os
import random
import argparse
from datetime import datetime
from pathlib import Path

import wandb
import torch
from utils.plotresults import plot_training_results
from torch import no_grad
from pkg_ddpg_td3.utils.map import (
    generate_map_dynamic,
    generate_map_corridor,
    generate_map_mpc,
    generate_map_eval
)
from pkg_ddpg_td3.environment import MapDescription

from utils.torchrl.env import make_env, render_rollout
from utils.torchrl.sac import SAC

from configs import BaseConfig


# TODO (kilian):
# - fix running on gpu device
# - generalize RL part to easily implement other algos

def process_args():
    parser = argparse.ArgumentParser(
                    prog='DRL-Traj-Planner',
                    description='Mobile robot navigation',)
    parser.add_argument('-lc', '--load-checkpoint',
                    action='store_true')
    parser.add_argument('-p', '--path', default=None, type=str)

    return parser.parse_args()


def run():
    args = process_args()
    config = BaseConfig()
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_env = make_env(config, generate_map=generate_map_eval, use_wandb=True)
    eval_env = make_env(config, generate_map=generate_map_eval)#, use_wandb=True)

    model = SAC(config.sac, train_env, eval_env)
    models_path = Path('../Model/testing')

    if args.load_checkpoint:
        if args.path is None:
            # get latest path
            path = max(models_path.glob('*/'), key=os.path.getmtime)
        else:
            path = models_path / args.path

        print(f"Loading {path}")
        path = path / "final_model.pth"
        model.load(path)
        render_rollout(eval_env, model, config, n_steps=3_000)
    else:
        timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        path = models_path / f"{timestamp}_SAC"
        path.mkdir(exist_ok=True, parents=True)

        _ = wandb.init(
            project="DRL-Traj-Planner",
            config=config.model_dump(),
            tags="sac_exploration",
        )

        model.train()
        model.save(f"{path}/final_model.pth")
        render_rollout(eval_env, model, config, n_steps=1_000)


if __name__ == "__main__":
    run()

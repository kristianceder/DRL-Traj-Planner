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


# def generate_map() -> MapDescription:
    # return random.choice([generate_map_dynamic, generate_map_corridor, generate_map_mpc()])()

def run():
    parser = argparse.ArgumentParser(
                    prog='DRL-Traj-Planner',
                    description='Mobile robot navigation',)
    parser.add_argument('-lc', '--load-checkpoint',
                    action='store_true')
    parser.add_argument('-p', '--path', default=None, type=str)

    args = parser.parse_args()
    
    config = BaseConfig()
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    train_env = make_env(generate_map_eval, config)
    eval_env = make_env(generate_map_eval, config)

    model = SAC(config.sac, train_env, eval_env)

    if args.load_checkpoint:
        # get latest path
        models_path = Path('../Model/testing')
        if args.path is None:
            path = max(models_path.glob('*/'), key=os.path.getmtime)
        else:
            path = models_path / args.path


        path = path / "final_model.pth"
        # path = "../Model/testing/24_07_10_12_07_58_SAC/final_model.pth"
        model.load(path)
        # plot_training_results(path)
        render_rollout(eval_env, model, config)
    
    else:
        timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        path = Path(f"../Model/testing/{timestamp}_SAC")
        path.mkdir(exist_ok=True, parents=True)

        run = wandb.init(
            project="DRL-Traj-Planner",
            config=config.model_dump(),
            tags="sac_exploration",
        )

        model.train()
        model.save(f"{path}/final_model.pth")


if __name__ == "__main__":
    run()

"""
Code used to train the continous DRL agents, DDPG and TD3.

Eight different example agent variants are present, the first four of which 
corresponds to the DDPG algorithm, while the second four are TD3. You can 
select which example agent to train and evaluate by setting the ``index`` 
varaible as the first argument from the command line. 
This is generally done by the slurm array function as seen in ``SLURM_jobscript.sh``.
"""

import os
import copy
import random
import argparse
from datetime import datetime
from pathlib import Path

import wandb
import torch
from pkg_ddpg_td3.utils.map import (
    generate_map_dynamic,
    generate_map_corridor,
    generate_map_static_nonconvex_obstacle,
    generate_map_eval
)
import gymnasium as gym
from stable_baselines3 import PPO

from configs import BaseConfig


def process_args():
    parser = argparse.ArgumentParser(
                    prog='DRL-Traj-Planner',
                    description='Mobile robot navigation',)
    parser.add_argument('-lc', '--load-checkpoint',
                    action='store_true')
    parser.add_argument('-p', '--path', default=None, type=str)

    return parser.parse_args()


def render_rollout(eval_env, model, config, n_steps=1_000):
    with torch.no_grad():
        state, _ = eval_env.reset()
        steps = 0
        ep_rwd = torch.zeros(1)
        for i in range(n_steps):
            action, _ = model.predict(state)
            next_state, rwd, trunc, term, info = eval_env.step(action)

            steps += 1
            ep_rwd += rwd

            # Only render every third frame for performance (matplotlib is slow)
            if i % 3 == 0 and i > 0:
                eval_env.render()

            if trunc or term or steps > config.sac.max_eps_steps:
                print(f'reset, ep reward {ep_rwd.item()}')
                state, _ = eval_env.reset()
                steps = 0
                ep_rwd = torch.zeros(1)
            else:
                state = next_state


def run():
    args = process_args()
    config = BaseConfig()
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    generate_map = generate_map_static_nonconvex_obstacle

    train_env = gym.make(config.env_name, generate_map=generate_map, max_episode_steps=config.sac.max_eps_steps)
    eval_env = gym.make(config.env_name, generate_map=generate_map, max_episode_steps=config.sac.max_eps_steps)

    # model = SAC(config.sac, train_env, eval_env)
    model = PPO('MultiInputPolicy', train_env, verbose=1)
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
        if args.path is not None:
            path = models_path / args.path / "final_model.pth"
            model.load(path)

        timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        path = models_path / f"{timestamp}_PPO_SB3"
        path.mkdir(exist_ok=True, parents=True)

        # _ = wandb.init(
        #     project="DRL-Traj-Planner",
        #     config=config.model_dump(),
        #     tags="sac_exploration",
        # )

        model.learn(50_000)
        model.save(f"{path}/final_model.pth")
        render_rollout(eval_env, model, config, n_steps=1_000)


if __name__ == "__main__":
    run()

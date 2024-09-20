"""
Code used to train the continous DRL agents with torchrl.

Eight different example agent variants are present, the first four of which 
corresponds to the DDPG algorithm, while the second four are TD3. You can 
select which example agent to train and evaluate by setting the ``index`` 
varaible as the first argument from the command line. 
This is generally done by the slurm array function as seen in ``SLURM_jobscript.sh``.
"""

import os
import sys
import gymnasium as gym
import numpy as np
import random
import argparse
from datetime import datetime
from pathlib import Path

import wandb
import torch
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import DDPG, TD3, SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from utils.plotresults import plot_training_results
from stable_baselines3.common.env_checker import check_env
from torch import no_grad
from pkg_ddpg_td3.utils.map import generate_map_dynamic, generate_map_corridor, generate_map_mpc, generate_map_eval
from pkg_ddpg_td3.environment import MapDescription
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from typing import Callable
from pkg_ddpg_td3.utils.per_ddpg import PerDDPG
from pkg_ddpg_td3.utils.per_td3 import PerTD3

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
    parser.add_argument('-i', '--index', default=-1, type=int)
    parser.add_argument('-p', '--path', default=None, type=str)

    args = parser.parse_args()
    
    # Selects which model variant to use
    index = args.index#6                   
    
    # Load a pre-trained model
    load_checkpoint = args.load_checkpoint#True

    # Select the path where the model should be stored
    # path = f'./Model/local_training/variant-{index}'
    # path = './Model/td3/image'
    # path = './Model/testing'
    # path = './Model/ddpg/image'
    # path = './Model/ddpg/ray'
    
    # Parameters for different example agent models 
    variant = [
        {
            'algorithm' : "DDPG",
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward1-v0',
            'net_arch': [64, 64],
            'per': True,
            'device': 'auto',
        },
        {
            'algorithm' : "DDPG",
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward2-v0',
            'net_arch': [64, 64],
            'per': True,
            'device': 'auto',
        },
        {
            'algorithm' : "DDPG",
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward1-v0',
            'net_arch': [16, 16],
            'per': True,
            'device': 'cpu',
        },
        {
            'algorithm' : "DDPG",
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward2-v0',
            'net_arch': [16, 16],
            'per': True,
            'device': 'cpu',
        },
        # TD3
        {
            'algorithm' : "TD3",
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward1-v0',
            'net_arch': [64, 64],
            'per': True,
            'device': 'auto',
        },
        {
            'algorithm' : "TD3",
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward2-v0',
            'net_arch': [64, 64],
            'per': True,
            'device': 'auto',
        },
        {
            'algorithm' : "TD3",
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward1-v0',
            'net_arch': [16, 16],
            'per': True,
            'device': 'cpu',
        },
        {
            'algorithm' : "TD3",
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward2-v0',
            'net_arch': [16, 16],
            'per': True,
            'device': 'cpu',
        },
        {
            'algorithm' : "SAC",
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward1-v0',
            'net_arch': [32, 32],
            'per': False,
            'device': 'mps',
        },
        {
            'algorithm' : "TD3",
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward1-v0',
            'net_arch': [32, 32],
            'per': False,
            'device': 'mps',
        },
        {
            'algorithm' : "PPO",
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward1-v0',
            'net_arch': [32, 32],
            'per': False,
            'device': 'mps',
        },
    ][index]


    tot_timesteps = 10e5
    n_cpu = 20

    env_eval = gym.make(variant['env_name'], generate_map=generate_map_dynamic, max_episode_steps=500)
    vec_env = make_vec_env(variant['env_name'], n_envs=n_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'generate_map': generate_map})
    vec_env_eval = make_vec_env(variant['env_name'], n_envs=n_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'generate_map': generate_map})
    # check_env(vec_env)



    if variant["algorithm"] == "DDPG" and not variant["per"]:
        Algorithm = DDPG
    elif variant["algorithm"] == "DDPG" and variant["per"]:
        Algorithm = PerDDPG
    elif variant["algorithm"] == "TD3" and not variant["per"]:
        Algorithm = TD3
    elif variant["algorithm"] == "TD3" and variant["per"]:
        Algorithm = PerTD3
    elif variant["algorithm"] == "SAC":
        Algorithm = SAC
    elif variant["algorithm"] == "PPO":
        Algorithm = PPO

    if load_checkpoint:
        # get latest path
        if args.path is None:
            path = max(Path('Model/testing').glob('*/'), key=os.path.getmtime)
        else:
            path = args.path
        model = Algorithm.load(f"{path}/best_model", env=env_eval)
        # plot_training_results(path)

        with no_grad():
            # while True:
            obs, _ = env_eval.reset()
            for i in range(0, 5000):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env_eval.step(action)

                if i % 3 == 0 and i > 0: # Only render every third frame for performance (matplotlib is slow)
                    # vec_env.render("human")
                    env_eval.render()

                if term or trunc:
                    print('reset')
                    obs, _ = env_eval.reset()
                    # break
    
    
    else:

        timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
        path = f"./Model/testing/{timestamp}_{variant['algorithm']}"

        sac_config = {
            "learning_rate": 3e-4,
            "learning_starts": 100,
            "batch_size": 1024, # bigger batch size (1024 vs 256) stabilizes training
            "buffer_size": int(5e6),
            "train_freq": 1,
            "gradient_steps": 1,
            # "action_noise_sigma": 0.5,
            "gamma": 0.99,
            "time_penalty": True,
        }

        run = wandb.init(
            project="DRL-Traj-Planner",
            # config=variant,
            sync_tensorboard=True,
            config=sac_config,
            tags="sac_exploration",
        )
        wandb_callback = WandbCallback(verbose=2)
        # vec_env_eval = VecVideoRecorder(
        #     vec_env_eval,
        #     f"videos/{run.id}",
        #     record_video_trigger=lambda x: x % 5_000 == 0,
        #     video_length=200,
        # )

        # cleanup vars not used by algo
        del sac_config['time_penalty']
            
        eval_callback = EvalCallback(vec_env_eval,
                                    best_model_save_path=path,
                                    log_path=path,
                                    eval_freq=max((tot_timesteps / 100) // n_cpu, 1),
                                    n_eval_episodes=n_cpu,
                                    callback_after_eval=wandb_callback)
        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=variant['net_arch'])



        n_actions  = vec_env.action_space.shape[-1]
        # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=sac_config["action_noise_sigma"] * np.ones(n_actions))
        # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
        # action_noise = NormalActionNoise(mean=np.zeros(n_actions),sigma=0.1 * np.ones(n_actions))

        # del sac_config['action_noise_sigma']

        model = Algorithm("MultiInputPolicy",
                    vec_env, 
                    # learning_rate=linear_schedule(0.0001),
                    # learning_rate=0.0001, 
                    # buffer_size=int(1e6),
                    # learning_starts=100_000,
                    # gamma=0.98,
                    # action_noise = action_noise,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    device=variant['device'],
                    tensorboard_log=f"runs/{run.id}",
                    **sac_config,
                )

        # Train the model
        callback = CallbackList([wandb_callback, eval_callback])
        model.learn(total_timesteps=tot_timesteps,
                    log_interval=4,
                    progress_bar=True,
                    callback=callback)

        # Save the model
        model.save(f"{path}/final_model")

                    
    
if __name__ == "__main__":
    run()

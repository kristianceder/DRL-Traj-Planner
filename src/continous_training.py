"""
Code used to train the continous DRL agents, DDPG and TD3.

Eight different example agent variants are present, the first four of which 
corresponds to the DDPG algorithm, while the second four are TD3. You can 
select which example agent to train and evaluate by setting the ``index`` 
varaible as the first argument from the command line. 
This is generally done by the slurm array function as seen in ``SLURM_jobscript.sh``.
"""

import sys
import gym
import numpy as np
import random

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
# from src.pkg_ddpg_td3.utils.plotresults import plot_training_results
from stable_baselines3.common.env_checker import check_env
from torch import no_grad
from pkg_ddpg_td3.utils.map import generate_map_dynamic, generate_map_corridor, generate_map_mpc, generate_map_eval
from pkg_ddpg_td3.utils.map_simple import generate_simple_map_dynamic, generate_simple_map_nonconvex, generate_simple_map_static
from pkg_ddpg_td3.environment import MapDescription
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from pkg_ddpg_td3.utils.per_ddpg import PerDDPG
from pkg_ddpg_td3.utils.per_td3 import PerTD3

def generate_map() -> MapDescription:
    return random.choice([generate_map_dynamic, generate_map_corridor, generate_map_mpc(), generate_simple_map_static, generate_simple_map_dynamic, generate_simple_map_nonconvex])()

def run():
    # Selects which predefined agent model to use
    index = int(sys.argv[1])
    run_index = int(sys.argv[2])

    # Select the path where the model should be stored
    path = f'/cephyr/users/cederk/Vera/github/DRL-Traj-Planner/Model/training/variant-{index}/run{run_index}'

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
    ][index]

    tot_timesteps = 1e7
    n_cpu = 32
    time_step = 0.1
    
    # Load a pre-trained model
    load_checkpoint = 0

    env_eval = gym.make(variant['env_name'], generate_map=generate_map, time_step = time_step)
    vec_env = make_vec_env(variant['env_name'], n_envs=n_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'generate_map': generate_map})
    vec_env_eval = make_vec_env(variant['env_name'], n_envs=n_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'generate_map': generate_map})
    check_env(env_eval)

    n_actions  = vec_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    if variant["algorithm"] == "DDPG" and not variant["per"]:
        Algorithm = DDPG
    elif variant["algorithm"] == "DDPG" and variant["per"]:
        Algorithm = PerDDPG
    elif variant["algorithm"] == "TD3" and not variant["per"]:
        Algorithm = TD3
    elif variant["algorithm"] == "TD3" and variant["per"]:
        Algorithm = PerTD3

    eval_callback = EvalCallback(vec_env_eval,
                                 best_model_save_path=path,
                                 log_path=path,
                                 eval_freq=max((tot_timesteps / 100) // n_cpu, 1),
				                n_eval_episodes=n_cpu)

    if load_checkpoint:
        model = Algorithm.load(f"{path}/best_model", env=vec_env)
    else:
        model = Algorithm("MultiInputPolicy",
                    vec_env, learning_rate=0.0001, buffer_size=int(1e6), 
                    learning_starts=1_000_000, gamma=0.98,
		            tau=0.01,
                    gradient_steps=-1,
                    action_noise = action_noise,
                    policy_kwargs={'net_arch': variant['net_arch']},
                    verbose=1,
                    device=variant['device'],
                )

    # Train the model    
    model.learn(total_timesteps=tot_timesteps, log_interval=4, progress_bar=True, callback=eval_callback)

    # Save the model
    model.save(f"{path}/final_model")

                    
    
if __name__ == "__main__":
    run()

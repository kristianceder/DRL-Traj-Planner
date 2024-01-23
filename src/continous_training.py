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
from utils.plotresults import plot_training_results
from stable_baselines3.common.env_checker import check_env
from torch import no_grad
from pkg_ddpg_td3.utils.map import generate_map_dynamic, generate_map_corridor, generate_map_mpc, generate_map_eval, generate_map_easy, generate_map_dynamic_2
from pkg_ddpg_td3.environment import MapDescription
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from pkg_ddpg_td3.utils.per_ddpg import PerDDPG
from pkg_ddpg_td3.utils.per_td3 import PerTD3

def generate_map() -> MapDescription:
    return random.choice([generate_map_dynamic, generate_map_corridor, generate_map_mpc()])()

def run():
    # Selects which predefined agent model to use
    index = 2#int(sys.argv[1])
    # Load a pre-trained model
    load_checkpoint = True
    # Select the path where the model should be stored
    path = f'./example/path/to/storage/folder/variant-{index}'
    # path = './Model/td3/image'
    # path = './Model/td3/ray'
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
            'per': False,
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

    tot_timesteps = 10e4
    n_cpu = 20
    time_step = 0.1
    
    

    env_eval = gym.make(variant['env_name'], generate_map=generate_map_dynamic_2, time_step = time_step)
    vec_env = make_vec_env(variant['env_name'], n_envs=1, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'generate_map': generate_map_dynamic_2})
    vec_env_eval = make_vec_env(variant['env_name'], n_envs=n_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'generate_map': generate_map_dynamic_2})
    # check_env(vec_env)

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
                                 eval_freq=max((tot_timesteps / 10) // n_cpu, 1),
                                 n_eval_episodes=n_cpu)

    if load_checkpoint:
        model = Algorithm.load(f"{path}/best_model", env=env_eval)
        plot_training_results(path)

        with no_grad():
            while True:
                obs = env_eval.reset()
                for i in range(0, 1000):
                    action, _states = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env_eval.step(action)
                    if i % 3 == 0: # Only render every third frame for performance (matplotlib is slow)
                        # vec_env.render("human")
                        env_eval.render()
                    if done:
                        break
    
    
    else:
        model = Algorithm("MultiInputPolicy",
                    vec_env, learning_rate=0.0001, buffer_size=int(1e6), 
                    learning_starts=10_000, gamma=0.98,
                    gradient_steps=-1,
                    #action_noise = action_noise,
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

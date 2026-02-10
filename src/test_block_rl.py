"""
Example code showing how to train DRL agents.

Five different example agent variants are present, the first four of which are
discussed in detail in the associated project report. You can select which
example agent to train and evaluate by setting the ``index`` varaible
"""

import random
import numpy as np
import copy

import gymnasium as gym
from torch import no_grad

from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

from pkg_dqn.utils.plotresults import plot_training_results
from pkg_dqn.utils.map import generate_map_dynamic, generate_map_corridor, generate_map_mpc

from pkg_dqn.environment import MapDescription, MobileRobot

TO_TRAIN = False
TO_SAVE = False

def generate_map() -> MapDescription:
    """
    MapDescription = Tuple[MobileRobot, Boundary, List[Obstacle], Goal]
    """
    return random.choice([generate_map_dynamic])()

def run():
    # Selects which predefined agent model to use
    index = 1

    # Parameters for different example agent models 
    variant = [
        {
            'env_name': 'TrajectoryPlannerEnvironmentImgsReward1-v0',
            'net_arch': [64, 64],
            'per': False,
            'device': 'auto',
        },
        {
            'env_name': 'TrajectoryPlannerEnvironmentRaysReward1-v0',
            'net_arch': [16, 16],
            'per': False,
            'device': 'cpu',
        },
    ][index]

    if index == 0:
        path = f'./././Model/image'
    elif index == 1:
        path = f'./././Model/ray'
    else:
        raise ValueError('Invalid index')

    tot_timesteps = 10e6
    n_cpu = 4

    env_eval = gym.make(variant['env_name'], generate_map=generate_map)
    check_env(env_eval)

    vec_env      = make_vec_env(variant['env_name'], n_envs=n_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'generate_map': generate_map})
    vec_env_eval = make_vec_env(variant['env_name'], n_envs=n_cpu, seed=0, vec_env_cls=SubprocVecEnv, env_kwargs={'generate_map': generate_map_mpc(11)})
    
    n_actions  = vec_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    Algorithm = PerDQN if variant['per'] else DDPG
    eval_callback = EvalCallback(vec_env_eval,
                                 best_model_save_path=path,
                                 log_path=path,
                                 eval_freq=max((tot_timesteps / 1000) // n_cpu, 1))
    model = Algorithm("MultiInputPolicy",
                    vec_env, learning_rate=0.0001, 
                    learning_starts=tot_timesteps/10, gamma=0.98,
                    gradient_steps=-1,
                    action_noise = action_noise,
                    policy_kwargs={'net_arch': variant['net_arch']},
                    verbose=1,
                    device=variant['device'],
                )

    # Train the model
    if TO_TRAIN:
        model.learn(total_timesteps=tot_timesteps, log_interval=4, progress_bar=True, callback=eval_callback)
        # Save the model
        if TO_SAVE:
            model.save(f"{path}/final_model")

    ### Validation
    model = Algorithm.load(f"{path}/best_model", env=vec_env)

    # Plot the learning curve of the training run
    plot_training_results(f"{path}/evaluations.npz")
    
    # Run the agent in an evaluation environemnt indefinately to show off the
    # the final trained agent
    env_eval_sim = gym.make(variant['env_name'], generate_map=generate_map)

    with no_grad():
        while True:
            obsv = env_eval.reset()
            # init_state = list([env_eval.agent.position, env_eval.agent.angle, env_eval.agent.speed, env_eval.agent.angular_velocity])
            for i in range(0, 1000):
                # env_eval.set_agent_state(*init_state)
                # obsv = env_eval.get_observation()
                action, _states = model.predict(obsv, deterministic=True)
                obsv, reward, done, info = env_eval.step(action)

                real_robot:MobileRobot = env_eval.agent
                pred_positions = []
                robot_sim = copy.deepcopy(real_robot)
                for j in range(20):
                    if j == 0:
                        robot_sim.step(action, 0.1)
                    else:
                        robot_sim.step_with_decay_angular_velocity(0.1, j+1)
                    pred_positions.append(list(robot_sim.position))

                if i % 3 == 0: # Only render every third frame for performance (matplotlib is slow)
                    env_eval.render(dqn_ref=pred_positions)
                if done:
                    break
    

if __name__ == "__main__":
    run()

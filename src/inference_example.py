"""
Code used to train the continous DRL agents, DDPG and TD3.

Eight different example agent variants are present, the first four of which 
corresponds to the DDPG algorithm, while the second four are TD3. You can 
select which example agent to train and evaluate by setting the ``index`` 
varaible as the first argument from the command line. 
This is generally done by the slurm array function as seen in ``SLURM_jobscript.sh``.
"""

import gym

from pkg_ddpg_td3.utils.map import generate_map_eval
from pkg_ddpg_td3.inference_model import inference_model

def run():
    
    # Select the path where the model should be stored
    # path = './Model/testing/variant-0/run1'

    path = './Model/ddpg/image'
    env_eval = gym.make('TrajectoryPlannerEnvironmentImgsReward1-v0', generate_map=generate_map_eval, time_step = 0.1)

    model = inference_model(path,env_eval)

    while True:
        obs = env_eval.reset()
        for i in range(0, 1000):
            action = model.get_action(obs)
            obs, reward, done, info = env_eval.step(action)
            if i % 3 == 0: # Only render every third frame for performance (matplotlib is slow)
                # vec_env.render("human")
                env_eval.render()
            if done:
                break
    
if __name__ == "__main__":
    run()

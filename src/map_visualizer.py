
import gym
import numpy as np

from pkg_ddpg_td3.utils.map import generate_map_dynamic, generate_map_corridor, generate_map_mpc, generate_map_eval
# from pkg_ddpg_td3.utils.map_simple import  generate_simple_map_easy, generate_simple_map_static, generate_simple_map_nonconvex, generate_simple_map_dynamic,generate_simple_map_nonconvex_static, generate_simple_map_static1
from pkg_ddpg_td3.utils.map_simple import *
from pkg_ddpg_td3.utils.map_multi_robot import generate_map_multi_robot1, generate_map_multi_robot2, generate_map_multi_robot3, generate_map_multi_robot3_eval
from main_pre_continous import generate_map

variant_list = [
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
    ]

def generate_map() -> MapDescription:
    # return random.choice([generate_map_dynamic, generate_map_corridor, generate_map_mpc(), generate_simple_map_static, generate_simple_map_dynamic, generate_simple_map_nonconvex])()
    return random.choice([generate_map_dynamic, generate_simple_map_nonconvex,generate_simple_map_dynamic,generate_simple_map_static])()




def run():

    index = 0
    variant = variant_list[index]
    
    scene_option_list = [
                         (1, 1, 2), 
                         (1, 1, 3), 
                         (1, 2, 1), 
                         (1, 2, 2), 
                         (1, 3, 1), 
                         (1, 3, 2), 
                         (1, 4, 1),
                         (2, 1, 1), 
                         (2, 1, 2), 
                         (2, 1, 3),
                         ][3]
    #generate_map(*scene_option_list)
    env_eval = gym.make(variant['env_name'], generate_map=generate_map)
    while True:
        env_eval.reset()
        # env_eval.step(np.array([0,0]))
        for i in range(10):
            env_eval.step(np.array([0,0]))
            env_eval.render()



if __name__ == "__main__":
    run()


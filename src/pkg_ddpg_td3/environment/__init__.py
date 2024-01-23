from typing import Callable
from .agent import MobileRobot
from .obstacle import Boundary, Obstacle, Animation
from .goal import Goal

MapDescription = tuple[MobileRobot, Boundary, list[Obstacle], Goal]
MapGenerator = Callable[[], MapDescription]

__all__ = ['MobileRobot', 'Boundary', 'Obstacle', 'Animation', 'Goal', 'MapDescription']

from gym.envs.registration import register

register(
    id='TrajectoryPlannerEnvironmentGoal-v0',
    entry_point='pkg_ddpg_td3.environment.variants.goal:TrajectoryPlannerEnvironmentGoal',
    max_episode_steps=300,
)

max_episode_steps = 300
register(
    id='TrajectoryPlannerEnvironmentImgsReward1-v0',
    entry_point='pkg_ddpg_td3.environment.variants.imgs_reward1:TrajectoryPlannerEnvironmentImgsReward1',
    max_episode_steps=max_episode_steps,
)
register(
    id='TrajectoryPlannerEnvironmentImgsReward2-v0',
    entry_point='pkg_ddpg_td3.environment.variants.imgs_reward2:TrajectoryPlannerEnvironmentImgsReward2',
    max_episode_steps=max_episode_steps,
)
register(
    id='TrajectoryPlannerEnvironmentRaysReward1-v0',
    entry_point='pkg_ddpg_td3.environment.variants.rays_reward1:TrajectoryPlannerEnvironmentRaysReward1',
    max_episode_steps=max_episode_steps,
)
register(
    id='TrajectoryPlannerEnvironmentRaysReward2-v0',
    entry_point='pkg_ddpg_td3.environment.variants.rays_reward2:TrajectoryPlannerEnvironmentRaysReward2',
    max_episode_steps=max_episode_steps,
)

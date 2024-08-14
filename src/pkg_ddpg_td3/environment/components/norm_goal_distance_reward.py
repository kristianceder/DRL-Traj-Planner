from numpy.linalg import norm
from . import Component
from .. import MobileRobot


class NormGoalDistanceReward(Component):
    """
    Rewards decreasing the distance to the goal and penalizes increasing said
    distance
    """
    def __init__(self, factor: float):
        self.factor = factor
        self.last_dist_to_goal = None
        self.v_max = MobileRobot().cfg.SPEED_MAX

    def reset(self) -> None:
        self.last_dist_to_goal = norm(self.env.goal.position - self.env.agent.position)

    def step(self, action: int) -> float:
        distance_to_goal = norm(self.env.goal.position - self.env.agent.position)

        if self.last_dist_to_goal is None:
            self.last_dist_to_goal = distance_to_goal

        goal_distance_diff = self.last_dist_to_goal - distance_to_goal

        reward = goal_distance_diff / (self.v_max * self.env.time_step)

        self.last_dist_to_goal = distance_to_goal
        return self.factor * reward

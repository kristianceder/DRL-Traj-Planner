from numpy.linalg import norm
from . import Component


class GoalDistanceReward(Component):
    """
    Rewards decreasing the distance to the goal and penalizes increasing said
    distance
    """
    def __init__(self, factor: float, strictly_pos: bool = False, bias: float = 0.):
        self.factor = factor
        self.strictly_pos = strictly_pos
        self.bias = bias

    def reset(self) -> None:
        self.distance_to_goal = norm(self.env.goal.position - self.env.agent.position)

    def step(self, action: int) -> float:
        self.last_distance_to_goal = self.distance_to_goal
        self.distance_to_goal = norm(self.env.goal.position - self.env.agent.position)
        if self.last_distance_to_goal is None:
            self.distance_to_goal = self.last_distance_to_goal
        
        dist_change = (self.last_distance_to_goal - self.distance_to_goal)

        dist_change += self.bias

        if self.strictly_pos:
            reward = max(0, dist_change)
        else:
            reward = self.factor * dist_change
        return reward

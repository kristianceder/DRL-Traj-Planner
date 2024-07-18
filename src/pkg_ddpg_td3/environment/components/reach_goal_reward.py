from . import Component


class ReachGoalReward(Component):
    """
    Gives a constant negative reward when the ATR reaches the goal
    """
    def __init__(self, factor: float, default_val: float = 0.):
        self.factor = factor
        self.default_val = default_val

    def step(self, action: int) -> float:
        reward = self.factor if self.env.reached_goal else self.default_val
        return reward
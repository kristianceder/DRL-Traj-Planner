import numpy as np
from . import Component


class NormCrossTrackReward(Component):
    """
    Gives a reward proportional to -(path cross track error)²
    """
    def __init__(self, factor: float, max_error: float = 5.):
        self.factor = factor
        self.max_error = max_error
    
    def step(self, action: int) -> float:
        closest_point = self.env.path.interpolate(self.env.path_progress)
        error = self.env.agent.point.distance(closest_point)
        reward = 1 - (2 * np.abs(error)) / self.max_error
        reward = np.clip(reward, -1, 1)
        return self.factor * reward

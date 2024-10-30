import numpy as np
from shapely import Point, distance
from . import Component


class NormObstacleDistanceReward(Component):
    """
    Rewards penalizing the distance to obstacles
    """
    def __init__(self, factor: float, max_dist: float = 1.):
        self.factor = factor
        self.max_dist = max_dist

    def step(self, action: int) -> float:
        min_dist = min([distance(self.env.agent.point, o.padded_polygon) for o in self.env.obstacles])
        raw_rwd = -1 + 2 * (min_dist / self.max_dist)
        # reward = min(1, max(-1, raw_rwd))
        reward = np.clip(raw_rwd, -1, 1)
        return self.factor * reward

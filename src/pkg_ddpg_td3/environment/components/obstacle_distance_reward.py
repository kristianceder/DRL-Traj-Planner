import numpy as np
from shapely import Point, distance
from . import Component


class ObstacleDistanceReward(Component):
    """
    Rewards penalizing the distance to obstacles
    """
    def __init__(self, factor: float, min_val: float = -10.):
        self.factor = factor
        self.min_val = min_val

    def step(self, action: int) -> float:
        min_dist = min([distance(self.env.agent.point, o.padded_polygon) for o in self.env.obstacles])

        reward = max(self.factor * np.log(min_dist), self.min_val)
        return reward

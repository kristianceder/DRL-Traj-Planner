from . import Component
import numpy as np


class NormAccelerationReward(Component):
    """
    Gives negative reward proportional to |acceleration|
    """
    def __init__(self, factor: float):
        self.factor = factor

    def step(self, action: list) -> float:
        reward = 1 - (np.abs(action).mean() * 2)
        return self.factor * reward

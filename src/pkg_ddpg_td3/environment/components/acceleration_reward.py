from . import Component
import numpy as np


class AccelerationReward(Component):
    """
    Gives negative reward proportional to (acceleration)^2
    """
    def __init__(self, factor: float, exponent: int = 2):
        self.factor = factor
        self.exponent = exponent
    
    def step(self, action: list) -> float:
        return -self.factor * np.abs(action[0])**self.exponent
        # return -self.factor * np.abs(self.env.agent.acceleration) ** self.exponent

from . import Component
import numpy as np

class AngularAccelerationReward(Component):
    """
    Gives negative reward proportional to (angular acceleration)^2
    """
    def __init__(self, factor: float, exponent: int = 2):
        self.factor = factor
        self.exponent = exponent
    
    def step(self, action: list) -> float:
        return -self.factor * np.abs(action[1])**self.exponent
        # return -self.factor * np.abs(self.env.agent.angular_acceleration) ** self.exponent

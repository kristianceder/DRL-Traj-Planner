from numpy import sign
from . import Component


class PosExcessiveSpeedReward(Component):
    """
    Gives a negative reward for exceeding a reference speed
    """
    def __init__(self, factor: float, reference_speed: float):
        self.factor = factor
        self.reference_speed = reference_speed
    
    def step(self, action: int) -> float:
        speed_diff = self.env.agent.speed - self.reference_speed
        # excessive_speed = max(0, speed_diff)
        penalty = self.factor * speed_diff**2
        return max(0, 1 - penalty)
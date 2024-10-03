from . import Component
from .. import MobileRobot

class PathProgressReward(Component):
    """
    Rewards making progress along the reference path, penalizes the opposite
    """
    def __init__(self, factor: float):
        self.factor = factor
        self.v_max = MobileRobot().cfg.SPEED_MAX
        self.last_path_progress = 0

    def reset(self) -> None:
        self.last_path_progress = 0
    
    def step(self, action: int) -> float:
        reward = (self.env.path_progress - self.last_path_progress) / (self.v_max * self.env.time_step)
        self.last_path_progress = self.env.path_progress
        return self.factor * reward
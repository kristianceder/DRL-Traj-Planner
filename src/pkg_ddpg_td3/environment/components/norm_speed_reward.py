from . import Component
from .. import MobileRobot
import numpy as np


class NormSpeedReward(Component):
    """
    Gives reward proportional to -(speed - reference speed)²
    """
    def __init__(self, factor: float, reference_speed: float, tau: float = .5):
        self.factor = factor
        self.tau = tau
        self.reference_speed = reference_speed
        v_max = MobileRobot().cfg.SPEED_MAX
        self.max_value = max(self.expectile(np.array(-reference_speed)),
                             self.expectile(np.array(v_max - reference_speed)))

    def expectile(self, error):
        if error > 0:
            return self.tau * error ** 2
        else:
            return (1 - self.tau) * error ** 2

    def step(self, action: int) -> float:
        error = self.env.agent.speed - self.reference_speed
        # print('---')
        # print(error)
        # print(self.max_value)
        reward = 1 - ((self.expectile(error) * 2) / self.max_value)
        # print(reward)

        return self.factor * reward

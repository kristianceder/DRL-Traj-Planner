import numpy.typing as npt
from . import Component
from .. import MobileRobot
from .utils import normalize
from collections import deque
import numpy as np


class SpeedObservation(Component):
    """
    Observes ATR speed
    """
    def __init__(self, max_len: int = 1):
        self.obs_queue = deque(maxlen=max_len)
        self.obs_queue.extend([0] * max_len)
        self.internal_obs_min: npt.ArrayLike = -np.ones(max_len)
        self.internal_obs_max: npt.ArrayLike = np.ones(max_len)

    def internal_obs(self) -> npt.ArrayLike:
        # [normalize(self.env.agent.speed, MobileRobot().cfg.SPEED_MIN, MobileRobot().cfg.SPEED_MAX)]
        speed_obs = normalize(self.env.agent.speed, MobileRobot().cfg.SPEED_MIN, MobileRobot().cfg.SPEED_MAX)
        self.obs_queue.append(speed_obs)
        return self.obs_queue

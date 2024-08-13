import numpy.typing as npt
from . import Component
from .. import MobileRobot
from .utils import normalize
from collections import deque
import numpy as np


class AngularVelocityObservation(Component):
    """
    Observes ATR angular velocity
    """
    # internal_obs_min: npt.ArrayLike = [-1]
    # internal_obs_max: npt.ArrayLike = [1]
    def __init__(self, max_len: int = 1):
        self.obs_queue = deque(maxlen=max_len)
        self.obs_queue.extend([0] * max_len)
        self.internal_obs_min: npt.ArrayLike = -np.ones(max_len)
        self.internal_obs_max: npt.ArrayLike = np.ones(max_len)

    def internal_obs(self) -> npt.ArrayLike:
        current_vel = normalize(self.env.agent.angular_velocity,
                                MobileRobot().cfg.ANGULAR_ACCELERATION_MIN,
                                MobileRobot().cfg.ANGULAR_ACCELERATION_MAX)
        self.obs_queue.append(current_vel)
        return self.obs_queue

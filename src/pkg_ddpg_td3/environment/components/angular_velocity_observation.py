import numpy.typing as npt
from . import Component
from .. import MobileRobot
from .utils import normalize


class AngularVelocityObservation(Component):
    """
    Observes ATR angular velocity
    """
    internal_obs_min: npt.ArrayLike = [-1]
    internal_obs_max: npt.ArrayLike = [1]

    def internal_obs(self) -> npt.ArrayLike:
        return [
            normalize(
                self.env.agent.angular_velocity,
                MobileRobot().cfg.ANGULAR_ACCELERATION_MIN,
                MobileRobot().cfg.ANGULAR_ACCELERATION_MAX
            )
        ]
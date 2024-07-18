import numpy as np
import numpy.typing as npt
from . import Component

class GoalPositionObservation(Component):
    """
    Observes goal position
    """
    internal_obs_min: npt.ArrayLike = np.zeros(2) - 100
    internal_obs_max: npt.ArrayLike = np.zeros(2) + 100

    def internal_obs(self) -> npt.ArrayLike:
        return self.env.goal.position
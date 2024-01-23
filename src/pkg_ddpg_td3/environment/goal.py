import numpy.typing as npt
import numpy as np

class Goal:
    """
    Describes the position of the reference path goal
    """
    def __init__(self, position: npt.ArrayLike):
        self.position = np.asarray(position, dtype=np.float32)

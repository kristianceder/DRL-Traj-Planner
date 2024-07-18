from . import Component


class BinaryCollisionReward(Component):
    """
    Gives a constant negative reward when the robot collides with something
    """
    def __init__(self, factor: float = None):
        self.factor = factor

    def step(self, action: int) -> float:
        reward = 0 if self.env.collided else 1
        return reward
from . import Component


class SpeedReward(Component):
    """
    Gives reward proportional to -(speed - reference speed)²
    """
    def __init__(self, factor: float, reference_speed: float, tau=None):
        self.factor = factor
        self.tau = tau
        self.reference_speed = reference_speed
    
    def step(self, action: int) -> float:
        error = self.env.agent.speed - self.reference_speed
        if self.tau is None:
            reward = error**2
        else:
            if error > 0:
                reward = self.tau * error**2
            else:
                reward = (1 - self.tau) * error**2

        return -self.env.time_step * self.factor * reward

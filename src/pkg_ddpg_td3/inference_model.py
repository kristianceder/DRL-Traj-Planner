"""
Code used to inference DDPG agents.

"""

from stable_baselines3 import DDPG
from torch import no_grad

class inference_model():
    def __init__(self,path_to_model, env) -> None:
        self.model = DDPG.load(f"{path_to_model}/best_model", env=env)

    def get_action(self,obs):
        with no_grad():
            return self.model.predict(obs, deterministic=True)[0]

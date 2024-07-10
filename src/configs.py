from pydantic import BaseModel
from typing import Optional

class SACConfig(BaseModel):
    seed: Optional[int] = None
    max_eps_steps: int = 500

    # collector
    total_frames: int = 150_000
    init_random_frames: int = 5000
    frames_per_batch: int = 1000
    init_env_steps: int = 1000
    env_per_collector: int = 1
    reset_at_each_iter: bool = False
    
    # replay
    replay_buffer_size: int = 1000000
    prioritize: int = 0
    scratch_dir: None = None

    # optim
    utd_ratio: float = 1.0
    gamma: float = 0.99
    loss_function: str = "l2"
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    weight_decay: float = 0.0
    batch_size: int = 256
    target_update_polyak: float = 0.995
    alpha_init: float = 1.0
    adam_eps: float = 1.0e-8
    
    # nets
    hidden_sizes: list = [256, 256]
    activation: str = "relu"
    default_policy_scale: float = 1.0
    scale_lb: float = 0.1
    device: str = "cpu"

    # eval
    eval_iter: int = 5000


class BaseConfig(BaseModel):
    env_name: str = "TrajectoryPlannerEnvironmentRaysReward1-v0"
    seed: int = 0

    sac: SACConfig = SACConfig()

    def __init__(self, **data):
        super().__init__(**data)
        self.sac.seed = self.seed
from pydantic import BaseModel
from typing import Optional

class RLConfig(BaseModel):
    seed: Optional[int] = None
    max_eps_steps: int = 400

    # collector
    total_frames: int = 50_000
    init_random_frames: Optional[int] = 5_000
    frames_per_batch: int = 1000
    init_env_steps: int = 5_000
    env_per_collector: int = 1
    reset_at_each_iter: bool = False

    # replay
    replay_buffer_size: int = 100_000
    prioritize: int = 0
    scratch_dir: None = None

    # nets
    hidden_sizes: list = [256, 256]
    activation: str = "relu"
    default_policy_scale: float = 1.0
    scale_lb: float = 0.1
    device: Optional[str] = None

    # optim
    utd_ratio: float = 1.0
    gamma: float = 0.99
    batch_size: int = 256
    weight_decay: float = 0.0
    adam_eps: float = 1.0e-8

    # eval
    eval_iter: int = 5000

    # lr schedule
    use_lr_schedule: bool = False


class SACConfig(RLConfig):
    # optim
    loss_function: str = "l2"
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    target_update_polyak: float = 0.995
    alpha_init: float = 1.0


class PPOConfig(RLConfig):
    lr: float = 3.0e-4
    clip_epsilon: float = 0.2,
    entropy_bonus: bool = True,
    samples_mc_entropy: int = 1,
    entropy_coef: float = 0.01,
    critic_coef: float = 1.0,
    loss_critic_type: str = "smooth_l1",
    normalize_advantage: bool = False,
    max_grad_norm: float = 1.0

    # GAE
    lmbda: float = 0.95


class PretrainConfig(BaseModel):
    lr: float = 3e-4
    bs: int = 128
    epochs: int = 200
    rollout_steps: int = 5_000


class BaseConfig(BaseModel):
    env_name: str = "TrajectoryPlannerEnvironmentRaysReward1-v0"
    seed: int = 10
    device: str = "cpu"
    use_vec_norm: bool = False

    algo: str = "sac"

    pretrain: PretrainConfig = PretrainConfig()
    sac: SACConfig = SACConfig()
    ppo: PPOConfig = PPOConfig()

    def __init__(self, **data):
        super().__init__(**data)
        # TODO solve this more elegently
        self.sac.seed = self.seed
        self.sac.device = self.device
        self.ppo.seed = self.seed
        self.ppo.device = self.device
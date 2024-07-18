from pydantic import BaseModel
from typing import Optional


class RLConfig(BaseModel):
    seed: Optional[int] = None
    max_eps_steps: int = 400

    # collector
    total_frames: int = 100_000
    init_random_frames: Optional[int] = 5_000
    frames_per_batch: int = 5_000
    init_env_steps: int = 5_000
    env_per_collector: int = 1
    reset_at_each_iter: bool = False
    use_multicollector: bool = False

    # replay
    replay_buffer_size: int = 5_000
    prioritize: bool = False
    scratch_dir: None = None
    prefetch: Optional[int] = None

    # nets
    hidden_sizes: list = [32, 32, 32]
    activation: str = "tanh"  # choices: "relu", "tanh", "leaky_relu"
    default_policy_scale: float = 1.0
    scale_lb: float = 0.1
    device: Optional[str] = None
    collector_device: Optional[str] = None

    # optim
    utd_ratio: float = 1.0
    gamma: float = 0.99
    batch_size: int = 256
    weight_decay: float = 0.0
    adam_eps: float = 1.0e-8

    # eval
    eval_iter: int = 10_000

    # lr schedule
    use_lr_schedule: bool = False
    first_reduce_frame: int = 10_000


class SACConfig(RLConfig):
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
    env_name: str = "TrajectoryPlannerEnvironmentRaysReward3-v0"
    seed: int = 10
    collector_device: str = "cpu"
    device: str = "cpu"
    use_vec_norm: bool = False
    n_envs: int = 1

    algo: str = "ppo"

    pretrain: PretrainConfig = PretrainConfig()
    sac: SACConfig = SACConfig()
    ppo: PPOConfig = PPOConfig()

    def __init__(self, **data):
        super().__init__(**data)
        # TODO solve this more elegently
        self.sac.seed = self.seed
        self.sac.device = self.device
        self.sac.collector_device = self.collector_device
        self.ppo.seed = self.seed
        self.ppo.device = self.device
        self.ppo.collector_device = self.collector_device

        if self.algo == "ppo":
            assert self.ppo.replay_buffer_size == self.ppo.frames_per_batch, \
                "Invalid ppo replay buffer size"

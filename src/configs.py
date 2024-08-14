from pydantic import BaseModel
from typing import Optional, List


class CurriculumConfig(BaseModel):
    steps_stage_1: int = 25_000  # add collision penalty
    steps_stage_2: int = 25_000  # add speed penalty
    steps_stage_3: int = 25_000  # add acceleration penalty

    reset_n_critic_layers: Optional[int] = 10
    reset_buffer: bool = True


class RLConfig(BaseModel):
    seed: Optional[int] = None
    max_eps_steps: int = 300
    reset_pretrained_actor: bool = False

    curriculum: CurriculumConfig = CurriculumConfig()

    # collector
    total_frames: int = 50_000
    init_random_frames: Optional[int] = 5_000
    # init_e_greedy: float = 0.2
    frames_per_batch: int = 1_000
    init_env_steps: int = 5_000
    env_per_collector: int = 1
    reset_at_each_iter: bool = False
    use_multicollector: bool = False

    # replay
    scratch_dir: None = None
    prefetch: Optional[int] = None

    # nets
    hidden_sizes: List[int] = [32, 32, 32]
    activation: str = "tanh"  # choices: "relu", "tanh", "leaky_relu"
    actor_dropout: Optional[float] = None
    critic_dropout: Optional[float] = None
    default_policy_scale: float = 1.0
    scale_lb: float = 0.1
    device: Optional[str] = None
    collector_device: Optional[str] = None

    # optim
    gamma: float = 0.99
    gamma_end: float = 0.99
    weight_decay: float = 0.0
    adam_eps: float = 1.0e-8
    max_grad_norm: float = 1.0
    loss_function: str = "smooth_l1"

    # shared parameters
    replay_buffer_size: int = 100_000
    prioritize: bool = False
    batch_size: int = 128
    utd_ratio: float = 1.0

    # network resets
    n_reset_layers: Optional[int] = None
    n_reset_layers_critic: Optional[int] = None

    # eval
    eval_iter: int = 200_000

    # lr schedule
    use_lr_schedule: bool = False
    first_reduce_frame: int = 10_000
    eta_min: Optional[float] = 1e-6  # minimum learning rate


class SACConfig(RLConfig):
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    target_update_polyak: float = 0.995
    alpha_init: float = 1.0
    min_alpha: Optional[float] = None


class TD3Config(RLConfig):
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    target_update_polyak: float = 0.995
    policy_noise: float = 0.2
    noise_clip: float = 0.5


class PPOConfig(RLConfig):
    lr: float = 3.0e-4
    clip_epsilon: float = 0.2
    entropy_bonus: bool = True
    samples_mc_entropy: int = 1
    entropy_coef: float = 0.01
    critic_coef: float = 1.0
    normalize_advantage: bool = False
    weight_decay: float = 0.0

    # GAE
    lmbda: float = 0.95

    # # shared parameters
    # frames_per_batch: int = 5_000
    # replay_buffer_size: int = 5_000
    # prioritize: bool = False
    # batch_size: int = 64
    # utd_ratio: float = 1.0


class PretrainConfig(BaseModel):
    lr: float = 3e-4
    bs: int = 128
    epochs: int = 200
    rollout_steps: int = 10_000


class BaseConfig(BaseModel):
    # v0 is original rewards, v1 is minimal
    # env 1 is original observations, 3 is updated
    env_name: str = "TrajectoryPlannerEnvironmentRaysReward3-v3"
    # map_key choices = ['dynamic_convex_obstacle', 'static_nonconvex_obstacle', 'corridor']
    map_key: str = 'dynamic_convex_obstacle'
    seed: int = 200  # 10, 100, 200
    collector_device: str = "cpu"
    device: str = "cpu"
    use_vec_norm: bool = False
    n_envs: int = 1

    algo: str = "td3"

    pretrain: PretrainConfig = PretrainConfig()
    sac: SACConfig = SACConfig()
    ppo: PPOConfig = PPOConfig()
    td3: TD3Config = TD3Config()

    def __init__(self, **data):
        super().__init__(**data)
        # TODO solve this more elegently
        self.sac.seed = self.seed
        self.sac.device = self.device
        self.sac.collector_device = self.collector_device
        self.ppo.seed = self.seed
        self.ppo.device = self.device
        self.ppo.collector_device = self.collector_device
        self.td3.seed = self.seed
        self.td3.device = self.device
        self.td3.collector_device = self.collector_device

        if self.algo == "ppo":
            assert self.ppo.replay_buffer_size == self.ppo.frames_per_batch, \
                "Invalid ppo replay buffer size"

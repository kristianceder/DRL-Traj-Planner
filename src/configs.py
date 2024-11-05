from pydantic import BaseModel
from typing import Optional, List


class CurriculumConfig(BaseModel):
    steps_stage_1: int = 25_000

    reset_n_critic_layers: Optional[int] = None
    reset_n_actor_layers: Optional[int] = None
    reset_buffer: bool = False
    reset_frames: bool = False
    num_updates_after_update: int = 25_000

    # "g": ReachGoal, "s": Speed, "d": PathProgress, "c": Collision, "a": Acceleration, "x": CrossTrack
    base_reward_keys: str = "gds"
    all_reward_keys: str = "gdcsax"


class RLConfig(BaseModel):
    seed: Optional[int] = None
    max_eps_steps: int = 400
    reset_pretrained_actor: bool = False
    reward_mode: str = ""  # will be overwritten in post init

    curriculum: CurriculumConfig = CurriculumConfig()

    # collector
    total_frames: int = 50_000
    init_random_frames: Optional[int] = 5_000
    frames_per_batch: int = 1_000
    init_env_steps: int = 5_000
    env_per_collector: int = 1
    reset_at_each_iter: bool = False
    use_multicollector: bool = False

    # replay
    scratch_dir: None = None
    prefetch: Optional[int] = 2

    # nets
    hidden_sizes: List[int] = [256, 256]
    activation: str = "relu"  # choices: "relu", "tanh", "leaky_relu"
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
    replay_buffer_size: int = 800_000 # NOTE this cannot be 1_000_000 cause the the ser
    prioritize: bool = False
    batch_size: int = 128
    utd_ratio: float = 1.0

    # network resets
    n_reset_layers: Optional[int] = None
    n_reset_layers_critic: Optional[int] = None

    # eval
    eval_iter: int = 100_000
    eval_rollout_steps: int = 5_000

    # lr schedule
    use_lr_schedule: bool = False
    first_reduce_frame: int = 10_000
    eta_min: Optional[float] = 1e-6  # minimum learning rate

    # kl approx
    kl_approx_method: str = "abs"  # choices: ["logp", "abs"]


class SACConfig(RLConfig):
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    alpha_lr: float = 3.0e-4
    target_update_polyak: float = 0.995
    alpha_init: float = 1.0
    min_alpha: Optional[float] = 0.01
    kl_beta: Optional[float] = None


class TD3Config(RLConfig):
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    target_update_polyak: float = 0.995
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    sigma_init: float = 0.9
    sigmn_end: float = 0.1


class DDPGConfig(RLConfig):
    actor_lr: float = 3.0e-4
    critic_lr: float = 3.0e-4
    target_update_polyak: float = 0.995
    sigma_init: float = 0.9
    sigmn_end: float = 0.1


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

    # shared parameters
    replay_buffer_size: int = 1_000
    # batch_size: int = 64


class PretrainConfig(BaseModel):
    lr: float = 3e-4
    bs: int = 128
    epochs: int = 200
    rollout_steps: int = 10_000


class BaseConfig(BaseModel):
    # v0 is original rewards, v1 is minimal, v2 multiply, v3 sum, v4 curriculum
    # env 1 is original observations, 3 is updated
    env_name: str = "TrajectoryPlannerEnvironmentRaysReward3-v3"
    # env_name: str = "TrajectoryPlannerEnvironmentImgsReward3-v0"
    reward_mode: Optional[str] = "sum"  # vals: sum, curriculum_step
    map_key: str = "dynamic_convex_obstacle"
    seed: int = 10  # 10, 100, 200
    collector_device: str = "cpu"
    device: str = "cpu"
    use_vec_norm: bool = False
    n_envs: int = 1
    use_wandb: bool = True

    w1: float = 0.1  # speed
    w2: float = 0.1  # acceleration
    w3: float = .25   # path progress goal distance
    w4: float = 0.1  # cross track
    w5: float = 0.    # NOT USED obstacle distance

    algo: str = "sac"  # choices: ["sac", "ppo", "td3", "ddpg"]

    pretrain: PretrainConfig = PretrainConfig()
    sac: SACConfig = SACConfig()
    ppo: PPOConfig = PPOConfig()
    td3: TD3Config = TD3Config()
    ddpg: DDPGConfig = DDPGConfig()

    def __init__(self, **data):
        super().__init__(**data)
        # Automatically propagate common attributes to all algorithm configurations
        for algo_config in [self.sac, self.ppo, self.td3, self.ddpg]:
            algo_config.seed = self.seed
            algo_config.device = self.device
            algo_config.collector_device = self.collector_device
            algo_config.reward_mode = self.reward_mode

        if self.algo == "ppo":
            assert self.ppo.replay_buffer_size == self.ppo.frames_per_batch, \
                "Invalid ppo replay buffer size"

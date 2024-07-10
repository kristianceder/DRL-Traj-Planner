from torchrl.envs import (
    GymEnv,
    CatTensors,
    Compose,
    DoubleToFloat,
    TransformedEnv,
)
from torchrl.envs.transforms import (
    InitTracker, 
    RewardSum, 
    StepCounter,
)

def make_env(generate_map, config):
    env = GymEnv(config.env_name, generate_map = generate_map)

    transformed_env = TransformedEnv(
        env,
        Compose(
            InitTracker(),
            StepCounter(config.sac.max_eps_steps),
            DoubleToFloat(),
            RewardSum(),
            CatTensors(in_keys=['internal', 'external'], out_key="observation")
        ),
    )
    return transformed_env
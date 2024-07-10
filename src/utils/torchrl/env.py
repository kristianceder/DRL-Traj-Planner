import torch
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
from torchrl.envs.utils import ExplorationType, set_exploration_type


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


def render_rollout(eval_env, model, config):
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        state = eval_env.reset()
        steps = 0
        ep_rwd = torch.zeros(1)
        for i in range(0, 2_000):
            action = state.copy()
            action['action'] = model.sample_action(state, sample_mean=True)
            # action = eval_env.rand_action(state)
            next_state = eval_env.step(action)

            steps += 1
            ep_rwd += next_state['next']['reward']

            # Only render every third frame for performance (matplotlib is slow)
            if i % 3 == 0 and i > 0:
                eval_env.render()

            if next_state['next']['done'] or steps > config.sac.max_eps_steps:
                print('reset')
                state = eval_env.reset()
                steps = 0
            else:
                state = next_state
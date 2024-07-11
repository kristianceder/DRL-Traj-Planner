import torch
from torchrl.envs import (
    default_info_dict_reader,
    step_mdp,
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


def make_env(config, **kwargs):
    env = GymEnv(config.env_name, **kwargs)

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
    reader = default_info_dict_reader(["success"])
    transformed_env.set_info_dict_reader(info_dict_reader=reader)

    return transformed_env


def render_rollout(eval_env, model, config, n_steps=2_000):
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        state = eval_env.reset()
        steps = 0
        ep_rwd = torch.zeros(1)
        for i in range(n_steps):
            action = model.model["policy"](state)
            next_state = eval_env.step(action)

            steps += 1
            ep_rwd += next_state['next']['reward']

            # Only render every third frame for performance (matplotlib is slow)
            if i % 3 == 0 and i > 0:
                eval_env.render()

            if next_state['next']['done'] or steps > config.sac.max_eps_steps:
                print(f'reset, ep reward {ep_rwd.item()}')
                state = eval_env.reset()
                steps = 0
                ep_rwd = torch.zeros(1)
            else:
                state = step_mdp(next_state)
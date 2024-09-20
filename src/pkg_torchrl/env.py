import torch
from torchrl.envs import (
    default_info_dict_reader,
    step_mdp,
    GymEnv,
    CatTensors,
    Compose,
    DoubleToFloat,
    TransformedEnv,
    ParallelEnv,
)
from torchrl.envs.transforms import (
    InitTracker,
    RewardSum,
    StepCounter,
    VecNorm
)
from torchrl.envs.utils import ExplorationType, set_exploration_type


def make_env(config, **kwargs):
    raw_env = GymEnv(config.env_name,
                     w1=config.w1,
                     w2=config.w2,
                     w3=config.w3,
                     w4=config.w4,
                     reward_mode=config.reward_mode,
                     config=config,
                     k0=config.k0,
                     kc=config.kc,
                     device=config.device,
                     **kwargs)

    def make_t_env():
        transform_list = [
            InitTracker(),
            StepCounter(config.sac.max_eps_steps),
            DoubleToFloat(),
            RewardSum(),
            CatTensors(in_keys=['internal', 'external'], out_key="observation"),
        ]

        if config.use_vec_norm:
            transform_list += [VecNorm(decay=0.9),]
        t_env = TransformedEnv(raw_env, Compose(*transform_list))
        reader = default_info_dict_reader(["success", "collided", "full_reward", "reward_tensor"])
        t_env.set_info_dict_reader(info_dict_reader=reader)
        return t_env

    if config.n_envs == 1:
        env = make_t_env()
    else:
        env = ParallelEnv(
            create_env_fn=lambda: make_t_env(),
            num_workers=config.n_envs,
            pin_memory=False,
        )

    return env


def render_rollout(eval_env, model, config, n_steps=2_000, is_torchrl=True):
    with set_exploration_type(ExplorationType.MODE), torch.no_grad():
        state = eval_env.reset()
        steps = 0
        ep_rwd = torch.zeros(1).to(config.device)
        for i in range(n_steps):
            action = model.model["policy"](state)
            next_state = eval_env.step(action)

            steps += 1
            ep_rwd += next_state['next']['reward']

            # Only render every third frame for performance (matplotlib is slow)
            if i % 3 == 0 and i > 0:
                eval_env.render()

            if next_state['next']['done'] or steps > config.sac.max_eps_steps:
                print(f'Episode reward {ep_rwd.item():.2f}')
                state = eval_env.reset()
                steps = 0
                ep_rwd = torch.zeros(1).to(config.device)
            else:
                state = step_mdp(next_state)
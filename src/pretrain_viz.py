from pkg_ddpg_td3.utils.map import generate_map_static
from pkg_torchrl.env import make_env
from pkg_torchrl.pretrain import SimpleController, rollout

from configs import BaseConfig


def main():
    config = BaseConfig()
    env = make_env(config, generate_map=generate_map_static)

    # load data
    ref_speed = 1.0

    model = SimpleController(ref_speed)
    data = rollout(env, model, config, n_steps=1_000, do_render=True)

    print(len(data['obs']))


if __name__ == '__main__':
    main()
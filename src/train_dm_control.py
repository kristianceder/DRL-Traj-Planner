"""
Code used to train the continous DRL agents, DDPG and TD3.

Eight different example agent variants are present, the first four of which 
corresponds to the DDPG algorithm, while the second four are TD3. You can 
select which example agent to train and evaluate by setting the ``index`` 
varaible as the first argument from the command line. 
This is generally done by the slurm array function as seen in ``SLURM_jobscript.sh``.
"""

# import random
import logging
# from datetime import datetime
# from pathlib import Path

from torchrl.envs import (
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
)

# import wandb
# import torch
# import numpy as np
# from pkg_torchrl.env import make_env
from pkg_torchrl.sac import SAC
# from pkg_torchrl.ppo import PPO
# from pkg_torchrl.td3 import TD3
# import shimmy # to find dm_control envs
# import gymnasium as gym
from torchrl.record import VideoRecorder
from torchrl.record.loggers.wandb import WandbLogger
from torchrl.envs import TransformedEnv, DMControlEnv


from configs import BaseConfig

logging.basicConfig(level=logging.ERROR)


def make_env():
    raw_env = DMControlEnv("finger", "spin", from_pixels=True, device='cpu') # ['spin', 'turn_easy', 'turn_hard']
    transform_list = [
        InitTracker(),
        StepCounter(),
        DoubleToFloat(),
        RewardSum(),
        CatTensors(in_keys=['position', 'velocity', 'touch'], out_key="observation"),
    ]
    env = TransformedEnv(raw_env, Compose(*transform_list))
    return env


def main():
    # _ = wandb.init(
    #     project="DM_control_finger",
    #     tags=["testing"],
    # )
    # config = BaseConfig(**wandb.config)
    config = BaseConfig()

    # wandb.config.update(config.model_dump())

    # print(f"seed: {config.seed}")
    # random.seed(config.seed)
    # np.random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # torch.cuda.manual_seed(config.seed)

    # TODO (kilian)
    # 1. Train simple algo
    # 2. Add acceleration-minimizing term
    # 3. Compare to curriculum

    logger = WandbLogger(project="dm_control_finger", exp_name="finger_spin", tags=["testing"])

    # train
    train_env = make_env()
    eval_env = make_env()
    agent = SAC(config.sac, train_env, eval_env)
    agent.train()

    video_env = TransformedEnv(eval_env, VideoRecorder(logger=logger, tag="run_video"))
    video_env.rollout(200)
    video_env.transform.dump()
    video_env.close()
    train_env.close()
    eval_env.close()

    # # train_env = make_env(config, use_wandb=True)
    # # eval_env = make_env(config)

    # algo_config = getattr(config, config.algo.lower())
    # model = eval(config.algo.upper())(algo_config, train_env, eval_env)
    # models_path = Path('../Model/testing')

    # timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    # path = models_path / f"{timestamp}_{config.algo.upper()}"
    # path.mkdir(exist_ok=True, parents=True)
    # wandb.config["path"] = path
    # wandb.config["map"] = generate_map.__name__

    # model.train()
    # model.save(f"{path}/final_model.pth")
    # logging.info(f"Final model saved to {path}")


if __name__ == "__main__":
    main()

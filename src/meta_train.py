import random
import logging
from datetime import datetime
from pathlib import Path

import wandb
import torch
import numpy as np
from pkg_map.utils import get_map
from pkg_torchrl.env import make_env
from pkg_torchrl.sac import SAC
from pkg_torchrl.meta.sac import MetaSAC

from configs import BaseConfig

logging.basicConfig(level=logging.INFO)

def set_seed(config):
    logging.info(f"seed: {config.seed}")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)


def main():
    _ = wandb.init(
        project="AutoCurriculum",
        tags=["meta_training"],
    )
    config = BaseConfig(**wandb.config)
    config.reward_mode = "curriculum_step"
    wandb.config.update(config.model_dump())
    set_seed(config)

    generate_map = get_map(config.map_key)

    train_env = make_env(config, generate_map=generate_map, use_wandb=True)
    eval_env = make_env(config, generate_map=generate_map)

    # load lower RL algo
    algo_config = getattr(config, config.algo.lower())
    algo_config.reward_mode = "curriculum_step"  # always have curriculum reward mode
    algo_class = eval(config.algo.upper())
    models_path = Path('../Model/testing')

    timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    path = models_path / f"{timestamp}_{config.algo.upper()}"
    path.mkdir(exist_ok=True, parents=True)
    wandb.config["path"] = path
    wandb.config["map"] = generate_map.__name__

    # load meta algo
    meta_algo = MetaSAC(config.meta)
    model = meta_algo.train(algo_class, algo_config, train_env, eval_env)

    model.save(f"{path}/final_model.pth")
    meta_algo.save(f"{path}/meta_model.pth")
    logging.info(f"Models saved to {path}")


if __name__ == '__main__':
    main()
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
from pkg_torchrl.meta_sac import MetaSAC

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
        project="DRL-Traj-Planner",
        tags=["meta_training"],
    )
    config = BaseConfig(**wandb.config)
    wandb.config.update(config.model_dump())
    set_seed(config)

    generate_map = get_map(config.map_key)

    train_env = make_env(config, generate_map=generate_map, use_wandb=True)
    eval_env = make_env(config, generate_map=generate_map)

    # load lower RL algo
    algo_config = getattr(config, config.algo.lower())
    model = eval(config.algo.upper())(algo_config, train_env, eval_env)
    models_path = Path('../Model/testing')

    timestamp = datetime.now().strftime("%y_%m_%d_%H_%M_%S")
    path = models_path / f"{timestamp}_{config.algo.upper()}"
    path.mkdir(exist_ok=True, parents=True)
    wandb.config["path"] = path
    wandb.config["map"] = generate_map.__name__

    # load meta algo
    meta_algo = MetaSAC(config.sac) # TODO have separate meta config
    meta_algo.train(model, config)
    
    model.save(f"{path}/final_model.pth")
    logging.info(f"Final model saved to {path}")


if __name__ == '__main__':
    main()
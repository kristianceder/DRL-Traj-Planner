import torch
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.value import GAE

from utils.torchrl.base import (
    AlgoBase,
    make_collector,
    make_replay_buffer
)


class PPO(AlgoBase):
    """PPO actor critic trainer"""
    def __init__(self, config, train_env, eval_env):
        super().__init__(config, train_env, eval_env,
                         in_keys_actor=['observation'],
                         in_keys_value=['observation'])

        self.advantage_module = GAE(
            gamma=self.config.gamma, lmbda=self.config.lmbda, value_network=self.model["value"], average_gae=True
        )
            
    def _init_loss_module(self):
        # Create SAC loss
        self.loss_module = ClipPPOLoss(
            actor_network=self.model["policy"],
            critic_network=self.model["value"],
            clip_epsilon=self.config.clip_epsilon,
            entropy_bonus=self.config.entropy_bonus,
            entropy_coef=self.config.entropy_coef,
            critic_coef=self.config.critic_coef,
            # loss_critic_type=self.config.loss_critic_type,
            normalize_advantage=self.config.normalize_advantage,
        )

    def _init_optimizer(self):
        self.optim = torch.optim.Adam(
            self.loss_module.parameters(),
            self.config.lr,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_eps,
        )

    def _loss_backward(self, loss_td):
        loss_value = (
            loss_td["loss_objective"]
            + loss_td["loss_critic"]
            + loss_td["loss_entropy"]
        )
        loss_value.backward()
        torch.nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.config.max_grad_norm)
        self.optim.step()
        self.optim.zero_grad()
        return loss_td.select(
                        "loss_objective", "loss_critic", "loss_entropy"
                    ).detach()


if __name__ == '__main__':
    from pkg_ddpg_td3.utils.map import generate_map_dynamic
    from utils.torchrl.env import make_env
    from configs import BaseConfig

    config =  BaseConfig()
    train_env = make_env(generate_map_dynamic, config)
    eval_env = make_env(generate_map_dynamic, config)

    model = PPO(config.ppo, train_env, eval_env)

    model.train(wandb=False)

    model.save('ppo.pth')
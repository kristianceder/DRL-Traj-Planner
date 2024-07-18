import torch
from torchrl.objectives.sac import SACLoss
from torchrl.objectives import SoftUpdate

from utils.torchrl.base import (
    AlgoBase,
    make_collector,
    make_replay_buffer
)


class SAC(AlgoBase):
    """Soft actor critic trainer"""
    def __init__(self, config, train_env, eval_env):
        super().__init__(config, train_env, eval_env,
                         in_keys_actor=['observation'],
                         in_keys_value=['action', 'observation'])
        self.replay_buffer = make_replay_buffer(
            batch_size=self.config.batch_size,
            prioritize=self.config.prioritize,
            buffer_size=self.config.replay_buffer_size,
            scratch_dir=self.config.scratch_dir,
            device="cpu",
        )
    
    def _init_loss_module(self):
        # Create SAC loss
        self.loss_module = SACLoss(
            actor_network=self.model["policy"],
            qvalue_network=self.model["value"],
            num_qvalue_nets=2,
            loss_function=self.config.loss_function,
            delay_actor=False,
            delay_qvalue=True,
            alpha_init=self.config.alpha_init,
        )
        self.loss_module.make_value_estimator(gamma=self.config.gamma)

        self.target_net_updater = SoftUpdate(
            self.loss_module, eps=self.config.target_update_polyak
        )

    def _init_optimizer(self):
        critic_params = list(self.loss_module.qvalue_network_params.flatten_keys().values())
        actor_params = list(self.loss_module.actor_network_params.flatten_keys().values())
        
        self.optimizers = {}
        self.optimizers["actor"] = torch.optim.Adam(
            actor_params,
            lr=self.config.actor_lr,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_eps,
        )
        self.optimizers["critic"] = torch.optim.Adam(
            critic_params,
            lr=self.config.critic_lr,
            weight_decay=self.config.weight_decay,
            eps=self.config.adam_eps,
        )
        self.optimizers["alpha"] = torch.optim.Adam(
            [self.loss_module.log_alpha],
            lr=self.config.alpha_lr,
        )

    def _loss_backward(self, loss_td):
        actor_loss = loss_td["loss_actor"]
        q_loss = loss_td["loss_qvalue"]
        alpha_loss = loss_td["loss_alpha"]

        # Update actor
        self.optimizers["actor"].zero_grad()
        actor_loss.backward()
        self.optimizers["actor"].step()

        # Update critic
        self.optimizers["critic"].zero_grad()
        q_loss.backward()
        self.optimizers["critic"].step()

        # Update alpha
        self.optimizers["alpha"].zero_grad()
        alpha_loss.backward()
        self.optimizers["alpha"].step()
        return loss_td.select(
                        "loss_actor", "loss_qvalue", "loss_alpha"
                    ).detach()
    
   


if __name__ == '__main__':
    from pkg_ddpg_td3.utils.map import generate_map_dynamic
    from utils.torchrl.env import make_env
    from configs import BaseConfig

    config =  BaseConfig()
    train_env = make_env(generate_map_dynamic, config)
    eval_env = make_env(generate_map_dynamic, config)

    model = SAC(config.sac, train_env, eval_env)

    model.train(wandb=False)

    model.save('sac.pth')
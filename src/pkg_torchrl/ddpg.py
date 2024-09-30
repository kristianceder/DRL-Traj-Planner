import torch
from torchrl.objectives.ddpg import DDPGLoss
from torchrl.objectives import SoftUpdate, ValueEstimators

from pkg_torchrl.base import AlgoBase


class DDPG(AlgoBase):
    """DDPG actor critic trainer"""
    def __init__(self, config, train_env, eval_env):
        if "pixels" in train_env.observation_spec.keys():
            in_keys_actor = ["pixels", "internal"]
            in_keys_value = ["pixels", "internal", "action"]
        else:
            in_keys_actor = ["observation"]
            in_keys_value = ["observation", "action"]
        super().__init__(config, train_env, eval_env,
                         in_keys_actor=in_keys_actor,
                         in_keys_value=in_keys_value,
                         deterministic=True)
            
    def _init_loss_module(self):
        self.loss_module = DDPGLoss(
            actor_network=self.model["policy"],
            value_network=self.model["value"],
            loss_function=self.config.loss_function,
        )
        self.loss_module.make_value_estimator(ValueEstimators.TD0, gamma=self.config.gamma)

        self.target_net_updater = SoftUpdate(
            self.loss_module, eps=self.config.target_update_polyak
        )

    def _init_optimizer(self):
        critic_params = list(self.loss_module.value_network_params.flatten_keys().values())
        actor_params = list(self.loss_module.actor_network_params.flatten_keys().values())
        self.optim = {
            "actor": torch.optim.Adam(
                actor_params,
                lr=self.config.actor_lr,
                weight_decay=self.config.weight_decay,
                eps=self.config.adam_eps,
            ),
            "critic": torch.optim.Adam(
                critic_params,
                lr=self.config.critic_lr,
                weight_decay=self.config.weight_decay,
                eps=self.config.adam_eps,
            ),
        }

    def _loss_backward(self, loss_td):
        # Update actor
        actor_loss = loss_td["loss_actor"]
        actor_loss.backward()
        params = self.optim["actor"].param_groups[0]["params"]
        torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)

        self.optim["actor"].step()
        self.optim["actor"].zero_grad()

        # Update critic
        q_loss = loss_td["loss_value"]
        q_loss.backward()
        params = self.optim["critic"].param_groups[0]["params"]
        torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)

        self.optim["critic"].step()
        self.optim["critic"].zero_grad()

        return loss_td.detach()


if __name__ == '__main__':
    from pkg_ddpg_td3.utils.map import generate_map_dynamic
    from pkg_torchrl.env import make_env
    from configs import BaseConfig

    config = BaseConfig()
    train_env = make_env(generate_map_dynamic, config)
    eval_env = make_env(generate_map_dynamic, config)

    model = DDPG(config.ddpg, train_env, eval_env)

    model.train(wandb=False)
    model.save('ddpg.pth')

import torch
from torchrl.objectives.sac import SACLoss
from torchrl.objectives import SoftUpdate

from pkg_torchrl.base import AlgoBase


class SAC(AlgoBase):
    """Soft actor critic trainer"""
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
                         deterministic=False)
    
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
            min_alpha=self.config.min_alpha,
        )
        self.loss_module.make_value_estimator(gamma=self.config.gamma)

        self.target_net_updater = SoftUpdate(
            self.loss_module, eps=self.config.target_update_polyak
        )

    def _init_optimizer(self):
        critic_params = list(self.loss_module.qvalue_network_params.flatten_keys().values())
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
            "alpha": torch.optim.Adam(
                [self.loss_module.log_alpha],
                lr=self.config.alpha_lr,
            )
        }

    def _loss_backward(self, loss_td):
        loss_keys = ["loss_actor", "loss_qvalue", "loss_alpha"]
        optim_keys = ["actor", "critic", "alpha"]

        for l_key, o_key in zip(loss_keys, optim_keys):
            loss = loss_td[l_key]

            # add kl loss if present
            if (self.config.kl_beta is not None
                    and l_key == "loss_actor"
                    and "kl_loss" in loss_td.keys()):
                loss += self.config.kl_beta * loss_td["kl_loss"]

            optim = self.optim[o_key]
            loss.backward()
            params = optim.param_groups[0]["params"]
            torch.nn.utils.clip_grad_norm_(params, self.config.max_grad_norm)
            optim.step()
            optim.zero_grad()

        return loss_td.detach()


if __name__ == '__main__':
    from pkg_ddpg_td3.utils.map import generate_map_dynamic
    from pkg_torchrl.env import make_env
    from configs import BaseConfig

    config =  BaseConfig()
    train_env = make_env(generate_map_dynamic, config)
    eval_env = make_env(generate_map_dynamic, config)

    model = SAC(config.sac, train_env, eval_env)

    model.train(wandb=False)

    model.save('sac.pth')
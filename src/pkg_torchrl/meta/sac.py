import logging
from tqdm import tqdm

import wandb
import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical, Bernoulli
from torchrl.data.tensor_specs import OneHotDiscreteTensorSpec, MultiDiscreteTensorSpec
# from torchrl.modules.distributions import NormalParamExtractor, OneHotCategorical, MaskedCategorical
from torchrl.modules.tensordict_module.actors import ProbabilisticActor
# from torchrl.modules import MLP
from torchrl.modules.tensordict_module.common import SafeModule
from torchrl.objectives.sac import DiscreteSACLoss
from torchrl.objectives import SoftUpdate
from torchrl.envs.utils import ExplorationType, set_exploration_type

from pkg_torchrl.utils import make_replay_buffer, get_activation
from pkg_torchrl.sac import SAC
from pkg_torchrl.meta.networks import MetaNetwork, MetaEncoder


import torch
import torch.nn as nn
import torch.nn.functional as F


def build_meta_actor(policy_net, action_spec, in_keys):
    module = SafeModule(policy_net, in_keys=in_keys, out_keys=["logits"])
    actor = ProbabilisticActor(
        module=module,
        in_keys=["logits"],
        out_keys=["meta_action"],
        spec=action_spec,
        distribution_class=Bernoulli,
        )#OneHotCategorical)
    return actor


class MetaSAC(SAC):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        in_keys_actor = ["meta_observation", "reward_tensor", "last_meta_action"]
        in_keys_value = ["meta_observation", "reward_tensor", "last_meta_action"]

        # TODO get this from env or config
        action_size = 4
        obs_size = 178

        action_spec = MultiDiscreteTensorSpec([2]*action_size)#OneHotDiscreteTensorSpec(action_size)
        action_size = action_spec.shape[-1]

        encoder = MetaEncoder(observation_dim=obs_size, reward_dim=4)

        # policy_net = MetaNetwork(encoder=encoder, reward_dim=4, out_activation=nn.Sigmoid())
        policy_net = MetaNetwork(encoder=encoder, reward_dim=4)
        qvalue_net = MetaNetwork(encoder=encoder, reward_dim=4)

        actor = build_meta_actor(policy_net, action_spec, in_keys_actor)

        qvalue = TensorDictModule(
            qvalue_net,
            in_keys=in_keys_value,
            out_keys=["action_value"],
        )

        self.model = nn.ModuleDict({
            "policy": actor,
            "value": qvalue
        }).to(self.device)

        # init input dims
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            for net in self.model.values():
                td = TensorDict({
                    "meta_observation": torch.rand(16, 1000, obs_size),
                    "last_meta_action": torch.rand(16, action_size),
                    "reward_tensor": torch.rand(16, 1000, action_size),
                })
                td = td.to(self.device)
                net(td)
        del td

        self._init_loss_module()
        self._init_optimizer()
        self._post_init_optimizer()
    
        self.meta_buffer = make_replay_buffer(
            batch_size=self.config.batch_size,
            prioritize=self.config.prioritize,
            buffer_size=self.config.replay_buffer_size,
            scratch_dir=self.config.scratch_dir,
            device="cpu",
            prefetch=self.config.prefetch,
        )

    def _init_loss_module(self):
        # Create SAC loss
        action_spec = self.model["policy"].spec["meta_action"]
        self.loss_module = DiscreteSACLoss(
            actor_network=self.model["policy"],
            qvalue_network=self.model["value"],
            num_qvalue_nets=2,
            loss_function=self.config.loss_function,
            action_space=action_spec,
            num_actions=len(action_spec.space)
        )
        self.loss_module.set_keys(action="meta_action")
        self.loss_module.make_value_estimator(gamma=self.config.gamma)

        self.target_net_updater = SoftUpdate(
            self.loss_module, eps=self.config.target_update_polyak
        )

    def update_w(self, reward_w, meta_action):
        # TODO logging?
        reward_w += meta_action

    def train(self, lower_model: SAC):
        max_phases = self.config.n_phases
        n_iters = self.config.n_iters
        collected_meta_iters = 0

        for iter_idx in tqdm(range(n_iters), desc="Training Iterations"):
            # reward_w = torch.zeros(4)
            # reward_w[0] = 1
            # reward_w[1] = 1
            # NOTE set all weights to 1 for debugging
            reward_w = torch.ones(4)
            lower_model.set_w(reward_w)

            collected_frames = 0
            episode_reward = 0

            # get initial random frames
            # TODO this should probably be a collector
            tensordict = lower_model.train_env.rollout(
                        self.config.init_random_frames,
                        auto_cast_to_device=True,
                        break_when_any_done=False,
                    ) 

            tensordict = tensordict.reshape(-1)
            current_frames = tensordict.numel()
            lower_model.replay_buffer.extend(tensordict.cpu())
            collected_frames += current_frames

            # build meta state
            last_td = TensorDict({
                "meta_observation": tensordict["observation"][-1000:,:].unsqueeze(0),
                "reward_tensor": tensordict["next", "reward_tensor"][-1000:,:].unsqueeze(0),
                "last_meta_action": reward_w.view(1, -1),
            }, batch_size=[1])
            
            # meta episode
            for p_idx in range(max_phases):
                logging.info(f"Phase {p_idx}")

                ### select w
                with torch.no_grad():
                    out = self.model["policy"](last_td)
                # FIXME keep weights at 1 for now for debugging
                # reward_w = out["meta_action"].squeeze()
                # model.set_w(reward_w) 
                logging.info(f"Reward weights: {reward_w}")

                ### grad steps on policy
                n_train_frames = 1_000
                losses = lower_model.grad_steps(n_train_frames)
                # TODO log losses

                ### rollout

                print(lower_model.model["policy"])
                rollout_steps = 1_000
                tensordict = lower_model.train_env.rollout(
                        rollout_steps,
                        lower_model.model["policy"],
                        auto_cast_to_device=True,
                        break_when_any_done=False,
                    )
                tensordict = tensordict.reshape(-1)
                current_frames = tensordict.numel()
                tensordict = tensordict.clone()
                # FIXME why does this not work?
                lower_model.replay_buffer.extend(tensordict.cpu())
                collected_frames += current_frames

                # only keep final performance as reward
                if p_idx == max_phases - 1:
                    meta_reward = tensordict["next", "true_reward"].mean().view(1,1)
                else:
                    meta_reward = torch.zeros(1, 1)

                meta_td = TensorDict({
                    "meta_observation": last_td["meta_observation"],
                    "reward_tensor": last_td["reward_tensor"],
                    "last_meta_action": last_td["last_meta_action"],
                    ("next", "meta_observation"): tensordict["next", "observation"].unsqueeze(0),
                    ("next", "reward_tensor"): tensordict["next", "reward_tensor"].unsqueeze(0),
                    ("next", "last_meta_action"): reward_w.view(1, -1),
                    ("next", "reward"): meta_reward,
                    ("next", "done"): torch.zeros(1, 1, dtype=torch.bool),
                    ("next", "terminated"): torch.zeros(1, 1, dtype=torch.bool), # FIXME make last entry 1
                }, batch_size=[1])

                self.meta_buffer.extend(meta_td.cpu())
                collected_meta_iters += 1

                r = tensordict["next", "reward"].mean().item()
                train_metrics_to_log = {
                    "meta_train/reward": r,
                    "meta_train/true_reward": tensordict["next", "true_reward"].mean().item(),
                }
                wandb.log(train_metrics_to_log, step=collected_frames)
                
                # last_td = meta_td
                last_td = TensorDict({
                    "meta_observation": tensordict["observation"][-1000:,:].unsqueeze(0),
                    "reward_tensor": tensordict["next", "reward_tensor"][-1000:,:].unsqueeze(0),
                    "last_meta_action": reward_w.view(1, -1),
                    "reward": torch.zeros(1,1),
                }, batch_size=[1])
                episode_reward += r

            ### meta gradients
            # UTD = 1.0, might want to change to a parameter later
            num_updates = self.config.n_phases

            # train
            if collected_meta_iters >= self.config.meta_init_env_steps:
                losses = TensorDict({}, batch_size=[num_updates])
                for i in range(num_updates):
                    sampled_tensordict = self.meta_buffer.sample()

                    if sampled_tensordict.device != self.device:
                        sampled_tensordict = sampled_tensordict.to(
                            self.device
                        )
                    else:
                        sampled_tensordict = sampled_tensordict.clone()

                    loss_td = self.loss_module(sampled_tensordict)

                    loss = self._loss_backward(loss_td)
                    losses[i] = loss

                    self.target_net_updater.step()

                    if self.config.prioritize:
                        loss_key = "loss_critic" if "loss_critic" in loss else "loss_qvalue"
                        sampled_tensordict.set(
                            loss_key, loss[loss_key] * torch.ones(sampled_tensordict.shape, device=self.device))
                        self.replay_buffer.update_tensordict_priority(sampled_tensordict)

            metrics_to_log = {}
            metrics_to_log["meta_train/episode_reward"] = episode_reward
            if collected_meta_iters >= self.config.meta_init_env_steps:
                for k in losses.keys():
                    metrics_to_log[f"train/{k}"] = losses.get(k).mean().item()

            wandb.log(metrics_to_log, step=collected_frames)

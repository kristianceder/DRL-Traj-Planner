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
        in_keys_actor = ["meta_observation", "reward_tensor", "w"]
        in_keys_value = ["meta_observation", "reward_tensor", "w"]

        # TODO get this from env or config
        action_size = 4
        obs_size = 178

        action_spec = MultiDiscreteTensorSpec([2]*action_size)#OneHotDiscreteTensorSpec(action_size)
        action_size = action_spec.shape[-1]

        encoder = MetaEncoder(observation_dim=obs_size, reward_dim=4)

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
                    "w": torch.rand(16, 4),
                    "reward_tensor": torch.rand(16, 1000, action_size),
                    "meta_action": torch.rand(16, action_size),
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

    def train(self, model):
        max_phases = self.config.n_phases
        n_iters = self.config.n_iters
        collected_frames = 0

        for iter_idx in tqdm(range(n_iters), desc="Training Iterations"):
            reward_w = torch.zeros(4)
            reward_w[0] = 1
            reward_w[1] = 1
            model.set_w(reward_w)
            last_td = None

            episode_reward = 0

            # rollout 1 episode
            for p_idx in range(max_phases):
                logging.info(f"Phase {p_idx}")
                logging.info(f"Reward weights: {reward_w}")

                # train policy
                n_train_frames = 1_000
                model.config.total_frames = n_train_frames
                model.train(use_wandb=False)
                base_eval_td = model.evaluate(1_000, return_means=False)
                # print(base_eval_td)
                r = base_eval_td["next", "reward"].mean().unsqueeze(0)

                eval_td = TensorDict({
                    "meta_observation": base_eval_td["observation"].unsqueeze(0),
                    "reward_tensor": base_eval_td["reward_tensor"].unsqueeze(0),
                    "w": reward_w.unsqueeze(0),
                    # "meta_action": torch.rand(1, 1000, 4),
                    "reward": r.unsqueeze(0),
                }, batch_size=[1])

                # append buffer
                if last_td is not None:
                    data = last_td.clone()
                    data["next", "meta_observation"] = eval_td["meta_observation"]
                    data["next", "reward_tensor"] = eval_td["reward_tensor"]
                    data["next", "w"] = eval_td["w"]
                    data["next", "reward"] = eval_td["reward"] # TODO check if this is the correct reward
                    term_func = torch.zeros if p_idx < max_phases - 1 else torch.ones
                    data["next", "done"] = term_func(1, 1, dtype=torch.bool)
                    data["next", "terminated"] = term_func(1, 1, dtype=torch.bool)

                    # print(data)

                    self.meta_buffer.extend(data.cpu())
                    collected_frames += 1

                # predict next action
                with torch.no_grad():
                    out = self.model["policy"](eval_td)
                reward_w = out["meta_action"].squeeze()
                print(reward_w)

                # save tuple for next iteration
                model.set_w(reward_w)
                eval_td["meta_action"] = reward_w.unsqueeze(0)

                
                last_td = eval_td

                episode_reward += r

            # UTD = 1.0, might want to change to a parameter later
            num_updates = self.config.n_phases

            # train
            if collected_frames >= self.config.init_env_steps:
                losses = TensorDict({}, batch_size=[num_updates])
                for i in range(num_updates):
                    sampled_tensordict = self.meta_buffer.sample()
                    # print("----")
                    # print(sampled_tensordict)

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
            if collected_frames >= self.config.init_env_steps:
                for k in losses.keys():
                    metrics_to_log[f"train/{k}"] = losses.get(k).mean().item()

            wandb.log(metrics_to_log)#, step=all_collected_frames)

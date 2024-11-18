import logging
from tqdm import tqdm
from time import time

import wandb
import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import InteractionType, TensorDictModule
from torch.distributions import Bernoulli
from torchrl.data.tensor_specs import MultiDiscreteTensorSpec
from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.modules.tensordict_module.common import SafeModule
from torchrl.objectives.sac import DiscreteSACLoss
from torchrl.objectives import SoftUpdate
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.collectors import SyncDataCollector

from pkg_torchrl.utils import make_replay_buffer, get_activation
from pkg_torchrl.sac import SAC
from pkg_torchrl.meta.networks import MetaNetwork, MetaEncoder


import torch
import torch.nn as nn


# TODO (kilian)
# - make this fast
#  - parallelize meta episodes
#  - torch compile?


def build_meta_actor(policy_net, action_spec, in_keys):
    module = SafeModule(policy_net, in_keys=in_keys, out_keys=["logits"])
    actor = ProbabilisticActor(
        module=module,
        in_keys=["logits"],
        out_keys=["meta_action"],
        spec=action_spec,
        distribution_class=Bernoulli,
        default_interaction_type=InteractionType.RANDOM,
        )#OneHotCategorical)
    return actor


class MetaSAC(SAC):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        in_keys_actor = ["meta_observation", "reward_tensor", "last_meta_action"]
        in_keys_value = ["meta_observation", "reward_tensor", "last_meta_action"]

        # TODO get this from env or config
        action_size = len(self.config.curriculum.all_reward_keys)
        self.action_size = action_size
        obs_size = 178

        action_spec = MultiDiscreteTensorSpec([2]*action_size)#OneHotDiscreteTensorSpec(action_size)
        action_size = action_spec.shape[-1]

        hidden_dim = config.meta_hidden_dim
        cnn_hidden_dim = config.meta_cnn_hidden_dim
        encoder = MetaEncoder(observation_dim=obs_size, reward_dim=action_size, hidden_dim=cnn_hidden_dim)
        policy_net = MetaNetwork(encoder=encoder, reward_dim=action_size,
                                 hidden_dim=hidden_dim, cnn_hidden_dim=cnn_hidden_dim)
        qvalue_net = MetaNetwork(encoder=encoder, reward_dim=action_size,
                                 hidden_dim=hidden_dim, cnn_hidden_dim=cnn_hidden_dim)


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
            batch_size=self.config.meta_batch_size,
            prioritize=self.config.meta_prioritize,
            buffer_size=self.config.meta_replay_buffer_size,
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

    def train(self, algo_class: SAC, algo_config, train_env, eval_env):
        n_iters = self.config.n_iters
        collected_meta_iters = 0
        overall_frames = 0

        debug_mode = False
        print_time = False

        for iter_idx in tqdm(range(n_iters), desc="Training Iterations"):
            iter_start_time = time()

            # reset lower agent
            lower_model = algo_class(algo_config, train_env, eval_env)

            # initial reward not used
            reward_w = torch.zeros(self.action_size)
            lower_model.set_w(reward_w)


            collected_frames = 0
            episode_reward = 0

            # get initial random frames
            collector = SyncDataCollector(
                lower_model.train_env,
                lower_model.model["policy"],
                init_random_frames=self.config.init_random_frames,
                frames_per_batch=self.config.frames_per_batch,
                total_frames=self.config.total_frames,
                device=self.config.collector_device,
                exploration_type=ExplorationType.RANDOM,
            )
            collector_iter = iter(collector)
            n_random_iters = self.config.init_random_frames // self.config.frames_per_batch
            for _ in range(n_random_iters):
                tensordict = next(collector_iter)
                current_frames = tensordict.numel()
                collected_frames += current_frames
                overall_frames += current_frames

            tensordict = tensordict.reshape(-1)
            lower_model.replay_buffer.extend(tensordict.cpu())

            # build meta state
            last_td = TensorDict({
                "meta_observation": tensordict["observation"][-1000:,:].unsqueeze(0),
                "reward_tensor": tensordict["next", "reward_tensor"][-1000:,:].unsqueeze(0),
                "last_meta_action": reward_w.view(1, -1),
            }, batch_size=[1])
            
            # select initial meta action
            if debug_mode:
                reward_w = torch.ones(self.action_size)
            else:
                out = self.predict(last_td, deterministic=False)
                reward_w = out["meta_action"].squeeze()
            lower_model.set_w(reward_w)
            logging.info(f"Reward weights: {reward_w}")

            rwd_updated = True

            # reduce random rollouts from max phases
            max_phases = self.config.total_frames // self.config.frames_per_batch

            init_time = time()
            if print_time:
                logging.info(f"Initialization took {init_time - iter_start_time:.2f}s")

            # meta episode
            # TODO parallelize this loop to speed up training
            for p_idx in range(n_random_iters, max_phases):
                phase_start_time = time()
                # logging.info(f"Phase {p_idx}")

                ### select w based on meta policy
                if p_idx % self.config.meta_action_ratio == 0:
                    if debug_mode:
                        reward_w = torch.ones(self.action_size)
                    else:
                        out = self.predict(last_td, deterministic=False)
                        reward_w = out["meta_action"].squeeze()

                    lower_model.set_w(reward_w)
                    rwd_updated = True
                    logging.info(f"Reward weights: {reward_w}")

                ### grad steps on policy
                # take different gradient steps depending if reward was updated
                n_train_frames = self.config.num_updates_after_update if rwd_updated else self.config.frames_per_batch
                # logging.info(f"Training base model for {n_train_frames} frames")
                before_grad_time = time()
                losses = lower_model.grad_steps(n_train_frames)
                rwd_updated = False
                grad_time = time()
                if print_time:
                    logging.info(f"Grad steps took {grad_time - before_grad_time:.2f}s")

                ### collect data
                tensordict = next(collector_iter)

                tensordict = tensordict.reshape(-1)
                current_frames = tensordict.numel()
                lower_model.replay_buffer.extend(tensordict.cpu())
                collected_frames += current_frames
                overall_frames += current_frames

                # only keep final performance as reward
                if p_idx == max_phases - 1:
                    # rollout more env steps for validation in last phase to get better estimate
                    logging.info("Evaluating final policy")
                    with set_exploration_type(ExplorationType.DETERMINISTIC):
                        base_eval_td = lower_model.eval_env.rollout(
                            max_steps=self.config.eval_rollout_steps,
                            policy=lower_model.model["policy"],
                            auto_cast_to_device=True,
                            break_when_any_done=False,
                        )

                    meta_reward = base_eval_td["next", "true_reward"].mean().view(1,1)
                    meta_reward *= self.config.meta_reward_scale
                    terminated = torch.ones(1, 1, dtype=torch.bool)
                    done = torch.ones(1, 1, dtype=torch.bool)
                else:
                    meta_reward = torch.zeros(1, 1)
                    terminated = torch.zeros(1, 1, dtype=torch.bool)
                    done = torch.zeros(1, 1, dtype=torch.bool)

                meta_td = TensorDict({
                    "meta_observation": last_td["meta_observation"],
                    "reward_tensor": last_td["reward_tensor"],
                    "last_meta_action": last_td["last_meta_action"],
                    "meta_action": reward_w.view(1, -1),
                    ("next", "meta_observation"): tensordict["next", "observation"].unsqueeze(0),
                    ("next", "reward_tensor"): tensordict["next", "reward_tensor"].unsqueeze(0),
                    ("next", "last_meta_action"): reward_w.view(1, -1),
                    ("next", "reward"): meta_reward,
                    ("next", "done"): done,
                    ("next", "terminated"): terminated,
                }, batch_size=[1])

                self.meta_buffer.extend(meta_td.cpu())

                r = tensordict["next", "reward"].mean().item()
                episode_end = tensordict["next", "done"]                    
                episode_rewards = tensordict["next", "episode_reward"][episode_end]
                episode_success = tensordict["next", "success"][episode_end]
                episode_collided = tensordict["next", "collided"][episode_end]
                episode_length = tensordict["next", "step_count"][episode_end]

                train_metrics_to_log = {
                    "train/reward": r,
                    "train/episode_reward": episode_rewards.mean().item(),
                    "train/episode_success": episode_success.float().mean().item(),
                    "train/episode_collided": episode_collided.float().mean().item(),
                    "train/episode_length": episode_length.float().mean().item(),
                    "meta_train/true_reward": tensordict["next", "true_reward"].mean().item(),
                    "meta_train/reward": meta_reward.item(),
                }
                for w_idx, _w in enumerate(reward_w):
                    train_metrics_to_log[f"meta_train/w{w_idx}"] = _w.item()
                for k, v in losses.items():
                    train_metrics_to_log[f"train/{k}"] = v.mean().item()

                phase_end_time = time()
                phase_dt = phase_end_time - phase_start_time
                if print_time:
                    logging.info(f"Phase {p_idx} took {phase_dt:.2f}s")
                train_metrics_to_log["train/base_phase_time"] = phase_dt

                wandb.log(train_metrics_to_log, step=overall_frames)
                
                # last_td = meta_td
                last_td = TensorDict({
                    "meta_observation": tensordict["observation"][-1000:,:].unsqueeze(0),
                    "reward_tensor": tensordict["next", "reward_tensor"][-1000:,:].unsqueeze(0),
                    "last_meta_action": reward_w.view(1, -1),
                    "reward": torch.zeros(1,1),
                }, batch_size=[1])
                episode_reward += r


            collected_meta_iters += 1

            ### meta gradients, i.e. train meta model
            # calculate meta env steps taken since last update
            meta_steps_taken = max_phases
            num_updates = 1 + (meta_steps_taken*self.config.utd_ratio) // self.config.meta_batch_size
            num_updates = int(num_updates)

            meta_grad_start_time = time()

            if collected_meta_iters >= self.config.meta_init_env_steps:
                logging.info(f"Training meta model for {num_updates} updates")
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

                    if self.config.meta_prioritize:
                        loss_key = "loss_critic" if "loss_critic" in loss else "loss_qvalue"
                        sampled_tensordict.set(
                            loss_key, loss[loss_key] * torch.ones(sampled_tensordict.shape, device=self.device))
                        self.replay_buffer.update_tensordict_priority(sampled_tensordict)

            meta_grad_dt = time() - meta_grad_start_time

            metrics_to_log = {}
            metrics_to_log["meta_train/grad_time"] = meta_grad_dt
            metrics_to_log["meta_train/episode_reward"] = episode_reward
            if collected_meta_iters >= self.config.meta_init_env_steps:
                for k in losses.keys():
                    metrics_to_log[f"meta_train/{k}"] = losses.get(k).mean().item()

            wandb.log(metrics_to_log, step=overall_frames)

            iter_end_time = time()
            if print_time:
                logging.info(f"Training iteration {iter_idx} took {iter_end_time - iter_start_time:.2f}s")

        if self.buffer_save_path is not None:
            self.meta_buffer.save(self.buffer_save_path)

        return lower_model
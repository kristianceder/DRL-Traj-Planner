from abc import ABC, abstractmethod
import time
import tqdm
import copy

import wandb
import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, ConvNet, AdditiveGaussianModule
from torchrl.modules.distributions import TanhNormal, TanhDelta
import torch.distributions as torch_dist

from .utils import get_activation, make_collector, make_replay_buffer, reset_actor, reset_critic


class ActorSequential(nn.Module):
    def __init__(self, feature, actor_mlp, actor_extractor):
        super().__init__()
        self.feature = feature
        self.actor_mlp = actor_mlp
        self.actor_extractor = actor_extractor

    def forward(self, *data):
        pixels, internal = data
        embed = self.feature(pixels)
        obs = torch.cat([embed, internal], dim=-1)
        x = self.actor_mlp(obs)
        if self.actor_extractor is None:
            return x, embed
        else:
            loc, scale = self.actor_extractor(x)
            return loc, scale, embed


class CriticSequential(nn.Module):
    def __init__(self, feature, critic_mlp):
        super().__init__()
        self.feature = feature
        self.critic_mlp = critic_mlp

    def forward(self, *data):
        pixels, internal, action = data
        embed = self.feature(pixels)
        obs = torch.cat([embed, internal], dim=-1)
        if action is not None:
            obs = torch.cat([obs, action], dim=-1)
        out = self.critic_mlp(obs)
        return out


def build_actor(encoder, img_mode, obs_size, action_spec, in_keys_actor, config, use_random_interaction: bool = True, deterministic: bool = False):
    action_size = action_spec.shape[-1]

    actor_net = MLP(
        # in_features = obs_size,
        num_cells = config.hidden_sizes,
        out_features = action_size if deterministic else 2 * action_size,
        activation_class = get_activation(config.activation),
        dropout = config.actor_dropout,
    )

    dist_kwargs = {
        "low": action_spec.space.low,
        "high": action_spec.space.high,
    }

    if deterministic:
        actor_extractor = None
        dist_class = TanhDelta
    else:
        dist_class = TanhNormal
        dist_kwargs["tanh_loc"] = False
        actor_extractor = NormalParamExtractor(
            scale_mapping=f"biased_softplus_{config.default_policy_scale}",
            scale_lb=config.scale_lb,
        )
    if img_mode:
        actor_net = ActorSequential(encoder, actor_net, actor_extractor)
    else:
        actor_net = nn.Sequential(actor_net, actor_extractor) if deterministic else actor_net

    action_keys = ["loc", "scale"] if not deterministic else ["param"]

    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys_actor,
        out_keys=action_keys + ["embed"],
    )
    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=action_keys,
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        default_interaction_type=InteractionType.RANDOM if use_random_interaction else InteractionType.MODE,
        return_log_prob=True,
        safe=True,  # this makes sure that predicted actions stay in action_spec bounds
    )
    return actor


def build_critic(encoder, img_mode, obs_size, action_size, in_keys_value, config):
    # Define Critic Network
    qvalue_mlp = MLP(
        # in_features = (obs_size + action_size) if 'action' in in_keys_value else obs_size,
        num_cells = config.hidden_sizes,
        out_features = 1,
        activation_class = get_activation(config.activation),
        dropout = config.critic_dropout,
    )

    if img_mode:
        qvalue_net = CriticSequential(encoder, qvalue_mlp)
    else:
        qvalue_net = qvalue_mlp

    qvalue = ValueOperator(
        in_keys=in_keys_value,
        module=qvalue_net,
    )
    return qvalue


class AlgoBase(ABC):
    def __init__(self, config, train_env, eval_env, in_keys_actor, in_keys_value, deterministic=False):
        self.config = config
        self.train_env = train_env
        self.eval_env = eval_env
        self.device = torch.device(config.device)
        self.in_keys_actor = in_keys_actor
        self.in_keys_value = in_keys_value
        self.deterministic = deterministic

        self.target_actor = None
        self.advantage_module = None
        self.target_net_updater = None
        self.is_pretrained = False
        self.updated_curriculum = False

        self.curriculum_stage = 0
        self.pretrained_actor_is_reset = False

        self._init_policy()
        self._init_loss_module()
        self._init_optimizer()
        self._post_init_optimizer()

        self.len_base_rewards = len(config.curriculum.base_reward_keys)
        self.len_constraint_rewards = len(config.curriculum.all_reward_keys) - self.len_base_rewards

        n_rewards = self.len_base_rewards + self.len_constraint_rewards
        if 'curriculum' in self.config.reward_mode:
            self.w = torch.zeros(n_rewards)
            self.w[:self.len_base_rewards] = 1.
        else:
            self.w = torch.ones(n_rewards)

        self.w.to(self.device)

        self.kl_loss_module = nn.KLDivLoss(reduction="batchmean", log_target=True)

        self.replay_buffer = make_replay_buffer(
            batch_size=self.config.batch_size,
            prioritize=self.config.prioritize,
            buffer_size=self.config.replay_buffer_size,
            scratch_dir=self.config.scratch_dir,
            device="cpu",
            prefetch=self.config.prefetch,
        )

    @abstractmethod
    def _init_loss_module(self):
        self.loss_module = None
        pass
    
    @abstractmethod
    def _init_optimizer(self):
        self.optim = None
        pass
    
    @abstractmethod
    def _loss_backward(self, loss_td) -> torch.Tensor:
        pass

    def get_action_dist(self, obs):
        if not isinstance(obs, TensorDict):
            obs = obs["observation"]
            if not isinstance(obs, torch.Tensor):
                obs = torch.from_numpy(obs).to(torch.float32)
            obs = TensorDict({"observation": obs})

        with torch.no_grad():
            act_dist = self.model.policy.get_dist(obs)
        return act_dist
    
    def sample_action(self, obs, sample_mean=False):
        act_dist = self.get_action_dist(obs)
        if sample_mean:
            act = act_dist.loc
        else:
            act = act_dist.sample()
        return act
    
    def set_pretrained(self):
        self.is_pretrained = True

    def _post_init_optimizer(self):
        if self.config.use_lr_schedule:
            num_schedule_steps = ((self.config.total_frames - self.config.first_reduce_frame)
                                  // self.config.frames_per_batch)
            if isinstance(self.optim, dict):
                self.scheduler = {}
                for k, opt in self.optim.items():
                    self.scheduler[f"{k}_lr"] = torch.optim.lr_scheduler.CosineAnnealingLR(
                                                    opt, num_schedule_steps, self.config.eta_min
                                                )
            else:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optim, num_schedule_steps, self.config.eta_min
                )
        else:
            self.scheduler = None

    def _init_policy(self):
        # Define Actor Network
        action_spec = self.train_env.action_spec
        obs_size = None
        action_size = None


        img_mode = "pixels" in self.train_env.observation_spec.keys()
    
        if img_mode:
            # TODO should have separate encoder for value and policy for off-policy algos
            # CNN from DQN Nature paper:
            #     Mnih, Volodymyr, et al.
            #     "Human-level control through deep reinforcement learning."
            #     Nature 518.7540 (2015): 529-533.
            # Addition is adaptive pooling
            encoder_actor = ConvNet(
                    num_cells = [32, 64, 64],
                    kernel_sizes = [8, 4, 3],
                    strides = [4, 2, 1],
                    activation_class = nn.ReLU,#nn.ELU,
                    squeeze_output=True,
                    aggregator_class=nn.AdaptiveAvgPool2d,
                    aggregator_kwargs={"output_size": (1, 1)},
                    device=self.config.device,
                )
            encoder_critic = ConvNet(
                    num_cells = [32, 64, 64],
                    kernel_sizes = [8, 4, 3],
                    strides = [4, 2, 1],
                    activation_class = nn.ReLU,#nn.ELU,
                    squeeze_output=True,
                    aggregator_class=nn.AdaptiveAvgPool2d,
                    aggregator_kwargs={"output_size": (1, 1)},
                    device=self.config.device,
                )
        else:
            # empty network that just passed through the input
            encoder_actor = nn.Identity()
            encoder_critic = nn.Identity()

        actor = build_actor(encoder_actor, img_mode, obs_size, action_spec, self.in_keys_actor, self.config, deterministic=self.deterministic)
        qvalue = build_critic(encoder_critic, img_mode, obs_size, action_size, self.in_keys_value, self.config)

        self.model = nn.ModuleDict({
            "policy": actor,
            "value": qvalue
        }).to(self.device)

        # init input dims
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            td = self.train_env.reset()
            td = td.to(self.device)
            for k, net in self.model.items():
                print(k)
                net(td)
        del td
        self.train_env.close()

    # def set_curriculum_stage(self, stage: int):
        # reset_n_critic_layers = self.config.curriculum.reset_n_critic_layers
        # reset_n_actor_layers = self.config.curriculum.reset_n_actor_layers

        # self.train_env.unwrapped.set_curriculum_stage(stage)
        # self.eval_env.unwrapped.set_curriculum_stage(stage)

        # if reset_n_actor_layers is not None:
        #     reset_actor(self.model["policy"], reset_n_actor_layers)

        # if reset_n_critic_layers is not None:
        #     reset_critic(self.model["value"], reset_n_critic_layers)

        # if self.config.curriculum.reset_buffer:
        #     self.replay_buffer.empty()
        #     print("Emptied replay buffer")
        # print(f"Curriculum stage: {stage}")

        # self.curriculum_stage = stage
        # self.updated_curriculum = True

    def _maybe_update_curriculum(self, collected_frames):
        if not self.config.reward_mode == "curriculum_step":
            return collected_frames

        if collected_frames >= self.config.curriculum.steps_stage_1 \
                and self.curriculum_stage < 1:
            if self.config.curriculum.reset_frames:
                collected_frames = 0
            if self.config.curriculum.reset_buffer:
                self.replay_buffer.empty()
                print("Emptied replay buffer")
            print(f"Curriculum stage: {1}")
            self.w = torch.ones_like(self.w).to(self.device)
            self.curriculum_stage = 1
            self.updated_curriculum = True
            self.target_actor = copy.deepcopy(self.model.policy)

        return collected_frames
    
    def step_curriculum(self, collected_frames):
        if self.config.reward_mode == 'sum':
            return None

        # this resets collected_frames = 0 if proceeds to next stage
        if self.config.reward_mode == 'curriculum_step':
            collected_frames = self._maybe_update_curriculum(collected_frames)
            return collected_frames

        elif self.config.reward_mode == 'curriculum':
            # TODO change w and step k
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported curriculum mode: {self.config.curriculum_mode}")

    def compute_kl_loss(self, sampled_tensordict):
        with set_exploration_type(ExplorationType.MODE), torch.no_grad():
            log_prior = self.target_actor.log_prob(sampled_tensordict)
        log_post = self.model.policy.log_prob(sampled_tensordict)

        if self.config.kl_approx_method == "logp":
            kl_loss = (log_post - log_prior).mean()
        elif self.config.kl_approx_method == "abs":
            kl_loss = 0.5 * nn.SmoothL1Loss()(log_post, log_prior)
        else:
            raise ValueError(f"Unsupported kl_approx_method {self.config.kl_approx_method}")
        return kl_loss
    
    def set_w(self, new_w):
        if not isinstance(new_w, torch.Tensor):
            new_w = torch.tensor(new_w)
        assert new_w.shape == self.w.shape, f"new_w shape {new_w.shape} != w shape {self.w.shape}"
        self.w = new_w.to(self.device)

    def compute_reward(self, reward_tensor):
        # computing the reward solely depends on self.w
        assert self.w.shape[-1] == reward_tensor.shape[-1]
        if not self.w.device == self.device:
            # print("Moving w to device")
            self.w = self.w.to(self.device)
        if not reward_tensor.device == self.device:
            # print("Moving reward_tensor to device")
            reward_tensor = reward_tensor.to(self.device)

        rewards = (self.w * reward_tensor).sum(dim=-1).unsqueeze(-1)
        return rewards

    def train(self, use_wandb=True, env_maker=None):
        # Create off-policy collector

        if self.deterministic:
            # TODO do want to add noise to TD3 as well?
            collector_policy = TensorDictSequential(
                    self.model["policy"],
                    AdditiveGaussianModule(
                        spec=self.model["policy"].spec,
                        annealing_num_steps=self.config.total_frames// 2,  # Number of frames after which sigma is sigma_end
                        sigma_init=self.config.sigma_init,  # Initial value of the sigma
                        sigma_end=self.config.sigmn_end,  # Final value of the sigma
                    ),
                )
        else:
            collector_policy = self.model["policy"]

        collector = make_collector(self.config, self.train_env, collector_policy,
                                   self.is_pretrained, env_maker=env_maker)

        # Main loop
        start_time = time.time()
        collected_frames = 0
        all_collected_frames = 0
        pbar = tqdm.tqdm(total=self.config.total_frames)
        last_success_rate = 0.0
        metrics_to_log = None

        num_updates = int(
            self.config.env_per_collector
            * self.config.frames_per_batch
            * self.config.utd_ratio
        )
        prioritize = self.config.prioritize
        eval_iter = self.config.eval_iter
        frames_per_batch = self.config.frames_per_batch
        eval_rollout_steps = self.config.eval_rollout_steps

        sampling_start = time.time()
        for i, tensordict in enumerate(collector):
            sampling_time = time.time() - sampling_start

            # Update weights of the inference policy
            collector.update_policy_weights_()

            if self.advantage_module is not None:
                self.advantage_module(tensordict)

            pbar.set_description(f"success: {last_success_rate:.2f}", refresh=False)
            pbar.update(tensordict.numel())

            tensordict = tensordict.reshape(-1)
            current_frames = tensordict.numel()
            # Add to replay buffer
            self.replay_buffer.extend(tensordict.cpu())
            collected_frames += current_frames
            all_collected_frames += current_frames

            # if not self.pretrained_actor_is_reset \
            #         and collected_frames >= self.config.init_random_frames \
            #         and self.is_pretrained \
            #         and self.config.reset_pretrained_actor:
            #     print('Resetting actor')
            #     reset_actor(self.model['policy'], 20)
            #     self.pretrained_actor_is_reset = True

            # Optimization steps
            training_start = time.time()
            if collected_frames >= self.config.init_env_steps:
                # print(f"Train {collected_frames} frames")
                # if self.config.n_reset_layers is not None:
                #     reset_actor(self.model["policy"], self.config.n_reset_layers)
                # if self.config.n_reset_layers_critic is not None:
                #     reset_critic(self.model["value"], self.config.n_reset_layers_critic)

                # train longer after updating curriculum
                if self.updated_curriculum:
                    temp_num_updates = self.config.curriculum.num_updates_after_update
                    self.updated_curriculum = False
                    print(f"Training for {temp_num_updates} updates after curriculum update")
                else:
                    temp_num_updates = num_updates

                losses = TensorDict({}, batch_size=[temp_num_updates])
                for i in range(temp_num_updates):
                    # Sample from replay buffer
                    sampled_tensordict = self.replay_buffer.sample()

                    if sampled_tensordict.device != self.device:
                        sampled_tensordict = sampled_tensordict.to(
                            self.device#, non_blocking=True
                        )
                    else:
                        sampled_tensordict = sampled_tensordict.clone()

                    # compute the reward during training as it depends on w which changes
                    # this way we can keep the whole replay buffer when changing reward weights

                    # sampled_tensordict["next", "reward"] = self.compute_reward(sampled_tensordict["next", "reward_tensor"])
                    # new_reward = self.compute_reward(sampled_tensordict["next", "reward_tensor"])
                    if self.curriculum_stage > 0 or "curriculum" not in self.config.reward_mode:
                        sampled_tensordict["next", "reward"] = sampled_tensordict["next", "full_reward"]
                    # else:
                    #     og_reward = sampled_tensordict["next", "reward"]

                    # sampled_tensordict["next", "reward"] = og_reward
                    # # sampled_tensordict["next", "reward"] = new_reward
                    # abs_err = (new_reward - og_reward)
                    # abs_err_map = abs_err.abs() > 1e-7
                    # if abs_err_map.any():
                    #     err_rwd_new = new_reward[abs_err_map]
                    #     err_rwd_old = og_reward[abs_err_map]

                    #     print(err_rwd_new)
                    #     print(err_rwd_old)
                    #     print(abs_err[abs_err_map])
                    #     print("----")

                    # Compute loss
                    loss_td = self.loss_module(sampled_tensordict)

                    # calculate target values if needed
                    if self.target_actor is not None:
                        loss_td['kl_loss'] = self.compute_kl_loss(sampled_tensordict)


                    loss = self._loss_backward(loss_td)
                    losses[i] = loss

                    # Update qnet_target params
                    if self.target_net_updater is not None:
                        self.target_net_updater.step()

                    # Update priority
                    if prioritize:
                        loss_key = "loss_critic" if "loss_critic" in loss else "loss_qvalue"
                        sampled_tensordict.set(
                            loss_key, loss[loss_key] * torch.ones(sampled_tensordict.shape))
                        self.replay_buffer.update_tensordict_priority(sampled_tensordict)

            training_time = time.time() - training_start
            episode_end = tensordict["next", "done"]
            episode_rewards = tensordict["next", "episode_reward"][episode_end]
            episode_success = tensordict["next", "success"][episode_end]
            episode_collided = tensordict["next", "collided"][episode_end]

            # Logging
            metrics_to_log = {}
            if len(episode_rewards) > 0:
                episode_length = tensordict["next", "step_count"][episode_end]
                metrics_to_log['train/w'] = self.w.sum().item()
                metrics_to_log["train/episode_reward"] = episode_rewards.mean().item()
                last_success_rate = episode_success.float().mean().item()
                metrics_to_log["train/episode_success"] = last_success_rate
                metrics_to_log["train/episode_collided"] = episode_collided.float().mean().item()
                metrics_to_log["train/reward"] = self.compute_reward(tensordict["next", "reward_tensor"]).mean().item() # this is a bit weird but should do the job
                metrics_to_log["train/full_reward"] = tensordict["next", "full_reward"].mean().item()
                metrics_to_log["train/episode_length"] = episode_length.sum().item() / len(
                    episode_length
                )
                if isinstance(self.optim, dict):
                    for k, opt in self.optim.items():
                        metrics_to_log[f"train/lr_{k}"] = opt.param_groups[0]["lr"]
                else:
                    metrics_to_log["train/lr"] = self.optim.param_groups[0]["lr"]
            if collected_frames >= self.config.init_env_steps:
                for k in losses.keys():
                    metrics_to_log[f"train/{k}"] = losses.get(k).mean().item()
                metrics_to_log["train/sampling_time"] = sampling_time
                metrics_to_log["train/training_time"] = training_time

            # Evaluation
            if abs(all_collected_frames % eval_iter) < frames_per_batch:
                eval_stats = self.evaluate(eval_rollout_steps)
                metrics_to_log.update(eval_stats)
            
            if use_wandb:
                wandb.log(metrics_to_log, step=all_collected_frames)

            if self.scheduler is not None and collected_frames >= self.config.first_reduce_frame:
                if isinstance(self.scheduler, dict):
                    for scheduler in self.scheduler.values():
                        scheduler.step()
                else:
                    self.scheduler.step()

            out = self.step_curriculum(collected_frames)
            if out is not None:
                collected_frames = out
            sampling_start = time.time()

        collector.shutdown()
        del collector
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Training took {execution_time:.2f} seconds to finish")

        return metrics_to_log
    
    def evaluate(self, eval_rollout_steps, return_means=True):
        # TODO record video of evaluation
        eval_metrics = {}
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            eval_start = time.time()
            eval_rollout = self.eval_env.rollout(
                eval_rollout_steps,
                self.model["policy"],
                auto_cast_to_device=True,
                break_when_any_done=False,
            )

            eval_time = time.time() - eval_start
            eval_reward = eval_rollout["next", "reward"].mean().item()
            eval_episode_end = eval_rollout["next", "done"]
            eval_episode_rewards = eval_rollout["next", "episode_reward"][eval_episode_end]
            eval_episode_lengths = eval_rollout["next", "step_count"][eval_episode_end]
            # if "success" in tensordict["next"].keys():
            eval_episode_len = eval_episode_lengths.sum().item() / len(eval_episode_lengths)

            entropy = 0.0
            kl = 0.0
            if not self.deterministic:
                o_dist = self.model["policy"].get_dist(eval_rollout)    
                entropy = torch_dist.Normal(o_dist.loc, o_dist.scale).entropy().sum(-1).mean().item()

                if self.target_actor is not None:
                    o_dist_base = self.target_actor.get_dist(eval_rollout)
                    log_prior = o_dist_base.log_prob(eval_rollout['action'])
                    log_post = o_dist.log_prob(eval_rollout['action'])
                    kl = (log_post - log_prior).mean().item()
                else:
                    kl = 0.0

            eval_episode_success = eval_rollout["next", "success"][eval_episode_end]
            eval_metrics["eval/episode_success"] = eval_episode_success.float().mean().item()
            eval_metrics["eval/episode_length"] = eval_episode_len
            eval_metrics["eval/entropy"] = entropy
            eval_metrics["eval/kl"] = kl
            eval_metrics["eval/episode_reward"] = eval_episode_rewards.mean().item()
            eval_metrics["eval/reward"] = eval_reward
            eval_metrics["eval/full_reward"] = eval_rollout["next", "full_reward"].mean().item()
            eval_metrics["eval/time"] = eval_time
            eval_metrics["eval/n_steps"] = eval_rollout["next", "reward"].shape[0]
        
        if return_means:
            return eval_metrics
        else:
            val_loss = self.loss_module(eval_rollout)
            r_hat = self.compute_reward(eval_rollout["next", "reward_tensor"])
            r = eval_rollout["next", "full_reward"]
            return val_loss, r_hat, r, eval_metrics

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
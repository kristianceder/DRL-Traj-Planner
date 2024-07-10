from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from tensordict import TensorDict
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError
    

def make_collector(config, train_env, actor_model_explore):
    collector = SyncDataCollector(
        train_env,
        actor_model_explore,
        init_random_frames=config.init_random_frames,
        frames_per_batch=config.frames_per_batch,
        total_frames=config.total_frames,
        device=config.device,
    )
    collector.set_seed(config.seed)
    return collector


def make_replay_buffer(
    batch_size,
    prioritize=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=3,
):
    if prioritize:
        replay_buffer = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=prefetch,
            storage=LazyMemmapStorage(
                buffer_size,
                scratch_dir=scratch_dir,
                device=device,
            ),
            batch_size=batch_size,
        )
    return replay_buffer


class AlgoBase(ABC):
    def __init__(self, config, train_env, eval_env, in_keys=["observation"]):
        self.config = config
        self.train_env = train_env
        self.eval_env = eval_env
        self.device = torch.device(config.device)
        self.in_keys = in_keys

        self._init_policy()
        self._init_loss_module()
        self._post_init_loss_module()
        self._init_optimizer()

    @abstractmethod
    def _init_loss_module(self):
        self.loss_module = None
        pass

    @abstractmethod
    def train(self, logger=None):
        pass
    
    def get_action_dist(self, obs):
        if not isinstance(obs, TensorDict):
            obs = obs["observation"]
            if not isinstance(obs, torch.Tensor):
                obs = torch.from_numpy(obs).to(torch.float32)
            obs = TensorDict({"observation": obs})#, [])

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
    
    def _post_init_loss_module(self):
        self.loss_module.make_value_estimator(gamma=self.config.gamma)

        self.target_net_updater = SoftUpdate(
            self.loss_module, eps=self.config.target_update_polyak
        )

    def _init_policy(self):
        # Define Actor Network
        in_keys = self.in_keys#["observation"]
        action_spec = self.train_env.action_spec
        obs_size = 0
        for key in self.in_keys:
            obs_size += self.train_env.observation_spec[key].shape[-1]
        action_size = action_spec.shape[-1]
        if self.train_env.batch_size:
            action_spec = action_spec[(0,) * len(self.train_env.batch_size)]
        actor_net_kwargs = {
            "in_features": obs_size,
            "num_cells": self.config.hidden_sizes,
            "out_features": 2 * action_size,
            "activation_class": get_activation(self.config.activation),
        }

        actor_net = MLP(**actor_net_kwargs)

        dist_class = TanhNormal
        dist_kwargs = {
            "min": action_spec.space.low,
            "max": action_spec.space.high,
            "tanh_loc": False,
        }

        actor_extractor = NormalParamExtractor(
            scale_mapping=f"biased_softplus_{self.config.default_policy_scale}",
            scale_lb=self.config.scale_lb,
        )
        actor_net = nn.Sequential(actor_net, actor_extractor)

        in_keys_actor = in_keys
        actor_module = TensorDictModule(
            actor_net,
            in_keys=in_keys_actor,
            out_keys=[
                "loc",
                "scale",
            ],
        )
        actor = ProbabilisticActor(
            spec=action_spec,
            in_keys=["loc", "scale"],
            module=actor_module,
            distribution_class=dist_class,
            distribution_kwargs=dist_kwargs,
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=False,
        )

        # Define Critic Network
        qvalue_net_kwargs = {
            "in_features": (obs_size + action_size),
            "num_cells": self.config.hidden_sizes,
            "out_features": 1,
            "activation_class": get_activation(self.config.activation),
        }

        qvalue_net = MLP(
            **qvalue_net_kwargs,
        )

        qvalue = ValueOperator(
            in_keys=["action"] + in_keys,
            module=qvalue_net,
        )

        self.model = nn.ModuleDict({
            "policy": actor,
            "value": qvalue
        }).to(self.device)

        # init input dims
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            td = self.train_env.reset()
            td = td.to(self.device)
            for net in self.model.values():
                net(td)
        del td
        self.train_env.close()


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
            lr=3.0e-4,
        )

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
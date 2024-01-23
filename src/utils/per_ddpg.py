"""
This file hacks Prioritized Experience Replay (PER) into stable_baselines3 by
extending the `DQN`` class. Lots of this file is copy-paste from the
stable_baselines3 `DQN`` class, in particular the ``train`` method of the class
``PerDQN`` is mostly copy-paste with minor changes.
"""

from typing import Any, List, Dict, Optional, Tuple, Type, Union

from gym import spaces
import numpy as np
import torch as th

from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3 import DDPG
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule

from stable_baselines3.td3.policies import CnnPolicy, TD3Policy, MlpPolicy, MultiInputPolicy
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.noise import NormalActionNoise

# See https://github.com/cloudpipe/cloudpickle/issues/460
from .type_aliases import PerReplayBufferSamples


class PerReplayBuffer(DictReplayBuffer):
    """
    Replay buffer with Prioritized experience replay using a sum-tree data
    structure. See https://doi.org/10.48550/arXiv.1511.05952 for details.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        alpha = 0.3,
        beta = 0.4,
        epsilon = 1e-3,
        update_max_freq = 1_000,
        refresh_tree_freq = 50_000,
        initial_priority = 1,
    ):
        assert optimize_memory_usage is False, "PerReplayBuffer does not support optimize_memory_usage"

        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            1,
            optimize_memory_usage,
            handle_timeout_termination
        )

        self._n_envs = n_envs
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.update_max_freq = update_max_freq
        self.refresh_tree_freq = refresh_tree_freq
        self.initial_priority = initial_priority

        self.tree = np.zeros((2 * self.buffer_size - 1,))
        self.update_max_count = self.update_max_freq - 1
        self.refresh_tree_count = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def update_priority(self, idx: int, delta: float) -> None:
        self._update_priority(idx, (np.abs(delta) + self.epsilon)**self.alpha)

    def _update_priority(self, idx: int, p: float) -> None:
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def _refresh_tree(self) -> None:
        self.tree[0:self.buffer_size - 1] = 0
        for idx in range(self.buffer_size - 1, len(self.tree)):
            self._propagate(idx, self.tree[idx])
    
    def add(
        self,
        obs: Dict[str, np.ndarray],
        next_obs: Dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]]
    ) -> None:
        for i in range(self._n_envs):
            self.update_max_count += 1
            if self.update_max_count >= self.update_max_freq:
                if self.pos == 0 and not self.full:
                    self.max_p = self.initial_priority
                else:
                    self.max_p = np.max(self.tree[self.buffer_size - 1:])
                self.update_max_count = 0

            idx = self.pos + self.buffer_size - 1

            super().add(
                {k: v[i:i + 1] for k, v in obs.items()},
                {k: v[i:i + 1] for k, v in next_obs.items()},
                action[i:i + 1],
                reward[i:i + 1],
                done[i:i + 1],
                infos[i:i + 1]
            )

            self._update_priority(idx, self.max_p)

            self.refresh_tree_count += 1
            if self.refresh_tree_count >= self.refresh_tree_freq:
                self._refresh_tree()
                self.refresh_tree_count = 0

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> PerReplayBufferSamples:
        indices = np.zeros((batch_size,), dtype=int)

        while True:
            segment = self.tree[0] / batch_size

            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)

                s = np.random.uniform(a, b)
                indices[i] = self._retrieve(0, s)
                if self.tree[indices[i]] == 0:
                    break
            else:
                break

            self._refresh_tree()

        data_idx = indices - self.buffer_size + 1

        n_entries = self.buffer_size if self.full else self.pos

        weights = np.power(n_entries * self.tree[indices] / self.tree[0], -self.beta)
        weights /= np.max(weights)

        sample = self._get_samples(data_idx, env)

        return PerReplayBufferSamples(*sample, indices, weights)

    def reset(self) -> None:
        super().reset()
        self.tree[:] = 0
        self.update_max_count = self.update_max_freq - 1
        self.refresh_tree_count = 0


class PerDDPG(DDPG):
    """
    Like DDPG but with PER

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically (Only available when passing string for the environment).
        Caution, this parameter is deprecated and will be removed in the future.
        Please use `EvalCallback` or a custom Callback instead.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        action_noise: np.float32 = NormalActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2)),
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,

    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=PerReplayBuffer,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device = device,
            _init_setup_model= _init_setup_model)
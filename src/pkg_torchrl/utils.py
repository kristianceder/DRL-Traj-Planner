import torch
import torch.nn as nn

from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torchrl.data import TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage


def reset_actor(actor, n_layers):
    linear_idxs = [i for i, layer in enumerate(actor.module[0].module[0])
                   if isinstance(layer, nn.Linear)]
    idxs_to_reset = linear_idxs[-n_layers:]
    for idx in idxs_to_reset:
        actor.module[0].module[0][idx].reset_parameters()


def get_activation(activation):
    if activation == "relu":
        return nn.ReLU
    elif activation == "tanh":
        return nn.Tanh
    elif activation == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise NotImplementedError


def make_collector(config, train_env, actor_model_explore, is_pretrained, env_maker=None):
    if config.use_multicollector:
        if env_maker is None:
            raise ValueError("env_maker must be provided if using multicollector")
        collector = MultiSyncDataCollector(
            create_env_fn=[env_maker, env_maker],
            policy=actor_model_explore,
            total_frames=config.total_frames,
            frames_per_batch=config.frames_per_batch,
            init_random_frames=None if is_pretrained else config.init_random_frames,
            reset_at_each_iter=False,
            device=config.collector_device,
            storing_device=config.collector_device,
        )
    else:
        collector = SyncDataCollector(
            train_env,
            actor_model_explore,
            init_random_frames=None if is_pretrained else config.init_random_frames,
            frames_per_batch=config.frames_per_batch,
            total_frames=config.total_frames,
            device=config.collector_device,
        )
    collector.set_seed(config.seed)

    return collector


def make_replay_buffer(
    batch_size,
    prioritize=False,
    buffer_size=1000000,
    scratch_dir=None,
    device="cpu",
    prefetch=1,
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

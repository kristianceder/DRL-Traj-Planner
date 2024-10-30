import logging
from tqdm import tqdm

import torch
import torch.nn as nn

from tensordict import TensorDict
from torch.distributions import Categorical, Bernoulli
from torchrl.data.tensor_specs import OneHotDiscreteTensorSpec, MultiDiscreteTensorSpec
from torchrl.modules.distributions import NormalParamExtractor, OneHotCategorical, MaskedCategorical
from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.modules import MLP
from torchrl.modules.tensordict_module.common import SafeModule
from torchrl.objectives.sac import DiscreteSACLoss
from torchrl.objectives import SoftUpdate
from torchrl.envs.utils import ExplorationType, set_exploration_type

from pkg_torchrl.utils import make_replay_buffer, get_activation
from pkg_torchrl.base import build_critic
from pkg_torchrl.sac import SAC


import torch
import torch.nn as nn
import torch.nn.functional as F

# class TimeSeriesAttentionNetwork(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_heads):
#         super(TimeSeriesAttentionNetwork, self).__init__()
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.num_heads = num_heads

#         # Embedding layer to transform input data
#         self.embedding = nn.Linear(input_dim, hidden_dim)

#         # Multi-head attention layer
#         self.attention = nn.MultiHeadAttention(hidden_dim, num_heads)

#         # Linear layer to transform attention output
#         self.linear = nn.Linear(hidden_dim, hidden_dim)

#         # Output layer
#         self.output = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         # Embed input data
#         x = self.embedding(x)

#         # Apply multi-head attention
#         x, _ = self.attention(x, x)

#         # Apply linear transformation
#         x = self.linear(x)

#         # Apply activation function
#         x = F.relu(x)

#         # Calculate output
#         output = self.output(x[:, -1, :])  # Take the last time step

#         return output

# # Initialize the network
# input_dim = 3  # 3 rows
# hidden_dim = 128
# output_dim = 2
# num_heads = 8
# num_samples = 3000

# network = TimeSeriesAttentionNetwork(input_dim, hidden_dim, output_dim, num_heads)

# # Create a random input tensor
# input_tensor = torch.randn(num_samples, input_dim)

# # Forward pass
# output = network(input_tensor)

# print(output.shape)


def build_meta_actor(obs_size, action_spec, in_keys, config):
    n_act = action_spec.shape[-1]

    net = MLP(in_features=obs_size,
                    num_cells=config.hidden_sizes,
                    out_features=n_act,
                    activation_class=get_activation(config.activation),
                    dropout=config.actor_dropout
                    )
    
    # net = nn.Sequential(actor_net, NormalParamExtractor())
    module = SafeModule(net, in_keys=in_keys, out_keys=["logits"])
    actor = ProbabilisticActor(
        module=module,
        in_keys=["logits"],
        out_keys=["meta_action"],
        spec=action_spec,
        distribution_class=Bernoulli,
        )#OneHotCategorical)
    return actor

def build_meta_state(eval_stats, reward_w):
    val_loss, r_hat, r, _ = eval_stats
    
    # TODO might be better to make this a gaussian input
    eval_state = torch.tensor([val_loss.mean(),
                        val_loss.std(),
                        r_hat.mean(),
                        r_hat.std(),
                        r.mean(),
                        r.std(),])
    state = torch.cat((eval_state, reward_w), 0)
    
    return state

class MetaSAC(SAC):
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        in_keys_actor = ["meta_observation"]
        in_keys_value = ["meta_observation", "meta_action"]

        action_size = 4
        obs_size = 6+4

        action_spec = MultiDiscreteTensorSpec([2]*action_size)#OneHotDiscreteTensorSpec(action_size)
        action_size = action_spec.shape[-1]

        actor = build_meta_actor(obs_size, action_spec, in_keys_actor, self.config)
        encoder_critic = nn.Identity()
        img_mode = False
        qvalue = build_critic(encoder_critic, img_mode, obs_size, action_size, in_keys_value, self.config)


        self.model = nn.ModuleDict({
            "policy": actor,
            "value": qvalue
        }).to(self.device)

        # init input dims
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            # td = self.train_env.reset()

            for net in self.model.values():
                td = TensorDict({
                    "meta_observation": torch.rand(16, obs_size),
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
        self.loss_module.make_value_estimator(gamma=self.config.gamma)

        self.target_net_updater = SoftUpdate(
            self.loss_module, eps=self.config.target_update_polyak
        )

    def update_w(self, reward_w, meta_action):
        # TODO log?
        reward_w += meta_action

    def train(self, model, config):

        max_phases = 6
        n_iters = 2

        for iter_idx in tqdm(range(n_iters), desc="Training Iterations"):
            reward_w = torch.zeros(4)
            reward_w[0] = 1
            reward_w[1] = 1
            model.set_w(reward_w)
            last_tuple = None

            # rollout 1 episode
            for p_idx in range(max_phases):
                logging.info(f"Phase {p_idx}")
                logging.info(f"Reward weights: {reward_w}")

                # TODO train for x iterations
                # model.train()
                # eval_stats = model.evaluate(3_000, return_means=False)
                # meta_state = build_meta_state(eval_stats, reward_w)

                eval_stats = [{"eval/full_reward": torch.rand(100)}]
                meta_state = torch.rand(10)


                # done = reward_w.all()
                # always loop until end
                done = torch.tensor(False)

                # append buffer
                if last_tuple is not None:
                    s, a, r = last_tuple
                    s_next = meta_state.unsqueeze(0)
                    batch_size = 1
                    data = TensorDict({
                        "observation": s,
                        "action": a,
                        ("next", "done"): done.unsqueeze(0),
                        ("next", "terminated"): torch.zeros(batch_size, 1, dtype=torch.bool),
                        ("next", "reward"): r,
                        ("next", "observation"): s_next, 
                        }, (batch_size,))

                    self.meta_buffer.extend(data.cpu())

                if done:
                    print("All rewards are 1, breaking")
                    break

                # predict next action
                # probs, _ = self.model["policy"](meta_state)
                td = TensorDict({
                    "meta_observation": meta_state.unsqueeze(0),
                })
                with torch.no_grad():
                    out = self.model["policy"](td)
                meta_action = out["meta_action"]
                print(meta_action)

                # save tuple for next iteration
                a = meta_action
                model.set_w(a.squeeze())
                s = meta_state.unsqueeze(0)

                r = eval_stats[-1]["eval/full_reward"].mean().unsqueeze(0)
                last_tuple = (s, a, r)

                # update reward weights
                # self.update_w(reward_w, meta_action)
                # model.set_w(reward_w)


            # train

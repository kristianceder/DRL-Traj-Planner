import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaEncoder(nn.Module):
    def __init__(self, observation_dim=10, reward_dim=4, hidden_dim=128):
        super(MetaEncoder, self).__init__()
        # Process observation sequence (1000x10)
        self.obs_encoder = nn.Sequential(
            nn.Conv1d(observation_dim, hidden_dim, kernel_size=5, stride=2),
            nn.ReLU(),
            # nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2),
            # nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Reduce temporal dimension to 1
        )
        
        # Process reward sequence (1000x4)
        self.reward_encoder = nn.Sequential(
            nn.Conv1d(reward_dim, hidden_dim, kernel_size=5, stride=2),
            nn.ReLU(),
            # nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2),
            # nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Reduce temporal dimension to 1
        )
        
        # Process current weights (1x4)
        self.weight_encoder = nn.Sequential(
            nn.Linear(reward_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, observation, rewards, current_weights):
        batch_size = observation.shape[0]
        
        # Reshape for 1D convolutions (batch, channels, length)
        obs = observation.transpose(1, 2)  # (batch_size, 10, 1000)
        rew = rewards.transpose(1, 2)  # (batch_size, 4, 1000)
        
        # Encode each input
        obs_encoding = self.obs_encoder(obs).view(batch_size, -1)
        reward_encoding = self.reward_encoder(rew).view(batch_size, -1)
        weight_encoding = self.weight_encoder(current_weights)
        
        # Combine encodings
        combined = torch.cat([obs_encoding, reward_encoding, weight_encoding], dim=1)

        return combined


class MetaNetwork(nn.Module):
    def __init__(self, encoder, reward_dim=4, hidden_dim=128, cnn_hidden_dim=32, out_activation=None):
        super(MetaNetwork, self).__init__()
        
        self.encoder = encoder

        # Combine all encodings and predict new weights
        self.fc_combined = nn.Sequential(
            nn.Linear(cnn_hidden_dim * 3, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, reward_dim)
        )
        self.out_activation = out_activation
        
    def forward(self, observation, rewards, current_weights):
        """
        Args:
            observation: tensor of shape (batch_size, 1000, 10)
            rewards: tensor of shape (batch_size, 1000, 4)
            current_weights: tensor of shape (batch_size, 4)
        Returns:
            mean: tensor of shape (batch_size, 4) - mean of new weights
            std: tensor of shape (batch_size, 4) - std of new weights
        """
        combined = self.encoder(observation, rewards, current_weights)
        
        # Predict new weights
        out = self.fc_combined(combined)

        if self.out_activation is not None:
            out = self.out_activation(out)

        return out


def prepare_input(observation, rewards, current_weights):
    """
    Convert numpy arrays to proper tensor format
    """
    # Add batch dimension if needed
    if len(observation.shape) == 2:
        observation = observation[None, ...]
        rewards = rewards[None, ...]
        current_weights = current_weights[None, ...]
        
    return (torch.FloatTensor(observation), 
            torch.FloatTensor(rewards),
            torch.FloatTensor(current_weights))
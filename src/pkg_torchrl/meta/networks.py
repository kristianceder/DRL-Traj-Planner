import torch
import torch.nn as nn
import torch.nn.functional as F


class MetaNetwork(nn.Module):
    def __init__(self, observation_dim=10, reward_dim=4, hidden_dim=128):
        super(MetaNetwork, self).__init__()
        
        # TODO those parameters could be shared between policy and value network    
        # Process observation sequence (1000x10)
        self.obs_encoder = nn.Sequential(
            nn.Conv1d(observation_dim, hidden_dim, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Reduce temporal dimension to 1
        )
        
        # Process reward sequence (1000x4)
        self.reward_encoder = nn.Sequential(
            nn.Conv1d(reward_dim, hidden_dim, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Reduce temporal dimension to 1
        )
        
        # Process current weights (1x4)
        self.weight_encoder = nn.Sequential(
            nn.Linear(reward_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combine all encodings and predict new weights
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, reward_dim)
        )
        
        # Optional: Add log_std for stochastic policy
        # self.log_std = nn.Parameter(torch.zeros(reward_dim))
        
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
        
        # Predict new weights
        mean = self.fc_combined(combined)
        # std = torch.exp(self.log_std)
        
        return mean#, std
    
    # def sample_action(self, observation, rewards, current_weights):
    #     """
    #     Sample new weights using the reparametrization trick
    #     """
    #     mean, std = self.forward(observation, rewards, current_weights)
    #     eps = torch.randn_like(mean)
    #     return mean + std * eps

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
"""
Neural Network Architectures for DQN Variants on Atari.
Implements:
1. Nature CNN Backbone (Mnih et al., 2015)
2. Vanilla DQN / Double DQN Network (identical network architecture, different target formulation)
3. Dueling DQN Network (Wang et al., 2016) with value and advantage streams
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NatureCNN(nn.Module):
    """
    Standard DeepMind Nature CNN feature extractor (Mnih et al., Nature 2015).
    Input: (batch_size, 4, 84, 84) - 4 stacked grayscale frames
    Output: (batch_size, 3136) flattened feature vector
    """
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Initialize convolutional layers using Kaiming (He) normal initialization
        for m in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Scale pixel inputs from [0, 255] uint8 to [0.0, 1.0] float32 if needed
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.0:
            x = x / 255.0
            
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.reshape(x.size(0), -1)  # Flatten: (batch_size, 64 * 7 * 7 = 3136)


class StandardDQN(nn.Module):
    """
    Standard Deep Q-Network (Mnih et al., 2015).
    Used for both Vanilla DQN and Double DQN (DDQN uses the same architecture,
    differing only in the Bellman update target formulation).
    """
    def __init__(self, in_channels: int = 4, num_actions: int = 4):
        super().__init__()
        self.feature_extractor = NatureCNN(in_channels=in_channels)
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        # Kaiming initialization for FC layers
        nn.init.kaiming_normal_(self.fc[0].weight, nonlinearity="relu")
        nn.init.constant_(self.fc[0].bias, 0.0)
        nn.init.kaiming_normal_(self.fc[2].weight, nonlinearity="linear")
        nn.init.constant_(self.fc[2].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        q_values = self.fc(features)
        return q_values


class DuelingDQN(nn.Module):
    """
    Dueling Network Architecture (Wang et al., 2016).
    Decomposes Q(s, a) into:
      V(s): Scalar state-value function
      A(s, a): Action advantage function
    
    Identifiability Constraint:
      Q(s, a) = V(s) + (A(s, a) - 1/|A| * sum_{a'} A(s, a'))
    Subtracting the mean advantage ensures stability in optimization
    and uniquely identifies V(s) and A(s, a).
    """
    def __init__(self, in_channels: int = 4, num_actions: int = 4):
        super().__init__()
        self.feature_extractor = NatureCNN(in_channels=in_channels)
        self.num_actions = num_actions

        # State Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        
        # Initialization
        for stream in [self.value_stream, self.advantage_stream]:
            nn.init.kaiming_normal_(stream[0].weight, nonlinearity="relu")
            nn.init.constant_(stream[0].bias, 0.0)
            nn.init.kaiming_normal_(stream[2].weight, nonlinearity="linear")
            nn.init.constant_(stream[2].bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        val = self.value_stream(features)         # Shape: (batch_size, 1)
        adv = self.advantage_stream(features)     # Shape: (batch_size, num_actions)

        # Q(s, a) = V(s) + (A(s, a) - mean_a(A(s, a)))
        q_values = val + (adv - adv.mean(dim=-1, keepdim=True))
        return q_values

    def get_value_and_advantage(self, x: torch.Tensor):
        """Helper to inspect the decoupled value and advantage streams for analysis."""
        features = self.feature_extractor(x)
        val = self.value_stream(features)
        adv = self.advantage_stream(features)
        return val, adv


def build_network(algo: str, in_channels: int, num_actions: int) -> nn.Module:
    """Factory function for building networks."""
    algo_lower = algo.lower().replace("-", "").replace("_", "")
    if algo_lower in ["vanilladqn", "dqn", "doubledqn", "ddqn"]:
        return StandardDQN(in_channels=in_channels, num_actions=num_actions)
    elif algo_lower in ["duelingdqn", "duelingdoubledqn", "dueling"]:
        return DuelingDQN(in_channels=in_channels, num_actions=num_actions)
    else:
        raise ValueError(f"Unknown algorithm: {algo}. Choose from 'dqn', 'double_dqn', 'dueling_dqn'.")


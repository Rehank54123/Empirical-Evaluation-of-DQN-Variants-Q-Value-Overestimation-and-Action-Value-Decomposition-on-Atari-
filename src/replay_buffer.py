"""
Experience Replay Buffer for Deep Q-Learning.
Uses pre-allocated NumPy arrays with uint8 frame storage for memory efficiency.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer.
    Observations are stored in uint8 to preserve system RAM.
    """
    def __init__(self, capacity: int, state_shape: tuple = (4, 84, 84), device: str = "cpu"):
        self.capacity = capacity
        self.device = torch.device(device)
        self.ptr = 0
        self.size = 0

        # Preallocate memory
        self.states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Append a new transition to the circular buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Uniformly sample a batch of transitions and convert to PyTorch tensors."""
        indices = np.random.randint(0, self.size, size=batch_size)

        states = torch.as_tensor(self.states[indices], device=self.device)
        actions = torch.as_tensor(self.actions[indices], device=self.device)
        rewards = torch.as_tensor(self.rewards[indices], device=self.device)
        next_states = torch.as_tensor(self.next_states[indices], device=self.device)
        dones = torch.as_tensor(self.dones[indices], device=self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size


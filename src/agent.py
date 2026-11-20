"""
Unified DQN Agent supporting Vanilla DQN, Double DQN, and Dueling Double DQN.
Implements:
- Linear epsilon-greedy exploration schedule
- Double Q-learning target decoupling
- Dueling architecture support
- Huber loss (Smooth L1) with gradient clipping
- Periodic target network synchronization
- Average Q-value estimation tracking for overestimation bias analysis
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import build_network


class DQNAgent:
    def __init__(
        self,
        algo: str,
        num_actions: int,
        in_channels: int = 4,
        gamma: float = 0.99,
        lr: float = 1e-4,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay_steps: int = 250_000,
        target_update_freq: int = 1000,
        device: str = "cpu"
    ):
        self.algo = algo.lower().replace("-", "").replace("_", "")
        self.num_actions = num_actions
        self.gamma = gamma
        self.device = torch.device(device)

        # Exploration hyperparameters
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_update_freq = target_update_freq
        self.total_steps = 0
        self.current_env_step = 0

        # Build online and target networks
        self.online_net = build_network(self.algo, in_channels, num_actions).to(self.device)
        self.target_net = copy.deepcopy(self.online_net).to(self.device)
        self.target_net.eval()
        for p in self.target_net.parameters():
            p.requires_grad = False

        # Optimizer: RMSProp (Mnih et al.) or Adam (standard modern practice)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr, eps=1e-5)

        # Mode flags
        self.is_double = "double" in self.algo

    def set_env_step(self, step: int):
        """Syncs the environment step count for precise exploration annealing."""
        self.current_env_step = step

    @property
    def epsilon(self) -> float:
        """Computes current decayed epsilon value based on environment steps."""
        step_count = self.current_env_step if self.current_env_step > 0 else self.total_steps
        fraction = min(1.0, step_count / float(self.eps_decay_steps))
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, eval_mode: bool = False, eval_eps: float = 0.01) -> int:
        """
        Epsilon-greedy action selection.
        If eval_mode is True, uses eval_eps (near-greedy policy as per assignment spec).
        """
        eps = eval_eps if eval_mode else self.epsilon

        if np.random.rand() < eps:
            return np.random.randint(self.num_actions)

        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_t)
            action = q_values.argmax(dim=-1).item()
        return action

    def update(self, batch) -> dict:
        """
        Performs one gradient descent step on a sampled batch of transitions.
        Computes Bellman targets according to algorithm variant (Vanilla vs Double).
        """
        states, actions, rewards, next_states, dones = batch
        self.total_steps += 1

        # Current Q-values: Q(s, a; theta)
        q_pred = self.online_net(states)
        q_pred_selected = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.is_double:
                # Double Q-Learning:
                # Action selected by ONLINE network: a* = argmax_a Q(s', a; theta)
                next_q_online = self.online_net(next_states)
                best_actions = next_q_online.argmax(dim=-1, keepdim=True)
                
                # Action evaluated by TARGET network: Q(s', a*; theta^-)
                next_q_target = self.target_net(next_states)
                next_q_value = next_q_target.gather(1, best_actions).squeeze(1)
            else:
                # Vanilla DQN:
                # Target network selects and evaluates max action: max_a Q(s', a; theta^-)
                next_q_target = self.target_net(next_states)
                next_q_value = next_q_target.max(dim=-1)[0]

            # Bellman target: y = r + gamma * (1 - done) * Q_next
            target = rewards + self.gamma * (1.0 - dones) * next_q_value

        # Huber loss (Smooth L1 Loss)
        loss = F.smooth_l1_loss(q_pred_selected, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent gradient explosion
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Update target network periodically
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {
            "loss": loss.item(),
            "mean_q": q_pred_selected.mean().item(),
            "epsilon": self.epsilon
        }

    def compute_average_q(self, states: torch.Tensor) -> float:
        """
        Computes average predicted Q-value across a fixed or sampled batch of states.
        Key metric for tracking Q-value overestimation bias.
        """
        self.online_net.eval()
        with torch.no_grad():
            q_values = self.online_net(states)
            max_q = q_values.max(dim=-1)[0].mean().item()
        self.online_net.train()
        return max_q

    def save(self, filepath: str):
        """Save model checkpoint."""
        torch.save({
            "algo": self.algo,
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps
        }, filepath)

    def load(self, filepath: str):
        """Load model checkpoint."""
        ckpt = torch.load(filepath, map_location=self.device)
        self.online_net.load_state_dict(ckpt["online_state_dict"])
        self.target_net.load_state_dict(ckpt["target_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.total_steps = ckpt.get("total_steps", 0)


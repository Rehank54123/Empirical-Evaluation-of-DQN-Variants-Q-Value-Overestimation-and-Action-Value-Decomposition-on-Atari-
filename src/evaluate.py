"""
Evaluation Protocol for Atari DQN Agents.
Conforms to project requirements:
- Evaluates agent over 30 episodes using near-greedy policy (epsilon = 0.01)
- Computes mean and median episode return
- Computes variance and standard error
- Records Q-value estimates along trajectories
- Converts raw returns to Human-Normalized Scores (HNS)
"""

import argparse
import numpy as np
import torch

from src.wrappers import make_atari_env
from src.normalization import compute_human_normalized_score


def evaluate_policy(agent, env, episodes: int = 30, eval_eps: float = 0.01):
    """
    Evaluates policy across `episodes` runs.
    Returns dictionary of evaluation statistics.
    """
    returns = []
    lengths = []
    avg_q_per_ep = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = False
        ep_return = 0.0
        ep_len = 0
        ep_q_vals = []

        while not done:
            action = agent.select_action(obs, eval_mode=True, eval_eps=eval_eps)

            # Record Q-value for chosen action
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
            with torch.no_grad():
                q_vals = agent.online_net(obs_t)
                ep_q_vals.append(q_vals[0, action].item())

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            ep_len += 1
            done = terminated or truncated

        returns.append(ep_return)
        lengths.append(ep_len)
        avg_q_per_ep.append(np.mean(ep_q_vals) if ep_q_vals else 0.0)

    returns = np.array(returns)
    lengths = np.array(lengths)
    avg_q_per_ep = np.array(avg_q_per_ep)

    stats = {
        "mean_return": float(np.mean(returns)),
        "median_return": float(np.median(returns)),
        "std_return": float(np.std(returns)),
        "sem_return": float(np.std(returns) / np.sqrt(episodes)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_eval_q": float(np.mean(avg_q_per_ep)),
        "raw_returns": returns.tolist(),
        "episodes": episodes
    }
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt model checkpoint")
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5", help="Gymnasium environment id")
    parser.add_argument("--episodes", type=int, default=30, help="Number of test episodes")
    parser.add_argument("--eval_eps", type=float, default=0.01, help="Epsilon for near-greedy policy")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    algo = ckpt.get("algo", "vanilla_dqn")

    eval_env = make_atari_env(args.env_id, is_eval=True)
    num_actions = eval_env.action_space.n

    from src.agent import DQNAgent
    agent = DQNAgent(algo=algo, num_actions=num_actions, device=args.device)
    agent.load(args.checkpoint)

    print(f"Evaluating {algo} on {args.env_id} over {args.episodes} episodes (eps={args.eval_eps})...")
    stats = evaluate_policy(agent, eval_env, episodes=args.episodes, eval_eps=args.eval_eps)
    hns = compute_human_normalized_score(args.env_id, stats["mean_return"])

    print("\n--- Evaluation Results ---")
    print(f"Mean Return:        {stats['mean_return']:.2f} +/- {stats['sem_return']:.2f}")
    print(f"Median Return:      {stats['median_return']:.2f}")
    print(f"Min / Max Return:   {stats['min_return']:.2f} / {stats['max_return']:.2f}")
    print(f"Mean Episode Steps: {stats['mean_length']:.1f}")
    print(f"Mean Predicted Q:   {stats['mean_eval_q']:.2f}")
    print(f"Human-Norm Score:   {hns * 100:.2f}%")


if __name__ == "__main__":
    main()


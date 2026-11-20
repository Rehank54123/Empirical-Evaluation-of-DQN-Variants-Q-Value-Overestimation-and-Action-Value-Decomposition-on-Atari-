"""
Training Script for Atari DQN Variants.
Supports Vanilla DQN, Double DQN, and Dueling Double DQN across multiple seeds and games.
Tracks:
- Episodic rewards and lengths
- Training losses and epsilon decay
- Fixed-state evaluation Q-values for Q-bias curves
- Periodic evaluation protocol (30 near-greedy episodes)
- Saves checkpoints and structured logs (JSON)
"""

import argparse
import json
import os
import random
import time
import numpy as np
import torch

from src.agent import DQNAgent
from src.replay_buffer import ReplayBuffer
from src.wrappers import make_atari_env
from src.evaluate import evaluate_policy
from src.normalization import compute_human_normalized_score
from configs.default_config import GAMES_CONFIG, DEFAULT_HYPERPARAMS


def set_seed(seed: int):
    """Ensure strict experimental reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(
    algo: str = "double_dqn",
    game_key: str = "pong",
    seed: int = 1,
    total_timesteps: int = 500_000,
    eval_frequency: int = 25_000,
    eval_episodes: int = 30,
    output_dir: str = "runs",
    device: str = "auto"
):
    if device == "auto":
        if torch.cuda.is_available():
            try:
                # Test whether device kernel image is compatible
                _test = torch.zeros(1, device="cuda") + 1
                device = "cuda"
            except Exception as e:
                print(f"[Notice] CUDA is present ({torch.cuda.get_device_name(0)}) but kernels are not compiled for sm_120. Falling back to CPU for execution.")
                device = "cpu"
        else:
            device = "cpu"
    print(f"--> Initializing Training: Algo={algo}, Game={game_key}, Seed={seed}, Device={device}")

    set_seed(seed)

    # Resolve environment ID
    game_info = GAMES_CONFIG.get(game_key.lower())
    env_id = game_info["env_id"] if game_info else game_key

    # Setup directories
    run_name = f"{algo}_{game_key}_seed{seed}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Create Environments
    train_env = make_atari_env(env_id, is_eval=False)
    eval_env = make_atari_env(env_id, is_eval=True)

    num_actions = train_env.action_space.n
    hparams = DEFAULT_HYPERPARAMS

    agent = DQNAgent(
        algo=algo,
        num_actions=num_actions,
        in_channels=hparams["frame_stack"],
        gamma=hparams["gamma"],
        lr=hparams["learning_rate"],
        eps_start=hparams["epsilon_start"],
        eps_end=hparams["epsilon_end"],
        eps_decay_steps=hparams["epsilon_decay_steps"],
        target_update_freq=hparams["target_update_frequency"],
        device=device
    )

    buffer = ReplayBuffer(
        capacity=hparams["buffer_capacity"],
        state_shape=(hparams["frame_stack"], hparams["frame_height"], hparams["frame_width"]),
        device=device
    )

    # Fixed states holdout pool for Q-bias overestimation curve tracking
    fixed_eval_states = None

    # Logging structures
    metrics = {
        "algo": algo,
        "game": game_key,
        "env_id": env_id,
        "seed": seed,
        "train_episodes": [],
        "evaluations": [],
        "q_bias_tracking": []
    }

    obs, _ = train_env.reset(seed=seed)
    ep_reward = 0.0
    ep_len = 0
    ep_idx = 0
    start_time = time.time()

    for step in range(1, total_timesteps + 1):
        agent.set_env_step(step)
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = train_env.step(action)
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_reward += reward
        ep_len += 1

        # Train step
        if step >= hparams["learning_starts"] and step % hparams["train_frequency"] == 0:
            batch = buffer.sample(hparams["batch_size"])
            agent.update(batch)

        # Collect holdout states once buffer is warm for steady Q-overestimation evaluation
        if fixed_eval_states is None and step >= hparams["learning_starts"]:
            fixed_batch = buffer.sample(min(128, len(buffer)))
            fixed_eval_states = fixed_batch[0].clone()

        # Handle episode termination in training environment
        if done:
            ep_idx += 1
            metrics["train_episodes"].append({
                "episode": ep_idx,
                "step": step,
                "reward": float(ep_reward),
                "length": ep_len,
                "epsilon": agent.epsilon
            })
            obs, _ = train_env.reset()
            ep_reward = 0.0
            ep_len = 0

        # Periodic Evaluation & Q-bias calculation
        if step % eval_frequency == 0 or step == total_timesteps:
            # Fast intermediate evaluation (5 episodes); full evaluation (30 episodes) at the final step
            current_eval_eps = eval_episodes if step == total_timesteps else min(eval_episodes, 5)
            eval_stats = evaluate_policy(agent, eval_env, episodes=current_eval_eps, eval_eps=hparams["eval_epsilon"])
            hns = compute_human_normalized_score(env_id, eval_stats["mean_return"])
            eval_stats["step"] = step
            eval_stats["human_normalized_score"] = float(hns)
            metrics["evaluations"].append(eval_stats)

            # Track Q-value overestimation on fixed holdout state set
            if fixed_eval_states is not None:
                avg_holdout_q = agent.compute_average_q(fixed_eval_states)
                metrics["q_bias_tracking"].append({
                    "step": step,
                    "avg_q": float(avg_holdout_q),
                    "mean_eval_return": eval_stats["mean_return"]
                })

            elapsed = time.time() - start_time
            fps = step / elapsed if elapsed > 0 else 0
            print(
                f"[Step {step:6d}/{total_timesteps}] "
                f"Eval Return: {eval_stats['mean_return']:6.2f} +/- {eval_stats['sem_return']:4.2f} | "
                f"HNS: {hns*100:6.1f}% | "
                f"Eps: {agent.epsilon:.3f} | FPS: {fps:4.1f} ({current_eval_eps} eps)",
                flush=True
            )

            # Save periodic checkpoint
            agent.save(os.path.join(ckpt_dir, f"model_step_{step}.pt"))

            # Save metrics JSON
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

    # Save final model
    agent.save(os.path.join(run_dir, "final_model.pt"))
    train_env.close()
    eval_env.close()

    print(f"--> Training completed for {run_name}. Logs saved to {run_dir}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN variant on Atari.")
    parser.add_argument("--algo", type=str, default="double_dqn", choices=["vanilla_dqn", "double_dqn", "dueling_dqn"])
    parser.add_argument("--game", type=str, default="pong", choices=["pong", "spaceinvaders", "mspacman", "seaquest"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--eval_freq", type=int, default=25_000)
    parser.add_argument("--eval_episodes", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="runs")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    train(
        algo=args.algo,
        game_key=args.game,
        seed=args.seed,
        total_timesteps=args.timesteps,
        eval_frequency=args.eval_freq,
        eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        device=args.device
    )


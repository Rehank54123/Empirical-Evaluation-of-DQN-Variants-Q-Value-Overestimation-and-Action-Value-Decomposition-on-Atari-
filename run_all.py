"""
Master Orchestrator to Run Atari DQN Variant Experiments.
Executes training across games and algorithms, saves logs to runs/,
and automatically generates publication figures upon completion.
"""

import argparse
import os
import sys
import time
from src.train import train
from src.visualize import load_runs_data, plot_learning_curves, plot_q_value_overestimation


def main():
    parser = argparse.ArgumentParser(description="Run Atari DQN Variant Experiment Matrix.")
    parser.add_argument("--games", nargs="+", default=["pong", "spaceinvaders", "mspacman", "seaquest"],
                        help="List of games to train on.")
    parser.add_argument("--algos", nargs="+", default=["dueling_dqn", "double_dqn", "vanilla_dqn"],
                        help="List of algorithms to evaluate.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1],
                        help="Random seeds to train.")
    parser.add_argument("--timesteps", type=int, default=250_000,
                        help="Total environment interaction steps per run (default: 250k for fast convergence).")
    parser.add_argument("--eval_freq", type=int, default=50_000,
                        help="Evaluation frequency in environment steps (default: 50k).")
    parser.add_argument("--eval_episodes", type=int, default=30,
                        help="Number of near-greedy evaluation episodes.")
    parser.add_argument("--output_dir", type=str, default="runs",
                        help="Root directory for saving checkpoints and metrics.")
    parser.add_argument("--device", type=str, default="auto",
                        help="Compute device ('cuda', 'cpu', or 'auto').")
    args = parser.parse_args()

    total_runs = len(args.games) * len(args.algos) * len(args.seeds)
    print("=" * 70)
    print(f"STARTING ATARI EXPERIMENT MATRIX: {total_runs} RUNS SCHEDULED")
    print(f"Games:      {args.games}")
    print(f"Algorithms: {args.algos}")
    print(f"Seeds:      {args.seeds}")
    print(f"Timesteps:  {args.timesteps:,} steps per run")
    print(f"Eval Freq:  Every {args.eval_freq:,} steps ({args.eval_episodes} episodes)")
    print(f"Device:     {args.device}")
    print("=" * 70)

    start_all = time.time()
    run_idx = 0

    for game in args.games:
        for algo in args.algos:
            for seed in args.seeds:
                run_idx += 1
                run_name = f"{algo}_{game}_seed{seed}"
                run_dir = os.path.join(args.output_dir, run_name)
                final_model = os.path.join(run_dir, "final_model.pt")
                metrics_file = os.path.join(run_dir, "metrics.json")

                # Check if this run is already completed
                if os.path.exists(final_model) and os.path.exists(metrics_file):
                    print(f"\n>>> [{run_idx}/{total_runs}] Skipping {run_name} (Already completed!) <<<", flush=True)
                    continue

                print(f"\n>>> [{run_idx}/{total_runs}] Running {algo.upper()} on {game.upper()} (Seed {seed}) <<<", flush=True)
                run_start = time.time()
                try:
                    train(
                        algo=algo,
                        game_key=game,
                        seed=seed,
                        total_timesteps=args.timesteps,
                        eval_frequency=args.eval_freq,
                        eval_episodes=args.eval_episodes,
                        output_dir=args.output_dir,
                        device=args.device
                    )
                    elapsed = time.time() - run_start
                    print(f"Completed run {run_idx}/{total_runs} in {elapsed / 60:.1f} minutes.")
                except Exception as e:
                    print(f"Error during run {algo}_{game}_seed{seed}: {e}", file=sys.stderr)

                # Generate updated plots after each completed run
                try:
                    runs_data = load_runs_data(args.output_dir)
                    if runs_data:
                        os.makedirs("figures", exist_ok=True)
                        plot_learning_curves(runs_data, os.path.join("figures", "learning_curves.png"))
                        plot_q_value_overestimation(runs_data, os.path.join("figures", "q_overestimation_curves.png"))
                except Exception as pe:
                    print(f"Plotting notice: {pe}")

    total_elapsed = time.time() - start_all
    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETED in {total_elapsed / 3600:.2f} hours.")
    print("=" * 70)


if __name__ == "__main__":
    main()


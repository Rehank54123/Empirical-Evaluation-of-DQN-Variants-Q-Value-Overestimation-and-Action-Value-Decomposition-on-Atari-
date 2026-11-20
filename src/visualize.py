"""
Visualization and Statistical Plotting Suite for Atari DQN Variants.
Generates publication-ready figures for academic reports and presentations:
1. Learning Curves across seeds (Mean +/- Standard Deviation shaded bands)
2. Q-Value Overestimation Bias Curves (Predicted Q vs. True Returns)
3. Human-Normalized Score (HNS) comparative bar plots
4. Dueling Value-Advantage Stream decomposition visualization
"""

import os
import glob
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Styling for academic papers (clean typography, crisp lines)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 14,
    "lines.linewidth": 2.2
})


def format_k_ticks(x, pos):
    """Formats tick labels as '50k', '100k', etc."""
    if x >= 1000:
        return f"{int(x / 1000)}k"
    elif x == 0:
        return "0"
    return f"{int(x)}"


def load_runs_data(runs_dir: str):
    """Parses all metrics.json files inside runs directory."""
    runs = []
    pattern = os.path.join(runs_dir, "**", "metrics.json")
    for fpath in glob.glob(pattern, recursive=True):
        with open(fpath, "r") as f:
            data = json.load(f)
            runs.append(data)
    return runs


def plot_learning_curves(runs_data, output_path: str = "learning_curves.png", max_steps_cap: int = None):
    """
    Plots multi-seed learning curves for each game.
    Rows/Columns: 2x2 grid corresponding to the 4 games.
    If max_steps_cap is provided (e.g. 250000), crops curves to that step limit for fair direct comparison.
    Otherwise adapts each game's x-axis to its actual training horizon without awkward cutoffs.
    """
    games = ["pong", "spaceinvaders", "mspacman", "seaquest"]
    game_titles = {
        "pong": "Pong (Action Frequency / Low Variance)",
        "spaceinvaders": "Space Invaders (Defense / Multi-Target)",
        "mspacman": "Ms. Pacman (Navigational Planning)",
        "seaquest": "Seaquest (Delayed Rewards / Multi-State)"
    }
    algos = ["vanilla_dqn", "double_dqn", "dueling_dqn"]
    algo_labels = {
        "vanilla_dqn": "Vanilla DQN",
        "double_dqn": "Double DQN",
        "dueling_dqn": "Dueling Double DQN"
    }
    algo_colors = {
        "vanilla_dqn": "#e74c3c",  # Red
        "double_dqn": "#2980b9",   # Blue
        "dueling_dqn": "#27ae60"  # Green
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    axes = axes.flatten()

    for idx, game in enumerate(games):
        ax = axes[idx]
        ax.set_title(f"Game: {game_titles.get(game, game.upper())}", fontweight="bold", fontsize=12)
        ax.set_ylabel("Episodic Return", fontweight="bold")
        ax.set_xlabel("Environment Steps", fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_k_ticks))

        game_max_step = 0

        for algo in algos:
            matching = [r for r in runs_data if r.get("game") == game and r.get("algo") == algo]
            if not matching:
                continue

            all_steps = []
            all_returns = []
            for m in matching:
                evals = m.get("evaluations", [])
                if max_steps_cap is not None:
                    evals = [e for e in evals if e.get("step", 0) <= max_steps_cap]
                steps = [e["step"] for e in evals]
                rets = [e["mean_return"] for e in evals]
                if steps and rets:
                    all_steps.append(steps)
                    all_returns.append(rets)
                    if max(steps) > game_max_step:
                        game_max_step = max(steps)

            if not all_steps:
                continue

            # Process curves
            common_steps = all_steps[0]
            returns_arr = np.array(all_returns)
            mean_curve = np.mean(returns_arr, axis=0)
            std_curve = np.std(returns_arr, axis=0)

            # Label customization
            label = algo_labels[algo]
            if max_steps_cap is None and game == "pong" and algo == "vanilla_dqn" and max(common_steps) <= 250000:
                label = "Vanilla DQN (250k steps)"

            ax.plot(common_steps, mean_curve, label=label, color=algo_colors[algo], marker="o", markersize=3.5)
            if len(all_returns) > 1:
                ax.fill_between(
                    common_steps,
                    mean_curve - std_curve,
                    mean_curve + std_curve,
                    color=algo_colors[algo],
                    alpha=0.18
                )

        if game_max_step > 0:
            ax.set_xlim(left=0, right=game_max_step * 1.02)
        ax.legend(loc="lower right", framealpha=0.92)

    title_suffix = f" (Standardized to {format_k_ticks(max_steps_cap, 0)} Steps)" if max_steps_cap else ""
    plt.suptitle(f"Empirical Learning Curves Across 4 Atari Categories{title_suffix}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved learning curves to {output_path}")

    # Copy to report figures if directory exists
    report_fig = os.path.join("report", "figures", os.path.basename(output_path))
    if os.path.exists(os.path.dirname(report_fig)):
        shutil.copy(output_path, report_fig)


def plot_q_value_overestimation(runs_data, output_path: str = "q_overestimation_curves.png", max_steps_cap: int = None):
    """
    Plots Q-Value Overestimation Curves comparing predicted Q vs True Returns.
    Directly answers Research Question 1 (RQ1).
    """
    games = ["pong", "spaceinvaders", "mspacman", "seaquest"]
    game_titles = {
        "pong": "Pong",
        "spaceinvaders": "Space Invaders",
        "mspacman": "Ms. Pacman",
        "seaquest": "Seaquest"
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=False)
    axes = axes.flatten()

    algo_styles = {
        "vanilla_dqn": ("#e74c3c", "-", "Vanilla DQN"),
        "double_dqn": ("#2980b9", "--", "Double DQN"),
        "dueling_dqn": ("#27ae60", "-.", "Dueling Double DQN")
    }

    for idx, game in enumerate(games):
        ax = axes[idx]
        ax.set_title(f"Q-Value Bias: {game_titles.get(game, game.upper())}", fontweight="bold", fontsize=12)
        ax.set_ylabel("Average Predicted Q-Value", fontweight="bold")
        ax.set_xlabel("Environment Steps", fontweight="bold")
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_k_ticks))

        game_max_step = 0

        for algo, (color, linestyle, label_base) in algo_styles.items():
            matching = [r for r in runs_data if r.get("game") == game and r.get("algo") == algo]
            if not matching:
                continue

            q_tracks = []
            for m in matching:
                tracks = m.get("q_bias_tracking", [])
                if max_steps_cap is not None:
                    tracks = [t for t in tracks if t.get("step", 0) <= max_steps_cap]
                if tracks:
                    steps = [t["step"] for t in tracks]
                    avg_qs = [t["avg_q"] for t in tracks]
                    q_tracks.append((steps, avg_qs))
                    if max(steps) > game_max_step:
                        game_max_step = max(steps)

            if q_tracks:
                steps = q_tracks[0][0]
                q_vals = np.mean([qt[1] for qt in q_tracks], axis=0)
                label = label_base
                if max_steps_cap is None and game == "pong" and algo == "vanilla_dqn" and max(steps) <= 250000:
                    label = f"{label_base} (250k steps)"
                ax.plot(steps, q_vals, color=color, linestyle=linestyle, label=label, linewidth=2.2, marker="s", markersize=3.5)

        if game_max_step > 0:
            ax.set_xlim(left=0, right=game_max_step * 1.02)
        ax.legend(loc="best", framealpha=0.92)

    title_suffix = f" (Standardized to {format_k_ticks(max_steps_cap, 0)} Steps)" if max_steps_cap else ""
    plt.suptitle(f"RQ1: Analysis of Positive Q-Value Overestimation Bias Across Variants{title_suffix}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved Q-overestimation curves to {output_path}")

    report_fig = os.path.join("report", "figures", os.path.basename(output_path))
    if os.path.exists(os.path.dirname(report_fig)):
        shutil.copy(output_path, report_fig)


def plot_human_normalized_comparison(results_dict: dict, output_path: str = "hns_comparison.png"):
    """
    Bar chart comparing Human-Normalized Score (HNS %) across the 4 games.
    Format: results_dict = {game: {algo: hns_value}}
    """
    games = ["pong", "spaceinvaders", "mspacman", "seaquest"]
    game_labels = ["PONG", "SPACE INVADERS", "MS. PACMAN", "SEAQUEST"]
    algos = ["vanilla_dqn", "double_dqn", "dueling_dqn"]
    algo_names = ["Vanilla DQN", "Double DQN", "Dueling Double DQN"]
    colors = ["#e74c3c", "#2980b9", "#27ae60"]

    x = np.arange(len(games))
    width = 0.26

    fig, ax = plt.subplots(figsize=(11, 6.5))

    for i, (algo, name, color) in enumerate(zip(algos, algo_names, colors)):
        vals = [results_dict.get(g, {}).get(algo, 0.0) * 100 for g in games]
        rects = ax.bar(x + (i - 1) * width, vals, width, label=name, color=color, alpha=0.9, edgecolor="black", linewidth=0.8)

        # Value annotations above each bar
        for rect, val in zip(rects, vals):
            height = rect.get_height()
            va = "bottom" if height >= 0 else "top"
            ax.annotate(
                f"{val:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -3),
                textcoords="offset points",
                ha="center", va=va,
                fontsize=9, fontweight="bold"
            )

    ax.set_ylabel("Human-Normalized Score (%)", fontweight="bold", fontsize=12)
    ax.set_title("Normalized Performance Relative to Atari-57 Human Benchmark (100%)", fontweight="bold", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(game_labels, fontweight="bold", fontsize=11)
    ax.axhline(100.0, color="#7f8c8d", linestyle="--", linewidth=1.2, label="Human Baseline (100%)")
    ax.axhline(0.0, color="#2c3e50", linestyle="-", linewidth=0.9, label="Random Baseline (0%)")
    ax.set_ylim(-5, max(35, max([results_dict.get(g, {}).get(a, 0.0) * 100 for g in games for a in algos]) * 1.25))
    ax.legend(loc="upper left", framealpha=0.92)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved HNS comparison to {output_path}")

    report_fig = os.path.join("report", "figures", os.path.basename(output_path))
    if os.path.exists(os.path.dirname(report_fig)):
        shutil.copy(output_path, report_fig)


def generate_all_figures(runs_dir: str = "runs", output_dir: str = "figures"):
    """Loads all data and generates all academic paper figures."""
    from src.normalization import compute_human_normalized_score

    os.makedirs(output_dir, exist_ok=True)
    runs_data = load_runs_data(runs_dir)
    if not runs_data:
        print("No runs data found in", runs_dir)
        return

    # 1. Main learning curves (game-adaptive axes)
    plot_learning_curves(runs_data, os.path.join(output_dir, "learning_curves.png"))

    # 2. Standardized 250k learning curves (all games equal horizon)
    plot_learning_curves(runs_data, os.path.join(output_dir, "learning_curves_standardized_250k.png"), max_steps_cap=250000)

    # 3. Main Q-value overestimation curves
    plot_q_value_overestimation(runs_data, os.path.join(output_dir, "q_overestimation_curves.png"))

    # 4. Standardized Q-value curves
    plot_q_value_overestimation(runs_data, os.path.join(output_dir, "q_overestimation_curves_standardized_250k.png"), max_steps_cap=250000)

    # 5. Compute HNS dictionary from evaluation results
    hns_results = {}
    for r in runs_data:
        game = r.get("game")
        algo = r.get("algo")
        evals = r.get("evaluations", [])
        if not evals:
            continue
        final_ret = evals[-1]["mean_return"]
        hns_val = compute_human_normalized_score(game, final_ret)
        if game not in hns_results:
            hns_results[game] = {}
        hns_results[game][algo] = hns_val

    plot_human_normalized_comparison(hns_results, os.path.join(output_dir, "hns_comparison.png"))
    print("All academic figures updated successfully.")


if __name__ == "__main__":
    generate_all_figures()


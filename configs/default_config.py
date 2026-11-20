"""
Central Configuration for Atari DQN Variant Experiments.
Specifies environments, 4 categorical assignments, algorithms,
hyperparameters, and compute budgets.
"""

# The 4 Selected Atari Games representing 4 Distinct Categories
GAMES_CONFIG = {
    "pong": {
        "env_id": "ALE/Pong-v5",
        "alt_env_id": "PongNoFrameskip-v4",
        "category": "Reflexive / 2D Reactive Ball & Paddle",
        "action_space": 6,
        "effective_actions": 3,
        "description": "Fast continuous reaction task with low action complexity and immediate reward feedback."
    },
    "spaceinvaders": {
        "env_id": "ALE/SpaceInvaders-v5",
        "alt_env_id": "SpaceInvadersNoFrameskip-v4",
        "category": "Fixed-Screen Tactical Shooter",
        "action_space": 6,
        "effective_actions": 6,
        "description": "Defensive swarm shooting with high-variance targets and spatial risk management."
    },
    "mspacman": {
        "env_id": "ALE/MsPacman-v5",
        "alt_env_id": "MsPacmanNoFrameskip-v4",
        "category": "Maze Navigation & Dense Spatial Foraging",
        "action_space": 9,
        "effective_actions": 9,
        "description": "Spatial corridor traversal with dense rewards and frequent action redundancy."
    },
    "seaquest": {
        "env_id": "ALE/Seaquest-v5",
        "alt_env_id": "SeaquestNoFrameskip-v4",
        "category": "Multi-Objective Resource Management & Delayed Horizon",
        "action_space": 18,
        "effective_actions": 18,
        "description": "Complex 18-action submarine game requiring oxygen management, diver rescue, and tactical shooting."
    }
}

ALGORITHMS = ["vanilla_dqn", "double_dqn", "dueling_dqn"]

# Experimental seeds (Puneet Sharma requires at least 3 random seeds)
DEFAULT_SEEDS = [1, 2, 3]

# Hyperparameters conforming to Nature / Double DQN standards
DEFAULT_HYPERPARAMS = {
    # Environment
    "frame_height": 84,
    "frame_width": 84,
    "frame_stack": 4,
    "frame_skip": 4,
    "noop_max": 30,

    # Optimization
    "batch_size": 32,
    "learning_rate": 1e-4,
    "gamma": 0.99,
    "clip_grad_norm": 10.0,
    "loss_fn": "smooth_l1",  # Huber loss

    # Replay Buffer
    "buffer_capacity": 100_000,
    "learning_starts": 10_000,
    "train_frequency": 4,

    # Target Network
    "target_update_frequency": 1_000,

    # Exploration schedule
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay_steps": 150_000,

    # Training and Evaluation Budgets (Fast optimized settings)
    "total_timesteps": 250_000,
    "eval_frequency": 50_000,
    "eval_episodes": 30,         # Full 30-episode evaluation conducted at the final step
    "eval_epsilon": 0.01,        # Near-greedy policy
}


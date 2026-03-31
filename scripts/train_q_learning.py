from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import matplotlib

# Use a non-interactive backend so plots can be saved from terminal runs.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.q_learning_agent import QLearningAgent
from utils.experiment_utils import build_env


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Q-learning training."""
    parser = argparse.ArgumentParser(
        description="Train a tabular Q-learning agent for SweepAgent."
    )
    parser.add_argument(
        "--map-name",
        type=str,
        default="default",
        help="Map preset name to train on.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the agent and checkpoint naming.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Print one compact progress line every N episodes.",
    )

    # Hyperparameters exposed to the UI.
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.10,
        help="Q-learning update step size.",
    )
    parser.add_argument(
        "--discount-factor",
        type=float,
        default=0.95,
        help="Discount factor (gamma).",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.00,
        help="Initial epsilon for epsilon-greedy exploration.",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="Multiplicative epsilon decay applied after each episode.",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.05,
        help="Minimum epsilon floor.",
    )
    return parser.parse_args()


def ensure_output_dirs() -> tuple[Path, Path]:
    """Create output directories for checkpoints and plots if needed."""
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    plot_dir = PROJECT_ROOT / "outputs" / "plots"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, plot_dir


def get_checkpoint_path(map_name: str, seed: int) -> Path:
    """Return the map-specific checkpoint path used by the project."""
    checkpoint_dir, _ = ensure_output_dirs()
    return checkpoint_dir / f"q_learning_agent_{map_name}_seed_{seed}.json"


def moving_average(values: list[float], window: int = 100) -> list[float]:
    """Compute a simple trailing moving average."""
    if not values:
        return []

    output: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        output.append(mean(values[start : idx + 1]))
    return output


def save_line_plot(
    values: list[float],
    title: str,
    y_label: str,
    output_path: Path,
) -> Path:
    """Save a single line plot for one metric."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def serialize_q_table(q_table: Any) -> dict[str, list[float]]:
    """
    Convert an in-memory Q-table into a JSON-safe dictionary.

    This assumes q_table behaves like:
        {state_tuple: [q0, q1, q2, q3], ...}
    """
    serialized: dict[str, list[float]] = {}
    for state, q_values in q_table.items():
        state_key = str(tuple(state))
        serialized[state_key] = [float(x) for x in q_values]
    return serialized


def maybe_get_q_table(agent: QLearningAgent) -> dict[Any, Any]:
    """
    Retrieve the internal q_table from the agent.

    This project has used JSON checkpoint serialization already, so a q_table-like
    structure should exist. If your agent uses a different attribute name, adjust here.
    """
    if hasattr(agent, "q_table"):
        return getattr(agent, "q_table")
    raise AttributeError("QLearningAgent does not expose a 'q_table' attribute.")


def build_agent(args: argparse.Namespace) -> QLearningAgent:
    """
    Create the Q-learning agent with UI-configurable hyperparameters.

    If your QLearningAgent constructor uses different keyword names,
    update only this function.
    """
    return QLearningAgent(
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon_start,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        seed=args.seed,
    )


def train_one_episode(env, agent: QLearningAgent) -> dict[str, float]:
    """Run one full training episode and return summary metrics."""
    state = env.reset()
    done = False
    total_reward = 0.0
    final_info: dict[str, float] = {}

    while not done:
        action = agent.select_action(state, training=True)
        next_state, reward, done, info = env.step(action)

        # Standard one-step tabular Q-learning update.
        agent.update(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward
        final_info = info

    return {
        "total_reward": total_reward,
        "steps_taken": float(final_info["steps_taken"]),
        "cleaned_ratio": float(final_info["cleaned_ratio"]),
        "success": 1.0 if float(final_info["cleaned_ratio"]) == 1.0 else 0.0,
    }


def maybe_decay_epsilon(agent: QLearningAgent) -> None:
    """
    Decay epsilon once per episode if the agent exposes either:
    - decay_epsilon()
    - epsilon / epsilon_decay / epsilon_min fields
    """
    if hasattr(agent, "decay_epsilon") and callable(getattr(agent, "decay_epsilon")):
        agent.decay_epsilon()
        return

    if all(hasattr(agent, name) for name in ("epsilon", "epsilon_decay", "epsilon_min")):
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)


def get_current_epsilon(agent: QLearningAgent) -> float:
    """Read the current epsilon value if available."""
    if hasattr(agent, "epsilon"):
        return float(agent.epsilon)
    return 0.0


def save_checkpoint(
    checkpoint_path: Path,
    args: argparse.Namespace,
    agent: QLearningAgent,
) -> Path:
    """
    Save checkpoint in the same schema expected by QLearningAgent.load().
    """
    if hasattr(agent, "save") and callable(getattr(agent, "save")):
        agent.save(checkpoint_path)
        return checkpoint_path

    if hasattr(agent, "to_dict") and callable(getattr(agent, "to_dict")):
        payload = agent.to_dict()
        payload["metadata"] = {
            "algorithm": "q_learning",
            "map_name": args.map_name,
            "episodes": args.episodes,
            "seed": args.seed,
            "hyperparameters": {
                "learning_rate": args.learning_rate,
                "discount_factor": args.discount_factor,
                "epsilon_start": args.epsilon_start,
                "epsilon_decay": args.epsilon_decay,
                "epsilon_min": args.epsilon_min,
            },
        }

        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return checkpoint_path

    raise AttributeError("QLearningAgent must provide save() or to_dict() for checkpoint serialization.")


def main() -> None:
    args = parse_args()

    env = build_env(map_name=args.map_name)
    agent = build_agent(args)

    _, plot_dir = ensure_output_dirs()
    checkpoint_path = get_checkpoint_path(map_name=args.map_name, seed=args.seed)

    rewards: list[float] = []
    cleaned_ratios: list[float] = []
    successes: list[float] = []
    epsilons: list[float] = []

    print(f"Training Q-learning agent on map='{args.map_name}'...")
    print(f"episodes: {args.episodes}")
    print(f"seed: {args.seed}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"discount_factor: {args.discount_factor}")
    print(f"epsilon_start: {args.epsilon_start}")
    print(f"epsilon_decay: {args.epsilon_decay}")
    print(f"epsilon_min: {args.epsilon_min}")

    for episode_idx in range(1, args.episodes + 1):
        episode_result = train_one_episode(env=env, agent=agent)

        rewards.append(episode_result["total_reward"])
        cleaned_ratios.append(episode_result["cleaned_ratio"])
        successes.append(episode_result["success"])
        epsilons.append(get_current_epsilon(agent))

        maybe_decay_epsilon(agent)

        if episode_idx % args.print_every == 0 or episode_idx == args.episodes:
            start_idx = max(0, len(rewards) - args.print_every)
            avg_reward = mean(rewards[start_idx:])
            avg_cleaned_ratio = mean(cleaned_ratios[start_idx:])
            success_rate = mean(successes[start_idx:])
            current_epsilon = get_current_epsilon(agent)

            print(
                f"Episode {episode_idx}/{args.episodes} | "
                f"avg_reward={avg_reward:.2f} | "
                f"avg_cleaned_ratio={avg_cleaned_ratio * 100:.2f}% | "
                f"success_rate={success_rate * 100:.2f}% | "
                f"epsilon={current_epsilon:.4f}"
            )

    reward_plot_path = save_line_plot(
        values=moving_average(rewards, window=100),
        title=f"SweepAgent Training Reward ({args.map_name})",
        y_label="Reward (100-episode moving average)",
        output_path=plot_dir / f"training_reward_{args.map_name}.png",
    )
    cleaned_plot_path = save_line_plot(
        values=moving_average(cleaned_ratios, window=100),
        title=f"SweepAgent Training Cleaned Ratio ({args.map_name})",
        y_label="Cleaned Ratio (100-episode moving average)",
        output_path=plot_dir / f"training_cleaned_ratio_{args.map_name}.png",
    )
    success_plot_path = save_line_plot(
        values=moving_average(successes, window=100),
        title=f"SweepAgent Training Success Rate ({args.map_name})",
        y_label="Success Rate (100-episode moving average)",
        output_path=plot_dir / f"training_success_{args.map_name}.png",
    )
    epsilon_plot_path = save_line_plot(
        values=epsilons,
        title=f"SweepAgent Epsilon Decay ({args.map_name})",
        y_label="Epsilon",
        output_path=plot_dir / f"training_epsilon_{args.map_name}.png",
    )

    save_checkpoint(checkpoint_path=checkpoint_path, args=args, agent=agent)

    q_table = maybe_get_q_table(agent)
    final_window = min(100, len(rewards))

    print("\n=== Training Complete ===")
    print(f"checkpoint: {checkpoint_path}")
    print(f"reward_plot: {reward_plot_path}")
    print(f"cleaned_plot: {cleaned_plot_path}")
    print(f"success_plot: {success_plot_path}")
    print(f"epsilon_plot: {epsilon_plot_path}")
    print(f"learned_q_states: {len(q_table)}")
    print(f"final_epsilon: {get_current_epsilon(agent):.4f}")
    print(f"last_{final_window}_avg_reward: {mean(rewards[-final_window:]):.2f}")
    print(f"last_{final_window}_avg_cleaned_ratio: {mean(cleaned_ratios[-final_window:]) * 100:.2f}%")
    print(f"last_{final_window}_success_rate: {mean(successes[-final_window:]) * 100:.2f}%")


if __name__ == "__main__":
    main()
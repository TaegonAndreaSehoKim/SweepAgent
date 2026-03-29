from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Add the project root so local modules can be imported when run as a script.
    sys.path.append(str(PROJECT_ROOT))

from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from configs.map_presets import MAP_PRESETS
from env.grid_clean_env import GridCleanEnv
from utils.experiment_utils import build_env, load_or_train_q_agent


def parse_args() -> argparse.Namespace:
    # Parse command-line arguments for multi-map benchmarking.
    parser = argparse.ArgumentParser(
        description="Benchmark random and learned SweepAgent policies across map presets."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Training episodes used only when a checkpoint does not already exist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for checkpoint lookup, training, and random baseline evaluation.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes per map and per agent.",
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        default=list(MAP_PRESETS.keys()),
        help="One or more map preset names to benchmark (for example: default harder).",
    )
    return parser.parse_args()


def ensure_output_dirs() -> None:
    # Create output folders used by benchmark logs and plots.
    (PROJECT_ROOT / "outputs" / "plots").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / "logs").mkdir(parents=True, exist_ok=True)


def run_episode(
    env: GridCleanEnv,
    agent,
    training: bool = False,
) -> Dict[str, float]:
    # Run one full episode and collect summary statistics.
    state = env.reset()
    done = False
    total_reward = 0.0
    final_info: Dict[str, float] = {}

    while not done:
        if isinstance(agent, QLearningAgent):
            action = agent.select_action(state, training=training)
        else:
            action = agent.select_action(state)

        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        final_info = info

    return {
        "avg_reward": total_reward,
        "avg_steps": final_info["steps_taken"],
        "avg_cleaned_ratio": final_info["cleaned_ratio"],
        "success_rate": 1.0 if final_info["cleaned_ratio"] == 1.0 else 0.0,
    }


def evaluate_agent(
    env: GridCleanEnv,
    agent,
    num_episodes: int = 100,
) -> Dict[str, float]:
    # Evaluate one agent over multiple episodes and average the metrics.
    rewards: List[float] = []
    steps: List[float] = []
    cleaned_ratios: List[float] = []
    successes: List[float] = []

    for _ in range(num_episodes):
        result = run_episode(env=env, agent=agent, training=False)
        rewards.append(result["avg_reward"])
        steps.append(result["avg_steps"])
        cleaned_ratios.append(result["avg_cleaned_ratio"])
        successes.append(result["success_rate"])

    return {
        "avg_reward": mean(rewards),
        "avg_steps": mean(steps),
        "avg_cleaned_ratio": mean(cleaned_ratios),
        "success_rate": mean(successes),
    }


def print_summary(title: str, summary: Dict[str, float]) -> None:
    # Print a compact summary block for one benchmark result.
    print(f"\n=== {title} ===")
    print(f"average reward: {summary['avg_reward']:.2f}")
    print(f"average steps: {summary['avg_steps']:.2f}")
    print(f"average cleaned ratio: {summary['avg_cleaned_ratio']:.2%}")
    print(f"success rate: {summary['success_rate']:.2%}")


def save_results_csv(results: List[Dict[str, float | str]]) -> Path:
    # Save benchmark results to a CSV file.
    ensure_output_dirs()
    output_path = PROJECT_ROOT / "outputs" / "logs" / "map_benchmark_results.csv"

    fieldnames = [
        "map_name",
        "agent_type",
        "avg_reward",
        "avg_steps",
        "avg_cleaned_ratio",
        "success_rate",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return output_path


def save_bar_plot(
    results: List[Dict[str, float | str]],
    metric_key: str,
    y_label: str,
    title: str,
    output_filename: str,
) -> Path:
    # Save a grouped bar chart for one benchmark metric.
    ensure_output_dirs()
    map_names = [str(row["map_name"]) for row in results if row["agent_type"] == "random"]

    random_values = []
    learned_values = []

    for map_name in map_names:
        random_row = next(
            row
            for row in results
            if row["map_name"] == map_name and row["agent_type"] == "random"
        )
        learned_row = next(
            row
            for row in results
            if row["map_name"] == map_name and row["agent_type"] == "learned"
        )

        random_values.append(float(random_row[metric_key]))
        learned_values.append(float(learned_row[metric_key]))

    x_positions = list(range(len(map_names)))
    bar_width = 0.36

    plt.figure(figsize=(10, 5.5))
    plt.bar(
        [x - bar_width / 2 for x in x_positions],
        random_values,
        width=bar_width,
        label="Random Agent",
    )
    plt.bar(
        [x + bar_width / 2 for x in x_positions],
        learned_values,
        width=bar_width,
        label="Learned Greedy Agent",
    )

    plt.xticks(x_positions, map_names)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    output_path = PROJECT_ROOT / "outputs" / "plots" / output_filename
    plt.savefig(output_path)
    plt.close()

    return output_path


def main() -> None:
    args = parse_args()

    invalid_maps = [map_name for map_name in args.maps if map_name not in MAP_PRESETS]
    if invalid_maps:
        supported_maps = ", ".join(MAP_PRESETS.keys())
        invalid_str = ", ".join(invalid_maps)
        raise ValueError(
            f"Unknown map name(s): {invalid_str}. Supported maps: {supported_maps}"
        )

    # Benchmark both policies across the selected map presets.
    all_results: List[Dict[str, float | str]] = []

    print("=== Benchmark Configuration ===")
    print(f"training episodes: {args.episodes}")
    print(f"seed: {args.seed}")
    print(f"evaluation episodes: {args.eval_episodes}")
    print(f"maps: {', '.join(args.maps)}")

    for map_name in args.maps:
        print("\n==============================")
        print(f"Running benchmark for map: {map_name}")
        print("==============================")

        random_env = build_env(map_name=map_name)
        random_agent = RandomAgent(
            action_space_size=len(random_env.ACTIONS),
            seed=args.seed,
        )
        random_result = evaluate_agent(
            env=random_env,
            agent=random_agent,
            num_episodes=args.eval_episodes,
        )
        print_summary(f"{map_name} Random Agent", random_result)
        all_results.append(
            {
                "map_name": map_name,
                "agent_type": "random",
                **random_result,
            }
        )

        learned_env = build_env(map_name=map_name)
        learned_agent = load_or_train_q_agent(
            map_name=map_name,
            num_episodes=args.episodes,
            seed=args.seed,
        )
        learned_result = evaluate_agent(
            env=learned_env,
            agent=learned_agent,
            num_episodes=args.eval_episodes,
        )
        print_summary(f"{map_name} Learned Greedy Agent", learned_result)
        all_results.append(
            {
                "map_name": map_name,
                "agent_type": "learned",
                **learned_result,
            }
        )

    csv_path = save_results_csv(all_results)

    success_plot_path = save_bar_plot(
        results=all_results,
        metric_key="success_rate",
        y_label="Success Rate",
        title="SweepAgent Success Rate by Map",
        output_filename="map_benchmark_success_rate.png",
    )
    reward_plot_path = save_bar_plot(
        results=all_results,
        metric_key="avg_reward",
        y_label="Average Reward",
        title="SweepAgent Reward by Map",
        output_filename="map_benchmark_reward.png",
    )
    steps_plot_path = save_bar_plot(
        results=all_results,
        metric_key="avg_steps",
        y_label="Average Steps",
        title="SweepAgent Steps by Map",
        output_filename="map_benchmark_steps.png",
    )
    cleaned_plot_path = save_bar_plot(
        results=all_results,
        metric_key="avg_cleaned_ratio",
        y_label="Average Cleaned Ratio",
        title="SweepAgent Cleaned Ratio by Map",
        output_filename="map_benchmark_cleaned_ratio.png",
    )

    print("\n=== Saved Benchmark Artifacts ===")
    print(csv_path.relative_to(PROJECT_ROOT))
    print(success_plot_path.relative_to(PROJECT_ROOT))
    print(reward_plot_path.relative_to(PROJECT_ROOT))
    print(steps_plot_path.relative_to(PROJECT_ROOT))
    print(cleaned_plot_path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()

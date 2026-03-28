from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Add the project root so local imports work from this script.
    sys.path.append(str(PROJECT_ROOT))

from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from configs.map_presets import (
    DISCOUNT_FACTOR,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    LEARNING_RATE,
    MAP_PRESETS,
    PRINT_EVERY,
    REWARD_CLEAN,
    REWARD_FINISH,
    REWARD_INVALID,
    REWARD_MOVE,
    REWARD_REVISIT,
    TRAIN_EPISODES,
    TRAIN_SEED,
)
from env.grid_clean_env import GridCleanEnv


def ensure_output_dirs() -> None:
    # Create output directories if they do not exist.
    (PROJECT_ROOT / "outputs" / "plots").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / "logs").mkdir(parents=True, exist_ok=True)


def build_env(map_name: str) -> GridCleanEnv:
    # Build one environment from the selected preset.
    if map_name not in MAP_PRESETS:
        raise ValueError(f"Unknown map_name: {map_name}")

    preset = MAP_PRESETS[map_name]

    return GridCleanEnv(
        grid_map=preset["grid_map"],
        max_steps=preset["max_steps"],
        reward_clean=REWARD_CLEAN,
        reward_move=REWARD_MOVE,
        reward_revisit=REWARD_REVISIT,
        reward_invalid=REWARD_INVALID,
        reward_finish=REWARD_FINISH,
    )


def evaluate_random_agent(
    env: GridCleanEnv,
    num_episodes: int = 100,
    seed: int = 42,
) -> Dict[str, float]:
    # Evaluate the random baseline on one map.
    agent = RandomAgent(action_space_size=len(env.ACTIONS), seed=seed)
    episode_results: List[Dict[str, float]] = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        final_info: Dict[str, float] = {}

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            state = next_state
            final_info = info

        episode_results.append(
            {
                "avg_reward": total_reward,
                "avg_steps": final_info["steps_taken"],
                "avg_cleaned_ratio": final_info["cleaned_ratio"],
                "success_rate": 1.0 if final_info["cleaned_ratio"] == 1.0 else 0.0,
            }
        )

    return {
        "avg_reward": mean(result["avg_reward"] for result in episode_results),
        "avg_steps": mean(result["avg_steps"] for result in episode_results),
        "avg_cleaned_ratio": mean(
            result["avg_cleaned_ratio"] for result in episode_results
        ),
        "success_rate": mean(result["success_rate"] for result in episode_results),
    }


def train_q_learning_agent(
    env: GridCleanEnv,
    num_episodes: int = TRAIN_EPISODES,
    seed: int = TRAIN_SEED,
    print_every: int = PRINT_EVERY,
) -> QLearningAgent:
    # Train a Q-learning agent on one map.
    agent = QLearningAgent(
        action_space_size=len(env.ACTIONS),
        learning_rate=LEARNING_RATE,
        discount_factor=DISCOUNT_FACTOR,
        epsilon=EPSILON_START,
        epsilon_decay=EPSILON_DECAY,
        epsilon_min=EPSILON_MIN,
        seed=seed,
    )

    rewards: List[float] = []
    cleaned_ratios: List[float] = []
    success_rates: List[float] = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        final_info: Dict[str, float] = {}

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            total_reward += reward
            state = next_state
            final_info = info

        agent.decay_epsilon()

        rewards.append(total_reward)
        cleaned_ratios.append(final_info["cleaned_ratio"])
        success_rates.append(1.0 if final_info["cleaned_ratio"] == 1.0 else 0.0)

        if episode % print_every == 0:
            print(
                f"Episode {episode}/{num_episodes} | "
                f"avg_reward={mean(rewards[-print_every:]):.2f} | "
                f"avg_cleaned_ratio={mean(cleaned_ratios[-print_every:]):.2%} | "
                f"success_rate={mean(success_rates[-print_every:]):.2%} | "
                f"epsilon={agent.epsilon:.4f}"
            )

    return agent


def evaluate_learned_agent(
    env: GridCleanEnv,
    agent: QLearningAgent,
    num_episodes: int = 100,
) -> Dict[str, float]:
    # Evaluate the learned greedy policy on one map.
    episode_results: List[Dict[str, float]] = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        final_info: Dict[str, float] = {}

        while not done:
            action = agent.get_policy_action(state)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            state = next_state
            final_info = info

        episode_results.append(
            {
                "avg_reward": total_reward,
                "avg_steps": final_info["steps_taken"],
                "avg_cleaned_ratio": final_info["cleaned_ratio"],
                "success_rate": 1.0 if final_info["cleaned_ratio"] == 1.0 else 0.0,
            }
        )

    return {
        "avg_reward": mean(result["avg_reward"] for result in episode_results),
        "avg_steps": mean(result["avg_steps"] for result in episode_results),
        "avg_cleaned_ratio": mean(
            result["avg_cleaned_ratio"] for result in episode_results
        ),
        "success_rate": mean(result["success_rate"] for result in episode_results),
    }


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


def save_benchmark_plot(results: List[Dict[str, float | str]]) -> Path:
    # Save a grouped bar chart for success rate by map and agent.
    ensure_output_dirs()
    output_path = PROJECT_ROOT / "outputs" / "plots" / "map_benchmark_success_rate.png"

    map_names = list(MAP_PRESETS.keys())
    random_success = []
    learned_success = []

    for map_name in map_names:
        random_row = next(
            row for row in results
            if row["map_name"] == map_name and row["agent_type"] == "random"
        )
        learned_row = next(
            row for row in results
            if row["map_name"] == map_name and row["agent_type"] == "learned"
        )

        random_success.append(float(random_row["success_rate"]))
        learned_success.append(float(learned_row["success_rate"]))

    x_positions = list(range(len(map_names)))
    bar_width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(
        [x - bar_width / 2 for x in x_positions],
        random_success,
        width=bar_width,
        label="Random",
    )
    plt.bar(
        [x + bar_width / 2 for x in x_positions],
        learned_success,
        width=bar_width,
        label="Learned",
    )
    plt.xticks(x_positions, map_names)
    plt.xlabel("Map")
    plt.ylabel("Success Rate")
    plt.title("SweepAgent Benchmark: Success Rate by Map")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def print_result_block(map_name: str, title: str, result: Dict[str, float]) -> None:
    # Print one compact result block.
    print(f"\n[{map_name}] {title}")
    print(f"average reward: {result['avg_reward']:.2f}")
    print(f"average steps: {result['avg_steps']:.2f}")
    print(f"average cleaned ratio: {result['avg_cleaned_ratio']:.2%}")
    print(f"success rate: {result['success_rate']:.2%}")


def benchmark_all_maps() -> None:
    # Train and compare random vs learned policy across all presets.
    all_results: List[Dict[str, float | str]] = []

    for map_name in MAP_PRESETS:
        print(f"\n==============================")
        print(f"Running benchmark for map: {map_name}")
        print(f"==============================")

        random_env = build_env(map_name)
        train_env = build_env(map_name)
        eval_env = build_env(map_name)

        random_result = evaluate_random_agent(
            env=random_env,
            num_episodes=100,
            seed=42,
        )
        print_result_block(map_name, "Random Agent", random_result)

        learned_agent = train_q_learning_agent(
            env=train_env,
            num_episodes=TRAIN_EPISODES,
            seed=TRAIN_SEED,
            print_every=PRINT_EVERY,
        )
        learned_result = evaluate_learned_agent(
            env=eval_env,
            agent=learned_agent,
            num_episodes=100,
        )
        print_result_block(map_name, "Learned Greedy Agent", learned_result)

        all_results.append(
            {
                "map_name": map_name,
                "agent_type": "random",
                "avg_reward": random_result["avg_reward"],
                "avg_steps": random_result["avg_steps"],
                "avg_cleaned_ratio": random_result["avg_cleaned_ratio"],
                "success_rate": random_result["success_rate"],
            }
        )
        all_results.append(
            {
                "map_name": map_name,
                "agent_type": "learned",
                "avg_reward": learned_result["avg_reward"],
                "avg_steps": learned_result["avg_steps"],
                "avg_cleaned_ratio": learned_result["avg_cleaned_ratio"],
                "success_rate": learned_result["success_rate"],
            }
        )

    csv_path = save_results_csv(all_results)
    plot_path = save_benchmark_plot(all_results)

    print("\nSaved benchmark files:")
    print(csv_path)
    print(plot_path)


if __name__ == "__main__":
    benchmark_all_maps()
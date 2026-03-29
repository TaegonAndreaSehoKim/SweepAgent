from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Add the project root so local modules can be imported when run as a script.
    sys.path.append(str(PROJECT_ROOT))

from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from env.grid_clean_env import GridCleanEnv
from utils.experiment_utils import build_env, load_or_train_q_agent


def parse_args() -> argparse.Namespace:
    # Parse command-line arguments for comparison configuration.
    parser = argparse.ArgumentParser(
        description="Compare the random baseline and Q-learning agent on a selected map."
    )
    parser.add_argument(
        "--map-name",
        type=str,
        default="default",
        help="Map preset name (for example: default, harder, wide_room, corridor).",
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
        help="Seed used for the Q-learning checkpoint lookup or training.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes for each agent.",
    )
    return parser.parse_args()


def run_episode(
    env: GridCleanEnv,
    agent,
    training: bool = False,
) -> Dict[str, float]:
    # Run one full episode and collect summary statistics.
    state = env.reset()
    done = False
    total_reward = 0
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
        "total_reward": total_reward,
        "steps_taken": final_info["steps_taken"],
        "cleaned_tiles": final_info["cleaned_tiles"],
        "total_dirty_tiles": final_info["total_dirty_tiles"],
        "cleaned_ratio": final_info["cleaned_ratio"],
        "success": 1.0 if final_info["cleaned_ratio"] == 1.0 else 0.0,
    }


def evaluate_agent(
    env: GridCleanEnv,
    agent,
    num_episodes: int = 100,
) -> Dict[str, float]:
    # Evaluate an agent over multiple episodes and average the results.
    rewards: List[float] = []
    steps: List[float] = []
    cleaned_ratios: List[float] = []
    successes: List[float] = []

    for _ in range(num_episodes):
        result = run_episode(env=env, agent=agent, training=False)
        rewards.append(result["total_reward"])
        steps.append(result["steps_taken"])
        cleaned_ratios.append(result["cleaned_ratio"])
        successes.append(result["success"])

    return {
        "average_reward": mean(rewards),
        "average_steps": mean(steps),
        "average_cleaned_ratio": mean(cleaned_ratios),
        "success_rate": mean(successes),
    }


def print_summary(title: str, summary: Dict[str, float]) -> None:
    # Print a compact summary block for an evaluated agent.
    print(f"\n=== {title} ===")
    print(f"average reward: {summary['average_reward']:.2f}")
    print(f"average steps: {summary['average_steps']:.2f}")
    print(f"average cleaned ratio: {summary['average_cleaned_ratio']:.2%}")
    print(f"success rate: {summary['success_rate']:.2%}")


def main() -> None:
    args = parse_args()

    # Use the shared environment preset selected by the user.
    env = build_env(map_name=args.map_name)

    print(f"Evaluating random baseline on map='{args.map_name}'...")
    random_agent = RandomAgent(action_space_size=len(env.ACTIONS), seed=args.seed)
    random_summary = evaluate_agent(
        env=env,
        agent=random_agent,
        num_episodes=args.eval_episodes,
    )
    print_summary("Random Agent", random_summary)

    print(f"\nPreparing Q-learning agent for map='{args.map_name}'...")
    q_agent = load_or_train_q_agent(
        map_name=args.map_name,
        num_episodes=args.episodes,
        seed=args.seed,
    )

    print("\nEvaluating Q-learning agent...")
    q_summary = evaluate_agent(
        env=env,
        agent=q_agent,
        num_episodes=args.eval_episodes,
    )
    print_summary("Q-Learning Agent", q_summary)

    reward_gain = q_summary["average_reward"] - random_summary["average_reward"]
    step_delta = random_summary["average_steps"] - q_summary["average_steps"]
    cleaned_gain = (
        q_summary["average_cleaned_ratio"] - random_summary["average_cleaned_ratio"]
    )
    success_gain = q_summary["success_rate"] - random_summary["success_rate"]

    print("\n=== Comparison ===")
    print(f"map_name: {args.map_name}")
    print(f"eval_episodes: {args.eval_episodes}")
    print(f"reward improvement: {reward_gain:.2f}")
    print(f"step reduction: {step_delta:.2f}")
    print(f"cleaned ratio improvement: {cleaned_gain:.2%}")
    print(f"success rate improvement: {success_gain:.2%}")


if __name__ == "__main__":
    main()

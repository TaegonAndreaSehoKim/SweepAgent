from __future__ import annotations

import argparse
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
from env.grid_clean_env import GridCleanEnv
from utils.experiment_utils import build_env, get_checkpoint_path


def ensure_output_dirs() -> None:
    # Create output folders before saving plots, logs, or checkpoints.
    (PROJECT_ROOT / "outputs" / "plots").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / "checkpoints").mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    # Parse command-line arguments for training configuration.
    parser = argparse.ArgumentParser(
        description="Train a Q-learning SweepAgent on a selected map preset."
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
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the Q-learning agent.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Print rolling training metrics every N episodes.",
    )
    return parser.parse_args()


def run_training_episode(
    env: GridCleanEnv,
    agent: QLearningAgent,
) -> Dict[str, float]:
    # Reset the environment and training metrics for one episode.
    state = env.reset()
    done = False
    total_reward = 0
    final_info: Dict[str, float] = {}

    while not done:
        # Choose an action, step the environment, and update the Q-table.
        action = agent.select_action(state, training=True)
        next_state, reward, done, info = env.step(action)

        agent.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        state = next_state
        total_reward += reward
        final_info = info

    # Decay epsilon after the episode ends.
    agent.decay_epsilon()

    return {
        "total_reward": total_reward,
        "steps_taken": final_info["steps_taken"],
        "cleaned_tiles": final_info["cleaned_tiles"],
        "total_dirty_tiles": final_info["total_dirty_tiles"],
        "cleaned_ratio": final_info["cleaned_ratio"],
        "success": 1.0 if final_info["cleaned_ratio"] == 1.0 else 0.0,
        "epsilon": agent.epsilon,
    }


def run_greedy_episode(
    env: GridCleanEnv,
    agent: QLearningAgent,
    render: bool = True,
) -> Dict[str, float]:
    # Run one evaluation episode with exploration disabled.
    state = env.reset()
    done = False
    total_reward = 0
    final_info: Dict[str, float] = {}
    action_history: List[str] = []

    if render:
        print("\n=== Greedy Policy Episode Start ===")
        env.render()

    while not done:
        # Always choose the current best action from the learned policy.
        action = agent.get_policy_action(state)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        action_history.append(env.ACTION_NAMES[action])

        state = next_state
        final_info = info

        if render:
            print(f"\naction={action} ({env.ACTION_NAMES[action]})")
            print(f"reward={reward}")
            env.render()

    result = {
        "total_reward": total_reward,
        "steps_taken": final_info["steps_taken"],
        "cleaned_tiles": final_info["cleaned_tiles"],
        "total_dirty_tiles": final_info["total_dirty_tiles"],
        "cleaned_ratio": final_info["cleaned_ratio"],
        "success": 1.0 if final_info["cleaned_ratio"] == 1.0 else 0.0,
    }

    if render:
        print("\n=== Greedy Policy Episode Summary ===")
        print(result)
        print("action_history:", action_history)

    return result


def moving_average(values: List[float], window_size: int) -> List[float]:
    # Smooth a metric series with a simple trailing moving average.
    smoothed: List[float] = []

    for idx in range(len(values)):
        start_idx = max(0, idx - window_size + 1)
        window = values[start_idx : idx + 1]
        smoothed.append(sum(window) / len(window))

    return smoothed


def save_training_plots(
    rewards: List[float],
    cleaned_ratios: List[float],
    success_rates: List[float],
    epsilons: List[float],
    map_name: str,
) -> None:
    # Create output folders before saving figures.
    ensure_output_dirs()

    reward_ma = moving_average(rewards, window_size=50)
    cleaned_ma = moving_average(cleaned_ratios, window_size=50)
    success_ma = moving_average(success_rates, window_size=50)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="episode reward")
    plt.plot(reward_ma, label="reward (moving average 50)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(f"Q-Learning Training Reward ({map_name} map)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs" / "plots" / f"training_reward_{map_name}.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(cleaned_ratios, label="episode cleaned ratio")
    plt.plot(cleaned_ma, label="cleaned ratio (moving average 50)")
    plt.xlabel("Episode")
    plt.ylabel("Cleaned Ratio")
    plt.title(f"Q-Learning Cleaning Performance ({map_name} map)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PROJECT_ROOT / "outputs" / "plots" / f"training_cleaned_ratio_{map_name}.png"
    )
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(success_rates, label="episode success")
    plt.plot(success_ma, label="success rate (moving average 50)")
    plt.xlabel("Episode")
    plt.ylabel("Success")
    plt.title(f"Q-Learning Success Trend ({map_name} map)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs" / "plots" / f"training_success_{map_name}.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epsilons, label="epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title(f"Exploration Decay ({map_name} map)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs" / "plots" / f"training_epsilon_{map_name}.png")
    plt.close()


def train_q_learning(
    map_name: str = "default",
    num_episodes: int = 1000,
    seed: int = 42,
    print_every: int = 100,
) -> QLearningAgent:
    # Build the shared environment preset and initialize the agent.
    env = build_env(map_name=map_name)
    agent = QLearningAgent(
        action_space_size=len(env.ACTIONS),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        seed=seed,
    )

    rewards: List[float] = []
    cleaned_ratios: List[float] = []
    success_rates: List[float] = []
    epsilons: List[float] = []

    for episode in range(1, num_episodes + 1):
        result = run_training_episode(env=env, agent=agent)

        rewards.append(result["total_reward"])
        cleaned_ratios.append(result["cleaned_ratio"])
        success_rates.append(result["success"])
        epsilons.append(result["epsilon"])

        if episode % print_every == 0:
            recent_rewards = rewards[-print_every:]
            recent_cleaned = cleaned_ratios[-print_every:]
            recent_success = success_rates[-print_every:]

            print(
                f"Episode {episode}/{num_episodes} | "
                f"avg_reward={mean(recent_rewards):.2f} | "
                f"avg_cleaned_ratio={mean(recent_cleaned):.2%} | "
                f"success_rate={mean(recent_success):.2%} | "
                f"epsilon={agent.epsilon:.4f}"
            )

    save_training_plots(
        rewards=rewards,
        cleaned_ratios=cleaned_ratios,
        success_rates=success_rates,
        epsilons=epsilons,
        map_name=map_name,
    )

    checkpoint_path = get_checkpoint_path(map_name=map_name, seed=seed)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(checkpoint_path)

    print("\n=== Final Training Summary ===")
    print(f"map_name: {map_name}")
    print(f"episodes: {num_episodes}")
    print(f"final epsilon: {agent.epsilon:.4f}")
    print(f"q_table states learned: {len(agent.q_table)}")
    print(f"last 100 avg reward: {mean(rewards[-100:]):.2f}")
    print(f"last 100 avg cleaned ratio: {mean(cleaned_ratios[-100:]):.2%}")
    print(f"last 100 success rate: {mean(success_rates[-100:]):.2%}")
    print(f"checkpoint saved to: {checkpoint_path.relative_to(PROJECT_ROOT)}")

    run_greedy_episode(env=env, agent=agent, render=True)

    print("\nSaved plots:")
    print(f"outputs/plots/training_reward_{map_name}.png")
    print(f"outputs/plots/training_cleaned_ratio_{map_name}.png")
    print(f"outputs/plots/training_success_{map_name}.png")
    print(f"outputs/plots/training_epsilon_{map_name}.png")

    return agent


if __name__ == "__main__":
    args = parse_args()
    train_q_learning(
        map_name=args.map_name,
        num_episodes=args.episodes,
        seed=args.seed,
        print_every=args.print_every,
    )
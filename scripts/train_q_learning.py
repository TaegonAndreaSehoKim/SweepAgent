from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.q_learning_agent import QLearningAgent
from env.grid_clean_env import GridCleanEnv


def ensure_output_dirs() -> None:
    (PROJECT_ROOT / "outputs" / "plots").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "outputs" / "logs").mkdir(parents=True, exist_ok=True)


def run_training_episode(
    env: GridCleanEnv,
    agent: QLearningAgent,
) -> Dict[str, float]:
    state = env.reset()
    done = False
    total_reward = 0
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

        state = next_state
        total_reward += reward
        final_info = info

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
    state = env.reset()
    done = False
    total_reward = 0
    final_info: Dict[str, float] = {}
    action_history: List[str] = []

    if render:
        print("\n=== Greedy Policy Episode Start ===")
        env.render()

    while not done:
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
) -> None:
    ensure_output_dirs()

    reward_ma = moving_average(rewards, window_size=50)
    cleaned_ma = moving_average(cleaned_ratios, window_size=50)
    success_ma = moving_average(success_rates, window_size=50)

    # Reward plot
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="episode reward")
    plt.plot(reward_ma, label="reward (moving average 50)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Training Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs" / "plots" / "training_reward.png")
    plt.close()

    # Cleaned ratio plot
    plt.figure(figsize=(10, 5))
    plt.plot(cleaned_ratios, label="episode cleaned ratio")
    plt.plot(cleaned_ma, label="cleaned ratio (moving average 50)")
    plt.xlabel("Episode")
    plt.ylabel("Cleaned Ratio")
    plt.title("Q-Learning Cleaning Performance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs" / "plots" / "training_cleaned_ratio.png")
    plt.close()

    # Success rate plot
    plt.figure(figsize=(10, 5))
    plt.plot(success_rates, label="episode success")
    plt.plot(success_ma, label="success rate (moving average 50)")
    plt.xlabel("Episode")
    plt.ylabel("Success")
    plt.title("Q-Learning Success Trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs" / "plots" / "training_success.png")
    plt.close()

    # Epsilon plot
    plt.figure(figsize=(10, 5))
    plt.plot(epsilons, label="epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.title("Exploration Decay")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROJECT_ROOT / "outputs" / "plots" / "training_epsilon.png")
    plt.close()


def train_q_learning(
    num_episodes: int = 1000,
    seed: int = 42,
    print_every: int = 100,
) -> None:
    env = GridCleanEnv()
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
    )

    print("\n=== Final Training Summary ===")
    print(f"episodes: {num_episodes}")
    print(f"final epsilon: {agent.epsilon:.4f}")
    print(f"q_table states learned: {len(agent.q_table)}")
    print(f"last 100 avg reward: {mean(rewards[-100:]):.2f}")
    print(f"last 100 avg cleaned ratio: {mean(cleaned_ratios[-100:]):.2%}")
    print(f"last 100 success rate: {mean(success_rates[-100:]):.2%}")

    run_greedy_episode(env=env, agent=agent, render=True)

    print("\nSaved plots:")
    print("outputs/plots/training_reward.png")
    print("outputs/plots/training_cleaned_ratio.png")
    print("outputs/plots/training_success.png")
    print("outputs/plots/training_epsilon.png")


if __name__ == "__main__":
    train_q_learning(num_episodes=1000, seed=42, print_every=100)
from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Add the project root so local modules can be imported from the script.
    sys.path.append(str(PROJECT_ROOT))

from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from configs.default_config import (
    DEFAULT_GRID_MAP,
    DEFAULT_MAX_STEPS,
    DISCOUNT_FACTOR,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    LEARNING_RATE,
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


def build_default_env() -> GridCleanEnv:
    # Build the default experiment environment from shared config values.
    return GridCleanEnv(
        grid_map=DEFAULT_GRID_MAP,
        max_steps=DEFAULT_MAX_STEPS,
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
    # Evaluate the random baseline on the given environment.
    agent = RandomAgent(action_space_size=len(env.ACTIONS), seed=seed)
    episode_results: List[Dict[str, float]] = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        final_info: Dict[str, float] = {}

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            state = next_state
            final_info = info

        episode_results.append(
            {
                "total_reward": total_reward,
                "steps_taken": final_info["steps_taken"],
                "cleaned_ratio": final_info["cleaned_ratio"],
                "success": 1.0 if final_info["cleaned_ratio"] == 1.0 else 0.0,
            }
        )

    return {
        "avg_reward": mean(result["total_reward"] for result in episode_results),
        "avg_steps": mean(result["steps_taken"] for result in episode_results),
        "avg_cleaned_ratio": mean(
            result["cleaned_ratio"] for result in episode_results
        ),
        "success_rate": mean(result["success"] for result in episode_results),
    }


def train_q_learning_agent(
    env: GridCleanEnv,
    num_episodes: int = TRAIN_EPISODES,
    seed: int = TRAIN_SEED,
    print_every: int = PRINT_EVERY,
) -> QLearningAgent:
    # Train a Q-learning agent on the given environment.
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
    # Evaluate the learned greedy policy without exploration.
    episode_results: List[Dict[str, float]] = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        final_info: Dict[str, float] = {}

        while not done:
            action = agent.get_policy_action(state)
            next_state, reward, done, info = env.step(action)

            total_reward += reward
            state = next_state
            final_info = info

        episode_results.append(
            {
                "total_reward": total_reward,
                "steps_taken": final_info["steps_taken"],
                "cleaned_ratio": final_info["cleaned_ratio"],
                "success": 1.0 if final_info["cleaned_ratio"] == 1.0 else 0.0,
            }
        )

    return {
        "avg_reward": mean(result["total_reward"] for result in episode_results),
        "avg_steps": mean(result["steps_taken"] for result in episode_results),
        "avg_cleaned_ratio": mean(
            result["cleaned_ratio"] for result in episode_results
        ),
        "success_rate": mean(result["success"] for result in episode_results),
    }


def print_summary(title: str, result: Dict[str, float]) -> None:
    # Print one result block in a compact format.
    print(f"\n=== {title} ===")
    print(f"average reward: {result['avg_reward']:.2f}")
    print(f"average steps: {result['avg_steps']:.2f}")
    print(f"average cleaned ratio: {result['avg_cleaned_ratio']:.2%}")
    print(f"success rate: {result['success_rate']:.2%}")


def main() -> None:
    # Build separate environment instances for each phase.
    random_env = build_default_env()
    train_env = build_default_env()
    eval_env = build_default_env()

    print("Evaluating random baseline...")
    random_result = evaluate_random_agent(env=random_env, num_episodes=100, seed=42)
    print_summary("Random Agent", random_result)

    print("\nTraining Q-learning agent...")
    learned_agent = train_q_learning_agent(
        env=train_env,
        num_episodes=TRAIN_EPISODES,
        seed=TRAIN_SEED,
        print_every=PRINT_EVERY,
    )

    print("\nEvaluating learned greedy policy...")
    learned_result = evaluate_learned_agent(
        env=eval_env,
        agent=learned_agent,
        num_episodes=100,
    )
    print_summary("Learned Greedy Agent", learned_result)

    print("\n=== Improvement Summary ===")
    print(f"reward gain: {learned_result['avg_reward'] - random_result['avg_reward']:.2f}")
    print(f"step reduction: {random_result['avg_steps'] - learned_result['avg_steps']:.2f}")
    print(
        f"cleaned ratio gain: "
        f"{(learned_result['avg_cleaned_ratio'] - random_result['avg_cleaned_ratio']) * 100:.2f} percentage points"
    )
    print(
        f"success rate gain: "
        f"{(learned_result['success_rate'] - random_result['success_rate']) * 100:.2f} percentage points"
    )
    print("\nTesting on the harder map...")
    harder_random_env = build_harder_env()
    harder_train_env = build_harder_env()
    harder_eval_env = build_harder_env()

    harder_random_result = evaluate_random_agent(
        env=harder_random_env,
        num_episodes=100,
        seed=42,
    )
    print_summary("Random Agent on Harder Map", harder_random_result)

    harder_learned_agent = train_q_learning_agent(
        env=harder_train_env,
        num_episodes=TRAIN_EPISODES,
        seed=TRAIN_SEED,
        print_every=PRINT_EVERY,
    )

    harder_learned_result = evaluate_learned_agent(
        env=harder_eval_env,
        agent=harder_learned_agent,
        num_episodes=100,
    )
    print_summary("Learned Greedy Agent on Harder Map", harder_learned_result)

def build_harder_env() -> GridCleanEnv:
    # Build a slightly harder room layout for a tougher test.
    from configs.default_config import HARDER_GRID_MAP, HARDER_MAX_STEPS

    return GridCleanEnv(
        grid_map=HARDER_GRID_MAP,
        max_steps=HARDER_MAX_STEPS,
        reward_clean=REWARD_CLEAN,
        reward_move=REWARD_MOVE,
        reward_revisit=REWARD_REVISIT,
        reward_invalid=REWARD_INVALID,
        reward_finish=REWARD_FINISH,
    )

if __name__ == "__main__":
    main()


from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Add the project root so local modules can be imported when run as a script.
    sys.path.append(str(PROJECT_ROOT))

from agents.random_agent import RandomAgent
from env.grid_clean_env import GridCleanEnv


def run_episode(
    env: GridCleanEnv,
    agent: RandomAgent,
    render: bool = False,
) -> Dict[str, float]:
    # Reset the environment and tracking values for one episode.
    state = env.reset()
    total_reward = 0
    done = False
    final_info: Dict[str, float] = {}
    action_history: List[str] = []

    if render:
        print("\n=== Episode Start ===")
        env.render()

    while not done:
        # Sample an action, apply it, and record the transition.
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        action_history.append(env.ACTION_NAMES[action])

        state = next_state
        final_info = info

        if render:
            print(f"\naction={action} ({env.ACTION_NAMES[action]})")
            print(f"reward={reward}")
            env.render()

    # Package the episode metrics for downstream averaging.
    result = {
        "total_reward": total_reward,
        "steps_taken": final_info["steps_taken"],
        "cleaned_tiles": final_info["cleaned_tiles"],
        "total_dirty_tiles": final_info["total_dirty_tiles"],
        "cleaned_ratio": final_info["cleaned_ratio"],
        "success": 1.0 if final_info["cleaned_ratio"] == 1.0 else 0.0,
    }

    if render:
        print("\n=== Episode Summary ===")
        print(result)
        print("action_history:", action_history)

    return result


def evaluate_random_agent(
    num_episodes: int = 100,
    seed: int = 42,
    render_first_episode: bool = True,
) -> None:
    # Build the environment and the random baseline agent.
    env = GridCleanEnv()
    agent = RandomAgent(action_space_size=len(env.ACTIONS), seed=seed)

    episode_results = []

    for episode_idx in range(num_episodes):
        # Render only the first episode to keep the output readable.
        render = render_first_episode and episode_idx == 0
        result = run_episode(env=env, agent=agent, render=render)
        episode_results.append(result)

    # Compute average baseline metrics across all episodes.
    avg_reward = mean(result["total_reward"] for result in episode_results)
    avg_steps = mean(result["steps_taken"] for result in episode_results)
    avg_cleaned_tiles = mean(result["cleaned_tiles"] for result in episode_results)
    avg_cleaned_ratio = mean(result["cleaned_ratio"] for result in episode_results)
    success_rate = mean(result["success"] for result in episode_results)

    print("\n=== Random Agent Evaluation Summary ===")
    print(f"episodes: {num_episodes}")
    print(f"average total reward: {avg_reward:.2f}")
    print(f"average steps taken: {avg_steps:.2f}")
    print(f"average cleaned tiles: {avg_cleaned_tiles:.2f}")
    print(f"average cleaned ratio: {avg_cleaned_ratio:.2%}")
    print(f"success rate: {success_rate:.2%}")


if __name__ == "__main__":
    evaluate_random_agent(num_episodes=100, seed=42, render_first_episode=True)

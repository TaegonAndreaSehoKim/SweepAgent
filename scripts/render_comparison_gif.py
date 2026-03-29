from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import matplotlib

# Use a non-interactive backend for file rendering.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Add the project root so local imports work from this script.
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
    HARDER_GRID_MAP,
    HARDER_MAX_STEPS,
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


State = Tuple[int, int, int]


def build_env(map_name: str) -> GridCleanEnv:
    # Build the selected environment preset.
    if map_name == "default":
        grid_map = DEFAULT_GRID_MAP
        max_steps = DEFAULT_MAX_STEPS
    elif map_name == "harder":
        grid_map = HARDER_GRID_MAP
        max_steps = HARDER_MAX_STEPS
    else:
        raise ValueError("map_name must be either 'default' or 'harder'.")

    return GridCleanEnv(
        grid_map=grid_map,
        max_steps=max_steps,
        reward_clean=REWARD_CLEAN,
        reward_move=REWARD_MOVE,
        reward_revisit=REWARD_REVISIT,
        reward_invalid=REWARD_INVALID,
        reward_finish=REWARD_FINISH,
    )


def get_checkpoint_path(map_name: str, seed: int) -> Path:
    # Keep separate checkpoints for each map so policies are not mixed.
    return (
        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / f"q_learning_agent_{map_name}_seed_{seed}.json"
    )


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

            state = next_state
            total_reward += reward
            final_info = info

        agent.decay_epsilon()
        rewards.append(total_reward)

        if episode % print_every == 0:
            recent_rewards = rewards[-print_every:]
            print(
                f"Episode {episode}/{num_episodes} | "
                f"avg_reward={sum(recent_rewards) / len(recent_rewards):.2f} | "
                f"cleaned_ratio={final_info['cleaned_ratio']:.2%} | "
                f"epsilon={agent.epsilon:.4f}"
            )

    return agent


def load_or_train_q_agent(
    map_name: str,
    num_episodes: int = TRAIN_EPISODES,
    seed: int = TRAIN_SEED,
) -> QLearningAgent:
    # Reuse an existing checkpoint for the selected map when available.
    checkpoint_path = get_checkpoint_path(map_name=map_name, seed=seed)

    if checkpoint_path.exists():
        print(
            "Loading trained Q-learning agent from "
            f"{checkpoint_path.relative_to(PROJECT_ROOT)}"
        )
        return QLearningAgent.load(checkpoint_path)

    print(f"No checkpoint found for map='{map_name}'. Training Q-learning agent...")
    train_env = build_env(map_name=map_name)
    agent = train_q_learning_agent(
        env=train_env,
        num_episodes=num_episodes,
        seed=seed,
        print_every=PRINT_EVERY,
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(checkpoint_path)
    print(f"Saved checkpoint to: {checkpoint_path.relative_to(PROJECT_ROOT)}")

    return agent


def get_tile_color(env: GridCleanEnv, row: int, col: int) -> str:
    # Choose the tile color based on wall and dirty/clean status.
    if env.grid[row][col] == "#":
        return "#2f2f2f"

    position = (row, col)

    if position in env.dirty_index_map:
        dirty_idx = env.dirty_index_map[position]
        is_cleaned = ((env.cleaned_mask >> dirty_idx) & 1) == 1
        return "#b7e4c7" if is_cleaned else "#ffd166"

    return "#f8f9fa"


def draw_env_on_axis(
    ax: plt.Axes,
    env: GridCleanEnv,
    panel_title: str,
    step_idx: int,
    total_reward: float,
) -> None:
    # Draw one environment state on the given axis.
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")

    for row in range(env.rows):
        for col in range(env.cols):
            tile_color = get_tile_color(env, row, col)

            ax.add_patch(
                Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=tile_color,
                    edgecolor="#adb5bd",
                    linewidth=1.2,
                )
            )

    robot_row, robot_col = env.robot_pos
    ax.add_patch(
        Circle(
            (robot_col + 0.5, robot_row + 0.5),
            0.28,
            facecolor="#3a86ff",
            edgecolor="#1d3557",
            linewidth=2,
        )
    )

    cleaned_tiles = env._count_cleaned_tiles()
    total_dirty_tiles = env.total_dirty_tiles

    ax.set_title(
        f"{panel_title}\n"
        f"Step: {step_idx} | "
        f"Reward: {total_reward:.0f} | "
        f"Cleaned: {cleaned_tiles}/{total_dirty_tiles}",
        fontsize=11,
        pad=10,
    )


def draw_comparison_frame(
    random_env: GridCleanEnv,
    learned_env: GridCleanEnv,
    random_step_idx: int,
    learned_step_idx: int,
    random_total_reward: float,
    learned_total_reward: float,
    main_title: str,
) -> np.ndarray:
    # Draw one side-by-side comparison frame.
    fig_width = max(10, (random_env.cols + learned_env.cols) * 0.8)
    fig_height = max(4.5, max(random_env.rows, learned_env.rows) * 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    fig.suptitle(main_title, fontsize=14, y=0.98)

    draw_env_on_axis(
        ax=axes[0],
        env=random_env,
        panel_title="Random Agent",
        step_idx=random_step_idx,
        total_reward=random_total_reward,
    )
    draw_env_on_axis(
        ax=axes[1],
        env=learned_env,
        panel_title="Learned Greedy Agent",
        step_idx=learned_step_idx,
        total_reward=learned_total_reward,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.canvas.draw()

    # Convert the matplotlib canvas to an RGB image array.
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def rollout_random_episode(
    env: GridCleanEnv,
    seed: int = 42,
) -> List[Dict[str, object]]:
    # Run one random episode and record all states.
    agent = RandomAgent(action_space_size=len(env.ACTIONS), seed=seed)

    records: List[Dict[str, object]] = []
    state = env.reset()
    done = False
    total_reward = 0.0
    step_idx = 0

    records.append(
        {
            "step_idx": step_idx,
            "total_reward": total_reward,
            "robot_pos": env.robot_pos,
            "cleaned_mask": env.cleaned_mask,
        }
    )

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)

        total_reward += reward
        step_idx += 1
        state = next_state

        records.append(
            {
                "step_idx": step_idx,
                "total_reward": total_reward,
                "robot_pos": env.robot_pos,
                "cleaned_mask": env.cleaned_mask,
            }
        )

    return records


def rollout_learned_episode(
    env: GridCleanEnv,
    agent: QLearningAgent,
) -> List[Dict[str, object]]:
    # Run one greedy learned episode and record all states.
    records: List[Dict[str, object]] = []
    state = env.reset()
    done = False
    total_reward = 0.0
    step_idx = 0

    records.append(
        {
            "step_idx": step_idx,
            "total_reward": total_reward,
            "robot_pos": env.robot_pos,
            "cleaned_mask": env.cleaned_mask,
        }
    )

    while not done:
        action = agent.get_policy_action(state)
        next_state, reward, done, _ = env.step(action)

        total_reward += reward
        step_idx += 1
        state = next_state

        records.append(
            {
                "step_idx": step_idx,
                "total_reward": total_reward,
                "robot_pos": env.robot_pos,
                "cleaned_mask": env.cleaned_mask,
            }
        )

    return records


def apply_record(env: GridCleanEnv, record: Dict[str, object]) -> None:
    # Apply one recorded state to an environment instance.
    env.robot_pos = record["robot_pos"]  # type: ignore[assignment]
    env.cleaned_mask = int(record["cleaned_mask"])
    env.steps_taken = int(record["step_idx"])


def build_comparison_frames(
    map_name: str,
    random_records: List[Dict[str, object]],
    learned_records: List[Dict[str, object]],
) -> List[np.ndarray]:
    # Build synchronized side-by-side frames for both policies.
    frames: List[np.ndarray] = []

    max_len = max(len(random_records), len(learned_records))
    main_title = f"SweepAgent Policy Comparison ({map_name} map)"

    for idx in range(max_len):
        random_record = random_records[min(idx, len(random_records) - 1)]
        learned_record = learned_records[min(idx, len(learned_records) - 1)]

        random_env = build_env(map_name=map_name)
        learned_env = build_env(map_name=map_name)

        apply_record(random_env, random_record)
        apply_record(learned_env, learned_record)

        frame = draw_comparison_frame(
            random_env=random_env,
            learned_env=learned_env,
            random_step_idx=int(random_record["step_idx"]),
            learned_step_idx=int(learned_record["step_idx"]),
            random_total_reward=float(random_record["total_reward"]),
            learned_total_reward=float(learned_record["total_reward"]),
            main_title=main_title,
        )
        frames.append(frame)

    # Hold the final frame a bit longer at the end.
    for _ in range(6):
        frames.append(frames[-1])

    return frames


def save_gif(
    frames: List[np.ndarray],
    output_path: Path,
    frame_duration: float = 0.6,
) -> None:
    # Save the captured frames as a GIF file.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=frame_duration, loop=0)


def render_comparison_gif(
    map_name: str = "harder",
    num_episodes: int = TRAIN_EPISODES,
    train_seed: int = TRAIN_SEED,
    random_seed: int = 42,
    frame_duration: float = 0.6,
) -> Path:
    # Load or train the learned agent, run both policies, and save a comparison GIF.
    learned_rollout_env = build_env(map_name=map_name)
    random_rollout_env = build_env(map_name=map_name)

    learned_agent = load_or_train_q_agent(
        map_name=map_name,
        num_episodes=num_episodes,
        seed=train_seed,
    )

    print("Collecting random episode...")
    random_records = rollout_random_episode(env=random_rollout_env, seed=random_seed)

    print("Collecting learned greedy episode...")
    learned_records = rollout_learned_episode(
        env=learned_rollout_env,
        agent=learned_agent,
    )

    print("Rendering comparison frames...")
    frames = build_comparison_frames(
        map_name=map_name,
        random_records=random_records,
        learned_records=learned_records,
    )

    output_path = PROJECT_ROOT / "outputs" / "gifs" / f"comparison_{map_name}.gif"
    save_gif(frames=frames, output_path=output_path, frame_duration=frame_duration)

    return output_path


if __name__ == "__main__":
    saved_path = render_comparison_gif(
        map_name="harder",
        num_episodes=1000,
        train_seed=42,
        random_seed=42,
        frame_duration=0.6,
    )
    print(f"Saved comparison GIF to: {saved_path}")
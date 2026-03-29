from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

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
from configs.default_config import TRAIN_EPISODES, TRAIN_SEED
from env.grid_clean_env import GridCleanEnv
from utils.experiment_utils import build_env, load_or_train_q_agent


def parse_args() -> argparse.Namespace:
    # Parse command-line arguments for GIF rendering configuration.
    parser = argparse.ArgumentParser(
        description="Render a learned SweepAgent policy as a GIF."
    )
    parser.add_argument(
        "--map-name",
        type=str,
        default="harder",
        help="Map preset name (for example: default, harder, wide_room, corridor).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=TRAIN_EPISODES,
        help="Training episodes used only when a checkpoint does not already exist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=TRAIN_SEED,
        help="Seed used for the Q-learning checkpoint lookup or training.",
    )
    parser.add_argument(
        "--frame-duration",
        type=float,
        default=0.7,
        help="Duration of each GIF frame in seconds.",
    )
    return parser.parse_args()


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


def draw_frame(
    env: GridCleanEnv,
    step_idx: int,
    total_reward: float,
    title: str,
) -> np.ndarray:
    # Draw one frame of the current environment state.
    fig_width = max(5, env.cols * 0.9)
    fig_height = max(4, env.rows * 0.9)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
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
                    linewidth=1.5,
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
        f"{title}\n"
        f"Step: {step_idx} | "
        f"Reward: {total_reward:.0f} | "
        f"Cleaned: {cleaned_tiles}/{total_dirty_tiles}",
        fontsize=12,
        pad=16,
    )

    fig.tight_layout()
    fig.canvas.draw()

    # Convert the matplotlib canvas to an RGB image array.
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def collect_greedy_episode_frames(
    env: GridCleanEnv,
    agent: QLearningAgent,
    title: str,
) -> List[np.ndarray]:
    # Run one greedy episode and capture every step as a frame.
    frames: List[np.ndarray] = []

    state = env.reset()
    done = False
    total_reward = 0.0
    step_idx = 0

    frames.append(
        draw_frame(
            env=env,
            step_idx=step_idx,
            total_reward=total_reward,
            title=title,
        )
    )

    while not done:
        action = agent.get_policy_action(state)
        next_state, reward, done, _ = env.step(action)

        total_reward += reward
        step_idx += 1
        state = next_state

        frames.append(
            draw_frame(
                env=env,
                step_idx=step_idx,
                total_reward=total_reward,
                title=title,
            )
        )

    return frames


def save_gif(
    frames: List[np.ndarray],
    output_path: Path,
    frame_duration: float = 0.7,
) -> None:
    # Save the captured frames as a GIF file.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=frame_duration, loop=0)


def render_policy_gif(
    map_name: str = "harder",
    num_episodes: int = TRAIN_EPISODES,
    seed: int = TRAIN_SEED,
    frame_duration: float = 0.7,
) -> Path:
    # Load or train the agent, then render one greedy episode to a GIF.
    render_env = build_env(map_name=map_name)

    agent = load_or_train_q_agent(
        map_name=map_name,
        num_episodes=num_episodes,
        seed=seed,
    )

    print("Collecting frames from the learned greedy policy...")
    title = f"SweepAgent Learned Policy ({map_name} map)"
    frames = collect_greedy_episode_frames(
        env=render_env,
        agent=agent,
        title=title,
    )

    output_path = PROJECT_ROOT / "outputs" / "gifs" / f"learned_policy_{map_name}.gif"
    save_gif(frames=frames, output_path=output_path, frame_duration=frame_duration)

    return output_path


if __name__ == "__main__":
    args = parse_args()
    saved_path = render_policy_gif(
        map_name=args.map_name,
        num_episodes=args.episodes,
        seed=args.seed,
        frame_duration=args.frame_duration,
    )
    print(f"Saved GIF to: {saved_path}")

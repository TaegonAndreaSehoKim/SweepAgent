from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.q_learning_agent import QLearningAgent
from agents.random_agent import RandomAgent
from configs.default_config import TRAIN_EPISODES, TRAIN_SEED
from env.grid_clean_env import GridCleanEnv
from utils.experiment_utils import build_env, load_or_train_q_agent


State = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a side-by-side GIF comparing random and learned SweepAgent policies."
    )
    parser.add_argument("--map-name", type=str, default="harder")
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--train-seed", type=int, default=TRAIN_SEED)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--frame-duration", type=float, default=0.6)
    return parser.parse_args()


def get_tile_color(env: GridCleanEnv, row: int, col: int) -> str:
    if env.grid[row][col] == "#":
        return "#2f2f2f"

    if env._is_charger(row, col):
        return "#cdb4db"

    position = (row, col)
    if position in env.dirty_index_map:
        dirty_idx = env.dirty_index_map[position]
        is_cleaned = ((env.cleaned_mask >> dirty_idx) & 1) == 1
        return "#b7e4c7" if is_cleaned else "#ffd166"

    return "#f8f9fa"


def format_battery_text(env: GridCleanEnv) -> str:
    if env.battery_capacity is None:
        return "Battery: off"
    return f"Battery: {env.battery_remaining}/{env.battery_capacity}"


def draw_env_on_axis(
    ax: plt.Axes,
    env: GridCleanEnv,
    panel_title: str,
    step_idx: int,
    total_reward: float,
) -> None:
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

            if env._is_charger(row, col):
                ax.text(
                    col + 0.5,
                    row + 0.52,
                    "C",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
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
    battery_text = format_battery_text(env)

    ax.set_title(
        f"{panel_title}\n"
        f"Step: {step_idx} | Reward: {total_reward:.0f}\n"
        f"Cleaned: {cleaned_tiles}/{total_dirty_tiles} | {battery_text}",
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

    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return frame


def rollout_random_episode(
    env: GridCleanEnv,
    seed: int = 42,
) -> List[Dict[str, object]]:
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
            "battery_remaining": env.battery_remaining,
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
                "battery_remaining": env.battery_remaining,
            }
        )

    return records


def rollout_learned_episode(
    env: GridCleanEnv,
    agent: QLearningAgent,
) -> List[Dict[str, object]]:
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
            "battery_remaining": env.battery_remaining,
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
                "battery_remaining": env.battery_remaining,
            }
        )

    return records


def apply_record(env: GridCleanEnv, record: Dict[str, object]) -> None:
    env.robot_pos = record["robot_pos"]  # type: ignore[assignment]
    env.cleaned_mask = int(record["cleaned_mask"])
    env.steps_taken = int(record["step_idx"])
    env.battery_remaining = record["battery_remaining"]  # type: ignore[assignment]


def build_comparison_frames(
    map_name: str,
    random_records: List[Dict[str, object]],
    learned_records: List[Dict[str, object]],
) -> List[np.ndarray]:
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

    for _ in range(6):
        frames.append(frames[-1])

    return frames


def save_gif(
    frames: List[np.ndarray],
    output_path: Path,
    frame_duration: float = 0.6,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, duration=frame_duration, loop=0)


def render_comparison_gif(
    map_name: str = "harder",
    num_episodes: int = TRAIN_EPISODES,
    train_seed: int = TRAIN_SEED,
    random_seed: int = 42,
    frame_duration: float = 0.6,
) -> Path:
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
    args = parse_args()
    saved_path = render_comparison_gif(
        map_name=args.map_name,
        num_episodes=args.episodes,
        train_seed=args.train_seed,
        random_seed=args.random_seed,
        frame_duration=args.frame_duration,
    )
    print(f"Saved comparison GIF to: {saved_path}")
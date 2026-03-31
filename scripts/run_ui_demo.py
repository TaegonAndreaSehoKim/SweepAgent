from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.experiment_utils import build_env, load_or_train_q_agent
from utils.ui_utils import (
    BACKGROUND_COLOR,
    MAX_STEP_DELAY,
    MIN_STEP_DELAY,
    STEP_DELAY_DELTA,
    TOP_PANEL_HEIGHT,
    WINDOW_PADDING_X,
    WINDOW_PADDING_Y,
    compute_bottom_info_height,
    draw_bottom_info,
    draw_single_panel,
    draw_top_controls,
    get_hover_info,
    load_fonts,
    reset_panel_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple pygame UI demo for a learned SweepAgent policy."
    )
    parser.add_argument("--map-name", type=str, default="charge_required_v2")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cell-size", type=int, default=72)
    parser.add_argument("--step-delay", type=float, default=0.5)
    return parser.parse_args()


def run_greedy_demo(
    map_name: str,
    episodes: int,
    seed: int,
    cell_size: int,
    initial_step_delay: float,
) -> None:
    pygame.init()
    pygame.display.set_caption("SweepAgent UI Demo")

    fonts = load_fonts()

    env = build_env(map_name=map_name)
    env.map_name = map_name

    agent = load_or_train_q_agent(
        map_name=map_name,
        num_episodes=episodes,
        seed=seed,
    )

    state, done, total_reward, step_idx, path_history, visit_counts = reset_panel_state(env)

    preview_left_lines = [
        (f"Map: {map_name}", (0, 0, 0)),
        ("Step: 9999", (0, 0, 0)),
        ("Reward: 999", (0, 0, 0)),
    ]
    preview_right_lines = [
        (f"Cleaned: {env.total_dirty_tiles}/{env.total_dirty_tiles}", (0, 0, 0)),
        (
            f"Battery: {env.battery_capacity}/{env.battery_capacity}"
            if env.battery_capacity is not None
            else "Battery: off",
            (0, 0, 0),
        ),
        ("Status: paused", (0, 0, 0)),
        ("Delay: 0.5s", (0, 0, 0)),
    ]

    from utils.ui_utils import compute_panel_layout

    preview_layout = compute_panel_layout(
        env=env,
        cell_size=cell_size,
        fonts=fonts,
        left_lines=preview_left_lines,
        right_lines=preview_right_lines,
        title_text="SweepAgent UI Demo",
        subtitle_text=None,
    )

    bottom_info_height = compute_bottom_info_height(fonts)
    width = WINDOW_PADDING_X + preview_layout.panel_width + WINDOW_PADDING_X
    height = TOP_PANEL_HEIGHT + preview_layout.total_height + bottom_info_height + WINDOW_PADDING_Y

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    step_delay = initial_step_delay
    is_paused = False
    last_step_time = time.time()

    panel_left = WINDOW_PADDING_X
    panel_top = TOP_PANEL_HEIGHT

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return

                if event.key == pygame.K_SPACE:
                    is_paused = not is_paused
                    last_step_time = time.time()

                if event.key == pygame.K_r:
                    state, done, total_reward, step_idx, path_history, visit_counts = reset_panel_state(env)
                    is_paused = False
                    last_step_time = time.time()

                if event.key == pygame.K_LEFTBRACKET:
                    step_delay = min(MAX_STEP_DELAY, step_delay + STEP_DELAY_DELTA)

                if event.key == pygame.K_RIGHTBRACKET:
                    step_delay = max(MIN_STEP_DELAY, step_delay - STEP_DELAY_DELTA)

        now = time.time()

        if not done and not is_paused and now - last_step_time >= step_delay:
            action = agent.get_policy_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            step_idx += 1
            last_step_time = now

            path_history.append(env.robot_pos)
            visit_counts[env.robot_pos] = visit_counts.get(env.robot_pos, 0) + 1

        screen.fill(BACKGROUND_COLOR)

        draw_top_controls(
            screen=screen,
            width=width,
            step_delay=step_delay,
            is_paused=is_paused,
            fonts=fonts,
            title_text="SweepAgent UI Demo",
        )

        grid_rect = draw_single_panel(
            screen=screen,
            env=env,
            total_reward=total_reward,
            step_idx=step_idx,
            done=done,
            is_paused=is_paused,
            step_delay=step_delay,
            visit_counts=visit_counts,
            path_history=path_history,
            panel_left=panel_left,
            panel_top=panel_top,
            cell_size=cell_size,
            fonts=fonts,
        )

        mouse_pos = pygame.mouse.get_pos()
        hover_lines = get_hover_info(
            env=env,
            visit_counts=visit_counts,
            grid_rect=grid_rect,
            mouse_pos=mouse_pos,
            cell_size=cell_size,
        )

        hover_title = "Hover: Learned Agent" if hover_lines is not None else None

        draw_bottom_info(
            screen=screen,
            width=width,
            height=height,
            fonts=fonts,
            hover_title=hover_title,
            hover_lines=hover_lines,
            bottom_info_height=bottom_info_height,
        )

        pygame.display.flip()
        clock.tick(60)


def main() -> None:
    args = parse_args()
    run_greedy_demo(
        map_name=args.map_name,
        episodes=args.episodes,
        seed=args.seed,
        cell_size=args.cell_size,
        initial_step_delay=args.step_delay,
    )


if __name__ == "__main__":
    main()
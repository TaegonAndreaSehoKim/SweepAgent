from __future__ import annotations

import argparse
import random
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
    PANEL_GAP,
    STEP_DELAY_DELTA,
    TOP_PANEL_HEIGHT,
    WINDOW_PADDING_X,
    WINDOW_PADDING_Y,
    compute_bottom_info_height,
    compute_panel_layout,
    draw_bottom_info,
    draw_compare_panel,
    draw_top_controls,
    get_hover_info,
    load_fonts,
    reset_panel_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a side-by-side pygame UI comparison for random vs learned SweepAgent policies."
    )
    parser.add_argument("--map-name", type=str, default="charge_required_v2")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cell-size", type=int, default=56)
    parser.add_argument("--step-delay", type=float, default=0.5)
    return parser.parse_args()


def run_ui_compare(
    map_name: str,
    episodes: int,
    seed: int,
    random_seed: int,
    cell_size: int,
    initial_step_delay: float,
) -> None:
    pygame.init()
    pygame.display.set_caption("SweepAgent UI Comparison")

    fonts = load_fonts()

    random_env = build_env(map_name=map_name)
    learned_env = build_env(map_name=map_name)
    random_env.map_name = map_name
    learned_env.map_name = map_name

    learned_agent = load_or_train_q_agent(
        map_name=map_name,
        num_episodes=episodes,
        seed=seed,
    )

    rng = random.Random(random_seed)

    random_state, random_done, random_reward, random_step, random_path, random_visits = reset_panel_state(random_env)
    learned_state, learned_done, learned_reward, learned_step, learned_path, learned_visits = reset_panel_state(learned_env)

    preview_left_lines = [
        ("Step: 9999", (0, 0, 0)),
        ("Reward: 999", (0, 0, 0)),
    ]
    preview_right_lines = [
        (f"Cleaned: {random_env.total_dirty_tiles}/{random_env.total_dirty_tiles}", (0, 0, 0)),
        (
            f"Battery: {random_env.battery_capacity}/{random_env.battery_capacity}"
            if random_env.battery_capacity is not None
            else "Battery: off",
            (0, 0, 0),
        ),
        ("Status: success", (0, 0, 0)),
    ]

    random_layout = compute_panel_layout(
        env=random_env,
        cell_size=cell_size,
        fonts=fonts,
        left_lines=preview_left_lines,
        right_lines=preview_right_lines,
        title_text="Random Agent",
        subtitle_text=f"Map: {map_name}",
    )
    learned_layout = compute_panel_layout(
        env=learned_env,
        cell_size=cell_size,
        fonts=fonts,
        left_lines=preview_left_lines,
        right_lines=preview_right_lines,
        title_text="Learned Greedy Agent",
        subtitle_text=f"Map: {map_name}",
    )

    panel_width = max(random_layout.panel_width, learned_layout.panel_width)
    panel_height = max(random_layout.total_height, learned_layout.total_height)

    bottom_info_height = compute_bottom_info_height(fonts)

    width = (
        WINDOW_PADDING_X
        + panel_width
        + PANEL_GAP
        + panel_width
        + WINDOW_PADDING_X
    )
    height = TOP_PANEL_HEIGHT + panel_height + bottom_info_height + WINDOW_PADDING_Y

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    step_delay = initial_step_delay
    is_paused = False
    last_step_time = time.time()

    left_panel_left = WINDOW_PADDING_X
    right_panel_left = WINDOW_PADDING_X + panel_width + PANEL_GAP
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
                    rng = random.Random(random_seed)
                    random_state, random_done, random_reward, random_step, random_path, random_visits = reset_panel_state(random_env)
                    learned_state, learned_done, learned_reward, learned_step, learned_path, learned_visits = reset_panel_state(learned_env)
                    is_paused = False
                    last_step_time = time.time()

                if event.key == pygame.K_LEFTBRACKET:
                    step_delay = min(MAX_STEP_DELAY, step_delay + STEP_DELAY_DELTA)

                if event.key == pygame.K_RIGHTBRACKET:
                    step_delay = max(MIN_STEP_DELAY, step_delay - STEP_DELAY_DELTA)

        now = time.time()

        if not is_paused and now - last_step_time >= step_delay:
            if not random_done:
                random_action = rng.randrange(len(random_env.ACTIONS))
                random_state, reward, random_done, _ = random_env.step(random_action)
                random_reward += reward
                random_step += 1
                random_path.append(random_env.robot_pos)
                random_visits[random_env.robot_pos] = random_visits.get(random_env.robot_pos, 0) + 1

            if not learned_done:
                learned_action = learned_agent.get_policy_action(learned_state)
                learned_state, reward, learned_done, _ = learned_env.step(learned_action)
                learned_reward += reward
                learned_step += 1
                learned_path.append(learned_env.robot_pos)
                learned_visits[learned_env.robot_pos] = learned_visits.get(learned_env.robot_pos, 0) + 1

            last_step_time = now

        screen.fill(BACKGROUND_COLOR)

        draw_top_controls(
            screen=screen,
            width=width,
            step_delay=step_delay,
            is_paused=is_paused,
            fonts=fonts,
            title_text="SweepAgent UI Comparison",
        )

        random_grid_rect, _ = draw_compare_panel(
            screen=screen,
            panel_title="Random Agent",
            env=random_env,
            total_reward=random_reward,
            step_idx=random_step,
            done=random_done,
            visit_counts=random_visits,
            path_history=random_path,
            panel_left=left_panel_left,
            panel_top=panel_top,
            cell_size=cell_size,
            fonts=fonts,
        )

        learned_grid_rect, _ = draw_compare_panel(
            screen=screen,
            panel_title="Learned Greedy Agent",
            env=learned_env,
            total_reward=learned_reward,
            step_idx=learned_step,
            done=learned_done,
            visit_counts=learned_visits,
            path_history=learned_path,
            panel_left=right_panel_left,
            panel_top=panel_top,
            cell_size=cell_size,
            fonts=fonts,
        )

        mouse_pos = pygame.mouse.get_pos()

        hover_title = None
        hover_lines = None

        random_hover = get_hover_info(
            env=random_env,
            visit_counts=random_visits,
            grid_rect=random_grid_rect,
            mouse_pos=mouse_pos,
            cell_size=cell_size,
        )
        learned_hover = get_hover_info(
            env=learned_env,
            visit_counts=learned_visits,
            grid_rect=learned_grid_rect,
            mouse_pos=mouse_pos,
            cell_size=cell_size,
        )

        if random_hover is not None:
            hover_title = "Hover: Random Agent"
            hover_lines = random_hover
        elif learned_hover is not None:
            hover_title = "Hover: Learned Agent"
            hover_lines = learned_hover

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
    run_ui_compare(
        map_name=args.map_name,
        episodes=args.episodes,
        seed=args.seed,
        random_seed=args.random_seed,
        cell_size=args.cell_size,
        initial_step_delay=args.step_delay,
    )


if __name__ == "__main__":
    main()
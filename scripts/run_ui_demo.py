from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple

import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.q_learning_agent import QLearningAgent
from env.grid_clean_env import GridCleanEnv
from utils.experiment_utils import build_env, load_or_train_q_agent


BACKGROUND_COLOR = (245, 247, 250)
WALL_COLOR = (47, 47, 47)
FLOOR_COLOR = (248, 249, 250)
DIRTY_COLOR = (255, 209, 102)
CLEANED_COLOR = (183, 228, 199)
CHARGER_COLOR = (205, 180, 219)
GRID_LINE_COLOR = (173, 181, 189)

ROBOT_COLOR = (58, 134, 255)
ROBOT_BORDER_COLOR = (29, 53, 87)

PATH_COLOR = (64, 145, 255)
PATH_RECENT_COLOR = (29, 78, 216)

HEATMAP_COLOR = (255, 99, 71)

TEXT_COLOR = (33, 37, 41)
SUCCESS_COLOR = (25, 135, 84)
FAIL_COLOR = (220, 53, 69)

WINDOW_PADDING_X = 24
WINDOW_PADDING_Y = 24
STATUS_PANEL_HEIGHT = 190

MIN_STEP_DELAY = 0.1
MAX_STEP_DELAY = 1.5
STEP_DELAY_DELTA = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple pygame UI demo for a learned SweepAgent policy."
    )
    parser.add_argument(
        "--map-name",
        type=str,
        default="charge_required_v2",
        help="Map preset name to visualize.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5000,
        help="Training episodes used only if a checkpoint does not already exist.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used for checkpoint lookup or training.",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=72,
        help="Pixel size of each grid cell.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.5,
        help="Delay in seconds between greedy policy actions.",
    )
    return parser.parse_args()


def load_fonts() -> tuple[pygame.font.Font, pygame.font.Font, pygame.font.Font]:
    candidate_names = [
        "Segoe UI",
        "Malgun Gothic",
        "맑은 고딕",
        "Arial",
        "Calibri",
    ]

    def build_font(size: int, bold: bool = False) -> pygame.font.Font:
        for name in candidate_names:
            matched = pygame.font.match_font(name, bold=bold)
            if matched:
                return pygame.font.Font(matched, size)
        return pygame.font.Font(None, size)

    title_font = build_font(34, bold=True)
    body_font = build_font(22, bold=False)
    small_font = build_font(18, bold=False)
    return title_font, body_font, small_font


def get_tile_type(env: GridCleanEnv, row: int, col: int) -> str:
    if env.grid[row][col] == "#":
        return "wall"
    if env._is_charger(row, col):
        return "charger"
    if (row, col) in env.dirty_index_map:
        dirty_idx = env.dirty_index_map[(row, col)]
        is_cleaned = ((env.cleaned_mask >> dirty_idx) & 1) == 1
        return "cleaned" if is_cleaned else "dirty"
    return "floor"


def get_tile_color(tile_type: str) -> Tuple[int, int, int]:
    if tile_type == "wall":
        return WALL_COLOR
    if tile_type == "dirty":
        return DIRTY_COLOR
    if tile_type == "cleaned":
        return CLEANED_COLOR
    if tile_type == "charger":
        return CHARGER_COLOR
    return FLOOR_COLOR


def cell_center(
    row: int,
    col: int,
    cell_size: int,
    top_margin: int,
    left_margin: int,
) -> Tuple[int, int]:
    return (
        left_margin + col * cell_size + cell_size // 2,
        top_margin + row * cell_size + cell_size // 2,
    )


def blend_colors(
    base_color: Tuple[int, int, int],
    overlay_color: Tuple[int, int, int],
    alpha: float,
) -> Tuple[int, int, int]:
    alpha = max(0.0, min(1.0, alpha))
    return (
        int(base_color[0] * (1 - alpha) + overlay_color[0] * alpha),
        int(base_color[1] * (1 - alpha) + overlay_color[1] * alpha),
        int(base_color[2] * (1 - alpha) + overlay_color[2] * alpha),
    )


def draw_grid(
    screen: pygame.Surface,
    env: GridCleanEnv,
    cell_size: int,
    top_margin: int,
    left_margin: int,
    body_font: pygame.font.Font,
    visit_counts: dict[Tuple[int, int], int],
) -> None:
    max_visits = max(visit_counts.values()) if visit_counts else 0

    for row in range(env.rows):
        for col in range(env.cols):
            tile_type = get_tile_type(env, row, col)
            base_color = get_tile_color(tile_type)

            if tile_type != "wall":
                visits = visit_counts.get((row, col), 0)
                if visits > 0 and max_visits > 0:
                    alpha = 0.15 + 0.45 * (visits / max_visits)
                    base_color = blend_colors(base_color, HEATMAP_COLOR, alpha)

            rect = pygame.Rect(
                left_margin + col * cell_size,
                top_margin + row * cell_size,
                cell_size,
                cell_size,
            )

            pygame.draw.rect(screen, base_color, rect)
            pygame.draw.rect(screen, GRID_LINE_COLOR, rect, width=2)

            if tile_type == "charger":
                label = body_font.render("C", True, TEXT_COLOR)
                label_rect = label.get_rect(center=rect.center)
                screen.blit(label, label_rect)


def draw_path_overlay(
    screen: pygame.Surface,
    path_history: list[Tuple[int, int]],
    cell_size: int,
    top_margin: int,
    left_margin: int,
) -> None:
    if len(path_history) < 2:
        return

    recent_start_idx = max(1, len(path_history) - 5)

    for idx in range(1, len(path_history)):
        prev_row, prev_col = path_history[idx - 1]
        curr_row, curr_col = path_history[idx]

        start_pos = cell_center(
            row=prev_row,
            col=prev_col,
            cell_size=cell_size,
            top_margin=top_margin,
            left_margin=left_margin,
        )
        end_pos = cell_center(
            row=curr_row,
            col=curr_col,
            cell_size=cell_size,
            top_margin=top_margin,
            left_margin=left_margin,
        )

        if idx >= recent_start_idx:
            color = PATH_RECENT_COLOR
            width = 6
        else:
            color = PATH_COLOR
            width = 4

        pygame.draw.line(screen, color, start_pos, end_pos, width)

    for idx, (row, col) in enumerate(path_history[:-1]):
        pos = cell_center(
            row=row,
            col=col,
            cell_size=cell_size,
            top_margin=top_margin,
            left_margin=left_margin,
        )

        if idx >= recent_start_idx - 1:
            radius = max(3, cell_size // 12)
            color = PATH_RECENT_COLOR
        else:
            radius = max(2, cell_size // 16)
            color = PATH_COLOR

        pygame.draw.circle(screen, color, pos, radius)


def draw_robot(
    screen: pygame.Surface,
    env: GridCleanEnv,
    cell_size: int,
    top_margin: int,
    left_margin: int,
) -> None:
    robot_row, robot_col = env.robot_pos
    robot_center = cell_center(
        row=robot_row,
        col=robot_col,
        cell_size=cell_size,
        top_margin=top_margin,
        left_margin=left_margin,
    )
    robot_radius = max(10, cell_size // 4)

    pygame.draw.circle(screen, ROBOT_COLOR, robot_center, robot_radius)
    pygame.draw.circle(screen, ROBOT_BORDER_COLOR, robot_center, robot_radius, width=3)


def build_left_status_lines(
    env: GridCleanEnv,
    total_reward: float,
    step_idx: int,
) -> list[tuple[str, Tuple[int, int, int]]]:
    return [
        (f"Map: {getattr(env, 'map_name', 'unknown')}", TEXT_COLOR),
        (f"Step: {step_idx}", TEXT_COLOR),
        (f"Reward: {total_reward:.0f}", TEXT_COLOR),
    ]


def build_right_status_lines(
    env: GridCleanEnv,
    done: bool,
    is_paused: bool,
    step_delay: float,
) -> list[tuple[str, Tuple[int, int, int]]]:
    cleaned_tiles = env._count_cleaned_tiles()
    total_dirty_tiles = env.total_dirty_tiles

    lines: list[tuple[str, Tuple[int, int, int]]] = [
        (f"Cleaned: {cleaned_tiles}/{total_dirty_tiles}", TEXT_COLOR),
    ]

    if env.battery_capacity is None:
        lines.append(("Battery: off", TEXT_COLOR))
    else:
        lines.append((f"Battery: {env.battery_remaining}/{env.battery_capacity}", TEXT_COLOR))

    if done:
        if cleaned_tiles == total_dirty_tiles:
            lines.append(("Status: success", SUCCESS_COLOR))
        else:
            lines.append(("Status: stopped", FAIL_COLOR))
    else:
        lines.append(("Status: paused" if is_paused else "Status: running", TEXT_COLOR))

    lines.append((f"Delay: {step_delay:.1f}s", TEXT_COLOR))
    return lines


def draw_status_panel(
    screen: pygame.Surface,
    env: GridCleanEnv,
    total_reward: float,
    step_idx: int,
    done: bool,
    is_paused: bool,
    step_delay: float,
    width: int,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    title = title_font.render("SweepAgent UI Demo", True, TEXT_COLOR)
    screen.blit(title, (WINDOW_PADDING_X, 18))

    help_text = small_font.render(
        "ESC quit | SPACE pause | R restart | [ ] speed",
        True,
        TEXT_COLOR,
    )
    help_rect = help_text.get_rect(topright=(width - WINDOW_PADDING_X, 26))
    screen.blit(help_text, help_rect)

    left_lines = build_left_status_lines(
        env=env,
        total_reward=total_reward,
        step_idx=step_idx,
    )
    right_lines = build_right_status_lines(
        env=env,
        done=done,
        is_paused=is_paused,
        step_delay=step_delay,
    )

    left_x = WINDOW_PADDING_X
    right_x = width - WINDOW_PADDING_X
    start_y = 72
    line_gap = 26

    y = start_y
    for text_value, color in left_lines:
        text_surface = body_font.render(text_value, True, color)
        screen.blit(text_surface, (left_x, y))
        y += line_gap

    y = start_y
    for text_value, color in right_lines:
        text_surface = body_font.render(text_value, True, color)
        text_rect = text_surface.get_rect(topright=(right_x, y))
        screen.blit(text_surface, text_rect)
        y += line_gap


def reset_demo_state(env: GridCleanEnv) -> tuple[Tuple[int, int, int, int], bool, float, int, list[Tuple[int, int]], dict[Tuple[int, int], int]]:
    state = env.reset()
    done = False
    total_reward = 0.0
    step_idx = 0
    path_history = [env.robot_pos]
    visit_counts = {env.robot_pos: 1}
    return state, done, total_reward, step_idx, path_history, visit_counts


def run_greedy_demo(
    env: GridCleanEnv,
    agent: QLearningAgent,
    cell_size: int,
    initial_step_delay: float,
) -> None:
    pygame.init()
    pygame.display.set_caption("SweepAgent UI Demo")

    title_font, body_font, small_font = load_fonts()

    left_margin = WINDOW_PADDING_X
    right_margin = WINDOW_PADDING_X
    bottom_margin = WINDOW_PADDING_Y
    top_margin = STATUS_PANEL_HEIGHT

    width = left_margin + env.cols * cell_size + right_margin
    height = top_margin + env.rows * cell_size + bottom_margin

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    state, done, total_reward, step_idx, path_history, visit_counts = reset_demo_state(env)
    step_delay = initial_step_delay
    is_paused = False
    last_step_time = time.time()

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
                    state, done, total_reward, step_idx, path_history, visit_counts = reset_demo_state(env)
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

        draw_status_panel(
            screen=screen,
            env=env,
            total_reward=total_reward,
            step_idx=step_idx,
            done=done,
            is_paused=is_paused,
            step_delay=step_delay,
            width=width,
            title_font=title_font,
            body_font=body_font,
            small_font=small_font,
        )

        draw_grid(
            screen=screen,
            env=env,
            cell_size=cell_size,
            top_margin=top_margin,
            left_margin=left_margin,
            body_font=body_font,
            visit_counts=visit_counts,
        )

        draw_path_overlay(
            screen=screen,
            path_history=path_history,
            cell_size=cell_size,
            top_margin=top_margin,
            left_margin=left_margin,
        )

        draw_robot(
            screen=screen,
            env=env,
            cell_size=cell_size,
            top_margin=top_margin,
            left_margin=left_margin,
        )

        pygame.display.flip()
        clock.tick(60)


def main() -> None:
    args = parse_args()

    env = build_env(map_name=args.map_name)
    env.map_name = args.map_name

    agent = load_or_train_q_agent(
        map_name=args.map_name,
        num_episodes=args.episodes,
        seed=args.seed,
    )

    run_greedy_demo(
        env=env,
        agent=agent,
        cell_size=args.cell_size,
        initial_step_delay=args.step_delay,
    )


if __name__ == "__main__":
    main()
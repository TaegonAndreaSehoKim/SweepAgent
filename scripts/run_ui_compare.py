from __future__ import annotations

import argparse
import random
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
MUTED_TEXT_COLOR = (90, 98, 108)
SUCCESS_COLOR = (25, 135, 84)
FAIL_COLOR = (220, 53, 69)
PANEL_BORDER_COLOR = (206, 212, 218)
INFO_BG_COLOR = (255, 255, 255)

WINDOW_PADDING_X = 24
WINDOW_PADDING_Y = 24
TOP_PANEL_HEIGHT = 80
PANEL_HEADER_HEIGHT = 120
PANEL_GAP = 24
BOTTOM_INFO_HEIGHT = 150

MIN_STEP_DELAY = 0.1
MAX_STEP_DELAY = 1.5
STEP_DELAY_DELTA = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a side-by-side pygame UI comparison for random vs learned SweepAgent policies."
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
        "--random-seed",
        type=int,
        default=42,
        help="Seed used by the random policy rollout.",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=56,
        help="Pixel size of each grid cell.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.5,
        help="Delay in seconds between actions.",
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

    title_font = build_font(30, bold=True)
    body_font = build_font(20, bold=False)
    small_font = build_font(16, bold=False)
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
    grid_top: int,
    grid_left: int,
) -> Tuple[int, int]:
    return (
        grid_left + col * cell_size + cell_size // 2,
        grid_top + row * cell_size + cell_size // 2,
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
    grid_top: int,
    grid_left: int,
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
                grid_left + col * cell_size,
                grid_top + row * cell_size,
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
    grid_top: int,
    grid_left: int,
) -> None:
    if len(path_history) < 2:
        return

    recent_start_idx = max(1, len(path_history) - 5)

    for idx in range(1, len(path_history)):
        prev_row, prev_col = path_history[idx - 1]
        curr_row, curr_col = path_history[idx]

        start_pos = cell_center(prev_row, prev_col, cell_size, grid_top, grid_left)
        end_pos = cell_center(curr_row, curr_col, cell_size, grid_top, grid_left)

        if idx >= recent_start_idx:
            color = PATH_RECENT_COLOR
            width = 5
        else:
            color = PATH_COLOR
            width = 3

        pygame.draw.line(screen, color, start_pos, end_pos, width)

    for idx, (row, col) in enumerate(path_history[:-1]):
        pos = cell_center(row, col, cell_size, grid_top, grid_left)

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
    grid_top: int,
    grid_left: int,
) -> None:
    robot_row, robot_col = env.robot_pos
    robot_center = cell_center(robot_row, robot_col, cell_size, grid_top, grid_left)
    robot_radius = max(9, cell_size // 4)

    pygame.draw.circle(screen, ROBOT_COLOR, robot_center, robot_radius)
    pygame.draw.circle(screen, ROBOT_BORDER_COLOR, robot_center, robot_radius, width=3)


def build_left_status_lines(
    total_reward: float,
    step_idx: int,
) -> list[tuple[str, Tuple[int, int, int]]]:
    return [
        (f"Step: {step_idx}", TEXT_COLOR),
        (f"Reward: {total_reward:.0f}", TEXT_COLOR),
    ]


def build_right_status_lines(
    env: GridCleanEnv,
    done: bool,
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
        lines.append(("Status: running", TEXT_COLOR))

    return lines


def draw_panel(
    screen: pygame.Surface,
    panel_title: str,
    env: GridCleanEnv,
    total_reward: float,
    step_idx: int,
    done: bool,
    visit_counts: dict[Tuple[int, int], int],
    path_history: list[Tuple[int, int]],
    panel_left: int,
    panel_top: int,
    panel_width: int,
    cell_size: int,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> pygame.Rect:
    title_surface = title_font.render(panel_title, True, TEXT_COLOR)
    screen.blit(title_surface, (panel_left, panel_top))

    subtitle_surface = small_font.render(
        f"Map: {getattr(env, 'map_name', 'unknown')}",
        True,
        MUTED_TEXT_COLOR,
    )
    screen.blit(subtitle_surface, (panel_left, panel_top + 36))

    left_lines = build_left_status_lines(
        total_reward=total_reward,
        step_idx=step_idx,
    )
    right_lines = build_right_status_lines(
        env=env,
        done=done,
    )

    left_x = panel_left
    right_x = panel_left + panel_width
    start_y = panel_top + 62
    line_gap = 18

    y = start_y
    for text_value, color in left_lines:
        text_surface = small_font.render(text_value, True, color)
        screen.blit(text_surface, (left_x, y))
        y += line_gap

    y = start_y
    for text_value, color in right_lines:
        text_surface = small_font.render(text_value, True, color)
        text_rect = text_surface.get_rect(topright=(right_x, y))
        screen.blit(text_surface, text_rect)
        y += line_gap

    grid_top = panel_top + PANEL_HEADER_HEIGHT
    grid_left = panel_left

    draw_grid(
        screen=screen,
        env=env,
        cell_size=cell_size,
        grid_top=grid_top,
        grid_left=grid_left,
        body_font=body_font,
        visit_counts=visit_counts,
    )

    draw_path_overlay(
        screen=screen,
        path_history=path_history,
        cell_size=cell_size,
        grid_top=grid_top,
        grid_left=grid_left,
    )

    draw_robot(
        screen=screen,
        env=env,
        cell_size=cell_size,
        grid_top=grid_top,
        grid_left=grid_left,
    )

    panel_rect = pygame.Rect(
        panel_left - 8,
        panel_top - 8,
        panel_width + 16,
        PANEL_HEADER_HEIGHT + env.rows * cell_size + 16,
    )
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=8)

    return pygame.Rect(
        grid_left,
        grid_top,
        env.cols * cell_size,
        env.rows * cell_size,
    )


def draw_top_controls(
    screen: pygame.Surface,
    width: int,
    step_delay: float,
    is_paused: bool,
    title_font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    title = title_font.render("SweepAgent UI Comparison", True, TEXT_COLOR)
    title_rect = title.get_rect(center=(width // 2, 26))
    screen.blit(title, title_rect)

    controls = small_font.render(
        "ESC quit | SPACE pause | R restart | [ ] speed",
        True,
        TEXT_COLOR,
    )
    controls_rect = controls.get_rect(center=(width // 2, 56))
    screen.blit(controls, controls_rect)

    state_text = f"State: {'paused' if is_paused else 'running'} | Delay: {step_delay:.1f}s"
    state_surface = small_font.render(state_text, True, TEXT_COLOR)
    state_rect = state_surface.get_rect(topright=(width - WINDOW_PADDING_X, 20))
    screen.blit(state_surface, state_rect)


def draw_legend(
    screen: pygame.Surface,
    left: int,
    top: int,
    small_font: pygame.font.Font,
) -> None:
    items = [
        ("Wall", WALL_COLOR),
        ("Dirty", DIRTY_COLOR),
        ("Cleaned", CLEANED_COLOR),
        ("Charger", CHARGER_COLOR),
        ("Heatmap", HEATMAP_COLOR),
        ("Path", PATH_COLOR),
    ]

    x = left
    for label, color in items:
        swatch_rect = pygame.Rect(x, top + 2, 16, 16)
        pygame.draw.rect(screen, color, swatch_rect)
        pygame.draw.rect(screen, GRID_LINE_COLOR, swatch_rect, width=1)

        text_surface = small_font.render(label, True, TEXT_COLOR)
        screen.blit(text_surface, (x + 24, top))
        x += 110


def get_hover_info(
    env: GridCleanEnv,
    visit_counts: dict[Tuple[int, int], int],
    grid_rect: pygame.Rect,
    mouse_pos: Tuple[int, int],
    cell_size: int,
) -> list[str] | None:
    if not grid_rect.collidepoint(mouse_pos):
        return None

    local_x = mouse_pos[0] - grid_rect.left
    local_y = mouse_pos[1] - grid_rect.top

    col = local_x // cell_size
    row = local_y // cell_size

    if row < 0 or row >= env.rows or col < 0 or col >= env.cols:
        return None

    tile_type = get_tile_type(env, row, col)
    visits = visit_counts.get((row, col), 0)

    details = [
        f"Tile: ({row}, {col})",
        f"Type: {tile_type}",
        f"Visits: {visits}",
    ]

    if (row, col) in env.dirty_index_map:
        dirty_idx = env.dirty_index_map[(row, col)]
        is_cleaned = ((env.cleaned_mask >> dirty_idx) & 1) == 1
        details.append(f"Dirty state: {'cleaned' if is_cleaned else 'dirty'}")

    if env._is_charger(row, col):
        details.append("Charger: yes")

    return details


def draw_bottom_info(
    screen: pygame.Surface,
    width: int,
    height: int,
    small_font: pygame.font.Font,
    hover_title: str | None,
    hover_lines: list[str] | None,
) -> None:
    panel_rect = pygame.Rect(
        WINDOW_PADDING_X,
        height - BOTTOM_INFO_HEIGHT + 8,
        width - 2 * WINDOW_PADDING_X,
        BOTTOM_INFO_HEIGHT - 16,
    )
    pygame.draw.rect(screen, INFO_BG_COLOR, panel_rect, border_radius=8)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=8)

    draw_legend(
        screen=screen,
        left=panel_rect.left + 16,
        top=panel_rect.top + 14,
        small_font=small_font,
    )

    if hover_title is None or hover_lines is None:
        info_text = small_font.render(
            "Hover over a tile to inspect its details.",
            True,
            MUTED_TEXT_COLOR,
        )
        info_rect = info_text.get_rect(topright=(panel_rect.right - 16, panel_rect.top + 14))
        screen.blit(info_text, info_rect)
        return

    title_surface = small_font.render(hover_title, True, TEXT_COLOR)
    title_rect = title_surface.get_rect(topright=(panel_rect.right - 16, panel_rect.top + 14))
    screen.blit(title_surface, title_rect)

    y = panel_rect.top + 38
    line_gap = 19
    for line in hover_lines:
        line_surface = small_font.render(line, True, MUTED_TEXT_COLOR)
        line_rect = line_surface.get_rect(topright=(panel_rect.right - 16, y))
        screen.blit(line_surface, line_rect)
        y += line_gap


def reset_panel_state(env: GridCleanEnv) -> tuple[Tuple[int, int, int, int], bool, float, int, list[Tuple[int, int]], dict[Tuple[int, int], int]]:
    state = env.reset()
    done = False
    total_reward = 0.0
    step_idx = 0
    path_history = [env.robot_pos]
    visit_counts = {env.robot_pos: 1}
    return state, done, total_reward, step_idx, path_history, visit_counts


def run_ui_compare(
    random_env: GridCleanEnv,
    learned_env: GridCleanEnv,
    learned_agent: QLearningAgent,
    random_seed: int,
    cell_size: int,
    initial_step_delay: float,
) -> None:
    pygame.init()
    pygame.display.set_caption("SweepAgent UI Comparison")

    title_font, body_font, small_font = load_fonts()

    panel_width = random_env.cols * cell_size
    panel_height = PANEL_HEADER_HEIGHT + random_env.rows * cell_size

    width = (
        WINDOW_PADDING_X
        + panel_width
        + PANEL_GAP
        + panel_width
        + WINDOW_PADDING_X
    )
    height = TOP_PANEL_HEIGHT + panel_height + BOTTOM_INFO_HEIGHT + WINDOW_PADDING_Y

    screen = pygame.display.set_mode((width, height))
    clock = pygame.time.Clock()

    rng = random.Random(random_seed)

    random_state, random_done, random_reward, random_step, random_path, random_visits = reset_panel_state(random_env)
    learned_state, learned_done, learned_reward, learned_step, learned_path, learned_visits = reset_panel_state(learned_env)

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
            title_font=title_font,
            small_font=small_font,
        )

        random_grid_rect = draw_panel(
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
            panel_width=panel_width,
            cell_size=cell_size,
            title_font=body_font,
            body_font=body_font,
            small_font=small_font,
        )

        learned_grid_rect = draw_panel(
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
            panel_width=panel_width,
            cell_size=cell_size,
            title_font=body_font,
            body_font=body_font,
            small_font=small_font,
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
            small_font=small_font,
            hover_title=hover_title,
            hover_lines=hover_lines,
        )

        pygame.display.flip()
        clock.tick(60)


def main() -> None:
    args = parse_args()

    random_env = build_env(map_name=args.map_name)
    learned_env = build_env(map_name=args.map_name)
    random_env.map_name = args.map_name
    learned_env.map_name = args.map_name

    learned_agent = load_or_train_q_agent(
        map_name=args.map_name,
        num_episodes=args.episodes,
        seed=args.seed,
    )

    run_ui_compare(
        random_env=random_env,
        learned_env=learned_env,
        learned_agent=learned_agent,
        random_seed=args.random_seed,
        cell_size=args.cell_size,
        initial_step_delay=args.step_delay,
    )


if __name__ == "__main__":
    main()
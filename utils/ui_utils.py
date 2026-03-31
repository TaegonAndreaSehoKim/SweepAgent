from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pygame

from env.grid_clean_env import GridCleanEnv


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
PANEL_GAP = 24

MIN_STEP_DELAY = 0.1
MAX_STEP_DELAY = 1.5
STEP_DELAY_DELTA = 0.1


@dataclass
class UIFonts:
    title_font: pygame.font.Font
    body_font: pygame.font.Font
    small_font: pygame.font.Font


@dataclass
class PanelLayout:
    panel_width: int
    panel_header_height: int
    grid_left: int
    grid_top: int
    grid_width: int
    grid_height: int
    total_height: int


def load_fonts() -> UIFonts:
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

    return UIFonts(
        title_font=build_font(30, bold=True),
        body_font=build_font(20, bold=False),
        small_font=build_font(16, bold=False),
    )


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


def build_left_status_lines(
    map_name: str,
    total_reward: float,
    step_idx: int,
    include_map_line: bool = True,
) -> List[Tuple[str, Tuple[int, int, int]]]:
    lines: List[Tuple[str, Tuple[int, int, int]]] = []
    if include_map_line:
        lines.append((f"Map: {map_name}", TEXT_COLOR))
    lines.extend(
        [
            (f"Step: {step_idx}", TEXT_COLOR),
            (f"Reward: {total_reward:.0f}", TEXT_COLOR),
        ]
    )
    return lines


def build_right_status_lines(
    env: GridCleanEnv,
    done: bool,
    is_paused: bool | None = None,
    step_delay: float | None = None,
) -> List[Tuple[str, Tuple[int, int, int]]]:
    cleaned_tiles = env._count_cleaned_tiles()
    total_dirty_tiles = env.total_dirty_tiles

    lines: List[Tuple[str, Tuple[int, int, int]]] = [
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
        if is_paused is None:
            lines.append(("Status: running", TEXT_COLOR))
        else:
            lines.append(("Status: paused" if is_paused else "Status: running", TEXT_COLOR))

    if step_delay is not None:
        lines.append((f"Delay: {step_delay:.1f}s", TEXT_COLOR))

    return lines


def estimate_text_width(font: pygame.font.Font, lines: List[Tuple[str, Tuple[int, int, int]]]) -> int:
    if not lines:
        return 0
    return max(font.size(text)[0] for text, _ in lines)


def compute_panel_layout(
    env: GridCleanEnv,
    cell_size: int,
    fonts: UIFonts,
    left_lines: List[Tuple[str, Tuple[int, int, int]]],
    right_lines: List[Tuple[str, Tuple[int, int, int]]],
    title_text: str,
    subtitle_text: str | None = None,
) -> PanelLayout:
    grid_width = env.cols * cell_size
    grid_height = env.rows * cell_size

    title_height = fonts.body_font.get_linesize()
    subtitle_height = fonts.small_font.get_linesize() if subtitle_text else 0
    status_rows = max(len(left_lines), len(right_lines))
    status_height = status_rows * 18

    panel_header_height = 24 + title_height + 10 + subtitle_height + 12 + status_height + 18
    panel_header_height = max(panel_header_height, 120)

    left_width = estimate_text_width(fonts.small_font, left_lines)
    right_width = estimate_text_width(fonts.small_font, right_lines)
    title_width = fonts.body_font.size(title_text)[0]
    subtitle_width = fonts.small_font.size(subtitle_text)[0] if subtitle_text else 0

    required_text_width = max(title_width, subtitle_width, left_width + 40 + right_width)
    panel_width = max(grid_width, required_text_width)

    return PanelLayout(
        panel_width=panel_width,
        panel_header_height=panel_header_height,
        grid_left=0,
        grid_top=panel_header_height,
        grid_width=grid_width,
        grid_height=grid_height,
        total_height=panel_header_height + grid_height,
    )


def draw_grid(
    screen: pygame.Surface,
    env: GridCleanEnv,
    cell_size: int,
    grid_top: int,
    grid_left: int,
    body_font: pygame.font.Font,
    visit_counts: Dict[Tuple[int, int], int],
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
    path_history: List[Tuple[int, int]],
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


def draw_status_block(
    screen: pygame.Surface,
    left_lines: List[Tuple[str, Tuple[int, int, int]]],
    right_lines: List[Tuple[str, Tuple[int, int, int]]],
    panel_left: int,
    panel_top: int,
    panel_width: int,
    fonts: UIFonts,
    title_text: str,
    subtitle_text: str | None = None,
) -> None:
    title_surface = fonts.body_font.render(title_text, True, TEXT_COLOR)
    screen.blit(title_surface, (panel_left, panel_top))

    current_y = panel_top + fonts.body_font.get_linesize() + 8

    if subtitle_text:
        subtitle_surface = fonts.small_font.render(subtitle_text, True, MUTED_TEXT_COLOR)
        screen.blit(subtitle_surface, (panel_left, current_y))
        current_y += fonts.small_font.get_linesize() + 10

    left_x = panel_left
    right_x = panel_left + panel_width
    line_gap = 18

    y = current_y
    for text_value, color in left_lines:
        text_surface = fonts.small_font.render(text_value, True, color)
        screen.blit(text_surface, (left_x, y))
        y += line_gap

    y = current_y
    for text_value, color in right_lines:
        text_surface = fonts.small_font.render(text_value, True, color)
        text_rect = text_surface.get_rect(topright=(right_x, y))
        screen.blit(text_surface, text_rect)
        y += line_gap


def draw_single_panel(
    screen: pygame.Surface,
    env: GridCleanEnv,
    total_reward: float,
    step_idx: int,
    done: bool,
    is_paused: bool,
    step_delay: float,
    visit_counts: Dict[Tuple[int, int], int],
    path_history: List[Tuple[int, int]],
    panel_left: int,
    panel_top: int,
    cell_size: int,
    fonts: UIFonts,
) -> pygame.Rect:
    left_lines = build_left_status_lines(
        map_name=getattr(env, "map_name", "unknown"),
        total_reward=total_reward,
        step_idx=step_idx,
        include_map_line=True,
    )
    right_lines = build_right_status_lines(
        env=env,
        done=done,
        is_paused=is_paused,
        step_delay=step_delay,
    )

    layout = compute_panel_layout(
        env=env,
        cell_size=cell_size,
        fonts=fonts,
        left_lines=left_lines,
        right_lines=right_lines,
        title_text="SweepAgent UI Demo",
        subtitle_text=None,
    )

    panel_rect = pygame.Rect(
        panel_left - 8,
        panel_top - 8,
        layout.panel_width + 16,
        layout.total_height + 16,
    )
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=8)

    draw_status_block(
        screen=screen,
        left_lines=left_lines,
        right_lines=right_lines,
        panel_left=panel_left,
        panel_top=panel_top,
        panel_width=layout.panel_width,
        fonts=fonts,
        title_text="SweepAgent UI Demo",
        subtitle_text=None,
    )

    grid_left = panel_left + (layout.panel_width - layout.grid_width) // 2
    grid_top = panel_top + layout.panel_header_height

    draw_grid(
        screen=screen,
        env=env,
        cell_size=cell_size,
        grid_top=grid_top,
        grid_left=grid_left,
        body_font=fonts.body_font,
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

    return pygame.Rect(grid_left, grid_top, layout.grid_width, layout.grid_height)


def draw_compare_panel(
    screen: pygame.Surface,
    panel_title: str,
    env: GridCleanEnv,
    total_reward: float,
    step_idx: int,
    done: bool,
    visit_counts: Dict[Tuple[int, int], int],
    path_history: List[Tuple[int, int]],
    panel_left: int,
    panel_top: int,
    cell_size: int,
    fonts: UIFonts,
) -> tuple[pygame.Rect, PanelLayout]:
    left_lines = build_left_status_lines(
        map_name=getattr(env, "map_name", "unknown"),
        total_reward=total_reward,
        step_idx=step_idx,
        include_map_line=False,
    )
    right_lines = build_right_status_lines(
        env=env,
        done=done,
        is_paused=None,
        step_delay=None,
    )

    layout = compute_panel_layout(
        env=env,
        cell_size=cell_size,
        fonts=fonts,
        left_lines=left_lines,
        right_lines=right_lines,
        title_text=panel_title,
        subtitle_text=f"Map: {getattr(env, 'map_name', 'unknown')}",
    )

    panel_rect = pygame.Rect(
        panel_left - 8,
        panel_top - 8,
        layout.panel_width + 16,
        layout.total_height + 16,
    )
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=8)

    draw_status_block(
        screen=screen,
        left_lines=left_lines,
        right_lines=right_lines,
        panel_left=panel_left,
        panel_top=panel_top,
        panel_width=layout.panel_width,
        fonts=fonts,
        title_text=panel_title,
        subtitle_text=f"Map: {getattr(env, 'map_name', 'unknown')}",
    )

    grid_left = panel_left + (layout.panel_width - layout.grid_width) // 2
    grid_top = panel_top + layout.panel_header_height

    draw_grid(
        screen=screen,
        env=env,
        cell_size=cell_size,
        grid_top=grid_top,
        grid_left=grid_left,
        body_font=fonts.body_font,
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

    return pygame.Rect(grid_left, grid_top, layout.grid_width, layout.grid_height), layout


def draw_top_controls(
    screen: pygame.Surface,
    width: int,
    step_delay: float,
    is_paused: bool,
    fonts: UIFonts,
    title_text: str,
) -> None:
    title = fonts.title_font.render(title_text, True, TEXT_COLOR)
    title_rect = title.get_rect(center=(width // 2, 26))
    screen.blit(title, title_rect)

    controls = fonts.small_font.render(
        "ESC quit | SPACE pause | R restart | [ ] speed",
        True,
        TEXT_COLOR,
    )
    controls_rect = controls.get_rect(center=(width // 2, 56))
    screen.blit(controls, controls_rect)

    state_text = f"State: {'paused' if is_paused else 'running'} | Delay: {step_delay:.1f}s"
    state_surface = fonts.small_font.render(state_text, True, TEXT_COLOR)
    state_rect = state_surface.get_rect(topright=(width - WINDOW_PADDING_X, 20))
    screen.blit(state_surface, state_rect)


def draw_legend(
    screen: pygame.Surface,
    left: int,
    top: int,
    small_font: pygame.font.Font,
) -> int:
    items = [
        ("Wall", WALL_COLOR),
        ("Dirty", DIRTY_COLOR),
        ("Cleaned", CLEANED_COLOR),
        ("Charger", CHARGER_COLOR),
        ("Heatmap", HEATMAP_COLOR),
        ("Path", PATH_COLOR),
    ]

    columns = 2
    row_gap = 10
    item_height = 22
    column_width = 170

    for idx, (label, color) in enumerate(items):
        row = idx % 3
        col = idx // 3

        x = left + col * column_width
        y = top + row * (item_height + row_gap)

        swatch_rect = pygame.Rect(x, y + 2, 16, 16)
        pygame.draw.rect(screen, color, swatch_rect)
        pygame.draw.rect(screen, GRID_LINE_COLOR, swatch_rect, width=1)

        text_surface = small_font.render(label, True, TEXT_COLOR)
        screen.blit(text_surface, (x + 24, y))

    total_height = 3 * item_height + 2 * row_gap
    return total_height


def get_hover_info(
    env: GridCleanEnv,
    visit_counts: Dict[Tuple[int, int], int],
    grid_rect: pygame.Rect,
    mouse_pos: Tuple[int, int],
    cell_size: int,
) -> List[str] | None:
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


def compute_bottom_info_height(fonts: UIFonts, hover_lines_count: int = 5) -> int:
    legend_rows_height = 3 * 22 + 2 * 10
    title_height = fonts.small_font.get_linesize()
    line_height = 19
    hover_block = title_height + 8 + hover_lines_count * line_height
    content_height = max(legend_rows_height, hover_block)
    return max(170, content_height + 32)


def draw_bottom_info(
    screen: pygame.Surface,
    width: int,
    height: int,
    fonts: UIFonts,
    hover_title: str | None,
    hover_lines: List[str] | None,
    bottom_info_height: int,
) -> None:
    panel_rect = pygame.Rect(
        WINDOW_PADDING_X,
        height - bottom_info_height + 8,
        width - 2 * WINDOW_PADDING_X,
        bottom_info_height - 16,
    )
    pygame.draw.rect(screen, INFO_BG_COLOR, panel_rect, border_radius=8)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=8)

    legend_height = draw_legend(
        screen=screen,
        left=panel_rect.left + 16,
        top=panel_rect.top + 14,
        small_font=fonts.small_font,
    )

    if hover_title is None or hover_lines is None:
        info_text = fonts.small_font.render(
            "Hover over a tile to inspect its details.",
            True,
            MUTED_TEXT_COLOR,
        )
        info_rect = info_text.get_rect(topright=(panel_rect.right - 16, panel_rect.top + 14))
        screen.blit(info_text, info_rect)
        return

    title_surface = fonts.small_font.render(hover_title, True, TEXT_COLOR)
    title_rect = title_surface.get_rect(topright=(panel_rect.right - 16, panel_rect.top + 14))
    screen.blit(title_surface, title_rect)

    y = panel_rect.top + 38
    line_gap = 19
    for line in hover_lines:
        line_surface = fonts.small_font.render(line, True, MUTED_TEXT_COLOR)
        line_rect = line_surface.get_rect(topright=(panel_rect.right - 16, y))
        screen.blit(line_surface, line_rect)
        y += line_gap


def reset_panel_state(env: GridCleanEnv) -> tuple[Tuple[int, int, int, int], bool, float, int, List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    state = env.reset()
    done = False
    total_reward = 0.0
    step_idx = 0
    path_history = [env.robot_pos]
    visit_counts = {env.robot_pos: 1}
    return state, done, total_reward, step_idx, path_history, visit_counts
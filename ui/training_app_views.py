from __future__ import annotations

from typing import Dict, List, Tuple

import pygame

from ui.training_app_core import (
    BUTTON_BG,
    BUTTON_HOVER_BG,
    BUTTON_ACTIVE_BG,
    PRIMARY_BUTTON_BG,
    PRIMARY_BUTTON_HOVER_BG,
    PRIMARY_BUTTON_TEXT,
    CONTROL_BAR_BG,
    DROPDOWN_BG,
    DROPDOWN_HOVER_BG,
    PROGRESS_BG,
    SCROLL_TRACK,
    SCROLL_THUMB,
    PLAYBACK_TITLE_AREA_HEIGHT,
    PLAYBACK_CONTROL_BAR_HEIGHT,
    PLAYBACK_TOP_RESERVED,
    TRAINING_TOP_Y,
    TRAINING_METRICS_HEIGHT,
    TRAINING_LOG_HEIGHT,
    TRAINING_GRAPH_HEIGHT,
    TRAINING_PREVIEW_HEIGHT,
    TRAINING_GAP,
)

BACKGROUND_COLOR = (245, 247, 250)
PANEL_BG = (255, 255, 255)
PANEL_BORDER_COLOR = (210, 218, 230)
TEXT_COLOR = (44, 52, 64)
MUTED_TEXT_COLOR = (126, 140, 156)
WALL_COLOR = (43, 45, 47)
FLOOR_COLOR = (248, 249, 250)
DIRTY_COLOR = (244, 208, 90)
CLEANED_COLOR = (183, 229, 205)
ROBOT_COLOR = (60, 132, 244)
CHARGER_COLOR = (202, 181, 228)
PATH_COLOR = (64, 147, 255)

WINDOW_PADDING_X = 24


def get_font(fonts, *names):
    for name in names:
        if hasattr(fonts, name):
            return getattr(fonts, name)
    raise AttributeError(f"UIFonts is missing all requested font attributes: {names}")


def small_font(fonts):
    return get_font(fonts, "small_font", "body_font", "medium_font", "title_font")


def title_font(fonts):
    return get_font(fonts, "title_font", "section_font", "medium_font", "small_font")


def section_font(fonts):
    return get_font(fonts, "subsection_font", "section_font", "title_font", "small_font")


def draw_button(screen: pygame.Surface, button, fonts, mouse_pos: tuple[int, int]) -> None:
    hovered = button.rect.collidepoint(mouse_pos) and button.enabled

    if button.primary:
        bg = PRIMARY_BUTTON_HOVER_BG if hovered else PRIMARY_BUTTON_BG
        text_color = PRIMARY_BUTTON_TEXT
    else:
        bg = BUTTON_HOVER_BG if hovered else BUTTON_BG
        text_color = TEXT_COLOR if button.enabled else MUTED_TEXT_COLOR

    pygame.draw.rect(screen, bg, button.rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, button.rect, width=2, border_radius=10)

    text_surface = small_font(fonts).render(button.label, True, text_color)
    text_rect = text_surface.get_rect(center=button.rect.center)
    screen.blit(text_surface, text_rect)


def get_env_map_name(env, fallback: str = "unknown") -> str:
    return getattr(env, "map_name", fallback)


def count_cleaned_tiles(env) -> int:
    if hasattr(env, "_count_cleaned_tiles"):
        try:
            return int(env._count_cleaned_tiles())
        except Exception:
            pass

    cleaned_mask = getattr(env, "cleaned_mask", 0)
    if isinstance(cleaned_mask, int):
        return int(cleaned_mask.bit_count())

    return 0


def total_dirty_tiles(env) -> int:
    if hasattr(env, "total_dirty_tiles"):
        return int(getattr(env, "total_dirty_tiles"))
    dirty_map = getattr(env, "dirty_index_map", {})
    if isinstance(dirty_map, dict):
        return len(dirty_map)
    return 0


def get_battery_text(env) -> str:
    if hasattr(env, "battery_remaining") and hasattr(env, "battery_capacity"):
        remaining = getattr(env, "battery_remaining")
        capacity = getattr(env, "battery_capacity")
        if remaining is not None and capacity is not None:
            return f"{remaining}/{capacity}"
    return "off"


def build_display_grid(env) -> list[list[str]]:
    grid = [row[:] for row in getattr(env, "grid", [])]
    if not grid:
        raw_map = getattr(env, "raw_map", [])
        grid = [list(row) for row in raw_map]

    dirty_index_map = getattr(env, "dirty_index_map", {})
    cleaned_mask = getattr(env, "cleaned_mask", 0)

    for (r, c), idx in dirty_index_map.items():
        cleaned = ((cleaned_mask >> idx) & 1) == 1
        if 0 <= r < len(grid) and 0 <= c < len(grid[0]):
            grid[r][c] = "." if cleaned else "D"

    raw_map = getattr(env, "raw_map", [])
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if raw_map and 0 <= r < len(raw_map) and 0 <= c < len(raw_map[r]):
                if raw_map[r][c] == "C":
                    if grid[r][c] == ".":
                        grid[r][c] = "C"

    robot_pos = getattr(env, "robot_pos", None)
    if robot_pos is not None:
        rr, rc = robot_pos
        if 0 <= rr < len(grid) and 0 <= rc < len(grid[0]):
            grid[rr][rc] = "R"

    return grid


def tile_base_color(tile: str) -> tuple[int, int, int]:
    if tile == "#":
        return WALL_COLOR
    if tile == "D":
        return DIRTY_COLOR
    if tile == "C":
        return CHARGER_COLOR
    if tile == ".":
        return FLOOR_COLOR
    if tile == "R":
        return FLOOR_COLOR
    return FLOOR_COLOR


def draw_grid(
    screen: pygame.Surface,
    env,
    top_left: tuple[int, int],
    cell_size: int,
    fonts,
    visit_counts: dict[tuple[int, int], int] | None = None,
    path_history: list[tuple[int, int]] | None = None,
) -> pygame.Rect:
    grid = build_display_grid(env)
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    left, top = top_left
    grid_rect = pygame.Rect(left, top, cols * cell_size, rows * cell_size)

    max_visits = 0
    if visit_counts:
        max_visits = max(visit_counts.values(), default=0)

    raw_map = getattr(env, "raw_map", [])
    dirty_index_map = getattr(env, "dirty_index_map", {})
    cleaned_mask = getattr(env, "cleaned_mask", 0)

    for r in range(rows):
        for c in range(cols):
            tile = grid[r][c]
            rect = pygame.Rect(
                left + c * cell_size,
                top + r * cell_size,
                cell_size,
                cell_size,
            )

            base_tile = tile
            if tile == "R":
                underlying = "."
                if (r, c) in dirty_index_map:
                    idx = dirty_index_map[(r, c)]
                    cleaned = ((cleaned_mask >> idx) & 1) == 1
                    underlying = "." if cleaned else "D"
                if raw_map and 0 <= r < len(raw_map) and 0 <= c < len(raw_map[r]):
                    if raw_map[r][c] == "C":
                        underlying = "C"
                base_tile = underlying

            color = tile_base_color(base_tile)

            if visit_counts and (r, c) in visit_counts and base_tile != "#":
                visits = visit_counts[(r, c)]
                if max_visits > 0:
                    boost = int(60 * visits / max_visits)
                    color = (
                        min(255, color[0] + boost),
                        max(0, color[1] - boost // 3),
                        max(0, color[2] - boost // 3),
                    )

            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, PANEL_BORDER_COLOR, rect, width=2)

            if base_tile == "C":
                label = small_font(fonts).render("C", True, TEXT_COLOR)
                label_rect = label.get_rect(center=rect.center)
                screen.blit(label, label_rect)

    if path_history and len(path_history) >= 2:
        for i in range(len(path_history) - 1):
            r1, c1 = path_history[i]
            r2, c2 = path_history[i + 1]
            p1 = (left + c1 * cell_size + cell_size // 2, top + r1 * cell_size + cell_size // 2)
            p2 = (left + c2 * cell_size + cell_size // 2, top + r2 * cell_size + cell_size // 2)
            pygame.draw.line(screen, PATH_COLOR, p1, p2, 4)

    robot_pos = getattr(env, "robot_pos", None)
    if robot_pos is not None:
        rr, rc = robot_pos
        center = (
            left + rc * cell_size + cell_size // 2,
            top + rr * cell_size + cell_size // 2,
        )
        pygame.draw.circle(screen, ROBOT_COLOR, center, max(10, cell_size // 3))
        pygame.draw.circle(screen, (24, 64, 140), center, max(2, cell_size // 16), width=2)

    return grid_rect


def draw_legend(screen: pygame.Surface, fonts, rect: pygame.Rect, hover_text: str | None = None) -> None:
    pygame.draw.rect(screen, PANEL_BG, rect, border_radius=12)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, rect, width=2, border_radius=12)

    entries = [
        ("Wall", WALL_COLOR),
        ("Dirty", DIRTY_COLOR),
        ("Cleaned", CLEANED_COLOR),
        ("Charger", CHARGER_COLOR),
        ("Path", PATH_COLOR),
    ]

    x = rect.left + 22
    y = rect.top + 24
    col_gap = 140
    row_gap = 36

    for idx, (label, color) in enumerate(entries):
        row = idx % 3
        col = idx // 3
        item_x = x + col * col_gap
        item_y = y + row * row_gap

        swatch = pygame.Rect(item_x, item_y, 18, 18)
        pygame.draw.rect(screen, color, swatch)
        pygame.draw.rect(screen, PANEL_BORDER_COLOR, swatch, width=1)

        label_surface = small_font(fonts).render(label, True, TEXT_COLOR)
        screen.blit(label_surface, (item_x + 28, item_y - 2))

    hover_display = hover_text or "Hover over a tile to inspect its details."
    hover_surface = small_font(fonts).render(hover_display, True, MUTED_TEXT_COLOR)
    hover_rect = hover_surface.get_rect(midright=(rect.right - 20, rect.top + 28))
    screen.blit(hover_surface, hover_rect)


def get_hover_text(env, grid_rect: pygame.Rect, cell_size: int, mouse_pos: tuple[int, int]) -> str | None:
    if not grid_rect.collidepoint(mouse_pos):
        return None

    col = (mouse_pos[0] - grid_rect.left) // cell_size
    row = (mouse_pos[1] - grid_rect.top) // cell_size

    rows = getattr(env, "rows", 0)
    cols = getattr(env, "cols", 0)
    if not (0 <= row < rows and 0 <= col < cols):
        return None

    raw_map = getattr(env, "raw_map", [])
    base = "."
    if raw_map and 0 <= row < len(raw_map) and 0 <= col < len(raw_map[row]):
        base = raw_map[row][col]

    tile_type = {
        "#": "wall",
        ".": "floor",
        "D": "dirty",
        "C": "charger",
        "R": "start",
    }.get(base, "floor")

    if (row, col) in getattr(env, "dirty_index_map", {}):
        idx = env.dirty_index_map[(row, col)]
        cleaned = ((getattr(env, "cleaned_mask", 0) >> idx) & 1) == 1
        tile_type = "cleaned" if cleaned else "dirty"

    return f"Tile: ({row}, {col}) | Type: {tile_type}"


def draw_dropdown_box(
    screen: pygame.Surface,
    fonts,
    label: str,
    value: str,
    rect: pygame.Rect,
    is_open: bool,
) -> None:
    label_surface = small_font(fonts).render(label, True, TEXT_COLOR)
    screen.blit(label_surface, (rect.left, rect.top - 24))

    box_bg = BUTTON_ACTIVE_BG if is_open else BUTTON_BG
    pygame.draw.rect(screen, box_bg, rect, border_radius=8)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, rect, width=2, border_radius=8)

    value_surface = small_font(fonts).render(value, True, TEXT_COLOR)
    value_rect = value_surface.get_rect(midleft=(rect.left + 12, rect.centery))
    screen.blit(value_surface, value_rect)

    pygame.draw.polygon(
        screen,
        MUTED_TEXT_COLOR,
        [
            (rect.right - 18, rect.centery - 3),
            (rect.right - 8, rect.centery - 3),
            (rect.right - 13, rect.centery + 4),
        ],
    )


def draw_input_box(
    screen: pygame.Surface,
    fonts,
    label: str,
    value: str,
    rect: pygame.Rect,
    is_active: bool,
) -> None:
    label_surface = small_font(fonts).render(label, True, TEXT_COLOR)
    screen.blit(label_surface, (rect.left, rect.top - 24))

    box_bg = BUTTON_ACTIVE_BG if is_active else BUTTON_BG
    pygame.draw.rect(screen, box_bg, rect, border_radius=8)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, rect, width=2, border_radius=8)

    display_value = value if value else ""
    if is_active:
        display_value += "|"
    value_surface = small_font(fonts).render(display_value, True, TEXT_COLOR)
    value_rect = value_surface.get_rect(midleft=(rect.left + 12, rect.centery))
    screen.blit(value_surface, value_rect)


def draw_dropdown_overlay(
    screen: pygame.Surface,
    fonts,
    anchor_rect: pygame.Rect,
    options: list[str],
    mouse_pos: tuple[int, int],
) -> list[pygame.Rect]:
    option_height = 34
    padding = 6
    screen_margin = 8

    dropdown_width = max(anchor_rect.width, 200)
    desired_height = padding * 2 + option_height * len(options)

    available_below = screen.get_height() - anchor_rect.bottom - 8 - screen_margin
    available_above = anchor_rect.top - 8 - screen_margin

    max_dropdown_height = max(
        option_height + padding * 2,
        min(420, max(available_below, available_above)),
    )
    dropdown_height = min(desired_height, max_dropdown_height)

    open_upward = desired_height > available_below and available_above > available_below

    dropdown_left = anchor_rect.left
    if dropdown_left + dropdown_width > screen.get_width() - screen_margin:
        dropdown_left = screen.get_width() - screen_margin - dropdown_width
    dropdown_left = max(screen_margin, dropdown_left)

    if open_upward:
        dropdown_top = anchor_rect.top - 8 - dropdown_height
        dropdown_top = max(screen_margin, dropdown_top)
    else:
        dropdown_top = anchor_rect.bottom + 8
        if dropdown_top + dropdown_height > screen.get_height() - screen_margin:
            dropdown_top = max(screen_margin, screen.get_height() - screen_margin - dropdown_height)

    dropdown_rect = pygame.Rect(
        dropdown_left,
        dropdown_top,
        dropdown_width,
        dropdown_height,
    )

    pygame.draw.rect(screen, DROPDOWN_BG, dropdown_rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, dropdown_rect, width=2, border_radius=10)

    option_rects: list[pygame.Rect] = []
    max_visible = max(1, (dropdown_rect.height - padding * 2) // option_height)

    for idx, option in enumerate(options[:max_visible]):
        option_rect = pygame.Rect(
            dropdown_rect.left + 6,
            dropdown_rect.top + padding + idx * option_height,
            dropdown_rect.width - 12,
            option_height - 2,
        )

        hovered = option_rect.collidepoint(mouse_pos)
        if hovered:
            pygame.draw.rect(screen, DROPDOWN_HOVER_BG, option_rect, border_radius=6)

        text_surface = small_font(fonts).render(option, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(midleft=(option_rect.left + 10, option_rect.centery))
        screen.blit(text_surface, text_rect)
        option_rects.append(option_rect)

    return option_rects


def draw_menu(
    screen: pygame.Surface,
    width: int,
    height: int,
    fonts,
    map_name: str,
    model_name: str,
    result_view: str,
    input_values: dict[str, str],
    open_dropdown: str | None,
    active_input: str | None,
    buttons: list,
    map_options: list[str],
    model_options: list[str],
    result_view_options: list[str],
) -> dict[str, pygame.Rect | list[pygame.Rect]]:
    screen.fill(BACKGROUND_COLOR)
    mouse_pos = pygame.mouse.get_pos()

    title = title_font(fonts).render("SweepAgent Training App", True, TEXT_COLOR)
    title_rect = title.get_rect(center=(width // 2, 42))
    screen.blit(title, title_rect)

    help_text = small_font(fonts).render(
        "Choose map, algorithm, seeds, and training hyperparameters from the control panel.",
        True,
        MUTED_TEXT_COLOR,
    )
    help_rect = help_text.get_rect(center=(width // 2, 74))
    screen.blit(help_text, help_rect)

    panel_rect = pygame.Rect(50, 110, width - 100, height - 180)
    pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=12)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=12)

    panel_inner_left = panel_rect.left + 24
    panel_inner_right = panel_rect.right - 24
    column_gap = 18
    field_w = (panel_inner_right - panel_inner_left - column_gap * 2) // 3

    col1_x = panel_inner_left
    col2_x = col1_x + field_w + column_gap
    col3_x = col2_x + field_w + column_gap

    row1_y = panel_rect.top + 60
    row2_y = panel_rect.top + 150
    row3_y = panel_rect.top + 240
    row4_y = panel_rect.top + 330

    field_h = 42

    map_rect = pygame.Rect(col1_x, row1_y, field_w, field_h)
    model_rect = pygame.Rect(col2_x, row1_y, field_w, field_h)
    result_rect = pygame.Rect(col3_x, row1_y, field_w, field_h)

    episodes_rect = pygame.Rect(col1_x, row2_y, field_w, field_h)
    train_seed_rect = pygame.Rect(col2_x, row2_y, field_w, field_h)
    playback_seed_rect = pygame.Rect(col3_x, row2_y, field_w, field_h)

    lr_rect = pygame.Rect(col1_x, row3_y, field_w, field_h)
    gamma_rect = pygame.Rect(col2_x, row3_y, field_w, field_h)
    eps_start_rect = pygame.Rect(col3_x, row3_y, field_w, field_h)

    eps_decay_rect = pygame.Rect(col1_x, row4_y, field_w, field_h)
    eps_min_rect = pygame.Rect(col2_x, row4_y, field_w, field_h)
    delay_rect = pygame.Rect(col3_x, row4_y, field_w, field_h)

    draw_dropdown_box(screen, fonts, "Map", map_name, map_rect, open_dropdown == "map")
    draw_dropdown_box(screen, fonts, "Algorithm", model_name, model_rect, open_dropdown == "model")
    draw_dropdown_box(screen, fonts, "Result View", result_view, result_rect, open_dropdown == "result_view")

    draw_input_box(screen, fonts, "Episodes", input_values.get("episodes", ""), episodes_rect, active_input == "episodes")
    draw_input_box(screen, fonts, "Train Seed", input_values.get("train_seed", ""), train_seed_rect, active_input == "train_seed")
    draw_input_box(screen, fonts, "Playback Seed", input_values.get("playback_seed", ""), playback_seed_rect, active_input == "playback_seed")

    draw_input_box(screen, fonts, "Learning Rate", input_values.get("learning_rate", ""), lr_rect, active_input == "learning_rate")
    draw_input_box(screen, fonts, "Discount Factor", input_values.get("discount_factor", ""), gamma_rect, active_input == "discount_factor")
    draw_input_box(screen, fonts, "Epsilon Start", input_values.get("epsilon_start", ""), eps_start_rect, active_input == "epsilon_start")

    draw_input_box(screen, fonts, "Epsilon Decay", input_values.get("epsilon_decay", ""), eps_decay_rect, active_input == "epsilon_decay")
    draw_input_box(screen, fonts, "Epsilon Min", input_values.get("epsilon_min", ""), eps_min_rect, active_input == "epsilon_min")
    draw_input_box(screen, fonts, "Playback Delay", input_values.get("delay", ""), delay_rect, active_input == "delay")

    notes = [
        "Numeric fields accept direct typing. Click a field, edit the value, then press Enter or click elsewhere.",
        "Recommended baseline for hard maps: 200000 episodes, gamma 0.99, epsilon_decay 0.99999.",
        "battery_adapt_q_learning adds a 50000-episode evaluation-battery finetune stage after the base run.",
        "random_baseline skips training and jumps directly to playback.",
    ]
    note_y = panel_rect.top + 455
    for note in notes:
        note_surface = small_font(fonts).render(note, True, MUTED_TEXT_COLOR)
        screen.blit(note_surface, (panel_rect.left + 32, note_y))
        note_y += 24

    for button in buttons:
        draw_button(screen, button, fonts, mouse_pos)

    dropdown_rects = {
        "map_rect": map_rect,
        "model_rect": model_rect,
        "result_rect": result_rect,
        "episodes_rect": episodes_rect,
        "train_seed_rect": train_seed_rect,
        "playback_seed_rect": playback_seed_rect,
        "learning_rate_rect": lr_rect,
        "discount_factor_rect": gamma_rect,
        "epsilon_start_rect": eps_start_rect,
        "epsilon_decay_rect": eps_decay_rect,
        "epsilon_min_rect": eps_min_rect,
        "delay_rect": delay_rect,
    }

    option_rects: dict[str, list[pygame.Rect]] = {
        "map_options": [],
        "model_options": [],
        "result_options": [],
    }

    if open_dropdown == "map":
        option_rects["map_options"] = draw_dropdown_overlay(screen, fonts, map_rect, map_options, mouse_pos)
    elif open_dropdown == "model":
        option_rects["model_options"] = draw_dropdown_overlay(screen, fonts, model_rect, model_options, mouse_pos)
    elif open_dropdown == "result_view":
        option_rects["result_options"] = draw_dropdown_overlay(screen, fonts, result_rect, result_view_options, mouse_pos)

    return {**dropdown_rects, **option_rects}


def draw_progress_bar(
    screen: pygame.Surface,
    rect: pygame.Rect,
    value: float,
    fill_color: tuple[int, int, int],
) -> None:
    value = max(0.0, min(1.0, value))
    pygame.draw.rect(screen, PROGRESS_BG, rect, border_radius=8)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, rect, width=1, border_radius=8)

    fill_rect = pygame.Rect(rect.left, rect.top, int(rect.width * value), rect.height)
    if fill_rect.width > 0:
        pygame.draw.rect(screen, fill_color, fill_rect, border_radius=8)


def draw_training_metrics_panel(screen: pygame.Surface, fonts, panel_rect: pygame.Rect, latest_metrics: dict[str, str]) -> None:
    pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=10)

    title_surface = section_font(fonts).render("Training Metrics", True, TEXT_COLOR)
    screen.blit(title_surface, (panel_rect.left + 16, panel_rect.top + 12))

    labels = [
        ("Episode", latest_metrics.get("episode", "-"), latest_metrics.get("total", "-")),
        ("Avg Reward", latest_metrics.get("avg_reward", "-"), None),
        ("Avg Cleaned Ratio", latest_metrics.get("avg_cleaned_ratio", "-"), None),
        ("Success Rate", latest_metrics.get("success_rate", "-"), None),
        ("Epsilon", latest_metrics.get("epsilon", "-"), None),
    ]

    y = panel_rect.top + 48
    for label, value, aux in labels:
        label_surface = small_font(fonts).render(label, True, MUTED_TEXT_COLOR)
        screen.blit(label_surface, (panel_rect.left + 16, y))

        value_text = f"{value}/{aux}" if label == "Episode" and aux is not None else value
        value_surface = small_font(fonts).render(value_text, True, TEXT_COLOR)
        screen.blit(value_surface, (panel_rect.left + 200, y))
        y += 24


def draw_training_log_panel(
    screen: pygame.Surface,
    fonts,
    panel_rect: pygame.Rect,
    log_lines: list[str],
    scroll_offset: int,
) -> int:
    pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=10)

    title_surface = section_font(fonts).render("Training Log", True, TEXT_COLOR)
    screen.blit(title_surface, (panel_rect.left + 16, panel_rect.top + 12))

    content_rect = pygame.Rect(panel_rect.left + 16, panel_rect.top + 48, panel_rect.width - 40, panel_rect.height - 64)
    line_height = 20

    visible_lines = max(1, content_rect.height // line_height)
    max_scroll = max(0, len(log_lines) - visible_lines)
    scroll_offset = max(0, min(scroll_offset, max_scroll))

    visible_slice = log_lines[scroll_offset : scroll_offset + visible_lines]
    y = content_rect.top
    for line in visible_slice:
        text_surface = small_font(fonts).render(line, True, TEXT_COLOR)
        screen.blit(text_surface, (content_rect.left, y))
        y += line_height

    if max_scroll > 0:
        track_rect = pygame.Rect(panel_rect.right - 16, content_rect.top, 8, content_rect.height)
        pygame.draw.rect(screen, SCROLL_TRACK, track_rect, border_radius=4)

        thumb_height = max(24, int(track_rect.height * (visible_lines / len(log_lines))))
        thumb_travel = track_rect.height - thumb_height
        thumb_y = track_rect.top + int(thumb_travel * (scroll_offset / max_scroll))
        thumb_rect = pygame.Rect(track_rect.left, thumb_y, track_rect.width, thumb_height)
        pygame.draw.rect(screen, SCROLL_THUMB, thumb_rect, border_radius=4)

    return max_scroll


def draw_training_graph_panel(
    screen: pygame.Surface,
    fonts,
    panel_rect: pygame.Rect,
    latest_metrics: dict[str, str],
    progress_fill: tuple[int, int, int],
    cleaned_fill: tuple[int, int, int],
    success_fill: tuple[int, int, int],
    epsilon_fill: tuple[int, int, int],
) -> None:
    pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=10)

    title_surface = section_font(fonts).render("Live Progress", True, TEXT_COLOR)
    screen.blit(title_surface, (panel_rect.left + 16, panel_rect.top + 12))

    def safe_percent(text: str) -> float:
        try:
            return float(text.replace("%", "")) / 100.0
        except ValueError:
            return 0.0

    def safe_value(text: str) -> float:
        try:
            return float(text)
        except ValueError:
            return 0.0

    success_value = safe_percent(latest_metrics.get("success_rate", "0%"))
    cleaned_value = safe_percent(latest_metrics.get("avg_cleaned_ratio", "0%"))
    epsilon_value = safe_value(latest_metrics.get("epsilon", "0"))

    episode_progress = 0.0
    try:
        episode_num = float(latest_metrics.get("episode", "0"))
        total_num = float(latest_metrics.get("total", "0"))
        if total_num > 0:
            episode_progress = episode_num / total_num
    except ValueError:
        episode_progress = 0.0

    bars = [
        ("Success Rate", success_value, success_fill),
        ("Cleaned Ratio", cleaned_value, cleaned_fill),
        ("Episode Progress", episode_progress, progress_fill),
        ("Epsilon", epsilon_value, epsilon_fill),
    ]

    y = panel_rect.top + 48
    bar_w = panel_rect.width - 190
    bar_h = 18

    for label, value, fill_color in bars:
        label_surface = small_font(fonts).render(label, True, TEXT_COLOR)
        screen.blit(label_surface, (panel_rect.left + 16, y))

        bar_rect = pygame.Rect(panel_rect.left + 150, y + 2, bar_w, bar_h)
        draw_progress_bar(screen, bar_rect, value, fill_color)

        value_surface = small_font(fonts).render(f"{value * 100:.1f}%", True, TEXT_COLOR)
        screen.blit(value_surface, (bar_rect.right - 56, y))
        y += 30


def draw_training_preview_panel(
    screen: pygame.Surface,
    fonts,
    panel_rect: pygame.Rect,
    preview_env,
    preview_reward: float,
    preview_step: int,
    preview_visits: dict[tuple[int, int], int],
    preview_path: list[tuple[int, int]],
    preview_mode_label: str,
    cell_size: int,
) -> None:
    pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=10)

    title_surface = section_font(fonts).render("Mini Rollout Preview", True, TEXT_COLOR)
    screen.blit(title_surface, (panel_rect.left + 16, panel_rect.top + 12))

    subtitle_surface = small_font(fonts).render(
        f"Mode: {preview_mode_label}",
        True,
        MUTED_TEXT_COLOR,
    )
    screen.blit(subtitle_surface, (panel_rect.left + 220, panel_rect.top + 14))

    if preview_env is None:
        return

    info_left_x = panel_rect.left + 18
    info_top_y = panel_rect.top + 52
    right_x = panel_rect.right - 18

    left_lines = [
        f"Map: {get_env_map_name(preview_env, 'preview')}",
        f"Step: {preview_step}",
        f"Reward: {preview_reward:.0f}",
    ]
    right_lines = [
        f"Cleaned: {count_cleaned_tiles(preview_env)}/{total_dirty_tiles(preview_env)}",
        f"Battery: {get_battery_text(preview_env)}",
        "Status: preview",
    ]

    for i, line in enumerate(left_lines):
        surface = small_font(fonts).render(line, True, TEXT_COLOR)
        screen.blit(surface, (info_left_x, info_top_y + i * 26))

    for i, line in enumerate(right_lines):
        surface = small_font(fonts).render(line, True, TEXT_COLOR)
        rect = surface.get_rect(topright=(right_x, info_top_y + i * 26))
        screen.blit(surface, rect)

    grid_top = panel_rect.top + 128
    grid_left = panel_rect.left + 18
    draw_grid(
        screen=screen,
        env=preview_env,
        top_left=(grid_left, grid_top),
        cell_size=max(20, min(cell_size, 42)),
        fonts=fonts,
        visit_counts=preview_visits,
        path_history=preview_path,
    )


def draw_training_screen(
    screen: pygame.Surface,
    width: int,
    height: int,
    fonts,
    map_name: str,
    model_name: str,
    episodes: int,
    latest_metrics: dict[str, str],
    log_lines: list[str],
    log_scroll_offset: int,
    preview_env,
    preview_reward: float,
    preview_step: int,
    preview_visits: dict[tuple[int, int], int],
    preview_path: list[tuple[int, int]],
    preview_mode_label: str,
    cell_size: int,
    buttons: list,
    progress_fill: tuple[int, int, int],
    cleaned_fill: tuple[int, int, int],
    success_fill: tuple[int, int, int],
    epsilon_fill: tuple[int, int, int],
) -> tuple[pygame.Rect, int]:
    screen.fill(BACKGROUND_COLOR)
    mouse_pos = pygame.mouse.get_pos()

    title_surface = title_font(fonts).render("SweepAgent Training", True, TEXT_COLOR)
    title_rect = title_surface.get_rect(center=(width // 2, 38))
    screen.blit(title_surface, title_rect)

    subtitle_surface = small_font(fonts).render(
        f"Map: {map_name} | Algorithm: {model_name} | Episodes: {episodes}",
        True,
        MUTED_TEXT_COLOR,
    )
    subtitle_rect = subtitle_surface.get_rect(center=(width // 2, 66))
    screen.blit(subtitle_surface, subtitle_rect)

    metrics_rect = pygame.Rect(40, TRAINING_TOP_Y, width - 80, TRAINING_METRICS_HEIGHT)
    log_rect = pygame.Rect(40, metrics_rect.bottom + TRAINING_GAP, width - 80, TRAINING_LOG_HEIGHT)
    graph_rect = pygame.Rect(40, log_rect.bottom + TRAINING_GAP, width - 80, TRAINING_GRAPH_HEIGHT)

    draw_training_metrics_panel(screen, fonts, metrics_rect, latest_metrics)
    max_scroll = draw_training_log_panel(screen, fonts, log_rect, log_lines, log_scroll_offset)
    draw_training_graph_panel(
        screen,
        fonts,
        graph_rect,
        latest_metrics,
        progress_fill,
        cleaned_fill,
        success_fill,
        epsilon_fill,
    )

    for button in buttons:
        draw_button(screen, button, fonts, mouse_pos)

    return log_rect, max_scroll


def draw_playback_title_bar(
    screen: pygame.Surface,
    width: int,
    fonts,
    title_text: str,
    subtitle_text: str | None = None,
) -> None:
    title_surface = title_font(fonts).render(title_text, True, TEXT_COLOR)
    title_rect = title_surface.get_rect(center=(width // 2, 30))
    screen.blit(title_surface, title_rect)

    if subtitle_text:
        subtitle_surface = small_font(fonts).render(subtitle_text, True, MUTED_TEXT_COLOR)
        subtitle_rect = subtitle_surface.get_rect(center=(width // 2, 58))
        screen.blit(subtitle_surface, subtitle_rect)


def draw_playback_control_bar(
    screen: pygame.Surface,
    width: int,
    fonts,
    buttons: list,
) -> None:
    control_rect = pygame.Rect(0, PLAYBACK_TITLE_AREA_HEIGHT, width, PLAYBACK_CONTROL_BAR_HEIGHT)
    pygame.draw.rect(screen, CONTROL_BAR_BG, control_rect)
    pygame.draw.line(screen, PANEL_BORDER_COLOR, (0, control_rect.bottom - 1), (width, control_rect.bottom - 1), 1)

    mouse_pos = pygame.mouse.get_pos()
    for button in buttons:
        draw_button(screen, button, fonts, mouse_pos)


def draw_single_playback_screen(
    screen: pygame.Surface,
    width: int,
    height: int,
    fonts,
    title_text: str,
    env,
    total_reward: float,
    step_idx: int,
    done: bool,
    is_paused: bool,
    step_delay: float,
    visit_counts: Dict[Tuple[int, int], int],
    path_history: List[Tuple[int, int]],
    cell_size: int,
    buttons: list,
) -> None:
    screen.fill(BACKGROUND_COLOR)

    subtitle_text = f"State: {'paused' if is_paused else ('done' if done else 'running')} | Delay: {step_delay:.1f}s"
    draw_playback_title_bar(screen, width, fonts, title_text, subtitle_text)
    draw_playback_control_bar(screen, width, fonts, buttons)

    panel_rect = pygame.Rect(20, PLAYBACK_TOP_RESERVED, width - 40, height - PLAYBACK_TOP_RESERVED - 170)
    pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=12)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=12)

    section_surface = section_font(fonts).render("SweepAgent UI Demo", True, TEXT_COLOR)
    screen.blit(section_surface, (panel_rect.left + 12, panel_rect.top + 12))

    left_lines = [
        f"Map: {get_env_map_name(env)}",
        f"Step: {step_idx}",
        f"Reward: {total_reward:.0f}",
    ]
    right_lines = [
        f"Cleaned: {count_cleaned_tiles(env)}/{total_dirty_tiles(env)}",
        f"Battery: {get_battery_text(env)}",
        f"Status: {'paused' if is_paused else ('done' if done else 'running')}",
        f"Delay: {step_delay:.1f}s",
    ]

    left_x = panel_rect.left + 14
    info_y = panel_rect.top + 52
    for i, line in enumerate(left_lines):
        surface = small_font(fonts).render(line, True, TEXT_COLOR)
        screen.blit(surface, (left_x, info_y + i * 24))

    right_x = panel_rect.right - 16
    for i, line in enumerate(right_lines):
        surface = small_font(fonts).render(line, True, TEXT_COLOR)
        rect = surface.get_rect(topright=(right_x, info_y + i * 24))
        screen.blit(surface, rect)

    grid_top = panel_rect.top + 118
    grid_left = panel_rect.left + 14
    grid_rect = draw_grid(
        screen=screen,
        env=env,
        top_left=(grid_left, grid_top),
        cell_size=cell_size,
        fonts=fonts,
        visit_counts=visit_counts,
        path_history=path_history,
    )

    legend_rect = pygame.Rect(20, height - 140, width - 40, 120)
    hover_text = get_hover_text(env, grid_rect, cell_size, pygame.mouse.get_pos())
    draw_legend(screen, fonts, legend_rect, hover_text)


def draw_compare_playback_screen(
    screen: pygame.Surface,
    width: int,
    height: int,
    fonts,
    random_env,
    learned_env,
    random_reward: float,
    learned_reward: float,
    random_step: int,
    learned_step: int,
    random_done: bool,
    learned_done: bool,
    is_paused: bool,
    step_delay: float,
    random_visits: Dict[Tuple[int, int], int],
    learned_visits: Dict[Tuple[int, int], int],
    random_path: List[Tuple[int, int]],
    learned_path: List[Tuple[int, int]],
    cell_size: int,
    buttons: list,
) -> None:
    screen.fill(BACKGROUND_COLOR)

    subtitle_text = f"State: {'paused' if is_paused else 'running'} | Delay: {step_delay:.1f}s"
    draw_playback_title_bar(screen, width, fonts, "SweepAgent Comparison", subtitle_text)
    draw_playback_control_bar(screen, width, fonts, buttons)

    panel_gap = 20
    panel_width = (width - 40 - panel_gap) // 2
    panel_height = height - PLAYBACK_TOP_RESERVED - 170

    left_panel = pygame.Rect(20, PLAYBACK_TOP_RESERVED, panel_width, panel_height)
    right_panel = pygame.Rect(left_panel.right + panel_gap, PLAYBACK_TOP_RESERVED, panel_width, panel_height)

    for panel_rect, title in [(left_panel, "Random Agent"), (right_panel, "Learned Greedy Agent")]:
        pygame.draw.rect(screen, PANEL_BG, panel_rect, border_radius=12)
        pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=12)
        title_surface = section_font(fonts).render(title, True, TEXT_COLOR)
        screen.blit(title_surface, (panel_rect.left + 12, panel_rect.top + 12))

    left_info = [
        f"Map: {get_env_map_name(random_env)}",
        f"Step: {random_step}",
        f"Reward: {random_reward:.0f}",
        f"Cleaned: {count_cleaned_tiles(random_env)}/{total_dirty_tiles(random_env)}",
    ]
    right_info = [
        f"Map: {get_env_map_name(learned_env)}",
        f"Step: {learned_step}",
        f"Reward: {learned_reward:.0f}",
        f"Cleaned: {count_cleaned_tiles(learned_env)}/{total_dirty_tiles(learned_env)}",
    ]

    for i, line in enumerate(left_info):
        surface = small_font(fonts).render(line, True, TEXT_COLOR)
        screen.blit(surface, (left_panel.left + 12, left_panel.top + 48 + i * 22))

    for i, line in enumerate(right_info):
        surface = small_font(fonts).render(line, True, TEXT_COLOR)
        screen.blit(surface, (right_panel.left + 12, right_panel.top + 48 + i * 22))

    compare_cell_size = max(18, min(cell_size, 42))

    left_grid_rect = draw_grid(
        screen=screen,
        env=random_env,
        top_left=(left_panel.left + 12, left_panel.top + 138),
        cell_size=compare_cell_size,
        fonts=fonts,
        visit_counts=random_visits,
        path_history=random_path,
    )
    right_grid_rect = draw_grid(
        screen=screen,
        env=learned_env,
        top_left=(right_panel.left + 12, right_panel.top + 138),
        cell_size=compare_cell_size,
        fonts=fonts,
        visit_counts=learned_visits,
        path_history=learned_path,
    )

    hover_text = get_hover_text(random_env, left_grid_rect, compare_cell_size, pygame.mouse.get_pos())
    if hover_text is None:
        hover_text = get_hover_text(learned_env, right_grid_rect, compare_cell_size, pygame.mouse.get_pos())

    legend_rect = pygame.Rect(20, height - 140, width - 40, 120)
    draw_legend(screen, fonts, legend_rect, hover_text)

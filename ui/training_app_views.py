from __future__ import annotations

from typing import Dict, List, Tuple

import pygame

from utils.ui_utils import (
    BACKGROUND_COLOR,
    INFO_BG_COLOR,
    MUTED_TEXT_COLOR,
    PANEL_BORDER_COLOR,
    TEXT_COLOR,
    WINDOW_PADDING_X,
    build_left_status_lines,
    build_right_status_lines,
    compute_bottom_info_height,
    compute_panel_layout,
    draw_bottom_info,
    draw_compare_panel,
    draw_single_panel,
    get_hover_info,
)
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

    text_surface = fonts.small_font.render(button.label, True, text_color)
    text_rect = text_surface.get_rect(center=button.rect.center)
    screen.blit(text_surface, text_rect)


def draw_dropdown_box(
    screen: pygame.Surface,
    fonts,
    label: str,
    value: str,
    rect: pygame.Rect,
    is_open: bool,
) -> None:
    label_surface = fonts.small_font.render(label, True, TEXT_COLOR)
    screen.blit(label_surface, (rect.left, rect.top - 24))

    box_bg = BUTTON_ACTIVE_BG if is_open else BUTTON_BG
    pygame.draw.rect(screen, box_bg, rect, border_radius=8)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, rect, width=2, border_radius=8)

    value_surface = fonts.small_font.render(value, True, TEXT_COLOR)
    value_rect = value_surface.get_rect(midleft=(rect.left + 12, rect.centery))
    screen.blit(value_surface, value_rect)

    arrow_surface = fonts.small_font.render("▲" if is_open else "▼", True, TEXT_COLOR)
    arrow_rect = arrow_surface.get_rect(center=(rect.right - 20, rect.centery))
    screen.blit(arrow_surface, arrow_rect)


def draw_dropdown_overlay(
    screen: pygame.Surface,
    fonts,
    rect: pygame.Rect,
    options: list[str],
    mouse_pos: tuple[int, int],
) -> list[pygame.Rect]:
    option_height = 34
    dropdown_rect = pygame.Rect(
        rect.left,
        rect.bottom + 8,
        rect.width,
        len(options) * option_height + 8,
    )

    pygame.draw.rect(screen, DROPDOWN_BG, dropdown_rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, dropdown_rect, width=2, border_radius=10)

    option_rects: list[pygame.Rect] = []
    for idx, option in enumerate(options):
        option_rect = pygame.Rect(
            dropdown_rect.left + 6,
            dropdown_rect.top + 4 + idx * option_height,
            dropdown_rect.width - 12,
            option_height - 2,
        )
        hovered = option_rect.collidepoint(mouse_pos)
        if hovered:
            pygame.draw.rect(screen, DROPDOWN_HOVER_BG, option_rect, border_radius=6)

        text_surface = fonts.small_font.render(option, True, TEXT_COLOR)
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
    episodes: int,
    step_delay: float,
    train_seed: int,
    playback_seed: int,
    learning_rate: float,
    discount_factor: float,
    epsilon_start: float,
    epsilon_decay: float,
    epsilon_min: float,
    open_dropdown: str | None,
    buttons: list,
    map_options: list[str],
    model_options: list[str],
    episode_options: list[int],
    delay_options: list[float],
    train_seed_options: list[int],
    playback_seed_options: list[int],
    learning_rate_options: list[float],
    discount_factor_options: list[float],
    epsilon_start_options: list[float],
    epsilon_decay_options: list[float],
    epsilon_min_options: list[float],
    result_view_options: list[str],
) -> dict[str, pygame.Rect | list[pygame.Rect]]:
    screen.fill(BACKGROUND_COLOR)
    mouse_pos = pygame.mouse.get_pos()

    title = fonts.title_font.render("SweepAgent Training App", True, TEXT_COLOR)
    title_rect = title.get_rect(center=(width // 2, 42))
    screen.blit(title, title_rect)

    help_text = fonts.small_font.render(
        "Choose map, algorithm, seeds, and training hyperparameters from the control panel.",
        True,
        MUTED_TEXT_COLOR,
    )
    help_rect = help_text.get_rect(center=(width // 2, 74))
    screen.blit(help_text, help_rect)

    panel_rect = pygame.Rect(50, 110, width - 100, height - 180)
    pygame.draw.rect(screen, INFO_BG_COLOR, panel_rect, border_radius=12)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=12)

    col1_x = panel_rect.left + 32
    col2_x = panel_rect.left + 380
    col3_x = panel_rect.left + 728

    row1_y = panel_rect.top + 60
    row2_y = panel_rect.top + 150
    row3_y = panel_rect.top + 240
    row4_y = panel_rect.top + 330

    field_w = 300
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

    draw_dropdown_box(screen, fonts, "Episodes", str(episodes), episodes_rect, open_dropdown == "episodes")
    draw_dropdown_box(screen, fonts, "Train Seed", str(train_seed), train_seed_rect, open_dropdown == "train_seed")
    draw_dropdown_box(screen, fonts, "Playback Seed", str(playback_seed), playback_seed_rect, open_dropdown == "playback_seed")

    draw_dropdown_box(screen, fonts, "Learning Rate", f"{learning_rate:.3f}", lr_rect, open_dropdown == "learning_rate")
    draw_dropdown_box(screen, fonts, "Discount Factor", f"{discount_factor:.3f}", gamma_rect, open_dropdown == "discount_factor")
    draw_dropdown_box(screen, fonts, "Epsilon Start", f"{epsilon_start:.2f}", eps_start_rect, open_dropdown == "epsilon_start")

    draw_dropdown_box(screen, fonts, "Epsilon Decay", f"{epsilon_decay:.3f}", eps_decay_rect, open_dropdown == "epsilon_decay")
    draw_dropdown_box(screen, fonts, "Epsilon Min", f"{epsilon_min:.2f}", eps_min_rect, open_dropdown == "epsilon_min")
    draw_dropdown_box(screen, fonts, "Playback Delay", f"{step_delay:.1f}s", delay_rect, open_dropdown == "delay")

    notes = [
        "q_learning: train from the selected seed and hyperparameters, then switch to playback",
        "random_baseline: skip training and preview a random rollout directly",
        "Large maps usually need longer training and slower epsilon decay",
    ]
    note_y = panel_rect.top + 455
    for note in notes:
        note_surface = fonts.small_font.render(note, True, MUTED_TEXT_COLOR)
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
        "episodes_options": [],
        "train_seed_options": [],
        "playback_seed_options": [],
        "learning_rate_options": [],
        "discount_factor_options": [],
        "epsilon_start_options": [],
        "epsilon_decay_options": [],
        "epsilon_min_options": [],
        "delay_options": [],
    }

    if open_dropdown == "map":
        option_rects["map_options"] = draw_dropdown_overlay(screen, fonts, map_rect, map_options, mouse_pos)
    elif open_dropdown == "model":
        option_rects["model_options"] = draw_dropdown_overlay(screen, fonts, model_rect, model_options, mouse_pos)
    elif open_dropdown == "result_view":
        option_rects["result_options"] = draw_dropdown_overlay(screen, fonts, result_rect, result_view_options, mouse_pos)
    elif open_dropdown == "episodes":
        option_rects["episodes_options"] = draw_dropdown_overlay(screen, fonts, episodes_rect, [str(x) for x in episode_options], mouse_pos)
    elif open_dropdown == "train_seed":
        option_rects["train_seed_options"] = draw_dropdown_overlay(screen, fonts, train_seed_rect, [str(x) for x in train_seed_options], mouse_pos)
    elif open_dropdown == "playback_seed":
        option_rects["playback_seed_options"] = draw_dropdown_overlay(screen, fonts, playback_seed_rect, [str(x) for x in playback_seed_options], mouse_pos)
    elif open_dropdown == "learning_rate":
        option_rects["learning_rate_options"] = draw_dropdown_overlay(screen, fonts, lr_rect, [f"{x:.3f}" for x in learning_rate_options], mouse_pos)
    elif open_dropdown == "discount_factor":
        option_rects["discount_factor_options"] = draw_dropdown_overlay(screen, fonts, gamma_rect, [f"{x:.3f}" for x in discount_factor_options], mouse_pos)
    elif open_dropdown == "epsilon_start":
        option_rects["epsilon_start_options"] = draw_dropdown_overlay(screen, fonts, eps_start_rect, [f"{x:.2f}" for x in epsilon_start_options], mouse_pos)
    elif open_dropdown == "epsilon_decay":
        option_rects["epsilon_decay_options"] = draw_dropdown_overlay(screen, fonts, eps_decay_rect, [f"{x:.3f}" for x in epsilon_decay_options], mouse_pos)
    elif open_dropdown == "epsilon_min":
        option_rects["epsilon_min_options"] = draw_dropdown_overlay(screen, fonts, eps_min_rect, [f"{x:.2f}" for x in epsilon_min_options], mouse_pos)
    elif open_dropdown == "delay":
        option_rects["delay_options"] = draw_dropdown_overlay(screen, fonts, delay_rect, [f"{x:.1f}s" for x in delay_options], mouse_pos)

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
    pygame.draw.rect(screen, INFO_BG_COLOR, panel_rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=10)

    lines = [
        f"Episode: {latest_metrics.get('episode', '-')}/{latest_metrics.get('total', '-')}",
        f"Avg reward: {latest_metrics.get('avg_reward', '-')}",
        f"Avg cleaned ratio: {latest_metrics.get('avg_cleaned_ratio', '-')}",
        f"Success rate: {latest_metrics.get('success_rate', '-')}",
        f"Epsilon: {latest_metrics.get('epsilon', '-')}",
    ]

    y = panel_rect.top + 18
    for text in lines:
        surface = fonts.body_font.render(text, True, TEXT_COLOR)
        screen.blit(surface, (panel_rect.left + 20, y))
        y += 30


def draw_training_graph_panel(
    screen: pygame.Surface,
    fonts,
    panel_rect: pygame.Rect,
    latest_metrics: dict[str, str],
    progress_fill,
    cleaned_fill,
    success_fill,
    epsilon_fill,
) -> None:
    pygame.draw.rect(screen, INFO_BG_COLOR, panel_rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=10)

    title = fonts.body_font.render("Training Snapshot", True, TEXT_COLOR)
    screen.blit(title, (panel_rect.left + 16, panel_rect.top + 12))

    total = latest_metrics.get("total")
    episode = latest_metrics.get("episode")
    cleaned = latest_metrics.get("avg_cleaned_ratio", "0%")
    success = latest_metrics.get("success_rate", "0%")
    epsilon = latest_metrics.get("epsilon", "0")

    progress = 0.0
    if total not in (None, "-", "") and episode not in (None, "-", ""):
        try:
            progress = float(episode) / max(1.0, float(total))
        except ValueError:
            progress = 0.0

    try:
        cleaned_value = float(cleaned.replace("%", "")) / 100.0
    except ValueError:
        cleaned_value = 0.0

    try:
        success_value = float(success.replace("%", "")) / 100.0
    except ValueError:
        success_value = 0.0

    try:
        epsilon_value = min(1.0, max(0.0, float(epsilon)))
    except ValueError:
        epsilon_value = 0.0

    rows = [
        ("Episode Progress", progress, progress_fill),
        ("Cleaned Ratio", cleaned_value, cleaned_fill),
        ("Success Rate", success_value, success_fill),
        ("Epsilon", epsilon_value, epsilon_fill),
    ]

    y = panel_rect.top + 48
    for label, value, color in rows:
        label_surface = fonts.small_font.render(label, True, TEXT_COLOR)
        screen.blit(label_surface, (panel_rect.left + 16, y))

        bar_rect = pygame.Rect(panel_rect.left + 180, y + 2, panel_rect.width - 250, 18)
        draw_progress_bar(screen, bar_rect, value, color)

        value_surface = fonts.small_font.render(f"{value * 100:.0f}%", True, MUTED_TEXT_COLOR)
        value_rect = value_surface.get_rect(midright=(panel_rect.right - 16, y + 11))
        screen.blit(value_surface, value_rect)

        y += 34


def draw_log_panel(
    screen: pygame.Surface,
    fonts,
    panel_rect: pygame.Rect,
    log_lines: list[str],
    log_scroll_offset: int,
) -> tuple[pygame.Rect, int]:
    pygame.draw.rect(screen, INFO_BG_COLOR, panel_rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=10)

    title = fonts.body_font.render("Training Log", True, TEXT_COLOR)
    screen.blit(title, (panel_rect.left + 16, panel_rect.top + 12))

    content_top = panel_rect.top + 46
    content_height = panel_rect.height - 62
    line_height = 18
    visible_count = max(1, content_height // line_height)

    max_scroll = max(0, len(log_lines) - visible_count)
    log_scroll_offset = max(0, min(log_scroll_offset, max_scroll))
    visible_lines = log_lines[log_scroll_offset:log_scroll_offset + visible_count]

    y = content_top
    for line in visible_lines:
        surface = fonts.small_font.render(line, True, MUTED_TEXT_COLOR)
        screen.blit(surface, (panel_rect.left + 16, y))
        y += line_height

    track_rect = pygame.Rect(panel_rect.right - 18, content_top, 10, content_height)
    pygame.draw.rect(screen, SCROLL_TRACK, track_rect, border_radius=6)

    thumb_height = max(28, int(content_height * min(1.0, visible_count / max(1, len(log_lines) if log_lines else 1))))
    thumb_range = max(0, content_height - thumb_height)
    thumb_top = track_rect.top
    if max_scroll > 0:
        thumb_top += int((log_scroll_offset / max_scroll) * thumb_range)
    thumb_rect = pygame.Rect(track_rect.left, thumb_top, track_rect.width, thumb_height)
    pygame.draw.rect(screen, SCROLL_THUMB, thumb_rect, border_radius=6)

    return track_rect, max_scroll


def draw_training_preview_panel(
    screen: pygame.Surface,
    fonts,
    panel_rect: pygame.Rect,
    env,
    total_reward: float,
    step_idx: int,
    visit_counts: Dict[Tuple[int, int], int],
    path_history: List[Tuple[int, int]],
    cell_size: int,
    preview_mode_label: str,
) -> pygame.Rect:
    pygame.draw.rect(screen, INFO_BG_COLOR, panel_rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=10)

    title = fonts.body_font.render("Mini Rollout Preview", True, TEXT_COLOR)
    screen.blit(title, (panel_rect.left + 16, panel_rect.top + 12))

    mode_surface = fonts.small_font.render(f"Mode: {preview_mode_label}", True, MUTED_TEXT_COLOR)
    mode_rect = mode_surface.get_rect(topright=(panel_rect.right - 16, panel_rect.top + 16))
    screen.blit(mode_surface, mode_rect)

    from utils.ui_utils import draw_grid, draw_path_overlay, draw_robot

    preview_cell_size = cell_size
    max_grid_width = panel_rect.width - 32
    max_grid_height = panel_rect.height - 84

    if env.cols * preview_cell_size > max_grid_width:
        preview_cell_size = max(20, max_grid_width // env.cols)
    if env.rows * preview_cell_size > max_grid_height:
        preview_cell_size = max(20, min(preview_cell_size, max_grid_height // env.rows))

    grid_width = env.cols * preview_cell_size
    grid_height = env.rows * preview_cell_size
    grid_left = panel_rect.left + (panel_rect.width - grid_width) // 2
    grid_top = panel_rect.top + 52

    draw_grid(
        screen=screen,
        env=env,
        cell_size=preview_cell_size,
        grid_top=grid_top,
        grid_left=grid_left,
        body_font=fonts.small_font,
        visit_counts=visit_counts,
    )
    draw_path_overlay(
        screen=screen,
        path_history=path_history,
        cell_size=preview_cell_size,
        grid_top=grid_top,
        grid_left=grid_left,
    )
    draw_robot(
        screen=screen,
        env=env,
        cell_size=preview_cell_size,
        grid_top=grid_top,
        grid_left=grid_left,
    )

    bottom_text = fonts.small_font.render(
        f"Step {step_idx} | Reward {total_reward:.0f}",
        True,
        MUTED_TEXT_COLOR,
    )
    screen.blit(bottom_text, (panel_rect.left + 16, panel_rect.bottom - 28))

    return pygame.Rect(grid_left, grid_top, grid_width, grid_height)


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
    preview_visits: Dict[Tuple[int, int], int],
    preview_path: List[Tuple[int, int]],
    preview_mode_label: str,
    cell_size: int,
    buttons: list,
    progress_fill,
    cleaned_fill,
    success_fill,
    epsilon_fill,
) -> tuple[pygame.Rect, int]:
    screen.fill(BACKGROUND_COLOR)
    mouse_pos = pygame.mouse.get_pos()

    title = fonts.title_font.render("Training in Progress", True, TEXT_COLOR)
    screen.blit(title, (WINDOW_PADDING_X, 20))

    subtitle = fonts.small_font.render(
        f"Map: {map_name} | Model: {model_name} | Episodes: {episodes}",
        True,
        MUTED_TEXT_COLOR,
    )
    screen.blit(subtitle, (WINDOW_PADDING_X, 58))

    metrics_panel = pygame.Rect(24, TRAINING_TOP_Y, width - 48, TRAINING_METRICS_HEIGHT)
    draw_training_metrics_panel(screen, fonts, metrics_panel, latest_metrics)

    logs_panel = pygame.Rect(24, metrics_panel.bottom + TRAINING_GAP, width - 48, TRAINING_LOG_HEIGHT)
    track_rect, max_scroll = draw_log_panel(screen, fonts, logs_panel, log_lines, log_scroll_offset)

    graph_panel = pygame.Rect(24, logs_panel.bottom + TRAINING_GAP, width - 48, TRAINING_GRAPH_HEIGHT)
    draw_training_graph_panel(
        screen,
        fonts,
        graph_panel,
        latest_metrics,
        progress_fill,
        cleaned_fill,
        success_fill,
        epsilon_fill,
    )

    preview_panel = pygame.Rect(24, graph_panel.bottom + TRAINING_GAP, width - 48, TRAINING_PREVIEW_HEIGHT)
    draw_training_preview_panel(
        screen,
        fonts,
        preview_panel,
        preview_env,
        preview_reward,
        preview_step,
        preview_visits,
        preview_path,
        cell_size,
        preview_mode_label,
    )

    for button in buttons:
        draw_button(screen, button, fonts, mouse_pos)

    return track_rect, max_scroll


def draw_playback_control_bar(
    screen: pygame.Surface,
    width: int,
    fonts,
    step_delay: float,
    is_paused: bool,
    buttons: list,
) -> None:
    title = fonts.title_font.render("SweepAgent Playback", True, TEXT_COLOR)
    title_rect = title.get_rect(center=(width // 2, 28))
    screen.blit(title, title_rect)

    subtitle = fonts.small_font.render(
        f"State: {'paused' if is_paused else 'running'} | Delay: {step_delay:.1f}s",
        True,
        MUTED_TEXT_COLOR,
    )
    subtitle_rect = subtitle.get_rect(center=(width // 2, 56))
    screen.blit(subtitle, subtitle_rect)

    bar_rect = pygame.Rect(0, PLAYBACK_TITLE_AREA_HEIGHT, width, PLAYBACK_CONTROL_BAR_HEIGHT)
    pygame.draw.rect(screen, CONTROL_BAR_BG, bar_rect)
    pygame.draw.line(screen, PANEL_BORDER_COLOR, (0, bar_rect.bottom), (width, bar_rect.bottom), 2)

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

    draw_playback_control_bar(screen, width, fonts, step_delay, is_paused, buttons)

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

    panel_left = (width - layout.panel_width) // 2
    panel_top = PLAYBACK_TOP_RESERVED

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

    hover_lines = get_hover_info(
        env=env,
        visit_counts=visit_counts,
        grid_rect=grid_rect,
        mouse_pos=pygame.mouse.get_pos(),
        cell_size=cell_size,
    )
    hover_title = f"Hover: {title_text}" if hover_lines is not None else None

    draw_bottom_info(
        screen=screen,
        width=width,
        height=height,
        fonts=fonts,
        hover_title=hover_title,
        hover_lines=hover_lines,
        bottom_info_height=compute_bottom_info_height(fonts),
    )


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

    draw_playback_control_bar(screen, width, fonts, step_delay, is_paused, buttons)

    preview_left_lines = [
        ("Step: 9999", TEXT_COLOR),
        ("Reward: 999", TEXT_COLOR),
    ]
    preview_right_lines = [
        (f"Cleaned: {random_env.total_dirty_tiles}/{random_env.total_dirty_tiles}", TEXT_COLOR),
        (
            f"Battery: {random_env.battery_capacity}/{random_env.battery_capacity}"
            if random_env.battery_capacity is not None
            else "Battery: off",
            TEXT_COLOR,
        ),
        ("Status: success", TEXT_COLOR),
    ]

    left_layout = compute_panel_layout(
        env=random_env,
        cell_size=cell_size,
        fonts=fonts,
        left_lines=preview_left_lines,
        right_lines=preview_right_lines,
        title_text="Random Agent",
        subtitle_text=f"Map: {getattr(random_env, 'map_name', 'unknown')}",
    )
    right_layout = compute_panel_layout(
        env=learned_env,
        cell_size=cell_size,
        fonts=fonts,
        left_lines=preview_left_lines,
        right_lines=preview_right_lines,
        title_text="Learned Greedy Agent",
        subtitle_text=f"Map: {getattr(learned_env, 'map_name', 'unknown')}",
    )

    panel_width = max(left_layout.panel_width, right_layout.panel_width)
    panel_gap = 24
    panel_top = PLAYBACK_TOP_RESERVED
    left_panel_left = WINDOW_PADDING_X
    right_panel_left = WINDOW_PADDING_X + panel_width + panel_gap

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

    hover_title = None
    hover_lines = None

    random_hover = get_hover_info(
        env=random_env,
        visit_counts=random_visits,
        grid_rect=random_grid_rect,
        mouse_pos=pygame.mouse.get_pos(),
        cell_size=cell_size,
    )
    learned_hover = get_hover_info(
        env=learned_env,
        visit_counts=learned_visits,
        grid_rect=learned_grid_rect,
        mouse_pos=pygame.mouse.get_pos(),
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
        bottom_info_height=compute_bottom_info_height(fonts),
    )
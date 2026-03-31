from __future__ import annotations

import argparse
import queue
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import pygame

# Add project root so local modules can be imported when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from configs.map_presets import MAP_PRESETS
from utils.experiment_utils import build_env, load_or_train_q_agent
from utils.ui_utils import (
    BACKGROUND_COLOR,
    INFO_BG_COLOR,
    MUTED_TEXT_COLOR,
    PANEL_BORDER_COLOR,
    TEXT_COLOR,
    WINDOW_PADDING_X,
    WINDOW_PADDING_Y,
    MAX_STEP_DELAY,
    MIN_STEP_DELAY,
    STEP_DELAY_DELTA,
    build_left_status_lines,
    build_right_status_lines,
    compute_bottom_info_height,
    compute_panel_layout,
    draw_bottom_info,
    draw_compare_panel,
    draw_single_panel,
    get_hover_info,
    load_fonts,
    reset_panel_state,
)

# Supported training / preview modes.
MODEL_OPTIONS = [
    "q_learning",
    "random_baseline",
]

# Result display mode after training.
RESULT_VIEW_OPTIONS = [
    "single_playback",
    "compare_playback",
]

# Fixed menu options for now.
EPISODE_OPTIONS = [500, 1000, 2000, 3000, 5000]
STEP_DELAY_OPTIONS = [0.1, 0.2, 0.3, 0.5, 0.8]

# Parse training progress lines emitted by scripts/train_q_learning.py.
TRAINING_LINE_PATTERN = re.compile(
    r"Episode\s+(\d+)/(\d+)\s+\|\s+avg_reward=([-\d.]+)\s+\|\s+avg_cleaned_ratio=([\d.]+)%\s+\|\s+success_rate=([\d.]+)%\s+\|\s+epsilon=([\d.]+)"
)

# UI colors for buttons / dropdowns / bars.
BUTTON_BG = (255, 255, 255)
BUTTON_HOVER_BG = (240, 244, 248)
BUTTON_ACTIVE_BG = (230, 239, 255)
PRIMARY_BUTTON_BG = (58, 134, 255)
PRIMARY_BUTTON_HOVER_BG = (47, 117, 223)
PRIMARY_BUTTON_TEXT = (255, 255, 255)

CONTROL_BAR_BG = (250, 251, 253)
DROPDOWN_BG = (255, 255, 255)
DROPDOWN_HOVER_BG = (242, 246, 252)

PROGRESS_BG = (230, 235, 241)
PROGRESS_FILL = (58, 134, 255)
SUCCESS_FILL = (25, 135, 84)
CLEANED_FILL = (255, 193, 7)
EPSILON_FILL = (111, 66, 193)

SCROLL_TRACK = (236, 240, 244)
SCROLL_THUMB = (173, 181, 189)

# Base menu size.
MENU_WIDTH = 1040
MENU_HEIGHT = 820

# Playback layout reserves a title band and a control bar.
PLAYBACK_TITLE_AREA_HEIGHT = 72
PLAYBACK_CONTROL_BAR_HEIGHT = 72
PLAYBACK_TOP_RESERVED = PLAYBACK_TITLE_AREA_HEIGHT + PLAYBACK_CONTROL_BAR_HEIGHT

# Training layout constants.
TRAINING_METRICS_HEIGHT = 180
TRAINING_LOG_HEIGHT = 220
TRAINING_GRAPH_HEIGHT = 180
TRAINING_PREVIEW_HEIGHT = 360

TRAINING_TOP_Y = 96
TRAINING_GAP = 18
TRAINING_BOTTOM_ACTION_HEIGHT = 72
TRAINING_BOTTOM_MARGIN = 24


@dataclass
class Button:
    """Simple clickable button model."""
    rect: pygame.Rect
    label: str
    on_click: Callable[[], None]
    primary: bool = False
    enabled: bool = True


class TrainingRunner:
    """
    Small wrapper around the training subprocess.

    It launches scripts/train_q_learning.py and streams stdout lines into a queue
    so the pygame UI can stay responsive while training is running.
    """

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.reader_thread: threading.Thread | None = None
        self.finished = False
        self.return_code: int | None = None

    def start(self, map_name: str, episodes: int, seed: int) -> None:
        command = [
            sys.executable,
            "scripts/train_q_learning.py",
            "--map-name",
            map_name,
            "--episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--print-every",
            "100",
        ]

        self.process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.finished = False
        self.return_code = None

        def _reader() -> None:
            assert self.process is not None
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.output_queue.put(line.rstrip("\n"))
            self.process.wait()
            self.return_code = self.process.returncode
            self.finished = True

        self.reader_thread = threading.Thread(target=_reader, daemon=True)
        self.reader_thread.start()

    def poll_lines(self) -> list[str]:
        """Drain any currently available stdout lines."""
        lines: list[str] = []
        while True:
            try:
                lines.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return lines

    def terminate(self) -> None:
        """Stop the running training subprocess if it is still alive."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SweepAgent training app.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cell-size", type=int, default=56)
    return parser.parse_args()


def clamp_step_delay(value: float) -> float:
    """Clamp playback speed into the allowed range."""
    return max(MIN_STEP_DELAY, min(MAX_STEP_DELAY, value))


def compute_training_window_height() -> int:
    """
    Compute the window height needed for the training screen only.

    Menu and playback screens keep their own sizes, but training uses a taller
    window so the mini rollout preview is fully visible.
    """
    return (
        TRAINING_TOP_Y
        + TRAINING_METRICS_HEIGHT
        + TRAINING_GAP
        + TRAINING_LOG_HEIGHT
        + TRAINING_GAP
        + TRAINING_GRAPH_HEIGHT
        + TRAINING_GAP
        + TRAINING_PREVIEW_HEIGHT
        + TRAINING_GAP
        + TRAINING_BOTTOM_ACTION_HEIGHT
        + TRAINING_BOTTOM_MARGIN
    )


def draw_button(
    screen: pygame.Surface,
    button: Button,
    fonts,
    mouse_pos: tuple[int, int],
) -> None:
    """Draw a standard or primary button with hover styling."""
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
    """Draw the closed/open dropdown field itself."""
    label_surface = fonts.body_font.render(label, True, TEXT_COLOR)
    screen.blit(label_surface, (rect.left, rect.top - 30))

    box_bg = BUTTON_ACTIVE_BG if is_open else BUTTON_BG
    pygame.draw.rect(screen, box_bg, rect, border_radius=8)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, rect, width=2, border_radius=8)

    value_surface = fonts.body_font.render(value, True, TEXT_COLOR)
    value_rect = value_surface.get_rect(midleft=(rect.left + 16, rect.centery))
    screen.blit(value_surface, value_rect)

    arrow_surface = fonts.body_font.render("▲" if is_open else "▼", True, TEXT_COLOR)
    arrow_rect = arrow_surface.get_rect(center=(rect.right - 24, rect.centery))
    screen.blit(arrow_surface, arrow_rect)


def draw_dropdown_overlay(
    screen: pygame.Surface,
    fonts,
    rect: pygame.Rect,
    options: list[str],
    mouse_pos: tuple[int, int],
) -> list[pygame.Rect]:
    """
    Draw the dropdown menu as an opaque overlay.

    This is intentionally fully opaque so the open menu does not visually mix
    with elements behind it.
    """
    option_height = 38
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
        text_rect = text_surface.get_rect(midleft=(option_rect.left + 12, option_rect.centery))
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
    open_dropdown: str | None,
    buttons: list[Button],
) -> dict[str, pygame.Rect | list[pygame.Rect]]:
    """Draw the selection menu screen."""
    screen.fill(BACKGROUND_COLOR)
    mouse_pos = pygame.mouse.get_pos()

    title = fonts.title_font.render("SweepAgent Training App", True, TEXT_COLOR)
    title_rect = title.get_rect(center=(width // 2, 42))
    screen.blit(title, title_rect)

    help_text = fonts.small_font.render(
        "Click each box to open options. Choose map, model, result view, and start.",
        True,
        MUTED_TEXT_COLOR,
    )
    help_rect = help_text.get_rect(center=(width // 2, 74))
    screen.blit(help_text, help_rect)

    panel_rect = pygame.Rect(70, 120, width - 140, height - 190)
    pygame.draw.rect(screen, INFO_BG_COLOR, panel_rect, border_radius=12)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=12)

    left_x = panel_rect.left + 40
    right_x = panel_rect.left + 520
    row1_y = panel_rect.top + 70
    row2_y = panel_rect.top + 200
    row3_y = panel_rect.top + 330

    map_rect = pygame.Rect(left_x, row1_y, 320, 46)
    model_rect = pygame.Rect(right_x, row1_y, 320, 46)
    episodes_rect = pygame.Rect(left_x, row2_y, 320, 46)
    delay_rect = pygame.Rect(right_x, row2_y, 320, 46)
    result_rect = pygame.Rect(left_x, row3_y, 320, 46)

    draw_dropdown_box(screen, fonts, "Map", map_name, map_rect, open_dropdown == "map")
    draw_dropdown_box(screen, fonts, "Model", model_name, model_rect, open_dropdown == "model")
    draw_dropdown_box(screen, fonts, "Episodes", str(episodes), episodes_rect, open_dropdown == "episodes")
    draw_dropdown_box(screen, fonts, "Playback Delay", f"{step_delay:.1f}s", delay_rect, open_dropdown == "delay")
    draw_dropdown_box(screen, fonts, "Result View", result_view, result_rect, open_dropdown == "result_view")

    notes = [
        "q_learning: train the selected map, show logs, then switch to playback",
        "random_baseline: skip training and directly preview a random rollout",
        "compare_playback: show random vs learned side by side after training",
    ]
    note_y = panel_rect.top + 450
    for note in notes:
        note_surface = fonts.small_font.render(note, True, MUTED_TEXT_COLOR)
        screen.blit(note_surface, (panel_rect.left + 40, note_y))
        note_y += 24

    for button in buttons:
        draw_button(screen, button, fonts, mouse_pos)

    map_option_rects: list[pygame.Rect] = []
    model_option_rects: list[pygame.Rect] = []
    episode_option_rects: list[pygame.Rect] = []
    delay_option_rects: list[pygame.Rect] = []
    result_option_rects: list[pygame.Rect] = []

    if open_dropdown == "map":
        map_option_rects = draw_dropdown_overlay(screen, fonts, map_rect, list(MAP_PRESETS.keys()), mouse_pos)
    elif open_dropdown == "model":
        model_option_rects = draw_dropdown_overlay(screen, fonts, model_rect, MODEL_OPTIONS, mouse_pos)
    elif open_dropdown == "episodes":
        episode_option_rects = draw_dropdown_overlay(screen, fonts, episodes_rect, [str(x) for x in EPISODE_OPTIONS], mouse_pos)
    elif open_dropdown == "delay":
        delay_option_rects = draw_dropdown_overlay(screen, fonts, delay_rect, [f"{x:.1f}s" for x in STEP_DELAY_OPTIONS], mouse_pos)
    elif open_dropdown == "result_view":
        result_option_rects = draw_dropdown_overlay(screen, fonts, result_rect, RESULT_VIEW_OPTIONS, mouse_pos)

    return {
        "map_rect": map_rect,
        "model_rect": model_rect,
        "episodes_rect": episodes_rect,
        "delay_rect": delay_rect,
        "result_rect": result_rect,
        "map_options": map_option_rects,
        "model_options": model_option_rects,
        "episodes_options": episode_option_rects,
        "delay_options": delay_option_rects,
        "result_options": result_option_rects,
    }


def draw_progress_bar(
    screen: pygame.Surface,
    rect: pygame.Rect,
    value: float,
    fill_color: tuple[int, int, int],
) -> None:
    """Draw a single horizontal bar used in the training snapshot panel."""
    value = max(0.0, min(1.0, value))
    pygame.draw.rect(screen, PROGRESS_BG, rect, border_radius=8)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, rect, width=1, border_radius=8)

    fill_rect = pygame.Rect(rect.left, rect.top, int(rect.width * value), rect.height)
    if fill_rect.width > 0:
        pygame.draw.rect(screen, fill_color, fill_rect, border_radius=8)


def draw_training_metrics_panel(
    screen: pygame.Surface,
    fonts,
    panel_rect: pygame.Rect,
    latest_metrics: dict[str, str],
) -> None:
    """Top metrics card for training status."""
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
) -> None:
    """Horizontal training bars for progress and current summary metrics."""
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
        ("Episode Progress", progress, PROGRESS_FILL),
        ("Cleaned Ratio", cleaned_value, CLEANED_FILL),
        ("Success Rate", success_value, SUCCESS_FILL),
        ("Epsilon", epsilon_value, EPSILON_FILL),
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
    """Scrollable log panel with a simple scrollbar on the right."""
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
    done: bool,
    visit_counts: Dict[Tuple[int, int], int],
    path_history: List[Tuple[int, int]],
    cell_size: int,
) -> pygame.Rect:
    """Mini rollout preview shown only on the training screen."""
    pygame.draw.rect(screen, INFO_BG_COLOR, panel_rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=10)

    title = fonts.body_font.render("Mini Rollout Preview", True, TEXT_COLOR)
    screen.blit(title, (panel_rect.left + 16, panel_rect.top + 12))

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
    preview_done: bool,
    preview_visits: Dict[Tuple[int, int], int],
    preview_path: List[Tuple[int, int]],
    cell_size: int,
    buttons: list[Button],
) -> tuple[pygame.Rect, int]:
    """Draw the whole training screen using the larger training-only window height."""
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
    draw_training_metrics_panel(
        screen=screen,
        fonts=fonts,
        panel_rect=metrics_panel,
        latest_metrics=latest_metrics,
    )

    logs_panel = pygame.Rect(24, metrics_panel.bottom + TRAINING_GAP, width - 48, TRAINING_LOG_HEIGHT)
    track_rect, max_scroll = draw_log_panel(
        screen=screen,
        fonts=fonts,
        panel_rect=logs_panel,
        log_lines=log_lines,
        log_scroll_offset=log_scroll_offset,
    )

    graph_panel = pygame.Rect(24, logs_panel.bottom + TRAINING_GAP, width - 48, TRAINING_GRAPH_HEIGHT)
    draw_training_graph_panel(
        screen=screen,
        fonts=fonts,
        panel_rect=graph_panel,
        latest_metrics=latest_metrics,
    )

    preview_panel = pygame.Rect(24, graph_panel.bottom + TRAINING_GAP, width - 48, TRAINING_PREVIEW_HEIGHT)
    draw_training_preview_panel(
        screen=screen,
        fonts=fonts,
        panel_rect=preview_panel,
        env=preview_env,
        total_reward=preview_reward,
        step_idx=preview_step,
        done=preview_done,
        visit_counts=preview_visits,
        path_history=preview_path,
        cell_size=cell_size,
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
    buttons: list[Button],
) -> None:
    """Shared header/control area for single and compare playback screens."""
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
    buttons: list[Button],
) -> None:
    """Single-agent playback screen."""
    screen.fill(BACKGROUND_COLOR)

    draw_playback_control_bar(
        screen=screen,
        width=width,
        fonts=fonts,
        step_delay=step_delay,
        is_paused=is_paused,
        buttons=buttons,
    )

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
    buttons: list[Button],
) -> None:
    """Side-by-side compare playback screen."""
    screen.fill(BACKGROUND_COLOR)

    draw_playback_control_bar(
        screen=screen,
        width=width,
        fonts=fonts,
        step_delay=step_delay,
        is_paused=is_paused,
        buttons=buttons,
    )

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


def main() -> None:
    args = parse_args()

    pygame.init()
    fonts = load_fonts()

    # Current selection state for the menu screen.
    map_name = "charge_required_v2" if "charge_required_v2" in MAP_PRESETS else list(MAP_PRESETS.keys())[0]
    model_name = "q_learning"
    result_view = "single_playback"
    episodes = 5000
    step_delay = 0.5

    # Start on the menu window size.
    screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))
    pygame.display.set_caption("SweepAgent Training App")
    clock = pygame.time.Clock()

    app_state = "menu"
    open_dropdown: str | None = None

    # Training state.
    trainer = TrainingRunner()
    latest_metrics: dict[str, str] = {}
    log_lines: list[str] = []
    log_scroll_offset = 0

    # Mini preview state shown on the training screen.
    training_preview_env = build_env(map_name=map_name)
    training_preview_env.map_name = map_name
    (
        training_preview_state,
        training_preview_done,
        training_preview_reward,
        training_preview_step,
        training_preview_path,
        training_preview_visits,
    ) = reset_panel_state(training_preview_env)
    training_preview_last_step_time = time.time()

    # Single playback state.
    playback_env = None
    playback_agent = None
    playback_model = None
    playback_state = None
    playback_done = False
    playback_reward = 0.0
    playback_step = 0
    playback_path: list[Tuple[int, int]] = []
    playback_visits: dict[Tuple[int, int], int] = {}
    playback_is_paused = False
    playback_last_step_time = time.time()
    playback_rng = random.Random(args.random_seed)

    # Compare playback state.
    compare_random_env = None
    compare_learned_env = None
    compare_learned_agent = None
    compare_random_state = None
    compare_learned_state = None
    compare_random_done = False
    compare_learned_done = False
    compare_random_reward = 0.0
    compare_learned_reward = 0.0
    compare_random_step = 0
    compare_learned_step = 0
    compare_random_path: list[Tuple[int, int]] = []
    compare_learned_path: list[Tuple[int, int]] = []
    compare_random_visits: dict[Tuple[int, int], int] = {}
    compare_learned_visits: dict[Tuple[int, int], int] = {}
    compare_is_paused = False
    compare_last_step_time = time.time()
    compare_rng = random.Random(args.random_seed)

    while True:
        mouse_clicked = False
        mouse_pos = pygame.mouse.get_pos()
        max_scroll = 0

        def rebuild_training_preview_env() -> None:
            """Reset the training mini-preview environment when the map selection changes."""
            nonlocal training_preview_env
            nonlocal training_preview_state, training_preview_done, training_preview_reward, training_preview_step
            nonlocal training_preview_path, training_preview_visits, training_preview_last_step_time

            training_preview_env = build_env(map_name=map_name)
            training_preview_env.map_name = map_name
            (
                training_preview_state,
                training_preview_done,
                training_preview_reward,
                training_preview_step,
                training_preview_path,
                training_preview_visits,
            ) = reset_panel_state(training_preview_env)
            training_preview_last_step_time = time.time()

        def start_action() -> None:
            """Start training or direct preview depending on the selected model."""
            nonlocal app_state, latest_metrics, log_lines, open_dropdown, log_scroll_offset
            nonlocal trainer
            nonlocal playback_env, playback_agent, playback_model
            nonlocal playback_state, playback_done, playback_reward, playback_step, playback_path, playback_visits
            nonlocal playback_is_paused, playback_last_step_time, playback_rng
            nonlocal compare_random_env, compare_learned_env, compare_learned_agent
            nonlocal compare_random_state, compare_learned_state
            nonlocal compare_random_done, compare_learned_done
            nonlocal compare_random_reward, compare_learned_reward
            nonlocal compare_random_step, compare_learned_step
            nonlocal compare_random_path, compare_learned_path
            nonlocal compare_random_visits, compare_learned_visits
            nonlocal compare_is_paused, compare_last_step_time, compare_rng

            open_dropdown = None
            latest_metrics = {}
            log_lines = []
            log_scroll_offset = 0
            rebuild_training_preview_env()

            if model_name == "q_learning":
                trainer = TrainingRunner()
                trainer.start(
                    map_name=map_name,
                    episodes=episodes,
                    seed=args.seed,
                )
                app_state = "training"
            else:
                if result_view == "single_playback":
                    playback_env = build_env(map_name=map_name)
                    playback_env.map_name = map_name
                    (
                        playback_state,
                        playback_done,
                        playback_reward,
                        playback_step,
                        playback_path,
                        playback_visits,
                    ) = reset_panel_state(playback_env)
                    playback_agent = None
                    playback_model = "random_baseline"
                    playback_is_paused = False
                    playback_last_step_time = time.time()
                    playback_rng = random.Random(args.random_seed)
                    app_state = "playback_single"
                else:
                    compare_random_env = build_env(map_name=map_name)
                    compare_learned_env = build_env(map_name=map_name)
                    compare_random_env.map_name = map_name
                    compare_learned_env.map_name = map_name
                    compare_learned_agent = None

                    (
                        compare_random_state,
                        compare_random_done,
                        compare_random_reward,
                        compare_random_step,
                        compare_random_path,
                        compare_random_visits,
                    ) = reset_panel_state(compare_random_env)
                    (
                        compare_learned_state,
                        compare_learned_done,
                        compare_learned_reward,
                        compare_learned_step,
                        compare_learned_path,
                        compare_learned_visits,
                    ) = reset_panel_state(compare_learned_env)

                    compare_is_paused = False
                    compare_last_step_time = time.time()
                    compare_rng = random.Random(args.random_seed)
                    app_state = "playback_compare"

        def cancel_training() -> None:
            """Cancel the running training process and go back to the menu."""
            nonlocal app_state
            trainer.terminate()
            app_state = "menu"

        def back_to_menu() -> None:
            """Return to the menu from any playback screen."""
            nonlocal app_state, open_dropdown
            app_state = "menu"
            open_dropdown = None

        def toggle_single_pause() -> None:
            """Pause/resume single playback."""
            nonlocal playback_is_paused, playback_last_step_time
            playback_is_paused = not playback_is_paused
            playback_last_step_time = time.time()

        def restart_single_playback() -> None:
            """Restart single playback from episode start."""
            nonlocal playback_state, playback_done, playback_reward, playback_step
            nonlocal playback_path, playback_visits, playback_is_paused, playback_last_step_time, playback_rng
            if playback_env is None:
                return
            playback_rng = random.Random(args.random_seed)
            (
                playback_state,
                playback_done,
                playback_reward,
                playback_step,
                playback_path,
                playback_visits,
            ) = reset_panel_state(playback_env)
            playback_is_paused = False
            playback_last_step_time = time.time()

        def toggle_compare_pause() -> None:
            """Pause/resume compare playback."""
            nonlocal compare_is_paused, compare_last_step_time
            compare_is_paused = not compare_is_paused
            compare_last_step_time = time.time()

        def restart_compare_playback() -> None:
            """Restart both compare playback environments."""
            nonlocal compare_random_state, compare_random_done, compare_random_reward, compare_random_step
            nonlocal compare_random_path, compare_random_visits
            nonlocal compare_learned_state, compare_learned_done, compare_learned_reward, compare_learned_step
            nonlocal compare_learned_path, compare_learned_visits
            nonlocal compare_is_paused, compare_last_step_time, compare_rng
            if compare_random_env is None or compare_learned_env is None:
                return
            compare_rng = random.Random(args.random_seed)
            (
                compare_random_state,
                compare_random_done,
                compare_random_reward,
                compare_random_step,
                compare_random_path,
                compare_random_visits,
            ) = reset_panel_state(compare_random_env)
            (
                compare_learned_state,
                compare_learned_done,
                compare_learned_reward,
                compare_learned_step,
                compare_learned_path,
                compare_learned_visits,
            ) = reset_panel_state(compare_learned_env)
            compare_is_paused = False
            compare_last_step_time = time.time()

        def slower() -> None:
            """Make playback slower."""
            nonlocal step_delay
            step_delay = clamp_step_delay(step_delay + STEP_DELAY_DELTA)

        def faster() -> None:
            """Make playback faster."""
            nonlocal step_delay
            step_delay = clamp_step_delay(step_delay - STEP_DELAY_DELTA)

        menu_buttons = [
            Button(
                pygame.Rect(390, 720, 260, 54),
                "Start Training" if model_name == "q_learning" else "Start Preview",
                start_action,
                primary=True,
            )
        ]

        training_height = compute_training_window_height()
        training_buttons = [
            Button(
                pygame.Rect(MENU_WIDTH - 220, training_height - 60, 180, 44),
                "Cancel Training",
                cancel_training,
            )
        ]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                trainer.terminate()
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                trainer.terminate()
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_clicked = True

            if event.type == pygame.MOUSEWHEEL and app_state == "training":
                log_scroll_offset -= event.y
                log_scroll_offset = max(0, min(log_scroll_offset, max_scroll))

        # --- Background updates per state ------------------------------------

        if app_state == "training":
            # Consume training logs from the subprocess.
            for line in trainer.poll_lines():
                log_lines.append(line)
                match = TRAINING_LINE_PATTERN.search(line)
                if match:
                    latest_metrics = {
                        "episode": match.group(1),
                        "total": match.group(2),
                        "avg_reward": match.group(3),
                        "avg_cleaned_ratio": match.group(4) + "%",
                        "success_rate": match.group(5) + "%",
                        "epsilon": match.group(6),
                    }

            # When training finishes, switch to the requested playback mode.
            if trainer.finished:
                if trainer.return_code == 0:
                    if result_view == "single_playback":
                        playback_env = build_env(map_name=map_name)
                        playback_env.map_name = map_name
                        playback_agent = load_or_train_q_agent(
                            map_name=map_name,
                            num_episodes=episodes,
                            seed=args.seed,
                        )
                        playback_model = "q_learning"
                        (
                            playback_state,
                            playback_done,
                            playback_reward,
                            playback_step,
                            playback_path,
                            playback_visits,
                        ) = reset_panel_state(playback_env)
                        playback_is_paused = False
                        playback_last_step_time = time.time()
                        app_state = "playback_single"
                    else:
                        compare_random_env = build_env(map_name=map_name)
                        compare_learned_env = build_env(map_name=map_name)
                        compare_random_env.map_name = map_name
                        compare_learned_env.map_name = map_name
                        compare_learned_agent = load_or_train_q_agent(
                            map_name=map_name,
                            num_episodes=episodes,
                            seed=args.seed,
                        )

                        (
                            compare_random_state,
                            compare_random_done,
                            compare_random_reward,
                            compare_random_step,
                            compare_random_path,
                            compare_random_visits,
                        ) = reset_panel_state(compare_random_env)
                        (
                            compare_learned_state,
                            compare_learned_done,
                            compare_learned_reward,
                            compare_learned_step,
                            compare_learned_path,
                            compare_learned_visits,
                        ) = reset_panel_state(compare_learned_env)

                        compare_is_paused = False
                        compare_last_step_time = time.time()
                        compare_rng = random.Random(args.random_seed)
                        app_state = "playback_compare"
                else:
                    app_state = "menu"

            # Keep the mini preview moving independently while training runs.
            now = time.time()
            if training_preview_env is not None and now - training_preview_last_step_time >= 0.35:
                action = random.randrange(len(training_preview_env.ACTIONS))
                (
                    training_preview_state,
                    reward,
                    training_preview_done,
                    _,
                ) = training_preview_env.step(action)
                training_preview_reward += reward
                training_preview_step += 1
                training_preview_path.append(training_preview_env.robot_pos)
                training_preview_visits[training_preview_env.robot_pos] = training_preview_visits.get(training_preview_env.robot_pos, 0) + 1
                training_preview_last_step_time = now

                if training_preview_done:
                    rebuild_training_preview_env()

        if app_state == "playback_single" and playback_env is not None:
            now = time.time()
            if not playback_done and not playback_is_paused and now - playback_last_step_time >= step_delay:
                if playback_model == "q_learning" and playback_agent is not None:
                    action = playback_agent.get_policy_action(playback_state)
                else:
                    action = playback_rng.randrange(len(playback_env.ACTIONS))

                (
                    playback_state,
                    reward,
                    playback_done,
                    _,
                ) = playback_env.step(action)
                playback_reward += reward
                playback_step += 1
                playback_path.append(playback_env.robot_pos)
                playback_visits[playback_env.robot_pos] = playback_visits.get(playback_env.robot_pos, 0) + 1
                playback_last_step_time = now

        if app_state == "playback_compare" and compare_random_env is not None and compare_learned_env is not None:
            now = time.time()
            if not compare_is_paused and now - compare_last_step_time >= step_delay:
                if not compare_random_done:
                    random_action = compare_rng.randrange(len(compare_random_env.ACTIONS))
                    (
                        compare_random_state,
                        reward,
                        compare_random_done,
                        _,
                    ) = compare_random_env.step(random_action)
                    compare_random_reward += reward
                    compare_random_step += 1
                    compare_random_path.append(compare_random_env.robot_pos)
                    compare_random_visits[compare_random_env.robot_pos] = compare_random_visits.get(compare_random_env.robot_pos, 0) + 1

                if not compare_learned_done:
                    if compare_learned_agent is not None:
                        learned_action = compare_learned_agent.get_policy_action(compare_learned_state)
                    else:
                        learned_action = compare_rng.randrange(len(compare_learned_env.ACTIONS))
                    (
                        compare_learned_state,
                        reward,
                        compare_learned_done,
                        _,
                    ) = compare_learned_env.step(learned_action)
                    compare_learned_reward += reward
                    compare_learned_step += 1
                    compare_learned_path.append(compare_learned_env.robot_pos)
                    compare_learned_visits[compare_learned_env.robot_pos] = compare_learned_visits.get(compare_learned_env.robot_pos, 0) + 1

                compare_last_step_time = now

        # --- Rendering -------------------------------------------------------

        if app_state == "menu":
            if screen.get_width() != MENU_WIDTH or screen.get_height() != MENU_HEIGHT:
                screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))

            dropdown_ui = draw_menu(
                screen=screen,
                width=MENU_WIDTH,
                height=MENU_HEIGHT,
                fonts=fonts,
                map_name=map_name,
                model_name=model_name,
                result_view=result_view,
                episodes=episodes,
                step_delay=step_delay,
                open_dropdown=open_dropdown,
                buttons=menu_buttons,
            )

            if mouse_clicked:
                handled = False

                for button in menu_buttons:
                    if button.rect.collidepoint(mouse_pos):
                        button.on_click()
                        handled = True
                        break

                if not handled:
                    dropdown_boxes = {
                        "map": dropdown_ui["map_rect"],
                        "model": dropdown_ui["model_rect"],
                        "episodes": dropdown_ui["episodes_rect"],
                        "delay": dropdown_ui["delay_rect"],
                        "result_view": dropdown_ui["result_rect"],
                    }

                    for key, rect in dropdown_boxes.items():
                        if isinstance(rect, pygame.Rect) and rect.collidepoint(mouse_pos):
                            open_dropdown = None if open_dropdown == key else key
                            handled = True
                            break

                if not handled and open_dropdown is not None:
                    option_key = {
                        "map": "map_options",
                        "model": "model_options",
                        "episodes": "episodes_options",
                        "delay": "delay_options",
                        "result_view": "result_options",
                    }[open_dropdown]

                    option_rects = dropdown_ui[option_key]
                    if isinstance(option_rects, list):
                        clicked_option = False
                        for idx, rect in enumerate(option_rects):
                            if rect.collidepoint(mouse_pos):
                                clicked_option = True
                                if open_dropdown == "map":
                                    map_name = list(MAP_PRESETS.keys())[idx]
                                    rebuild_training_preview_env()
                                elif open_dropdown == "model":
                                    model_name = MODEL_OPTIONS[idx]
                                elif open_dropdown == "episodes":
                                    episodes = EPISODE_OPTIONS[idx]
                                elif open_dropdown == "delay":
                                    step_delay = STEP_DELAY_OPTIONS[idx]
                                elif open_dropdown == "result_view":
                                    result_view = RESULT_VIEW_OPTIONS[idx]
                                open_dropdown = None
                                handled = True
                                break

                        if not clicked_option:
                            open_dropdown = None

        elif app_state == "training":
            # Training screen uses a taller window than the menu.
            if screen.get_width() != MENU_WIDTH or screen.get_height() != training_height:
                screen = pygame.display.set_mode((MENU_WIDTH, training_height))

            _, max_scroll = draw_training_screen(
                screen=screen,
                width=MENU_WIDTH,
                height=training_height,
                fonts=fonts,
                map_name=map_name,
                model_name=model_name,
                episodes=episodes,
                latest_metrics=latest_metrics,
                log_lines=log_lines,
                log_scroll_offset=log_scroll_offset,
                preview_env=training_preview_env,
                preview_reward=training_preview_reward,
                preview_step=training_preview_step,
                preview_done=training_preview_done,
                preview_visits=training_preview_visits,
                preview_path=training_preview_path,
                cell_size=args.cell_size,
                buttons=training_buttons,
            )

            if mouse_clicked:
                for button in training_buttons:
                    if button.rect.collidepoint(mouse_pos):
                        button.on_click()
                        break

        elif app_state == "playback_single" and playback_env is not None:
            preview_left_lines = build_left_status_lines(
                map_name=map_name,
                total_reward=playback_reward,
                step_idx=playback_step,
                include_map_line=True,
            )
            preview_right_lines = build_right_status_lines(
                env=playback_env,
                done=playback_done,
                is_paused=playback_is_paused,
                step_delay=step_delay,
            )
            preview_layout = compute_panel_layout(
                env=playback_env,
                cell_size=args.cell_size,
                fonts=fonts,
                left_lines=preview_left_lines,
                right_lines=preview_right_lines,
                title_text="SweepAgent UI Demo",
                subtitle_text=None,
            )
            bottom_info_height = compute_bottom_info_height(fonts)
            playback_width = WINDOW_PADDING_X + preview_layout.panel_width + WINDOW_PADDING_X
            playback_height = PLAYBACK_TOP_RESERVED + preview_layout.total_height + bottom_info_height + WINDOW_PADDING_Y

            if screen.get_width() != playback_width or screen.get_height() != playback_height:
                screen = pygame.display.set_mode((playback_width, playback_height))

            playback_buttons = [
                Button(pygame.Rect(24, 86, 120, 40), "Pause" if not playback_is_paused else "Resume", toggle_single_pause),
                Button(pygame.Rect(156, 86, 120, 40), "Restart", restart_single_playback),
                Button(pygame.Rect(288, 86, 110, 40), "Slower", slower),
                Button(pygame.Rect(410, 86, 110, 40), "Faster", faster),
                Button(pygame.Rect(playback_width - 160, 86, 136, 40), "Back to Menu", back_to_menu),
            ]

            draw_single_playback_screen(
                screen=screen,
                width=playback_width,
                height=playback_height,
                fonts=fonts,
                title_text="Learned Greedy Agent" if playback_model == "q_learning" else "Random Baseline",
                env=playback_env,
                total_reward=playback_reward,
                step_idx=playback_step,
                done=playback_done,
                is_paused=playback_is_paused,
                step_delay=step_delay,
                visit_counts=playback_visits,
                path_history=playback_path,
                cell_size=args.cell_size,
                buttons=playback_buttons,
            )

            if mouse_clicked:
                for button in playback_buttons:
                    if button.rect.collidepoint(mouse_pos):
                        button.on_click()
                        break

        elif app_state == "playback_compare" and compare_random_env is not None and compare_learned_env is not None:
            preview_left_lines = [
                ("Step: 9999", TEXT_COLOR),
                ("Reward: 999", TEXT_COLOR),
            ]
            preview_right_lines = [
                (f"Cleaned: {compare_random_env.total_dirty_tiles}/{compare_random_env.total_dirty_tiles}", TEXT_COLOR),
                (
                    f"Battery: {compare_random_env.battery_capacity}/{compare_random_env.battery_capacity}"
                    if compare_random_env.battery_capacity is not None
                    else "Battery: off",
                    TEXT_COLOR,
                ),
                ("Status: success", TEXT_COLOR),
            ]

            left_layout = compute_panel_layout(
                env=compare_random_env,
                cell_size=args.cell_size,
                fonts=fonts,
                left_lines=preview_left_lines,
                right_lines=preview_right_lines,
                title_text="Random Agent",
                subtitle_text=f"Map: {map_name}",
            )
            right_layout = compute_panel_layout(
                env=compare_learned_env,
                cell_size=args.cell_size,
                fonts=fonts,
                left_lines=preview_left_lines,
                right_lines=preview_right_lines,
                title_text="Learned Greedy Agent",
                subtitle_text=f"Map: {map_name}",
            )

            panel_width = max(left_layout.panel_width, right_layout.panel_width)
            panel_gap = 24
            bottom_info_height = compute_bottom_info_height(fonts)
            compare_width = WINDOW_PADDING_X + panel_width + panel_gap + panel_width + WINDOW_PADDING_X
            compare_height = PLAYBACK_TOP_RESERVED + max(left_layout.total_height, right_layout.total_height) + bottom_info_height + WINDOW_PADDING_Y

            if screen.get_width() != compare_width or screen.get_height() != compare_height:
                screen = pygame.display.set_mode((compare_width, compare_height))

            playback_buttons = [
                Button(pygame.Rect(24, 86, 120, 40), "Pause" if not compare_is_paused else "Resume", toggle_compare_pause),
                Button(pygame.Rect(156, 86, 120, 40), "Restart", restart_compare_playback),
                Button(pygame.Rect(288, 86, 110, 40), "Slower", slower),
                Button(pygame.Rect(410, 86, 110, 40), "Faster", faster),
                Button(pygame.Rect(compare_width - 160, 86, 136, 40), "Back to Menu", back_to_menu),
            ]

            draw_compare_playback_screen(
                screen=screen,
                width=compare_width,
                height=compare_height,
                fonts=fonts,
                random_env=compare_random_env,
                learned_env=compare_learned_env,
                random_reward=compare_random_reward,
                learned_reward=compare_learned_reward,
                random_step=compare_random_step,
                learned_step=compare_learned_step,
                random_done=compare_random_done,
                learned_done=compare_learned_done,
                is_paused=compare_is_paused,
                step_delay=step_delay,
                random_visits=compare_random_visits,
                learned_visits=compare_learned_visits,
                random_path=compare_random_path,
                learned_path=compare_learned_path,
                cell_size=args.cell_size,
                buttons=playback_buttons,
            )

            if mouse_clicked:
                for button in playback_buttons:
                    if button.rect.collidepoint(mouse_pos):
                        button.on_click()
                        break

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
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
    TOP_PANEL_HEIGHT,
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
    draw_single_panel,
    draw_top_controls,
    get_hover_info,
    load_fonts,
    reset_panel_state,
)

MODEL_OPTIONS = [
    "q_learning",
    "random_baseline",
]

EPISODE_OPTIONS = [500, 1000, 2000, 3000, 5000]
STEP_DELAY_OPTIONS = [0.1, 0.2, 0.3, 0.5, 0.8]

TRAINING_LINE_PATTERN = re.compile(
    r"Episode\s+(\d+)/(\d+)\s+\|\s+avg_reward=([-\d.]+)\s+\|\s+avg_cleaned_ratio=([\d.]+)%\s+\|\s+success_rate=([\d.]+)%\s+\|\s+epsilon=([\d.]+)"
)

BUTTON_BG = (255, 255, 255)
BUTTON_HOVER_BG = (240, 244, 248)
BUTTON_ACTIVE_BG = (230, 239, 255)
PRIMARY_BUTTON_BG = (58, 134, 255)
PRIMARY_BUTTON_HOVER_BG = (47, 117, 223)
PRIMARY_BUTTON_TEXT = (255, 255, 255)


@dataclass
class Button:
    rect: pygame.Rect
    label: str
    on_click: Callable[[], None]
    primary: bool = False
    enabled: bool = True


class TrainingRunner:
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
        lines: list[str] = []
        while True:
            try:
                lines.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return lines

    def terminate(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SweepAgent training app.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cell-size", type=int, default=64)
    return parser.parse_args()


def clamp_step_delay(value: float) -> float:
    return max(MIN_STEP_DELAY, min(MAX_STEP_DELAY, value))


def next_index(current: int, total: int, delta: int) -> int:
    return (current + delta) % total


def draw_button(
    screen: pygame.Surface,
    button: Button,
    fonts,
    mouse_pos: tuple[int, int],
) -> None:
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


def draw_menu(
    screen: pygame.Surface,
    width: int,
    height: int,
    fonts,
    map_name: str,
    model_name: str,
    episodes: int,
    step_delay: float,
    buttons: list[Button],
    row_arrow_rects: list[tuple[pygame.Rect, pygame.Rect]],
) -> None:
    screen.fill(BACKGROUND_COLOR)
    mouse_pos = pygame.mouse.get_pos()

    title = fonts.title_font.render("SweepAgent Training App", True, TEXT_COLOR)
    title_rect = title.get_rect(center=(width // 2, 42))
    screen.blit(title, title_rect)

    help_text = fonts.small_font.render(
        "Click arrows to change options, then start training or preview.",
        True,
        MUTED_TEXT_COLOR,
    )
    help_rect = help_text.get_rect(center=(width // 2, 74))
    screen.blit(help_text, help_rect)

    panel_rect = pygame.Rect(
        70,
        120,
        width - 140,
        height - 210,
    )
    pygame.draw.rect(screen, INFO_BG_COLOR, panel_rect, border_radius=12)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=12)

    items = [
        ("Map", map_name),
        ("Model", model_name),
        ("Episodes", str(episodes)),
        ("Playback Delay", f"{step_delay:.1f}s"),
    ]

    row_top = panel_rect.top + 34
    row_gap = 64

    for idx, ((label, value), (left_rect, right_rect)) in enumerate(zip(items, row_arrow_rects)):
        y = row_top + idx * row_gap

        label_surface = fonts.body_font.render(label, True, TEXT_COLOR)
        screen.blit(label_surface, (panel_rect.left + 28, y + 8))

        value_box = pygame.Rect(panel_rect.left + 280, y, 360, 42)
        pygame.draw.rect(screen, BUTTON_ACTIVE_BG, value_box, border_radius=8)
        pygame.draw.rect(screen, PANEL_BORDER_COLOR, value_box, width=2, border_radius=8)

        value_surface = fonts.body_font.render(value, True, TEXT_COLOR)
        value_rect = value_surface.get_rect(center=value_box.center)
        screen.blit(value_surface, value_rect)

        for rect, text in ((left_rect, "<"), (right_rect, ">")):
            hovered = rect.collidepoint(mouse_pos)
            bg = BUTTON_HOVER_BG if hovered else BUTTON_BG
            pygame.draw.rect(screen, bg, rect, border_radius=8)
            pygame.draw.rect(screen, PANEL_BORDER_COLOR, rect, width=2, border_radius=8)

            arrow_surface = fonts.body_font.render(text, True, TEXT_COLOR)
            arrow_rect = arrow_surface.get_rect(center=rect.center)
            screen.blit(arrow_surface, arrow_rect)

    notes = [
        "q_learning: train the selected map, then auto-play the learned policy",
        "random_baseline: skip training and directly preview a random rollout",
        "This is the first click-based app UI stage for SweepAgent.",
    ]
    note_y = panel_rect.top + 308
    for note in notes:
        note_surface = fonts.small_font.render(note, True, MUTED_TEXT_COLOR)
        screen.blit(note_surface, (panel_rect.left + 28, note_y))
        note_y += 24

    for button in buttons:
        draw_button(screen, button, fonts, mouse_pos)


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
    buttons: list[Button],
) -> None:
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

    metrics_panel = pygame.Rect(24, 96, width - 48, 120)
    pygame.draw.rect(screen, INFO_BG_COLOR, metrics_panel, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, metrics_panel, width=2, border_radius=10)

    metric_texts = [
        f"Episode: {latest_metrics.get('episode', '-')}/{latest_metrics.get('total', '-')}",
        f"Avg reward: {latest_metrics.get('avg_reward', '-')}",
        f"Avg cleaned ratio: {latest_metrics.get('avg_cleaned_ratio', '-')}",
        f"Success rate: {latest_metrics.get('success_rate', '-')}",
        f"Epsilon: {latest_metrics.get('epsilon', '-')}",
    ]

    y = metrics_panel.top + 16
    for text in metric_texts:
        surface = fonts.body_font.render(text, True, TEXT_COLOR)
        screen.blit(surface, (metrics_panel.left + 18, y))
        y += 20

    logs_panel = pygame.Rect(24, 236, width - 48, height - 330)
    pygame.draw.rect(screen, INFO_BG_COLOR, logs_panel, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, logs_panel, width=2, border_radius=10)

    logs_title = fonts.body_font.render("Training Log", True, TEXT_COLOR)
    screen.blit(logs_title, (logs_panel.left + 16, logs_panel.top + 12))

    y = logs_panel.top + 46
    visible_lines = log_lines[-16:]
    for line in visible_lines:
        surface = fonts.small_font.render(line, True, MUTED_TEXT_COLOR)
        screen.blit(surface, (logs_panel.left + 16, y))
        y += 18

    for button in buttons:
        draw_button(screen, button, fonts, mouse_pos)


def draw_playback_screen(
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
    screen.fill(BACKGROUND_COLOR)
    mouse_pos = pygame.mouse.get_pos()

    draw_top_controls(
        screen=screen,
        width=width,
        step_delay=step_delay,
        is_paused=is_paused,
        fonts=fonts,
        title_text=title_text,
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
    panel_top = TOP_PANEL_HEIGHT

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

    for button in buttons:
        draw_button(screen, button, fonts, mouse_pos)


def main() -> None:
    args = parse_args()

    pygame.init()
    fonts = load_fonts()

    map_names = list(MAP_PRESETS.keys())
    model_names = MODEL_OPTIONS

    map_idx = map_names.index("charge_required_v2") if "charge_required_v2" in map_names else 0
    model_idx = 0
    episodes_idx = EPISODE_OPTIONS.index(5000) if 5000 in EPISODE_OPTIONS else 0
    step_delay_idx = STEP_DELAY_OPTIONS.index(0.5) if 0.5 in STEP_DELAY_OPTIONS else 0

    menu_width = 980
    menu_height = 640

    screen = pygame.display.set_mode((menu_width, menu_height))
    pygame.display.set_caption("SweepAgent Training App")
    clock = pygame.time.Clock()

    app_state = "menu"
    trainer = TrainingRunner()
    latest_metrics: dict[str, str] = {}
    log_lines: list[str] = []

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

    while True:
        current_map = map_names[map_idx]
        current_model = model_names[model_idx]
        current_episodes = EPISODE_OPTIONS[episodes_idx]
        current_step_delay = STEP_DELAY_OPTIONS[step_delay_idx]

        def change_map(delta: int) -> None:
            nonlocal map_idx
            map_idx = next_index(map_idx, len(map_names), delta)

        def change_model(delta: int) -> None:
            nonlocal model_idx
            model_idx = next_index(model_idx, len(model_names), delta)

        def change_episodes(delta: int) -> None:
            nonlocal episodes_idx
            episodes_idx = next_index(episodes_idx, len(EPISODE_OPTIONS), delta)

        def change_delay(delta: int) -> None:
            nonlocal step_delay_idx
            step_delay_idx = next_index(step_delay_idx, len(STEP_DELAY_OPTIONS), delta)

        def start_action() -> None:
            nonlocal app_state, log_lines, latest_metrics
            nonlocal playback_env, playback_agent, playback_model
            nonlocal playback_state, playback_done, playback_reward, playback_step
            nonlocal playback_path, playback_visits, playback_is_paused, playback_last_step_time
            nonlocal trainer, playback_rng

            log_lines = []
            latest_metrics = {}
            playback_rng = random.Random(args.random_seed)

            if current_model == "q_learning":
                trainer = TrainingRunner()
                trainer.start(
                    map_name=current_map,
                    episodes=current_episodes,
                    seed=args.seed,
                )
                app_state = "training"
            else:
                playback_env = build_env(map_name=current_map)
                playback_env.map_name = current_map
                playback_state, playback_done, playback_reward, playback_step, playback_path, playback_visits = reset_panel_state(playback_env)
                playback_agent = None
                playback_model = "random_baseline"
                playback_is_paused = False
                playback_last_step_time = time.time()
                app_state = "playback"

        def cancel_training() -> None:
            nonlocal app_state
            trainer.terminate()
            app_state = "menu"

        def toggle_pause() -> None:
            nonlocal playback_is_paused, playback_last_step_time
            playback_is_paused = not playback_is_paused
            playback_last_step_time = time.time()

        def restart_playback() -> None:
            nonlocal playback_state, playback_done, playback_reward, playback_step
            nonlocal playback_path, playback_visits, playback_is_paused, playback_last_step_time
            nonlocal playback_rng
            if playback_env is None:
                return
            playback_rng = random.Random(args.random_seed)
            playback_state, playback_done, playback_reward, playback_step, playback_path, playback_visits = reset_panel_state(playback_env)
            playback_is_paused = False
            playback_last_step_time = time.time()

        def slower() -> None:
            nonlocal step_delay_idx
            new_delay = clamp_step_delay(current_step_delay + STEP_DELAY_DELTA)
            step_delay_idx = min(
                range(len(STEP_DELAY_OPTIONS)),
                key=lambda i: abs(STEP_DELAY_OPTIONS[i] - new_delay),
            )

        def faster() -> None:
            nonlocal step_delay_idx
            new_delay = clamp_step_delay(current_step_delay - STEP_DELAY_DELTA)
            step_delay_idx = min(
                range(len(STEP_DELAY_OPTIONS)),
                key=lambda i: abs(STEP_DELAY_OPTIONS[i] - new_delay),
            )

        def back_to_menu() -> None:
            nonlocal app_state
            app_state = "menu"

        panel_rect = pygame.Rect(70, 120, menu_width - 140, menu_height - 210)
        row_top = panel_rect.top + 34
        row_gap = 64

        row_arrow_rects = []
        for idx in range(4):
            y = row_top + idx * row_gap
            value_box_left = panel_rect.left + 280
            left_rect = pygame.Rect(value_box_left - 52, y, 42, 42)
            right_rect = pygame.Rect(value_box_left + 360 + 10, y, 42, 42)
            row_arrow_rects.append((left_rect, right_rect))

        menu_buttons: list[Button] = [
            Button(row_arrow_rects[0][0], "<", lambda: change_map(-1)),
            Button(row_arrow_rects[0][1], ">", lambda: change_map(1)),
            Button(row_arrow_rects[1][0], "<", lambda: change_model(-1)),
            Button(row_arrow_rects[1][1], ">", lambda: change_model(1)),
            Button(row_arrow_rects[2][0], "<", lambda: change_episodes(-1)),
            Button(row_arrow_rects[2][1], ">", lambda: change_episodes(1)),
            Button(row_arrow_rects[3][0], "<", lambda: change_delay(-1)),
            Button(row_arrow_rects[3][1], ">", lambda: change_delay(1)),
            Button(
                pygame.Rect(230, 500, 240, 52),
                "Start Training" if current_model == "q_learning" else "Start Preview",
                start_action,
                primary=True,
            ),
        ]

        training_buttons: list[Button] = [
            Button(
                pygame.Rect(menu_width - 220, menu_height - 76, 180, 44),
                "Cancel Training",
                cancel_training,
            )
        ]

        mouse_clicked = False
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                trainer.terminate()
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    trainer.terminate()
                    pygame.quit()
                    return

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_clicked = True

        if mouse_clicked:
            active_buttons: list[Button]
            if app_state == "menu":
                active_buttons = menu_buttons
            elif app_state == "training":
                active_buttons = training_buttons
            else:
                active_buttons = []

            for button in active_buttons:
                if button.enabled and button.rect.collidepoint(mouse_pos):
                    button.on_click()
                    break

        if app_state == "training":
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

            if trainer.finished:
                if trainer.return_code == 0:
                    playback_env = build_env(map_name=current_map)
                    playback_env.map_name = current_map
                    playback_agent = load_or_train_q_agent(
                        map_name=current_map,
                        num_episodes=current_episodes,
                        seed=args.seed,
                    )
                    playback_model = "q_learning"
                    playback_state, playback_done, playback_reward, playback_step, playback_path, playback_visits = reset_panel_state(playback_env)
                    playback_is_paused = False
                    playback_last_step_time = time.time()
                    app_state = "playback"
                else:
                    app_state = "menu"

        if app_state == "playback" and playback_env is not None:
            now = time.time()
            if not playback_done and not playback_is_paused and now - playback_last_step_time >= current_step_delay:
                if playback_model == "q_learning" and playback_agent is not None:
                    action = playback_agent.get_policy_action(playback_state)
                else:
                    action = playback_rng.randrange(len(playback_env.ACTIONS))

                playback_state, reward, playback_done, _ = playback_env.step(action)
                playback_reward += reward
                playback_step += 1
                playback_path.append(playback_env.robot_pos)
                playback_visits[playback_env.robot_pos] = playback_visits.get(playback_env.robot_pos, 0) + 1
                playback_last_step_time = now

        if app_state == "menu":
            if screen.get_width() != menu_width or screen.get_height() != menu_height:
                screen = pygame.display.set_mode((menu_width, menu_height))

            draw_menu(
                screen=screen,
                width=menu_width,
                height=menu_height,
                fonts=fonts,
                map_name=current_map,
                model_name=current_model,
                episodes=current_episodes,
                step_delay=current_step_delay,
                buttons=menu_buttons,
                row_arrow_rects=row_arrow_rects,
            )

        elif app_state == "training":
            if screen.get_width() != menu_width or screen.get_height() != menu_height:
                screen = pygame.display.set_mode((menu_width, menu_height))

            draw_training_screen(
                screen=screen,
                width=menu_width,
                height=menu_height,
                fonts=fonts,
                map_name=current_map,
                model_name=current_model,
                episodes=current_episodes,
                latest_metrics=latest_metrics,
                log_lines=log_lines,
                buttons=training_buttons,
            )

        elif app_state == "playback" and playback_env is not None:
            preview_left_lines = build_left_status_lines(
                map_name=current_map,
                total_reward=playback_reward,
                step_idx=playback_step,
                include_map_line=True,
            )
            preview_right_lines = build_right_status_lines(
                env=playback_env,
                done=playback_done,
                is_paused=playback_is_paused,
                step_delay=current_step_delay,
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
            playback_height = TOP_PANEL_HEIGHT + preview_layout.total_height + bottom_info_height + WINDOW_PADDING_Y

            if screen.get_width() != playback_width or screen.get_height() != playback_height:
                screen = pygame.display.set_mode((playback_width, playback_height))

            playback_buttons = [
                Button(pygame.Rect(24, 24, 120, 40), "Pause" if not playback_is_paused else "Resume", toggle_pause),
                Button(pygame.Rect(156, 24, 120, 40), "Restart", restart_playback),
                Button(pygame.Rect(288, 24, 110, 40), "Slower", slower),
                Button(pygame.Rect(410, 24, 110, 40), "Faster", faster),
                Button(pygame.Rect(playback_width - 160, 24, 136, 40), "Back to Menu", back_to_menu),
            ]

            if mouse_clicked:
                for button in playback_buttons:
                    if button.enabled and button.rect.collidepoint(mouse_pos):
                        button.on_click()
                        break

            draw_playback_screen(
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
                step_delay=current_step_delay,
                visit_counts=playback_visits,
                path_history=playback_path,
                cell_size=args.cell_size,
                buttons=playback_buttons,
            )

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
import queue
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
    PANEL_GAP,
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
    draw_compare_panel,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SweepAgent training app.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cell-size", type=int, default=64)
    return parser.parse_args()


def clamp_step_delay(value: float) -> float:
    return max(MIN_STEP_DELAY, min(MAX_STEP_DELAY, value))


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


def draw_menu(
    screen: pygame.Surface,
    width: int,
    height: int,
    fonts,
    selected_index: int,
    map_name: str,
    model_name: str,
    episodes: int,
    step_delay: float,
) -> None:
    screen.fill(BACKGROUND_COLOR)

    title = fonts.title_font.render("SweepAgent Training App", True, TEXT_COLOR)
    title_rect = title.get_rect(center=(width // 2, 40))
    screen.blit(title, title_rect)

    help_text = fonts.small_font.render(
        "UP/DOWN: move  LEFT/RIGHT: change option  ENTER: start  ESC: quit",
        True,
        TEXT_COLOR,
    )
    help_rect = help_text.get_rect(center=(width // 2, 72))
    screen.blit(help_text, help_rect)

    panel_rect = pygame.Rect(
        80,
        120,
        width - 160,
        height - 220,
    )
    pygame.draw.rect(screen, INFO_BG_COLOR, panel_rect, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, panel_rect, width=2, border_radius=10)

    items = [
        ("Map", map_name),
        ("Model", model_name),
        ("Episodes", str(episodes)),
        ("Playback Delay", f"{step_delay:.1f}s"),
    ]

    y = panel_rect.top + 40
    for idx, (label, value) in enumerate(items):
        color = TEXT_COLOR if idx == selected_index else MUTED_TEXT_COLOR
        prefix = ">" if idx == selected_index else " "
        line = f"{prefix} {label}: {value}"
        surface = fonts.body_font.render(line, True, color)
        screen.blit(surface, (panel_rect.left + 40, y))
        y += 50

    tips = [
        "q_learning: train selected map, then auto-play the learned policy",
        "random_baseline: skip training and directly preview a random rollout",
        "This is the first app-style UI stage for SweepAgent.",
    ]

    y += 20
    for tip in tips:
        surface = fonts.small_font.render(tip, True, MUTED_TEXT_COLOR)
        screen.blit(surface, (panel_rect.left + 40, y))
        y += 24


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
) -> None:
    screen.fill(BACKGROUND_COLOR)

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

    y = metrics_panel.top + 18
    for text in metric_texts:
        surface = fonts.body_font.render(text, True, TEXT_COLOR)
        screen.blit(surface, (metrics_panel.left + 18, y))
        y += 20

    logs_panel = pygame.Rect(24, 236, width - 48, height - 260)
    pygame.draw.rect(screen, INFO_BG_COLOR, logs_panel, border_radius=10)
    pygame.draw.rect(screen, PANEL_BORDER_COLOR, logs_panel, width=2, border_radius=10)

    logs_title = fonts.body_font.render("Training Log", True, TEXT_COLOR)
    screen.blit(logs_title, (logs_panel.left + 16, logs_panel.top + 12))

    y = logs_panel.top + 46
    visible_lines = log_lines[-18:]
    for line in visible_lines:
        surface = fonts.small_font.render(line, True, MUTED_TEXT_COLOR)
        screen.blit(surface, (logs_panel.left + 16, y))
        y += 18


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
) -> None:
    screen.fill(BACKGROUND_COLOR)

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
        title_text=title_text,
        subtitle_text=None,
    )

    bottom_info_height = compute_bottom_info_height(fonts)

    panel_left = (width - layout.panel_width) // 2
    panel_top = TOP_PANEL_HEIGHT

    grid_rect = draw_compare_panel(
        screen=screen,
        panel_title=title_text,
        env=env,
        total_reward=total_reward,
        step_idx=step_idx,
        done=done,
        visit_counts=visit_counts,
        path_history=path_history,
        panel_left=panel_left,
        panel_top=panel_top,
        cell_size=cell_size,
        fonts=fonts,
    )[0]

    mouse_pos = pygame.mouse.get_pos()
    hover_lines = get_hover_info(
        env=env,
        visit_counts=visit_counts,
        grid_rect=grid_rect,
        mouse_pos=mouse_pos,
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
        bottom_info_height=bottom_info_height,
    )


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
    selected_menu_idx = 0

    menu_width = 980
    menu_height = 620

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

                if app_state == "menu":
                    if event.key == pygame.K_UP:
                        selected_menu_idx = max(0, selected_menu_idx - 1)
                    elif event.key == pygame.K_DOWN:
                        selected_menu_idx = min(3, selected_menu_idx + 1)
                    elif event.key == pygame.K_LEFT:
                        if selected_menu_idx == 0:
                            map_idx = (map_idx - 1) % len(map_names)
                        elif selected_menu_idx == 1:
                            model_idx = (model_idx - 1) % len(model_names)
                        elif selected_menu_idx == 2:
                            episodes_idx = (episodes_idx - 1) % len(EPISODE_OPTIONS)
                        elif selected_menu_idx == 3:
                            step_delay_idx = (step_delay_idx - 1) % len(STEP_DELAY_OPTIONS)
                    elif event.key == pygame.K_RIGHT:
                        if selected_menu_idx == 0:
                            map_idx = (map_idx + 1) % len(map_names)
                        elif selected_menu_idx == 1:
                            model_idx = (model_idx + 1) % len(model_names)
                        elif selected_menu_idx == 2:
                            episodes_idx = (episodes_idx + 1) % len(EPISODE_OPTIONS)
                        elif selected_menu_idx == 3:
                            step_delay_idx = (step_delay_idx + 1) % len(STEP_DELAY_OPTIONS)
                    elif event.key == pygame.K_RETURN:
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

                elif app_state == "training":
                    pass

                elif app_state == "playback":
                    if event.key == pygame.K_SPACE:
                        playback_is_paused = not playback_is_paused
                        playback_last_step_time = time.time()
                    elif event.key == pygame.K_r:
                        if playback_env is not None:
                            playback_rng = random.Random(args.random_seed)
                            playback_state, playback_done, playback_reward, playback_step, playback_path, playback_visits = reset_panel_state(playback_env)
                            playback_is_paused = False
                            playback_last_step_time = time.time()
                    elif event.key == pygame.K_LEFTBRACKET:
                        new_delay = clamp_step_delay(current_step_delay + STEP_DELAY_DELTA)
                        nearest_idx = min(
                            range(len(STEP_DELAY_OPTIONS)),
                            key=lambda i: abs(STEP_DELAY_OPTIONS[i] - new_delay),
                        )
                        step_delay_idx = nearest_idx
                    elif event.key == pygame.K_RIGHTBRACKET:
                        new_delay = clamp_step_delay(current_step_delay - STEP_DELAY_DELTA)
                        nearest_idx = min(
                            range(len(STEP_DELAY_OPTIONS)),
                            key=lambda i: abs(STEP_DELAY_OPTIONS[i] - new_delay),
                        )
                        step_delay_idx = nearest_idx
                    elif event.key == pygame.K_BACKSPACE:
                        app_state = "menu"

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
            draw_menu(
                screen=screen,
                width=menu_width,
                height=menu_height,
                fonts=fonts,
                selected_index=selected_menu_idx,
                map_name=current_map,
                model_name=current_model,
                episodes=current_episodes,
                step_delay=current_step_delay,
            )

        elif app_state == "training":
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
            )

        elif app_state == "playback" and playback_env is not None:
            preview_left_lines = build_left_status_lines(
                map_name=current_map,
                total_reward=playback_reward,
                step_idx=playback_step,
                include_map_line=False,
            )
            preview_right_lines = build_right_status_lines(
                env=playback_env,
                done=playback_done,
                is_paused=playback_is_paused,
                step_delay=current_step_delay,
            )
            from utils.ui_utils import compute_panel_layout

            preview_layout = compute_panel_layout(
                env=playback_env,
                cell_size=args.cell_size,
                fonts=fonts,
                left_lines=preview_left_lines,
                right_lines=preview_right_lines,
                title_text="Learned Greedy Agent" if playback_model == "q_learning" else "Random Baseline",
                subtitle_text=f"Map: {current_map}",
            )

            bottom_info_height = compute_bottom_info_height(fonts)
            playback_width = WINDOW_PADDING_X + preview_layout.panel_width + WINDOW_PADDING_X
            playback_height = TOP_PANEL_HEIGHT + preview_layout.total_height + bottom_info_height + WINDOW_PADDING_Y

            if screen.get_width() != playback_width or screen.get_height() != playback_height:
                screen = pygame.display.set_mode((playback_width, playback_height))

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
            )

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
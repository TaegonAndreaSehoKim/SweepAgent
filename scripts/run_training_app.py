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

from configs.map_presets import MAP_PRESETS
from utils.experiment_utils import build_env, load_or_train_q_agent
from utils.ui_utils import (
    WINDOW_PADDING_X,
    build_left_status_lines,
    build_right_status_lines,
    compute_bottom_info_height,
    compute_panel_layout,
    load_fonts,
    reset_panel_state,
)
from ui.training_app_core import (
    Button,
    CLEANED_FILL,
    EPISODE_OPTIONS,
    EPSILON_FILL,
    MENU_HEIGHT,
    MENU_WIDTH,
    MODEL_OPTIONS,
    PLAYBACK_TOP_RESERVED,
    PREVIEW_STEP_INTERVAL_SEC,
    PROGRESS_FILL,
    RESULT_VIEW_OPTIONS,
    STEP_DELAY_OPTIONS,
    SUCCESS_FILL,
    TRAINING_LINE_PATTERN,
    PreviewPolicy,
    TrainingRunner,
    clamp_step_delay,
    compute_training_window_height,
    get_checkpoint_path,
)
from ui.training_app_views import (
    draw_compare_playback_screen,
    draw_menu,
    draw_single_playback_screen,
    draw_training_screen,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SweepAgent training app.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cell-size", type=int, default=56)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pygame.init()
    fonts = load_fonts()

    # Current menu selections.
    map_name = "charge_required_v2" if "charge_required_v2" in MAP_PRESETS else list(MAP_PRESETS.keys())[0]
    model_name = "q_learning"
    result_view = "single_playback"
    episodes = 5000
    step_delay = 0.5

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

    # Training mini preview state.
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
    training_preview_policy = PreviewPolicy(get_checkpoint_path(map_name, args.seed))
    training_preview_mode_label = "random fallback"

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

    def rebuild_training_preview_env() -> None:
        """Reset training preview environment and checkpoint policy."""
        nonlocal training_preview_env
        nonlocal training_preview_state, training_preview_done, training_preview_reward, training_preview_step
        nonlocal training_preview_path, training_preview_visits, training_preview_last_step_time
        nonlocal training_preview_policy, training_preview_mode_label

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
        training_preview_policy = PreviewPolicy(get_checkpoint_path(map_name, args.seed))
        training_preview_mode_label = "random fallback"

    def start_action() -> None:
        """Start training or direct preview depending on selected model."""
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
            trainer.start(map_name=map_name, episodes=episodes, seed=args.seed)
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
        """Cancel the training subprocess and go back to menu."""
        nonlocal app_state
        trainer.terminate()
        app_state = "menu"

    def back_to_menu() -> None:
        """Return to menu from playback screens."""
        nonlocal app_state, open_dropdown
        app_state = "menu"
        open_dropdown = None

    def toggle_single_pause() -> None:
        """Pause/resume single playback."""
        nonlocal playback_is_paused, playback_last_step_time
        playback_is_paused = not playback_is_paused
        playback_last_step_time = time.time()

    def restart_single_playback() -> None:
        """Restart single playback episode."""
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
        """Restart both compare playback episodes."""
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
        """Slow down playback speed."""
        nonlocal step_delay
        step_delay = clamp_step_delay(step_delay + 0.1, 0.1, 1.5)

    def faster() -> None:
        """Speed up playback speed."""
        nonlocal step_delay
        step_delay = clamp_step_delay(step_delay - 0.1, 0.1, 1.5)

    while True:
        mouse_clicked = False
        mouse_pos = pygame.mouse.get_pos()
        max_scroll = 0

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

        # Training background updates.
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

            now = time.time()
            if training_preview_env is not None and now - training_preview_last_step_time >= PREVIEW_STEP_INTERVAL_SEC:
                reloaded = training_preview_policy.maybe_reload()
                if reloaded and training_preview_policy.q_table:
                    training_preview_mode_label = "greedy checkpoint"
                elif not training_preview_policy.q_table:
                    training_preview_mode_label = "random fallback"

                if training_preview_policy.q_table:
                    action = training_preview_policy.get_greedy_action(
                        training_preview_state,
                        len(training_preview_env.ACTIONS),
                    )
                else:
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
                training_preview_visits[training_preview_env.robot_pos] = (
                    training_preview_visits.get(training_preview_env.robot_pos, 0) + 1
                )
                training_preview_last_step_time = now

                if training_preview_done:
                    rebuild_training_preview_env()

        # Single playback background updates.
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

        # Compare playback background updates.
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
                    compare_random_visits[compare_random_env.robot_pos] = (
                        compare_random_visits.get(compare_random_env.robot_pos, 0) + 1
                    )

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
                    compare_learned_visits[compare_learned_env.robot_pos] = (
                        compare_learned_visits.get(compare_learned_env.robot_pos, 0) + 1
                    )

                compare_last_step_time = now

        # Rendering.
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
                map_options=list(MAP_PRESETS.keys()),
                model_options=MODEL_OPTIONS,
                episode_options=EPISODE_OPTIONS,
                delay_options=STEP_DELAY_OPTIONS,
                result_view_options=RESULT_VIEW_OPTIONS,
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
                preview_visits=training_preview_visits,
                preview_path=training_preview_path,
                preview_mode_label=training_preview_mode_label,
                cell_size=args.cell_size,
                buttons=training_buttons,
                progress_fill=PROGRESS_FILL,
                cleaned_fill=CLEANED_FILL,
                success_fill=SUCCESS_FILL,
                epsilon_fill=EPSILON_FILL,
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
            playback_height = PLAYBACK_TOP_RESERVED + preview_layout.total_height + bottom_info_height + 24

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
                ("Step: 9999", (0, 0, 0)),
                ("Reward: 999", (0, 0, 0)),
            ]
            preview_right_lines = [
                (f"Cleaned: {compare_random_env.total_dirty_tiles}/{compare_random_env.total_dirty_tiles}", (0, 0, 0)),
                (
                    f"Battery: {compare_random_env.battery_capacity}/{compare_random_env.battery_capacity}"
                    if compare_random_env.battery_capacity is not None
                    else "Battery: off",
                    (0, 0, 0),
                ),
                ("Status: success", (0, 0, 0)),
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
            compare_height = PLAYBACK_TOP_RESERVED + max(left_layout.total_height, right_layout.total_height) + bottom_info_height + 24

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
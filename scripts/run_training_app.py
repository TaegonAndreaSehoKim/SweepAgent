from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pygame

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from configs.map_presets import MAP_PRESETS
from utils.ui_utils import (
    WINDOW_PADDING_X,
    build_left_status_lines,
    build_right_status_lines,
    compute_bottom_info_height,
    compute_panel_layout,
    load_fonts,
)
from ui.training_app_core import (
    Button,
    CLEANED_FILL,
    EPSILON_FILL,
    MENU_NUMERIC_FIELDS,
    MENU_HEIGHT,
    MENU_WIDTH,
    MODEL_OPTIONS,
    PLAYBACK_TOP_RESERVED,
    PROGRESS_FILL,
    RESULT_VIEW_OPTIONS,
    SUCCESS_FILL,
    PreviewPolicy,
    clamp_step_delay,
    commit_menu_numeric_input,
    compute_training_window_height,
    get_default_algorithm_params,
    sync_menu_numeric_inputs,
)
from ui.training_app_handlers import (
    cancel_training,
    maybe_reset_finished_preview,
    rebuild_training_preview_env,
    start_selected_flow,
    step_compare_playback,
    step_single_playback,
    step_training_preview,
    update_training_from_subprocess,
)
from ui.training_app_state import (
    ComparePlaybackState,
    MenuSelection,
    PreviewState,
    SinglePlaybackState,
    TrainingState,
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


def ui_button(x: int, y: int, w: int, h: int, label: str, on_click, primary: bool = False) -> Button:
    return Button(
        rect=pygame.Rect(x, y, w, h),
        label=label,
        on_click=on_click,
        primary=primary,
    )


def main() -> None:
    args = parse_args()

    pygame.init()
    fonts = load_fonts()

    menu = MenuSelection(
        map_name="charge_required_v2" if "charge_required_v2" in MAP_PRESETS else list(MAP_PRESETS.keys())[0],
        algorithm_name="q_learning",
        result_view="single_playback",
        episodes=200000,
        step_delay=0.5,
        train_seed=args.seed,
        playback_seed=args.random_seed,
        algorithm_params=get_default_algorithm_params("q_learning"),
    )
    sync_menu_numeric_inputs(menu)

    training = TrainingState()
    preview = PreviewState()
    single_playback = SinglePlaybackState()
    compare_playback = ComparePlaybackState()

    screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))
    pygame.display.set_caption("SweepAgent Training App")
    clock = pygame.time.Clock()

    app_state = "menu"

    rebuild_training_preview_env(
        preview_state=preview,
        menu=menu,
        preview_policy_cls=PreviewPolicy,
        seed=menu.train_seed,
    )

    def back_to_menu() -> None:
        nonlocal app_state
        app_state = "menu"
        menu.open_dropdown = None
        menu.active_input = None

    def toggle_single_pause() -> None:
        single_playback.is_paused = not single_playback.is_paused
        single_playback.last_step_time = time.time()

    def restart_single_playback() -> None:
        from utils.ui_utils import reset_panel_state
        import random

        if single_playback.env is None:
            return

        single_playback.rng = random.Random(menu.playback_seed)
        (
            single_playback.state,
            single_playback.done,
            single_playback.total_reward,
            single_playback.step_idx,
            single_playback.path_history,
            single_playback.visit_counts,
        ) = reset_panel_state(single_playback.env)
        single_playback.is_paused = False
        single_playback.last_step_time = time.time()

    def toggle_compare_pause() -> None:
        compare_playback.is_paused = not compare_playback.is_paused
        compare_playback.last_step_time = time.time()

    def restart_compare_playback() -> None:
        from utils.ui_utils import reset_panel_state
        import random

        if compare_playback.random_env is None or compare_playback.learned_env is None:
            return

        compare_playback.rng = random.Random(menu.playback_seed)

        (
            compare_playback.random_state,
            compare_playback.random_done,
            compare_playback.random_reward,
            compare_playback.random_step,
            compare_playback.random_path,
            compare_playback.random_visits,
        ) = reset_panel_state(compare_playback.random_env)

        (
            compare_playback.learned_state,
            compare_playback.learned_done,
            compare_playback.learned_reward,
            compare_playback.learned_step,
            compare_playback.learned_path,
            compare_playback.learned_visits,
        ) = reset_panel_state(compare_playback.learned_env)

        compare_playback.is_paused = False
        compare_playback.last_step_time = time.time()

    def slower() -> None:
        menu.step_delay = clamp_step_delay(menu.step_delay + 0.1, 0.1, 1.5)
        sync_menu_numeric_inputs(menu)

    def faster() -> None:
        menu.step_delay = clamp_step_delay(menu.step_delay - 0.1, 0.1, 1.5)
        sync_menu_numeric_inputs(menu)

    def commit_active_input() -> bool:
        if menu.active_input is None:
            return False

        field_name = menu.active_input
        previous_train_seed = menu.train_seed
        changed = commit_menu_numeric_input(menu, field_name)
        menu.active_input = None

        if field_name == "train_seed" and menu.train_seed != previous_train_seed:
            rebuild_training_preview_env(
                preview_state=preview,
                menu=menu,
                preview_policy_cls=PreviewPolicy,
                seed=menu.train_seed,
            )

        return changed

    def commit_all_inputs() -> None:
        for field_name in MENU_NUMERIC_FIELDS:
            menu.active_input = field_name
            commit_active_input()
        menu.active_input = None

    while True:
        mouse_clicked = False
        mouse_pos = pygame.mouse.get_pos()
        max_scroll = 0

        menu_buttons = [
            ui_button(
                x=390,
                y=MENU_HEIGHT - 80,
                w=260,
                h=54,
                label=(
                    "Start Curriculum"
                    if menu.algorithm_name == "curriculum_q_learning"
                    else "Start Training" if menu.algorithm_name != "random_baseline"
                    else "Start Preview"
                ),
                on_click=lambda: None,
                primary=True,
            )
        ]

        training_height = compute_training_window_height()
        training_buttons = [
            ui_button(
                x=MENU_WIDTH - 220,
                y=training_height - 60,
                w=180,
                h=44,
                label="Cancel Training",
                on_click=lambda: None,
            )
        ]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                training.runner.terminate()
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                training.runner.terminate()
                pygame.quit()
                return

            if event.type == pygame.KEYDOWN and app_state == "menu" and menu.active_input is not None:
                if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_TAB):
                    commit_active_input()
                elif event.key == pygame.K_BACKSPACE:
                    menu.input_values[menu.active_input] = menu.input_values.get(menu.active_input, "")[:-1]
                elif event.key == pygame.K_MINUS:
                    current_value = menu.input_values.get(menu.active_input, "")
                    if not current_value:
                        menu.input_values[menu.active_input] = "-"
                else:
                    if event.unicode and event.unicode in "0123456789.eE":
                        menu.input_values[menu.active_input] = (
                            menu.input_values.get(menu.active_input, "") + event.unicode
                        )

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_clicked = True

            if event.type == pygame.MOUSEWHEEL and app_state == "training":
                training.log_scroll_offset -= event.y
                training.log_scroll_offset = max(0, min(training.log_scroll_offset, max_scroll))

        if app_state == "training":
            next_state = update_training_from_subprocess(
                menu=menu,
                training=training,
                single_playback=single_playback,
                compare_playback=compare_playback,
                train_seed=menu.train_seed,
                random_seed=menu.playback_seed,
            )
            if next_state is not None:
                app_state = next_state

            step_training_preview(preview)
            maybe_reset_finished_preview(
                preview=preview,
                menu=menu,
                preview_policy_cls=PreviewPolicy,
                seed=menu.train_seed,
            )

        if app_state == "playback_single":
            step_single_playback(single_playback, menu.step_delay)

        if app_state == "playback_compare":
            step_compare_playback(compare_playback, menu.step_delay)

        if app_state == "menu":
            if screen.get_width() != MENU_WIDTH or screen.get_height() != MENU_HEIGHT:
                screen = pygame.display.set_mode((MENU_WIDTH, MENU_HEIGHT))

            dropdown_ui = draw_menu(
                screen=screen,
                width=MENU_WIDTH,
                height=MENU_HEIGHT,
                fonts=fonts,
                map_name=menu.map_name,
                model_name=menu.algorithm_name,
                result_view=menu.result_view,
                input_values=menu.input_values,
                open_dropdown=menu.open_dropdown,
                active_input=menu.active_input,
                buttons=menu_buttons,
                map_options=list(MAP_PRESETS.keys()),
                model_options=MODEL_OPTIONS,
                result_view_options=RESULT_VIEW_OPTIONS,
            )

            if mouse_clicked:
                handled = False

                # 1) Open dropdown option clicks must be handled first.
                if menu.open_dropdown is not None:
                    commit_active_input()
                    option_key = {
                        "map": "map_options",
                        "model": "model_options",
                        "result_view": "result_options",
                    }[menu.open_dropdown]

                    option_rects = dropdown_ui[option_key]
                    clicked_option = False

                    if isinstance(option_rects, list):
                        for idx, rect in enumerate(option_rects):
                            if rect.collidepoint(mouse_pos):
                                clicked_option = True

                                if menu.open_dropdown == "map":
                                    menu.map_name = list(MAP_PRESETS.keys())[idx]
                                    rebuild_training_preview_env(
                                        preview_state=preview,
                                        menu=menu,
                                        preview_policy_cls=PreviewPolicy,
                                        seed=menu.train_seed,
                                    )
                                elif menu.open_dropdown == "model":
                                    menu.algorithm_name = MODEL_OPTIONS[idx]
                                    menu.algorithm_params = get_default_algorithm_params(menu.algorithm_name)
                                    sync_menu_numeric_inputs(menu)
                                elif menu.open_dropdown == "result_view":
                                    menu.result_view = RESULT_VIEW_OPTIONS[idx]

                                menu.open_dropdown = None
                                handled = True
                                break

                    # Outside click closes the open dropdown before any lower header can steal the click.
                    if not handled and not clicked_option:
                        menu.open_dropdown = None
                        handled = True

                # 2) Then handle start button clicks.
                if not handled:
                    for button in menu_buttons:
                        if button.rect.collidepoint(mouse_pos):
                            commit_all_inputs()
                            app_state = start_selected_flow(
                                menu=menu,
                                training=training,
                                preview=preview,
                                single_playback=single_playback,
                                compare_playback=compare_playback,
                                preview_policy_cls=PreviewPolicy,
                                train_seed=menu.train_seed,
                                random_seed=menu.playback_seed,
                            )
                            handled = True
                            break

                # 3) Then handle direct input fields.
                if not handled:
                    input_boxes = {
                        "episodes": dropdown_ui["episodes_rect"],
                        "train_seed": dropdown_ui["train_seed_rect"],
                        "playback_seed": dropdown_ui["playback_seed_rect"],
                        "learning_rate": dropdown_ui["learning_rate_rect"],
                        "discount_factor": dropdown_ui["discount_factor_rect"],
                        "epsilon_start": dropdown_ui["epsilon_start_rect"],
                        "epsilon_decay": dropdown_ui["epsilon_decay_rect"],
                        "epsilon_min": dropdown_ui["epsilon_min_rect"],
                        "delay": dropdown_ui["delay_rect"],
                    }

                    for key, rect in input_boxes.items():
                        if isinstance(rect, pygame.Rect) and rect.collidepoint(mouse_pos):
                            if menu.active_input is not None and menu.active_input != key:
                                commit_active_input()
                            menu.active_input = key
                            menu.open_dropdown = None
                            handled = True
                            break

                # 4) Finally handle dropdown header toggles.
                if not handled:
                    dropdown_boxes = {
                        "map": dropdown_ui["map_rect"],
                        "model": dropdown_ui["model_rect"],
                        "result_view": dropdown_ui["result_rect"],
                    }

                    for key, rect in dropdown_boxes.items():
                        if isinstance(rect, pygame.Rect) and rect.collidepoint(mouse_pos):
                            commit_active_input()
                            menu.open_dropdown = None if menu.open_dropdown == key else key
                            handled = True
                            break

                if not handled and menu.active_input is not None:
                    commit_active_input()

        elif app_state == "training":
            if screen.get_width() != MENU_WIDTH or screen.get_height() != training_height:
                screen = pygame.display.set_mode((MENU_WIDTH, training_height))

            _, max_scroll = draw_training_screen(
                screen=screen,
                width=MENU_WIDTH,
                height=training_height,
                fonts=fonts,
                map_name=menu.map_name,
                model_name=menu.algorithm_name,
                episodes=menu.episodes,
                latest_metrics=training.latest_metrics,
                log_lines=training.log_lines,
                log_scroll_offset=training.log_scroll_offset,
                preview_env=preview.env,
                preview_reward=preview.total_reward,
                preview_step=preview.step_idx,
                preview_visits=preview.visit_counts,
                preview_path=preview.path_history,
                preview_mode_label=preview.mode_label,
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
                        app_state = cancel_training(training)
                        break

        elif app_state == "playback_single" and single_playback.env is not None:
            preview_left_lines = build_left_status_lines(
                map_name=menu.map_name,
                total_reward=single_playback.total_reward,
                step_idx=single_playback.step_idx,
                include_map_line=True,
            )
            preview_right_lines = build_right_status_lines(
                env=single_playback.env,
                done=single_playback.done,
                is_paused=single_playback.is_paused,
                step_delay=menu.step_delay,
            )
            preview_layout = compute_panel_layout(
                env=single_playback.env,
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
                ui_button(24, 86, 120, 40, "Pause" if not single_playback.is_paused else "Resume", toggle_single_pause),
                ui_button(156, 86, 120, 40, "Restart", restart_single_playback),
                ui_button(288, 86, 110, 40, "Slower", slower),
                ui_button(410, 86, 110, 40, "Faster", faster),
                ui_button(playback_width - 160, 86, 136, 40, "Back to Menu", back_to_menu),
            ]

            draw_single_playback_screen(
                screen=screen,
                width=playback_width,
                height=playback_height,
                fonts=fonts,
                title_text="Learned Greedy Agent" if single_playback.algorithm_name == "q_learning" else "Random Baseline",
                env=single_playback.env,
                total_reward=single_playback.total_reward,
                step_idx=single_playback.step_idx,
                done=single_playback.done,
                is_paused=single_playback.is_paused,
                step_delay=menu.step_delay,
                visit_counts=single_playback.visit_counts,
                path_history=single_playback.path_history,
                cell_size=args.cell_size,
                buttons=playback_buttons,
            )

            if mouse_clicked:
                for button in playback_buttons:
                    if button.rect.collidepoint(mouse_pos):
                        button.on_click()
                        break

        elif app_state == "playback_compare" and compare_playback.random_env is not None and compare_playback.learned_env is not None:
            preview_left_lines = [
                ("Step: 9999", (0, 0, 0)),
                ("Reward: 999", (0, 0, 0)),
            ]
            preview_right_lines = [
                (
                    f"Cleaned: {compare_playback.random_env.total_dirty_tiles}/{compare_playback.random_env.total_dirty_tiles}",
                    (0, 0, 0),
                ),
                (
                    f"Battery: {compare_playback.random_env.battery_capacity}/{compare_playback.random_env.battery_capacity}"
                    if compare_playback.random_env.battery_capacity is not None
                    else "Battery: off",
                    (0, 0, 0),
                ),
                ("Status: success", (0, 0, 0)),
            ]

            left_layout = compute_panel_layout(
                env=compare_playback.random_env,
                cell_size=args.cell_size,
                fonts=fonts,
                left_lines=preview_left_lines,
                right_lines=preview_right_lines,
                title_text="Random Agent",
                subtitle_text=f"Map: {menu.map_name}",
            )
            right_layout = compute_panel_layout(
                env=compare_playback.learned_env,
                cell_size=args.cell_size,
                fonts=fonts,
                left_lines=preview_left_lines,
                right_lines=preview_right_lines,
                title_text="Learned Greedy Agent",
                subtitle_text=f"Map: {menu.map_name}",
            )

            panel_width = max(left_layout.panel_width, right_layout.panel_width)
            panel_gap = 24
            bottom_info_height = compute_bottom_info_height(fonts)
            compare_width = WINDOW_PADDING_X + panel_width + panel_gap + panel_width + WINDOW_PADDING_X
            compare_height = PLAYBACK_TOP_RESERVED + max(left_layout.total_height, right_layout.total_height) + bottom_info_height + 24

            if screen.get_width() != compare_width or screen.get_height() != compare_height:
                screen = pygame.display.set_mode((compare_width, compare_height))

            playback_buttons = [
                ui_button(24, 86, 120, 40, "Pause" if not compare_playback.is_paused else "Resume", toggle_compare_pause),
                ui_button(156, 86, 120, 40, "Restart", restart_compare_playback),
                ui_button(288, 86, 110, 40, "Slower", slower),
                ui_button(410, 86, 110, 40, "Faster", faster),
                ui_button(compare_width - 160, 86, 136, 40, "Back to Menu", back_to_menu),
            ]

            draw_compare_playback_screen(
                screen=screen,
                width=compare_width,
                height=compare_height,
                fonts=fonts,
                random_env=compare_playback.random_env,
                learned_env=compare_playback.learned_env,
                random_reward=compare_playback.random_reward,
                learned_reward=compare_playback.learned_reward,
                random_step=compare_playback.random_step,
                learned_step=compare_playback.learned_step,
                random_done=compare_playback.random_done,
                learned_done=compare_playback.learned_done,
                is_paused=compare_playback.is_paused,
                step_delay=menu.step_delay,
                random_visits=compare_playback.random_visits,
                learned_visits=compare_playback.learned_visits,
                random_path=compare_playback.random_path,
                learned_path=compare_playback.learned_path,
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

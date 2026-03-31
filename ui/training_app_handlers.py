from __future__ import annotations

import random
import time

from utils.experiment_utils import build_env, get_checkpoint_path, load_or_train_q_agent
from utils.ui_utils import reset_panel_state
from ui.training_app_core import (
    PREVIEW_STEP_INTERVAL_SEC,
    TRAINING_LINE_PATTERN,
    build_training_command,
    is_trainable_algorithm,
)
from ui.training_app_state import (
    ComparePlaybackState,
    MenuSelection,
    PreviewState,
    SinglePlaybackState,
    TrainingState,
)


def rebuild_training_preview_env(
    preview_state: PreviewState,
    menu: MenuSelection,
    preview_policy_cls,
    seed: int,
) -> None:
    """Reset the training mini preview environment and checkpoint policy."""
    preview_state.env = build_env(map_name=menu.map_name)
    preview_state.env.map_name = menu.map_name
    (
        preview_state.state,
        preview_state.done,
        preview_state.total_reward,
        preview_state.step_idx,
        preview_state.path_history,
        preview_state.visit_counts,
    ) = reset_panel_state(preview_state.env)
    preview_state.last_step_time = time.time()
    preview_state.policy = preview_policy_cls(
        get_checkpoint_path(menu.map_name, menu.episodes, seed)
    )
    preview_state.mode_label = "random fallback"


def start_selected_flow(
    menu: MenuSelection,
    training: TrainingState,
    preview: PreviewState,
    single_playback: SinglePlaybackState,
    compare_playback: ComparePlaybackState,
    preview_policy_cls,
    train_seed: int,
    random_seed: int,
) -> str:
    """
    Start training or direct preview depending on the current menu selection.
    Returns the next app state string.
    """
    training.latest_metrics.clear()
    training.log_lines.clear()
    training.log_scroll_offset = 0

    rebuild_training_preview_env(
        preview_state=preview,
        menu=menu,
        preview_policy_cls=preview_policy_cls,
        seed=train_seed,
    )

    if is_trainable_algorithm(menu.algorithm_name):
        training.runner = type(training.runner)()
        training.runner.start(
            build_training_command(
                algorithm_name=menu.algorithm_name,
                map_name=menu.map_name,
                episodes=menu.episodes,
                seed=train_seed,
                algorithm_params=menu.algorithm_params,
            )
        )
        return "training"

    if menu.result_view == "single_playback":
        single_playback.env = build_env(map_name=menu.map_name)
        single_playback.env.map_name = menu.map_name
        (
            single_playback.state,
            single_playback.done,
            single_playback.total_reward,
            single_playback.step_idx,
            single_playback.path_history,
            single_playback.visit_counts,
        ) = reset_panel_state(single_playback.env)
        single_playback.agent = None
        single_playback.algorithm_name = "random_baseline"
        single_playback.is_paused = False
        single_playback.last_step_time = time.time()
        single_playback.rng = random.Random(random_seed)
        return "playback_single"

    compare_playback.random_env = build_env(map_name=menu.map_name)
    compare_playback.learned_env = build_env(map_name=menu.map_name)
    compare_playback.random_env.map_name = menu.map_name
    compare_playback.learned_env.map_name = menu.map_name
    compare_playback.learned_agent = None
    compare_playback.learned_algorithm_name = "random_baseline"

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
    compare_playback.rng = random.Random(random_seed)
    return "playback_compare"


def cancel_training(training: TrainingState) -> str:
    """Terminate training and return the menu app state."""
    training.runner.terminate()
    return "menu"


def update_training_from_subprocess(
    menu: MenuSelection,
    training: TrainingState,
    single_playback: SinglePlaybackState,
    compare_playback: ComparePlaybackState,
    train_seed: int,
    random_seed: int,
) -> str | None:
    """
    Consume training subprocess output and switch state if training finishes.
    Returns next app state or None if no transition is needed.
    """
    for line in training.runner.poll_lines():
        training.log_lines.append(line)
        match = TRAINING_LINE_PATTERN.search(line)
        if match:
            training.latest_metrics = {
                "episode": match.group(1),
                "total": match.group(2),
                "avg_reward": match.group(3),
                "avg_cleaned_ratio": match.group(4) + "%",
                "success_rate": match.group(5) + "%",
                "epsilon": match.group(6),
            }

    if not training.runner.finished:
        return None

    training.log_lines.append(f"[training process finished] return_code={training.runner.return_code}")

    if training.runner.return_code != 0:
        return "menu"

    if menu.result_view == "single_playback":
        single_playback.env = build_env(map_name=menu.map_name)
        single_playback.env.map_name = menu.map_name
        single_playback.agent = load_or_train_q_agent(
            map_name=menu.map_name,
            num_episodes=menu.episodes,
            seed=train_seed,
        )
        single_playback.algorithm_name = menu.algorithm_name
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
        single_playback.rng = random.Random(random_seed)
        return "playback_single"

    compare_playback.random_env = build_env(map_name=menu.map_name)
    compare_playback.learned_env = build_env(map_name=menu.map_name)
    compare_playback.random_env.map_name = menu.map_name
    compare_playback.learned_env.map_name = menu.map_name
    compare_playback.learned_agent = load_or_train_q_agent(
        map_name=menu.map_name,
        num_episodes=menu.episodes,
        seed=train_seed,
    )
    compare_playback.learned_algorithm_name = menu.algorithm_name

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
    compare_playback.rng = random.Random(random_seed)
    return "playback_compare"


def step_training_preview(preview: PreviewState) -> None:
    """Advance the mini preview using latest checkpoint policy or random fallback."""
    if preview.env is None or preview.state is None:
        return

    now = time.time()
    if now - preview.last_step_time < PREVIEW_STEP_INTERVAL_SEC:
        return

    if preview.policy is not None:
        reloaded = preview.policy.maybe_reload()
        if reloaded and preview.policy.q_table:
            preview.mode_label = "greedy checkpoint"
        elif not preview.policy.q_table:
            preview.mode_label = "random fallback"

    if preview.policy is not None and preview.policy.q_table:
        action = preview.policy.get_greedy_action(
            preview.state,
            len(preview.env.ACTIONS),
        )
    else:
        action = random.randrange(len(preview.env.ACTIONS))

    (
        preview.state,
        reward,
        preview.done,
        _,
    ) = preview.env.step(action)
    preview.total_reward += reward
    preview.step_idx += 1
    preview.path_history.append(preview.env.robot_pos)
    preview.visit_counts[preview.env.robot_pos] = preview.visit_counts.get(preview.env.robot_pos, 0) + 1
    preview.last_step_time = now


def maybe_reset_finished_preview(
    preview: PreviewState,
    menu: MenuSelection,
    preview_policy_cls,
    seed: int,
) -> None:
    """Reset the mini preview once the preview episode ends."""
    if preview.done:
        rebuild_training_preview_env(
            preview_state=preview,
            menu=menu,
            preview_policy_cls=preview_policy_cls,
            seed=seed,
        )


def step_single_playback(single_playback: SinglePlaybackState, step_delay: float) -> None:
    """Advance single playback by one step if enough time has passed."""
    if single_playback.env is None or single_playback.state is None:
        return

    now = time.time()
    if single_playback.done or single_playback.is_paused:
        return
    if now - single_playback.last_step_time < step_delay:
        return

    if single_playback.algorithm_name == "q_learning" and single_playback.agent is not None:
        action = single_playback.agent.get_policy_action(single_playback.state)
    else:
        action = single_playback.rng.randrange(len(single_playback.env.ACTIONS))

    (
        single_playback.state,
        reward,
        single_playback.done,
        _,
    ) = single_playback.env.step(action)
    single_playback.total_reward += reward
    single_playback.step_idx += 1
    single_playback.path_history.append(single_playback.env.robot_pos)
    single_playback.visit_counts[single_playback.env.robot_pos] = (
        single_playback.visit_counts.get(single_playback.env.robot_pos, 0) + 1
    )
    single_playback.last_step_time = now


def step_compare_playback(compare_playback: ComparePlaybackState, step_delay: float) -> None:
    """Advance side-by-side compare playback by one synchronized tick."""
    if compare_playback.random_env is None or compare_playback.learned_env is None:
        return

    now = time.time()
    if compare_playback.is_paused or now - compare_playback.last_step_time < step_delay:
        return

    if not compare_playback.random_done:
        random_action = compare_playback.rng.randrange(len(compare_playback.random_env.ACTIONS))
        (
            compare_playback.random_state,
            reward,
            compare_playback.random_done,
            _,
        ) = compare_playback.random_env.step(random_action)
        compare_playback.random_reward += reward
        compare_playback.random_step += 1
        compare_playback.random_path.append(compare_playback.random_env.robot_pos)
        compare_playback.random_visits[compare_playback.random_env.robot_pos] = (
            compare_playback.random_visits.get(compare_playback.random_env.robot_pos, 0) + 1
        )

    if not compare_playback.learned_done:
        if compare_playback.learned_algorithm_name == "q_learning" and compare_playback.learned_agent is not None:
            learned_action = compare_playback.learned_agent.get_policy_action(compare_playback.learned_state)
        else:
            learned_action = compare_playback.rng.randrange(len(compare_playback.learned_env.ACTIONS))

        (
            compare_playback.learned_state,
            reward,
            compare_playback.learned_done,
            _,
        ) = compare_playback.learned_env.step(learned_action)
        compare_playback.learned_reward += reward
        compare_playback.learned_step += 1
        compare_playback.learned_path.append(compare_playback.learned_env.robot_pos)
        compare_playback.learned_visits[compare_playback.learned_env.robot_pos] = (
            compare_playback.learned_visits.get(compare_playback.learned_env.robot_pos, 0) + 1
        )

    compare_playback.last_step_time = now
from __future__ import annotations

import random
import time

from utils.experiment_utils import build_env, load_or_train_q_agent
from utils.ui_utils import reset_panel_state
from ui.training_app_core import (
    PREVIEW_STEP_INTERVAL_SEC,
    TRAINING_LINE_PATTERN,
    build_training_command,
    get_playback_target_episodes,
    get_playback_target_map_name,
    get_preview_checkpoint_path,
    get_preview_map_name,
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
    preview_map_name = get_preview_map_name(menu.algorithm_name, menu.map_name)
    preview_state.env = build_env(map_name=preview_map_name)
    preview_state.env.map_name = preview_map_name
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
        get_preview_checkpoint_path(
            algorithm_name=menu.algorithm_name,
            selected_map_name=menu.map_name,
            selected_episodes=menu.episodes,
            seed=seed,
        )
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
    for line in training.runner.poll_lines():
        training.log_lines.append(line)
        match = TRAINING_LINE_PATTERN.search(line)
        if match:
            training.latest_metrics = {
                "episode": match.group(1),
                "total": match.group(2),
                "avg_reward": match.group(3),
                "avg_cleaned_ratio": f"{match.group(4)}%",
                "success_rate": f"{match.group(5)}%",
                "epsilon": match.group(6),
            }

    if not training.runner.finished:
        return None

    playback_map_name = get_playback_target_map_name(
        menu.algorithm_name,
        menu.map_name,
    )
    playback_episodes = get_playback_target_episodes(
        menu.algorithm_name,
        menu.episodes,
    )

    if menu.result_view == "single_playback":
        single_playback.env = build_env(map_name=playback_map_name)
        single_playback.env.map_name = playback_map_name
        (
            single_playback.state,
            single_playback.done,
            single_playback.total_reward,
            single_playback.step_idx,
            single_playback.path_history,
            single_playback.visit_counts,
        ) = reset_panel_state(single_playback.env)

        single_playback.agent = load_or_train_q_agent(
            map_name=playback_map_name,
            num_episodes=playback_episodes,
            seed=train_seed,
        )
        single_playback.algorithm_name = menu.algorithm_name
        single_playback.is_paused = False
        single_playback.last_step_time = time.time()
        single_playback.rng = random.Random(random_seed)
        return "playback_single"

    compare_playback.random_env = build_env(map_name=playback_map_name)
    compare_playback.learned_env = build_env(map_name=playback_map_name)
    compare_playback.random_env.map_name = playback_map_name
    compare_playback.learned_env.map_name = playback_map_name

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

    compare_playback.learned_agent = load_or_train_q_agent(
        map_name=playback_map_name,
        num_episodes=playback_episodes,
        seed=train_seed,
    )
    compare_playback.learned_algorithm_name = menu.algorithm_name
    compare_playback.is_paused = False
    compare_playback.last_step_time = time.time()
    compare_playback.rng = random.Random(random_seed)
    return "playback_compare"


def step_training_preview(preview: PreviewState) -> None:
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
    if not preview.done:
        return

    rebuild_training_preview_env(
        preview_state=preview,
        menu=menu,
        preview_policy_cls=preview_policy_cls,
        seed=seed,
    )


def step_single_playback(single_playback: SinglePlaybackState, step_delay: float) -> None:
    if single_playback.env is None or single_playback.done or single_playback.is_paused:
        return

    now = time.time()
    if now - single_playback.last_step_time < step_delay:
        return

    if single_playback.agent is None:
        action = single_playback.rng.randrange(len(single_playback.env.ACTIONS))
    else:
        action = single_playback.agent.select_action(single_playback.state, training=False)

    next_state, reward, done, _ = single_playback.env.step(action)
    single_playback.state = next_state
    single_playback.done = done
    single_playback.total_reward += reward
    single_playback.step_idx += 1
    single_playback.path_history.append(single_playback.env.robot_pos)
    single_playback.visit_counts[single_playback.env.robot_pos] = single_playback.visit_counts.get(single_playback.env.robot_pos, 0) + 1
    single_playback.last_step_time = now


def step_compare_playback(compare_playback: ComparePlaybackState, step_delay: float) -> None:
    if (
        compare_playback.random_env is None
        or compare_playback.learned_env is None
        or compare_playback.is_paused
    ):
        return

    now = time.time()
    if now - compare_playback.last_step_time < step_delay:
        return

    if not compare_playback.random_done:
        random_action = compare_playback.rng.randrange(len(compare_playback.random_env.ACTIONS))
        next_state, reward, done, _ = compare_playback.random_env.step(random_action)
        compare_playback.random_state = next_state
        compare_playback.random_done = done
        compare_playback.random_reward += reward
        compare_playback.random_step += 1
        compare_playback.random_path.append(compare_playback.random_env.robot_pos)
        compare_playback.random_visits[compare_playback.random_env.robot_pos] = compare_playback.random_visits.get(compare_playback.random_env.robot_pos, 0) + 1

    if not compare_playback.learned_done:
        if compare_playback.learned_agent is None:
            learned_action = compare_playback.rng.randrange(len(compare_playback.learned_env.ACTIONS))
        else:
            learned_action = compare_playback.learned_agent.select_action(compare_playback.learned_state, training=False)

        next_state, reward, done, _ = compare_playback.learned_env.step(learned_action)
        compare_playback.learned_state = next_state
        compare_playback.learned_done = done
        compare_playback.learned_reward += reward
        compare_playback.learned_step += 1
        compare_playback.learned_path.append(compare_playback.learned_env.robot_pos)
        compare_playback.learned_visits[compare_playback.learned_env.robot_pos] = compare_playback.learned_visits.get(compare_playback.learned_env.robot_pos, 0) + 1

    compare_playback.last_step_time = now
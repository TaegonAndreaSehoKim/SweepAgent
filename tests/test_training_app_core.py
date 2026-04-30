from __future__ import annotations

import random

from ui.training_app_core import (
    MODEL_OPTIONS,
    build_training_command,
    get_algorithm_display_name,
    is_trainable_algorithm,
    select_playback_action,
    split_episode_budget,
)


class StubPolicyAgent:
    def get_policy_action(self, state) -> int:
        return 2


def test_best_checkpoint_algorithms_are_playback_only() -> None:
    assert "dqn_best" in MODEL_OPTIONS
    assert "ppo_best" in MODEL_OPTIONS
    assert is_trainable_algorithm("dqn")
    assert is_trainable_algorithm("ppo")
    assert is_trainable_algorithm("ppo_guided")
    assert not is_trainable_algorithm("dqn_best")
    assert not is_trainable_algorithm("ppo_best")


def test_best_checkpoint_algorithms_have_display_names() -> None:
    assert get_algorithm_display_name("dqn_best") == "DQN Best Checkpoint"
    assert get_algorithm_display_name("ppo_best") == "PPO Best Checkpoint"
    assert get_algorithm_display_name("ppo_guided") == "Guided PPO Agent"


def test_select_playback_action_prefers_policy_action() -> None:
    action = select_playback_action(
        agent=StubPolicyAgent(),
        state=(1, 1, 0, 17),
        action_count=4,
        rng=random.Random(42),
    )

    assert action == 2


def test_build_training_command_supports_dqn() -> None:
    command = build_training_command(
        algorithm_name="dqn",
        map_name="default",
        episodes=10,
        seed=42,
        algorithm_params={"checkpoint_tag": "ui"},
    )

    assert "scripts/train_dqn.py" in command
    assert "--save-best-eval-checkpoint" in command
    assert command[command.index("--checkpoint-tag") + 1] == "ui"


def test_build_training_command_supports_guided_ppo_bastion_recipe() -> None:
    command = build_training_command(
        algorithm_name="ppo_guided",
        map_name="complex_charge_bastion",
        episodes=6500,
        seed=42,
        algorithm_params={"checkpoint_tag": "ui_guided"},
    )

    assert "scripts/train_ppo.py" in command
    assert "--curriculum-stage-keep-dirty-indices" in command
    assert "0,2" in command
    assert "--guided-dagger-bc-every" in command
    assert command[command.index("--checkpoint-tag") + 1] == "ui_guided"


def test_split_episode_budget_preserves_total() -> None:
    assert split_episode_budget(6500, [800, 1200, 2000, 2500]) == [
        800,
        1200,
        2000,
        2500,
    ]
    assert sum(split_episode_budget(5000, [800, 1200, 2000, 2500])) == 5000

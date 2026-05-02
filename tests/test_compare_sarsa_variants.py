from __future__ import annotations

import json
from argparse import Namespace

from scripts.compare_sarsa_variants import (
    build_checkpoint_tag,
    build_summary_row,
    build_train_command,
    build_variant_specs,
    normalize_variant_names,
)


def test_sarsa_variant_specs_split_guidance_and_shaping() -> None:
    variants = build_variant_specs(
        requested_variants=[],
        guided_exploration_ratio=0.9,
        relay_reward=0.5,
        relay_penalty=-0.75,
    )

    assert [variant.name for variant in variants] == [
        "plain",
        "shaping",
        "guided",
        "guided_shaping",
    ]
    assert variants[0].guided_exploration_ratio == 0.0
    assert variants[0].reward_move_toward_relay_charger == 0.0
    assert variants[1].guided_exploration_ratio == 0.0
    assert variants[1].reward_move_toward_relay_charger == 0.5
    assert variants[2].guided_exploration_ratio == 0.9
    assert variants[2].reward_move_toward_relay_charger == 0.0
    assert variants[3].guided_exploration_ratio == 0.9
    assert variants[3].penalty_move_away_from_relay_charger == -0.75


def test_normalize_variant_names_preserves_requested_order_without_duplicates() -> None:
    assert normalize_variant_names(["guided", "plain", "guided"]) == [
        "guided",
        "plain",
    ]
    assert normalize_variant_names(["guided", "all"]) == [
        "plain",
        "shaping",
        "guided",
        "guided_shaping",
    ]


def test_build_train_command_includes_variant_controls() -> None:
    args = Namespace(
        python_executable="python",
        map_name="complex_charge_bastion",
        battery_profile="evaluation",
        episodes=100000,
        seed=42,
        print_every=10000,
        learning_rate=0.05,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_decay=0.99995,
        epsilon_min=0.1,
        state_abstraction_mode="charger_context",
        safety_margin_bucket_size=5,
        eval_episodes=20,
        eval_every=10000,
        battery_capacity_override=0,
    )
    variant = build_variant_specs(["guided_shaping"], 0.9, 0.5, -0.75)[0]

    command = build_train_command(args, variant, "ablation_guided_shaping")

    assert "--guided-exploration-ratio" in command
    assert command[command.index("--guided-exploration-ratio") + 1] == "0.9"
    assert command[command.index("--reward-move-toward-relay-charger") + 1] == "0.5"
    assert command[command.index("--penalty-move-away-from-relay-charger") + 1] == "-0.75"
    assert command[command.index("--checkpoint-tag") + 1] == "ablation_guided_shaping"
    assert "--save-best-eval-checkpoint" in command


def test_build_summary_row_reads_best_eval_metadata(tmp_path) -> None:
    checkpoint_path = tmp_path / "final.json"
    best_checkpoint_path = tmp_path / "best.json"
    metadata = {
        "best_checkpoint_episodes": 70000,
        "best_eval_result": {
            "avg_reward": -43.5,
            "avg_steps": 155.0,
            "avg_cleaned_ratio": 1.0,
            "success_rate": 1.0,
        },
    }
    checkpoint_path.write_text(json.dumps({"metadata": metadata}), encoding="utf-8")
    best_checkpoint_path.write_text(json.dumps({"metadata": metadata}), encoding="utf-8")
    args = Namespace(
        map_name="complex_charge_bastion",
        seed=42,
        episodes=100000,
        comparison_eval_episodes=0,
        battery_capacity_override=0,
    )
    variant = build_variant_specs(["guided_shaping"], 0.9, 0.5, -0.75)[0]

    row = build_summary_row(
        args=args,
        variant=variant,
        checkpoint_tag=build_checkpoint_tag("ablation", "guided_shaping", 100000),
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
    )

    assert row["best_checkpoint_episodes"] == 70000
    assert row["best_eval_avg_steps"] == 155.0
    assert row["best_eval_success_rate"] == 1.0
    assert row["comparison_eval_episodes"] == 0

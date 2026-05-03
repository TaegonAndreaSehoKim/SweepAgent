from __future__ import annotations

from argparse import Namespace

from scripts.compare_all_algorithms import (
    AlgorithmCheckpointSpec,
    build_checkpoint_specs,
    default_checkpoint_specs,
    evaluated_checkpoint_row,
    generate_markdown_report,
    missing_checkpoint_row,
    selected_algorithms,
)


def test_default_bastion_checkpoint_specs_use_current_references() -> None:
    specs = default_checkpoint_specs("complex_charge_bastion")

    assert specs["q_learning"].checkpoint_path.name == (
        "q_learning_agent_complex_charge_bastion_ep_400000_seed_505.json"
    )
    assert specs["dqn"].checkpoint_path.name == (
        "dqn_agent_complex_charge_bastion_best_eval_seed_418_"
        "v2relay_shape_finalroute.pt"
    )
    assert specs["ppo"].checkpoint_path.name == (
        "ppo_agent_complex_charge_bastion_best_eval_seed_42_"
        "ppo_finalrelay_curriculum6500.pt"
    )
    assert specs["sarsa"].checkpoint_path.name == (
        "sarsa_agent_complex_charge_bastion_best_eval_seed_42_"
        "guided09_relay100k.json"
    )


def test_selected_algorithms_expands_all_and_deduplicates() -> None:
    assert selected_algorithms(["all"]) == ["q_learning", "dqn", "ppo", "sarsa"]
    assert selected_algorithms(["sarsa", "dqn", "sarsa"]) == ["sarsa", "dqn"]


def test_build_checkpoint_specs_applies_overrides() -> None:
    args = Namespace(
        map_name="complex_charge_bastion",
        include=["dqn", "sarsa"],
        q_checkpoint="",
        dqn_checkpoint="custom_dqn.pt",
        ppo_checkpoint="",
        sarsa_checkpoint="outputs/checkpoints/custom_sarsa.json",
    )

    specs = build_checkpoint_specs(args)

    assert [spec.algorithm for spec in specs] == ["dqn", "sarsa"]
    assert specs[0].label == "custom_dqn"
    assert specs[0].checkpoint_path.name == "custom_dqn.pt"
    assert specs[1].label == "custom_sarsa"
    assert specs[1].checkpoint_path.name == "custom_sarsa.json"


def test_evaluated_checkpoint_row_includes_metadata_and_metrics(tmp_path) -> None:
    args = Namespace(
        map_name="complex_charge_bastion",
        battery_profile="evaluation",
        eval_episodes=200,
    )
    spec = AlgorithmCheckpointSpec(
        algorithm="sarsa",
        label="sarsa_reference",
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    metadata = {
        "best_checkpoint_episodes": 80000,
        "checkpoint_tag": "guided09_relay100k",
        "seed": 42,
    }
    eval_result = {
        "avg_reward": -43.5,
        "avg_steps": 155.0,
        "avg_cleaned_ratio": 1.0,
        "success_rate": 1.0,
        "avg_recharges": 4.0,
        "termination_all_cleaned": 200,
        "termination_battery_depleted": 0,
        "termination_step_limit": 0,
        "termination_other": 0,
    }

    row = evaluated_checkpoint_row(args, spec, metadata, eval_result)

    assert row["status"] == "evaluated"
    assert row["checkpoint_episodes"] == 80000
    assert row["checkpoint_tag"] == "guided09_relay100k"
    assert row["source_seed"] == 42
    assert row["avg_steps"] == 155.0
    assert row["success_rate"] == 1.0


def test_missing_checkpoint_row_marks_status(tmp_path) -> None:
    args = Namespace(
        map_name="complex_charge_bastion",
        battery_profile="evaluation",
        eval_episodes=200,
    )
    spec = AlgorithmCheckpointSpec(
        algorithm="ppo",
        label="missing",
        checkpoint_path=tmp_path / "missing.pt",
    )

    row = missing_checkpoint_row(args, spec)

    assert row["status"] == "missing_checkpoint"
    assert row["algorithm"] == "ppo"
    assert row["success_rate"] == 0.0


def test_generate_markdown_report_summarizes_rankings(tmp_path) -> None:
    rows = [
        {
            "algorithm": "q_learning",
            "label": "q_reference",
            "status": "evaluated",
            "avg_cleaned_ratio": 0.666667,
            "success_rate": 0.0,
            "avg_steps": 87.0,
            "avg_reward": -100.5,
            "avg_recharges": 2.0,
        },
        {
            "algorithm": "ppo",
            "label": "ppo_reference",
            "status": "evaluated",
            "avg_cleaned_ratio": 1.0,
            "success_rate": 1.0,
            "avg_steps": 150.0,
            "avg_reward": -53.0,
            "avg_recharges": 3.0,
        },
        {
            "algorithm": "sarsa",
            "label": "sarsa_reference",
            "status": "evaluated",
            "avg_cleaned_ratio": 1.0,
            "success_rate": 1.0,
            "avg_steps": 155.0,
            "avg_reward": -43.5,
            "avg_recharges": 3.0,
        },
    ]

    report = generate_markdown_report(
        rows=rows,
        map_name="complex_charge_bastion",
        battery_profile="evaluation",
        eval_episodes=200,
        csv_path=tmp_path / "comparison.csv",
    )

    assert "| PPO | `ppo_reference` | 100.00% | 100.00% | 150.00 | -53.00 | 3.00 |" in report
    assert "- Solved references: PPO, SARSA." in report
    assert "- Fastest successful route: PPO at 150.00 steps." in report
    assert "- Best average reward: SARSA at -43.50." in report
    assert "Q-learning (66.67% cleaned, 0.00% success)" in report

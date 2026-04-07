from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.q_learning_agent import QLearningAgent  # noqa: E402
from configs.map_presets import MAP_PRESETS  # noqa: E402
from utils.experiment_utils import build_env, get_checkpoint_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multiple SweepAgent Q-learning checkpoints in batch.",
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        required=True,
        help="One or more map preset names to evaluate.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="One or more seeds to evaluate.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        required=True,
        help="Training episode count used in the checkpoint names.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=100,
        help="Number of greedy evaluation episodes per checkpoint.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if any expected checkpoint is missing.",
    )
    return parser.parse_args()


def validate_maps(map_names: list[str]) -> None:
    invalid_maps = [map_name for map_name in map_names if map_name not in MAP_PRESETS]
    if invalid_maps:
        supported_maps = ", ".join(MAP_PRESETS.keys())
        invalid_str = ", ".join(invalid_maps)
        raise ValueError(
            f"Unknown map name(s): {invalid_str}. Supported maps: {supported_maps}"
        )


def ensure_output_dirs() -> Path:
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def evaluate_checkpoint(
    map_name: str,
    seed: int,
    episodes: int,
    eval_episodes: int,
) -> dict[str, str | int | float]:
    checkpoint_path = get_checkpoint_path(
        map_name=map_name,
        episodes=episodes,
        seed=seed,
    )
    if not checkpoint_path.exists():
        return {
            "map_name": map_name,
            "seed": seed,
            "episodes": episodes,
            "eval_episodes": eval_episodes,
            "status": "missing_checkpoint",
            "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "avg_reward": 0.0,
            "avg_steps": 0.0,
            "avg_cleaned_ratio": 0.0,
            "success_rate": 0.0,
            "avg_recharges": 0.0,
            "termination_all_cleaned": 0,
            "termination_battery_depleted": 0,
            "termination_step_limit": 0,
            "termination_other": 0,
        }

    agent = QLearningAgent.load(checkpoint_path)
    env = build_env(map_name=map_name, battery_profile="evaluation")

    rewards: list[float] = []
    steps: list[float] = []
    cleaned_ratios: list[float] = []
    successes: list[float] = []
    recharges: list[float] = []
    terminations = {
        "all_cleaned": 0,
        "battery_depleted": 0,
        "step_limit": 0,
        "other": 0,
    }

    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        recharge_count = 0
        final_info: dict[str, float | str] = {}

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            total_reward += reward
            recharge_count += int(info["recharged"])
            final_info = info

        rewards.append(total_reward)
        steps.append(float(final_info["steps_taken"]))
        cleaned_ratios.append(float(final_info["cleaned_ratio"]))
        successes.append(1.0 if float(final_info["cleaned_ratio"]) == 1.0 else 0.0)
        recharges.append(float(recharge_count))

        termination_reason = str(final_info["termination_reason"])
        if termination_reason in terminations:
            terminations[termination_reason] += 1
        else:
            terminations["other"] += 1

    return {
        "map_name": map_name,
        "seed": seed,
        "episodes": episodes,
        "eval_episodes": eval_episodes,
        "status": "evaluated",
        "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "avg_reward": round(mean(rewards), 4),
        "avg_steps": round(mean(steps), 4),
        "avg_cleaned_ratio": round(mean(cleaned_ratios), 6),
        "success_rate": round(mean(successes), 6),
        "avg_recharges": round(mean(recharges), 4),
        "termination_all_cleaned": terminations["all_cleaned"],
        "termination_battery_depleted": terminations["battery_depleted"],
        "termination_step_limit": terminations["step_limit"],
        "termination_other": terminations["other"],
    }


def save_detailed_csv(results: list[dict[str, str | int | float]]) -> Path:
    output_path = ensure_output_dirs() / "q_batch_eval_results.csv"
    fieldnames = [
        "map_name",
        "seed",
        "episodes",
        "eval_episodes",
        "status",
        "checkpoint_path",
        "avg_reward",
        "avg_steps",
        "avg_cleaned_ratio",
        "success_rate",
        "avg_recharges",
        "termination_all_cleaned",
        "termination_battery_depleted",
        "termination_step_limit",
        "termination_other",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return output_path


def build_summary_rows(
    results: list[dict[str, str | int | float]],
) -> list[dict[str, str | int | float]]:
    summary_rows: list[dict[str, str | int | float]] = []

    evaluated_results = [row for row in results if row["status"] == "evaluated"]
    map_names = sorted({str(row["map_name"]) for row in results})

    for map_name in map_names:
        matching = [
            row for row in evaluated_results if str(row["map_name"]) == map_name
        ]
        missing_count = sum(
            1
            for row in results
            if str(row["map_name"]) == map_name and row["status"] == "missing_checkpoint"
        )

        if not matching:
            summary_rows.append(
                {
                    "map_name": map_name,
                    "evaluated_runs": 0,
                    "missing_runs": missing_count,
                    "mean_success_rate": 0.0,
                    "min_success_rate": 0.0,
                    "max_success_rate": 0.0,
                    "mean_cleaned_ratio": 0.0,
                    "mean_steps": 0.0,
                    "mean_recharges": 0.0,
                }
            )
            continue

        success_rates = [float(row["success_rate"]) for row in matching]
        cleaned_ratios = [float(row["avg_cleaned_ratio"]) for row in matching]
        step_counts = [float(row["avg_steps"]) for row in matching]
        recharge_counts = [float(row["avg_recharges"]) for row in matching]

        summary_rows.append(
            {
                "map_name": map_name,
                "evaluated_runs": len(matching),
                "missing_runs": missing_count,
                "mean_success_rate": round(mean(success_rates), 6),
                "min_success_rate": round(min(success_rates), 6),
                "max_success_rate": round(max(success_rates), 6),
                "mean_cleaned_ratio": round(mean(cleaned_ratios), 6),
                "mean_steps": round(mean(step_counts), 4),
                "mean_recharges": round(mean(recharge_counts), 4),
            }
        )

    return summary_rows


def save_summary_csv(summary_rows: list[dict[str, str | int | float]]) -> Path:
    output_path = ensure_output_dirs() / "q_batch_eval_summary.csv"
    fieldnames = [
        "map_name",
        "evaluated_runs",
        "missing_runs",
        "mean_success_rate",
        "min_success_rate",
        "max_success_rate",
        "mean_cleaned_ratio",
        "mean_steps",
        "mean_recharges",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    return output_path


def main() -> None:
    args = parse_args()
    validate_maps(args.maps)

    results: list[dict[str, str | int | float]] = []

    print("=== Batch Evaluation Configuration ===")
    print(f"maps: {', '.join(args.maps)}")
    print(f"seeds: {', '.join(str(seed) for seed in args.seeds)}")
    print(f"checkpoint episodes: {args.episodes}")
    print(f"eval_episodes: {args.eval_episodes}")

    for map_name in args.maps:
        for seed in args.seeds:
            result = evaluate_checkpoint(
                map_name=map_name,
                seed=seed,
                episodes=args.episodes,
                eval_episodes=args.eval_episodes,
            )
            results.append(result)
            print(
                f"[{result['status']}] map={map_name} "
                f"seed={seed} "
                f"success_rate={result['success_rate']}"
            )

    missing_count = sum(1 for row in results if row["status"] == "missing_checkpoint")
    if missing_count and args.strict_missing:
        raise SystemExit(
            f"Missing {missing_count} expected checkpoint(s). "
            "Re-run without --strict-missing to keep partial results."
        )

    results.sort(key=lambda row: (str(row["map_name"]), int(row["seed"])))
    detailed_path = save_detailed_csv(results)
    summary_rows = build_summary_rows(results)
    summary_path = save_summary_csv(summary_rows)

    print("\n=== Batch Evaluation Summary ===")
    print(f"evaluated: {sum(1 for row in results if row['status'] == 'evaluated')}")
    print(f"missing: {missing_count}")
    print(f"detailed_csv: {detailed_path.relative_to(PROJECT_ROOT)}")
    print(f"summary_csv: {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

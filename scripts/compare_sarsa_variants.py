from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.sarsa_agent import SarsaAgent
from configs.map_presets import (
    DISCOUNT_FACTOR,
    EPSILON_START,
)
from scripts.train_sarsa import (
    evaluate_sarsa_agent,
    format_sarsa_eval_result,
    get_sarsa_best_checkpoint_path,
    get_sarsa_checkpoint_path,
)

VARIANT_ORDER = ("plain", "shaping", "guided", "guided_shaping")


@dataclass(frozen=True)
class SarsaVariantSpec:
    name: str
    guided_exploration_ratio: float
    reward_move_toward_relay_charger: float
    penalty_move_away_from_relay_charger: float
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled SARSA ablation across plain, shaping-only, "
            "guided-only, and guided+shaping variants."
        )
    )
    parser.add_argument("--map-name", type=str, default="complex_charge_bastion")
    parser.add_argument("--episodes", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--discount-factor", type=float, default=DISCOUNT_FACTOR)
    parser.add_argument("--epsilon-start", type=float, default=EPSILON_START)
    parser.add_argument("--epsilon-decay", type=float, default=0.99995)
    parser.add_argument("--epsilon-min", type=float, default=0.10)
    parser.add_argument(
        "--state-abstraction-mode",
        type=str,
        choices=("identity", "safety_margin", "charger_context"),
        default="charger_context",
    )
    parser.add_argument("--safety-margin-bucket-size", type=int, default=5)
    parser.add_argument(
        "--battery-profile",
        type=str,
        choices=("training", "evaluation"),
        default="evaluation",
    )
    parser.add_argument("--battery-capacity-override", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument(
        "--comparison-eval-episodes",
        type=int,
        default=200,
        help="Greedy evaluation episodes run on each saved best checkpoint.",
    )
    parser.add_argument("--guided-exploration-ratio", type=float, default=0.9)
    parser.add_argument("--relay-reward", type=float, default=0.5)
    parser.add_argument("--relay-penalty", type=float, default=-0.75)
    parser.add_argument(
        "--variant",
        action="append",
        choices=(*VARIANT_ORDER, "all"),
        default=[],
        help="Variant to run. Repeat this flag or omit it to run all variants.",
    )
    parser.add_argument(
        "--checkpoint-tag-prefix",
        type=str,
        default="sarsa_ablation",
        help="Prefix used for variant checkpoint tags.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional output CSV path. Defaults under outputs/logs.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip training a variant when its final checkpoint already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the train commands without running them.",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used to launch train_sarsa.py.",
    )
    return parser.parse_args()


def normalize_variant_names(requested_variants: list[str]) -> list[str]:
    if not requested_variants or "all" in requested_variants:
        return list(VARIANT_ORDER)
    return list(dict.fromkeys(requested_variants))


def build_variant_specs(
    requested_variants: list[str],
    guided_exploration_ratio: float,
    relay_reward: float,
    relay_penalty: float,
) -> list[SarsaVariantSpec]:
    names = normalize_variant_names(requested_variants)
    specs = {
        "plain": SarsaVariantSpec(
            name="plain",
            guided_exploration_ratio=0.0,
            reward_move_toward_relay_charger=0.0,
            penalty_move_away_from_relay_charger=0.0,
            description="no guided exploration and no relay shaping",
        ),
        "shaping": SarsaVariantSpec(
            name="shaping",
            guided_exploration_ratio=0.0,
            reward_move_toward_relay_charger=relay_reward,
            penalty_move_away_from_relay_charger=relay_penalty,
            description="relay reward shaping only",
        ),
        "guided": SarsaVariantSpec(
            name="guided",
            guided_exploration_ratio=guided_exploration_ratio,
            reward_move_toward_relay_charger=0.0,
            penalty_move_away_from_relay_charger=0.0,
            description="relay-aware guided exploration only",
        ),
        "guided_shaping": SarsaVariantSpec(
            name="guided_shaping",
            guided_exploration_ratio=guided_exploration_ratio,
            reward_move_toward_relay_charger=relay_reward,
            penalty_move_away_from_relay_charger=relay_penalty,
            description="guided exploration plus relay reward shaping",
        ),
    }
    return [specs[name] for name in names]


def build_checkpoint_tag(prefix: str, variant_name: str, episodes: int) -> str:
    if not prefix:
        return f"{variant_name}_ep{episodes}"
    return f"{prefix}_{variant_name}_ep{episodes}"


def default_output_csv_path(map_name: str, episodes: int, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "logs"
        / f"sarsa_variant_comparison_{map_name}_ep_{episodes}_seed_{seed}.csv"
    )


def build_train_command(
    args: argparse.Namespace,
    variant: SarsaVariantSpec,
    checkpoint_tag: str,
) -> list[str]:
    command = [
        args.python_executable,
        str(PROJECT_ROOT / "scripts" / "train_sarsa.py"),
        "--map-name",
        args.map_name,
        "--battery-profile",
        args.battery_profile,
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--print-every",
        str(args.print_every),
        "--learning-rate",
        str(args.learning_rate),
        "--discount-factor",
        str(args.discount_factor),
        "--epsilon-start",
        str(args.epsilon_start),
        "--epsilon-decay",
        str(args.epsilon_decay),
        "--epsilon-min",
        str(args.epsilon_min),
        "--state-abstraction-mode",
        args.state_abstraction_mode,
        "--safety-margin-bucket-size",
        str(args.safety_margin_bucket_size),
        "--guided-exploration-ratio",
        str(variant.guided_exploration_ratio),
        "--reward-move-toward-relay-charger",
        str(variant.reward_move_toward_relay_charger),
        "--penalty-move-away-from-relay-charger",
        str(variant.penalty_move_away_from_relay_charger),
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-every",
        str(args.eval_every),
        "--checkpoint-tag",
        checkpoint_tag,
        "--save-best-eval-checkpoint",
    ]
    if args.battery_capacity_override > 0:
        command.extend(
            ["--battery-capacity-override", str(args.battery_capacity_override)]
        )
    return command


def read_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return {}
    with checkpoint_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    metadata = payload.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def evaluate_checkpoint(
    checkpoint_path: Path,
    map_name: str,
    eval_episodes: int,
    battery_capacity_override: int | None,
) -> dict[str, float]:
    if eval_episodes <= 0 or not checkpoint_path.exists():
        return {
            "avg_reward": 0.0,
            "avg_steps": 0.0,
            "avg_cleaned_ratio": 0.0,
            "success_rate": 0.0,
        }
    agent = SarsaAgent.load(checkpoint_path)
    return evaluate_sarsa_agent(
        map_name=map_name,
        eval_episodes=eval_episodes,
        agent=agent,
        battery_capacity_override=battery_capacity_override,
    )


def build_summary_row(
    args: argparse.Namespace,
    variant: SarsaVariantSpec,
    checkpoint_tag: str,
    checkpoint_path: Path,
    best_checkpoint_path: Path,
) -> dict[str, Any]:
    metadata = read_checkpoint_metadata(checkpoint_path)
    best_metadata = read_checkpoint_metadata(best_checkpoint_path)
    best_eval_result = metadata.get("best_eval_result", {})
    if not best_eval_result:
        best_eval_result = best_metadata.get("best_eval_result", {})
    if not isinstance(best_eval_result, dict):
        best_eval_result = {}

    comparison_checkpoint_path = (
        best_checkpoint_path if best_checkpoint_path.exists() else checkpoint_path
    )
    comparison_eval = evaluate_checkpoint(
        checkpoint_path=comparison_checkpoint_path,
        map_name=args.map_name,
        eval_episodes=args.comparison_eval_episodes,
        battery_capacity_override=(
            args.battery_capacity_override
            if args.battery_capacity_override > 0
            else None
        ),
    )

    return {
        "variant": variant.name,
        "description": variant.description,
        "map_name": args.map_name,
        "seed": args.seed,
        "episodes": args.episodes,
        "checkpoint_tag": checkpoint_tag,
        "checkpoint_path": str(checkpoint_path),
        "best_checkpoint_path": (
            str(best_checkpoint_path) if best_checkpoint_path.exists() else ""
        ),
        "best_checkpoint_episodes": int(
            metadata.get(
                "best_checkpoint_episodes",
                best_metadata.get("best_checkpoint_episodes", 0),
            )
            or 0
        ),
        "guided_exploration_ratio": variant.guided_exploration_ratio,
        "reward_move_toward_relay_charger": (
            variant.reward_move_toward_relay_charger
        ),
        "penalty_move_away_from_relay_charger": (
            variant.penalty_move_away_from_relay_charger
        ),
        "best_eval_avg_reward": metric_value(best_eval_result, "avg_reward"),
        "best_eval_avg_steps": metric_value(best_eval_result, "avg_steps"),
        "best_eval_avg_cleaned_ratio": metric_value(
            best_eval_result, "avg_cleaned_ratio"
        ),
        "best_eval_success_rate": metric_value(best_eval_result, "success_rate"),
        "comparison_eval_episodes": args.comparison_eval_episodes,
        "comparison_eval_avg_reward": comparison_eval["avg_reward"],
        "comparison_eval_avg_steps": comparison_eval["avg_steps"],
        "comparison_eval_avg_cleaned_ratio": comparison_eval["avg_cleaned_ratio"],
        "comparison_eval_success_rate": comparison_eval["success_rate"],
    }


def save_summary_csv(output_path: Path, rows: list[dict[str, Any]]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "description",
        "map_name",
        "seed",
        "episodes",
        "checkpoint_tag",
        "checkpoint_path",
        "best_checkpoint_path",
        "best_checkpoint_episodes",
        "guided_exploration_ratio",
        "reward_move_toward_relay_charger",
        "penalty_move_away_from_relay_charger",
        "best_eval_avg_reward",
        "best_eval_avg_steps",
        "best_eval_avg_cleaned_ratio",
        "best_eval_success_rate",
        "comparison_eval_episodes",
        "comparison_eval_avg_reward",
        "comparison_eval_avg_steps",
        "comparison_eval_avg_cleaned_ratio",
        "comparison_eval_success_rate",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def print_row_summary(row: dict[str, Any]) -> None:
    comparison_eval = {
        "avg_reward": row["comparison_eval_avg_reward"],
        "avg_steps": row["comparison_eval_avg_steps"],
        "avg_cleaned_ratio": row["comparison_eval_avg_cleaned_ratio"],
        "success_rate": row["comparison_eval_success_rate"],
    }
    print(
        f"{row['variant']}: "
        f"best_eval avg_steps={row['best_eval_avg_steps']:.2f}, "
        f"cleaned={row['best_eval_avg_cleaned_ratio'] * 100:.2f}%, "
        f"success={row['best_eval_success_rate'] * 100:.2f}% | "
        f"comparison {row['comparison_eval_episodes']}eps "
        f"{format_sarsa_eval_result(comparison_eval)}"
    )


def main() -> None:
    args = parse_args()
    output_path = (
        Path(args.output_csv)
        if args.output_csv
        else default_output_csv_path(args.map_name, args.episodes, args.seed)
    )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    rows: list[dict[str, Any]] = []
    variants = build_variant_specs(
        requested_variants=args.variant,
        guided_exploration_ratio=args.guided_exploration_ratio,
        relay_reward=args.relay_reward,
        relay_penalty=args.relay_penalty,
    )

    for variant in variants:
        checkpoint_tag = build_checkpoint_tag(
            prefix=args.checkpoint_tag_prefix,
            variant_name=variant.name,
            episodes=args.episodes,
        )
        checkpoint_path = get_sarsa_checkpoint_path(
            map_name=args.map_name,
            episodes=args.episodes,
            seed=args.seed,
            checkpoint_tag=checkpoint_tag,
        )
        best_checkpoint_path = get_sarsa_best_checkpoint_path(
            map_name=args.map_name,
            seed=args.seed,
            checkpoint_tag=checkpoint_tag,
        )
        command = build_train_command(
            args=args,
            variant=variant,
            checkpoint_tag=checkpoint_tag,
        )

        print(f"\n=== SARSA Variant: {variant.name} ===", flush=True)
        print(variant.description, flush=True)
        print(" ".join(command), flush=True)
        if args.dry_run:
            continue

        if args.reuse_existing and checkpoint_path.exists():
            print(f"Reusing existing checkpoint: {checkpoint_path}")
        else:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

        row = build_summary_row(
            args=args,
            variant=variant,
            checkpoint_tag=checkpoint_tag,
            checkpoint_path=checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
        )
        rows.append(row)
        print_row_summary(row)

    if args.dry_run:
        print("\nDry run complete. No training or CSV write performed.")
        return

    saved_path = save_summary_csv(output_path=output_path, rows=rows)
    print(f"\nSaved SARSA variant comparison CSV to: {saved_path}")


if __name__ == "__main__":
    main()

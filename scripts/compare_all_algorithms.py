from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from agents.q_learning_agent import QLearningAgent
from agents.sarsa_agent import SarsaAgent
from utils.experiment_utils import build_env


@dataclass(frozen=True)
class AlgorithmCheckpointSpec:
    algorithm: str
    label: str
    checkpoint_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Q-learning, DQN, PPO, and SARSA checkpoints under the "
            "same SweepAgent environment settings."
        )
    )
    parser.add_argument("--map-name", type=str, default="complex_charge_bastion")
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument(
        "--battery-profile",
        type=str,
        choices=("training", "evaluation"),
        default="evaluation",
    )
    parser.add_argument("--battery-capacity-override", type=int, default=0)
    parser.add_argument(
        "--q-checkpoint",
        type=str,
        default="",
        help="Optional Q-learning checkpoint path. Defaults to current bastion reference.",
    )
    parser.add_argument(
        "--dqn-checkpoint",
        type=str,
        default="",
        help="Optional DQN checkpoint path. Defaults to current bastion reference.",
    )
    parser.add_argument(
        "--ppo-checkpoint",
        type=str,
        default="",
        help="Optional PPO checkpoint path. Defaults to current bastion reference.",
    )
    parser.add_argument(
        "--sarsa-checkpoint",
        type=str,
        default="",
        help="Optional SARSA checkpoint path. Defaults to current bastion reference.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        choices=("q_learning", "dqn", "ppo", "sarsa", "all"),
        default=["all"],
        help="Algorithms to evaluate.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Optional output CSV path. Defaults under outputs/logs.",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="",
        help="Optional Markdown report path. Defaults under outputs/logs.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail when a requested checkpoint is missing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used to load DQN/PPO checkpoints. Defaults to cpu for comparable eval.",
    )
    return parser.parse_args()


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def default_checkpoint_specs(map_name: str) -> dict[str, AlgorithmCheckpointSpec]:
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    if map_name == "complex_charge_bastion":
        return {
            "q_learning": AlgorithmCheckpointSpec(
                algorithm="q_learning",
                label="q_learning_battery_adapt_400k_seed505",
                checkpoint_path=(
                    checkpoint_dir
                    / "q_learning_agent_complex_charge_bastion_ep_400000_seed_505.json"
                ),
            ),
            "dqn": AlgorithmCheckpointSpec(
                algorithm="dqn",
                label="dqn_seed418_v2relay_shape_finalroute",
                checkpoint_path=(
                    checkpoint_dir
                    / "dqn_agent_complex_charge_bastion_best_eval_seed_418_v2relay_shape_finalroute.pt"
                ),
            ),
            "ppo": AlgorithmCheckpointSpec(
                algorithm="ppo",
                label="ppo_finalrelay_curriculum6500",
                checkpoint_path=(
                    checkpoint_dir
                    / "ppo_agent_complex_charge_bastion_best_eval_seed_42_ppo_finalrelay_curriculum6500.pt"
                ),
            ),
            "sarsa": AlgorithmCheckpointSpec(
                algorithm="sarsa",
                label="sarsa_guided09_relay100k",
                checkpoint_path=(
                    checkpoint_dir
                    / "sarsa_agent_complex_charge_bastion_best_eval_seed_42_guided09_relay100k.json"
                ),
            ),
        }

    return {
        "q_learning": AlgorithmCheckpointSpec(
            algorithm="q_learning",
            label="q_learning_default",
            checkpoint_path=(
                checkpoint_dir / f"q_learning_agent_{map_name}_ep_10_seed_42.json"
            ),
        ),
        "dqn": AlgorithmCheckpointSpec(
            algorithm="dqn",
            label="dqn_default",
            checkpoint_path=checkpoint_dir / f"dqn_agent_{map_name}_best_eval_seed_42.pt",
        ),
        "ppo": AlgorithmCheckpointSpec(
            algorithm="ppo",
            label="ppo_default",
            checkpoint_path=checkpoint_dir / f"ppo_agent_{map_name}_best_eval_seed_42.pt",
        ),
        "sarsa": AlgorithmCheckpointSpec(
            algorithm="sarsa",
            label="sarsa_default",
            checkpoint_path=checkpoint_dir / f"sarsa_agent_{map_name}_ep_10000_seed_42.json",
        ),
    }


def selected_algorithms(include_values: list[str]) -> list[str]:
    if "all" in include_values:
        return ["q_learning", "dqn", "ppo", "sarsa"]
    return list(dict.fromkeys(include_values))


def build_checkpoint_specs(args: argparse.Namespace) -> list[AlgorithmCheckpointSpec]:
    defaults = default_checkpoint_specs(args.map_name)
    overrides = {
        "q_learning": args.q_checkpoint,
        "dqn": args.dqn_checkpoint,
        "ppo": args.ppo_checkpoint,
        "sarsa": args.sarsa_checkpoint,
    }
    specs: list[AlgorithmCheckpointSpec] = []

    for algorithm in selected_algorithms(args.include):
        default_spec = defaults[algorithm]
        override = overrides[algorithm]
        if override:
            checkpoint_path = resolve_project_path(override)
            label = checkpoint_path.stem
        else:
            checkpoint_path = default_spec.checkpoint_path
            label = default_spec.label
        specs.append(
            AlgorithmCheckpointSpec(
                algorithm=algorithm,
                label=label,
                checkpoint_path=checkpoint_path,
            )
        )

    return specs


def read_json_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    with checkpoint_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    metadata = payload.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def read_torch_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    metadata = payload.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def metadata_value(metadata: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in metadata and metadata[key] not in (None, ""):
            return metadata[key]
    return ""


def load_agent(spec: AlgorithmCheckpointSpec, device: str):
    if spec.algorithm == "q_learning":
        return QLearningAgent.load(spec.checkpoint_path)
    if spec.algorithm == "dqn":
        return DQNAgent.load(spec.checkpoint_path, device=device, training=False)
    if spec.algorithm == "ppo":
        return PPOAgent.load(spec.checkpoint_path, device=device, training=False)
    if spec.algorithm == "sarsa":
        return SarsaAgent.load(spec.checkpoint_path)
    raise ValueError(f"Unsupported algorithm: {spec.algorithm}")


def read_metadata(spec: AlgorithmCheckpointSpec) -> dict[str, Any]:
    if not spec.checkpoint_path.exists():
        return {}
    if spec.algorithm in {"q_learning", "sarsa"}:
        return read_json_checkpoint_metadata(spec.checkpoint_path)
    return read_torch_checkpoint_metadata(spec.checkpoint_path)


def policy_action_fn(spec: AlgorithmCheckpointSpec, agent) -> Callable[[tuple[int, int, int, int]], int]:
    if spec.algorithm == "dqn":
        return lambda state: int(agent.select_action(state, training=False))
    return lambda state: int(agent.get_policy_action(state))


def evaluate_policy(
    map_name: str,
    eval_episodes: int,
    action_fn: Callable[[tuple[int, int, int, int]], int],
    battery_profile: str = "evaluation",
    battery_capacity_override: int | None = None,
) -> dict[str, float | int]:
    if eval_episodes <= 0:
        return {
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

    env = build_env(
        map_name=map_name,
        battery_profile=battery_profile,  # type: ignore[arg-type]
        battery_capacity_override=battery_capacity_override,
    )
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
            action = action_fn(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
            recharge_count += int(info["recharged"])
            final_info = info

        cleaned_ratio = float(final_info["cleaned_ratio"])
        rewards.append(total_reward)
        steps.append(float(final_info["steps_taken"]))
        cleaned_ratios.append(cleaned_ratio)
        successes.append(1.0 if cleaned_ratio == 1.0 else 0.0)
        recharges.append(float(recharge_count))

        termination_reason = str(final_info["termination_reason"])
        if termination_reason in terminations:
            terminations[termination_reason] += 1
        else:
            terminations["other"] += 1

    return {
        "avg_reward": mean(rewards),
        "avg_steps": mean(steps),
        "avg_cleaned_ratio": mean(cleaned_ratios),
        "success_rate": mean(successes),
        "avg_recharges": mean(recharges),
        "termination_all_cleaned": terminations["all_cleaned"],
        "termination_battery_depleted": terminations["battery_depleted"],
        "termination_step_limit": terminations["step_limit"],
        "termination_other": terminations["other"],
    }


def missing_checkpoint_row(
    args: argparse.Namespace,
    spec: AlgorithmCheckpointSpec,
) -> dict[str, str | int | float]:
    return {
        "algorithm": spec.algorithm,
        "label": spec.label,
        "status": "missing_checkpoint",
        "map_name": args.map_name,
        "battery_profile": args.battery_profile,
        "eval_episodes": args.eval_episodes,
        "checkpoint_path": str(spec.checkpoint_path),
        "checkpoint_episodes": "",
        "checkpoint_tag": "",
        "source_seed": "",
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


def evaluated_checkpoint_row(
    args: argparse.Namespace,
    spec: AlgorithmCheckpointSpec,
    metadata: dict[str, Any],
    eval_result: dict[str, float | int],
) -> dict[str, str | int | float]:
    return {
        "algorithm": spec.algorithm,
        "label": spec.label,
        "status": "evaluated",
        "map_name": args.map_name,
        "battery_profile": args.battery_profile,
        "eval_episodes": args.eval_episodes,
        "checkpoint_path": str(spec.checkpoint_path),
        "checkpoint_episodes": metadata_value(
            metadata,
            "best_checkpoint_episodes",
            "checkpoint_episodes",
            "cumulative_episodes",
            "episodes",
        ),
        "checkpoint_tag": metadata_value(metadata, "checkpoint_tag"),
        "source_seed": metadata_value(metadata, "seed"),
        "avg_reward": round(float(eval_result["avg_reward"]), 4),
        "avg_steps": round(float(eval_result["avg_steps"]), 4),
        "avg_cleaned_ratio": round(float(eval_result["avg_cleaned_ratio"]), 6),
        "success_rate": round(float(eval_result["success_rate"]), 6),
        "avg_recharges": round(float(eval_result["avg_recharges"]), 4),
        "termination_all_cleaned": int(eval_result["termination_all_cleaned"]),
        "termination_battery_depleted": int(eval_result["termination_battery_depleted"]),
        "termination_step_limit": int(eval_result["termination_step_limit"]),
        "termination_other": int(eval_result["termination_other"]),
    }


def evaluate_checkpoint(
    args: argparse.Namespace,
    spec: AlgorithmCheckpointSpec,
) -> dict[str, str | int | float]:
    if not spec.checkpoint_path.exists():
        if args.strict_missing:
            raise FileNotFoundError(f"Missing checkpoint: {spec.checkpoint_path}")
        return missing_checkpoint_row(args, spec)

    metadata = read_metadata(spec)
    agent = load_agent(spec, device=args.device)
    eval_result = evaluate_policy(
        map_name=args.map_name,
        eval_episodes=args.eval_episodes,
        action_fn=policy_action_fn(spec, agent),
        battery_profile=args.battery_profile,
        battery_capacity_override=(
            args.battery_capacity_override
            if args.battery_capacity_override > 0
            else None
        ),
    )
    return evaluated_checkpoint_row(args, spec, metadata, eval_result)


def default_output_csv_path(map_name: str, eval_episodes: int) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "logs"
        / f"algorithm_comparison_{map_name}_eval_{eval_episodes}.csv"
    )


def default_output_markdown_path(map_name: str, eval_episodes: int) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "logs"
        / f"algorithm_comparison_{map_name}_eval_{eval_episodes}.md"
    )


def save_csv(output_path: Path, rows: list[dict[str, str | int | float]]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "label",
        "status",
        "map_name",
        "battery_profile",
        "eval_episodes",
        "checkpoint_path",
        "checkpoint_episodes",
        "checkpoint_tag",
        "source_seed",
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
        writer.writerows(rows)
    return output_path


def display_algorithm_name(algorithm: str) -> str:
    display_names = {
        "q_learning": "Q-learning",
        "dqn": "DQN",
        "ppo": "PPO",
        "sarsa": "SARSA",
    }
    return display_names.get(algorithm, algorithm)


def row_float(row: dict[str, str | int | float], key: str) -> float:
    return float(row.get(key, 0.0) or 0.0)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_metric(value: float) -> str:
    return f"{value:.2f}"


def generate_markdown_report(
    rows: list[dict[str, str | int | float]],
    map_name: str,
    battery_profile: str,
    eval_episodes: int,
    csv_path: Path,
) -> str:
    evaluated_rows = [row for row in rows if row["status"] == "evaluated"]
    solved_rows = [
        row for row in evaluated_rows if row_float(row, "success_rate") == 1.0
    ]
    partial_rows = [
        row for row in evaluated_rows if row_float(row, "success_rate") < 1.0
    ]

    lines = [
        f"# Algorithm Comparison: {map_name}",
        "",
        f"- evaluation profile: `{battery_profile}`",
        f"- evaluation episodes: `{eval_episodes}`",
        f"- source CSV: `{csv_path}`",
        "",
        "| Algorithm | Reference | Cleaned Ratio | Success Rate | Avg Steps | Avg Reward | Avg Recharges |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        if row["status"] != "evaluated":
            lines.append(
                "| "
                f"{display_algorithm_name(str(row['algorithm']))} | "
                f"`{row['label']}` | missing | missing | missing | missing | missing |"
            )
            continue

        lines.append(
            "| "
            f"{display_algorithm_name(str(row['algorithm']))} | "
            f"`{row['label']}` | "
            f"{format_percent(row_float(row, 'avg_cleaned_ratio'))} | "
            f"{format_percent(row_float(row, 'success_rate'))} | "
            f"{format_metric(row_float(row, 'avg_steps'))} | "
            f"{format_metric(row_float(row, 'avg_reward'))} | "
            f"{format_metric(row_float(row, 'avg_recharges'))} |"
        )

    lines.extend(["", "## Summary", ""])

    if solved_rows:
        solved_names = ", ".join(
            display_algorithm_name(str(row["algorithm"])) for row in solved_rows
        )
        fastest = min(solved_rows, key=lambda row: row_float(row, "avg_steps"))
        lines.append(f"- Solved references: {solved_names}.")
        lines.append(
            "- Fastest successful route: "
            f"{display_algorithm_name(str(fastest['algorithm']))} "
            f"at {format_metric(row_float(fastest, 'avg_steps'))} steps."
        )
    else:
        lines.append("- No evaluated reference fully solved the map.")

    if evaluated_rows:
        best_reward = max(evaluated_rows, key=lambda row: row_float(row, "avg_reward"))
        lines.append(
            "- Best average reward: "
            f"{display_algorithm_name(str(best_reward['algorithm']))} "
            f"at {format_metric(row_float(best_reward, 'avg_reward'))}."
        )

    if partial_rows:
        partial_summary = ", ".join(
            (
                f"{display_algorithm_name(str(row['algorithm']))} "
                f"({format_percent(row_float(row, 'avg_cleaned_ratio'))} cleaned, "
                f"{format_percent(row_float(row, 'success_rate'))} success)"
            )
            for row in partial_rows
        )
        lines.append(f"- Partial or failed references: {partial_summary}.")

    lines.append("")
    return "\n".join(lines)


def save_markdown_report(
    output_path: Path,
    rows: list[dict[str, str | int | float]],
    map_name: str,
    battery_profile: str,
    eval_episodes: int,
    csv_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        generate_markdown_report(
            rows=rows,
            map_name=map_name,
            battery_profile=battery_profile,
            eval_episodes=eval_episodes,
            csv_path=csv_path,
        ),
        encoding="utf-8",
    )
    return output_path


def print_summary_row(row: dict[str, str | int | float]) -> None:
    if row["status"] != "evaluated":
        print(f"{row['algorithm']}: missing checkpoint | {row['checkpoint_path']}")
        return
    print(
        f"{row['algorithm']} ({row['label']}): "
        f"avg_reward={float(row['avg_reward']):.2f} | "
        f"avg_steps={float(row['avg_steps']):.2f} | "
        f"avg_cleaned_ratio={float(row['avg_cleaned_ratio']) * 100:.2f}% | "
        f"success_rate={float(row['success_rate']) * 100:.2f}%"
    )


def main() -> None:
    args = parse_args()
    output_path = (
        resolve_project_path(args.output_csv)
        if args.output_csv
        else default_output_csv_path(args.map_name, args.eval_episodes)
    )
    markdown_output_path = (
        resolve_project_path(args.output_md)
        if args.output_md
        else default_output_markdown_path(args.map_name, args.eval_episodes)
    )
    rows: list[dict[str, str | int | float]] = []

    for spec in build_checkpoint_specs(args):
        print(f"Evaluating {spec.algorithm}: {spec.checkpoint_path}")
        row = evaluate_checkpoint(args, spec)
        rows.append(row)
        print_summary_row(row)

    saved_path = save_csv(output_path, rows)
    print(f"Saved algorithm comparison CSV to: {saved_path}")
    saved_markdown_path = save_markdown_report(
        output_path=markdown_output_path,
        rows=rows,
        map_name=args.map_name,
        battery_profile=args.battery_profile,
        eval_episodes=args.eval_episodes,
        csv_path=saved_path,
    )
    print(f"Saved algorithm comparison Markdown to: {saved_markdown_path}")


if __name__ == "__main__":
    main()

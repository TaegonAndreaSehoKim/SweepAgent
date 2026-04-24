from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from configs.map_presets import (  # noqa: E402
    DISCOUNT_FACTOR,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    PRINT_EVERY,
)
from utils.dqn_experiment_utils import (  # noqa: E402
    ensure_dqn_output_dirs,
    get_dqn_checkpoint_path,
    infer_dqn_checkpoint_episodes,
    read_dqn_checkpoint_metadata,
)


SUMMARY_FIELDS = [
    "slice_index",
    "slice_episodes",
    "checkpoint_episodes",
    "map_name",
    "seed",
    "feature_version",
    "battery_profile",
    "eval_episodes",
    "avg_reward",
    "avg_steps",
    "avg_cleaned_ratio",
    "success_rate",
    "checkpoint_path",
    "log_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train DQN in episode slices so intermediate checkpoints can be "
            "evaluated to locate regression or collapse windows."
        ),
    )
    parser.add_argument("--map-name", required=True, type=str)
    parser.add_argument(
        "--total-episodes",
        required=True,
        type=int,
        help="Total cumulative episode target after all slices finish.",
    )
    parser.add_argument(
        "--slice-episodes",
        required=True,
        type=int,
        help="Episodes per slice. The last slice is shortened if needed.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY)
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--starting-checkpoint-episodes", type=int, default=0)
    parser.add_argument(
        "--checkpoint-tag",
        type=str,
        default="",
        help="Optional suffix appended to each slice checkpoint filename.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--discount-factor", type=float, default=DISCOUNT_FACTOR)
    parser.add_argument("--epsilon-start", type=float, default=EPSILON_START)
    parser.add_argument("--epsilon-decay", type=float, default=EPSILON_DECAY)
    parser.add_argument("--epsilon-min", type=float, default=EPSILON_MIN)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-capacity", type=int, default=50000)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--train-every", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=1000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--guided-exploration-ratio", type=float, default=0.0)
    parser.add_argument("--feature-version", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--battery-profile",
        type=str,
        choices=("training", "evaluation"),
        default="training",
    )
    parser.add_argument("--battery-capacity-override", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run greedy evaluation every N training episodes inside each slice.",
    )
    parser.add_argument(
        "--save-best-eval-checkpoint",
        action="store_true",
        help="Save an extra best-eval checkpoint while the slice chain trains.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--summary-path",
        type=str,
        default="",
        help="Optional CSV output path for the slice summary.",
    )
    return parser.parse_args()


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_slice_schedule(
    total_episodes: int,
    slice_episodes: int,
    starting_checkpoint_episodes: int,
) -> list[int]:
    if slice_episodes <= 0:
        raise ValueError("--slice-episodes must be positive.")
    if total_episodes <= 0:
        raise ValueError("--total-episodes must be positive.")
    if starting_checkpoint_episodes < 0:
        raise ValueError("--starting-checkpoint-episodes cannot be negative.")
    if starting_checkpoint_episodes >= total_episodes:
        raise ValueError(
            "starting checkpoint episodes already reach or exceed --total-episodes."
        )

    remaining = total_episodes - starting_checkpoint_episodes
    schedule: list[int] = []
    while remaining > 0:
        current_slice = min(slice_episodes, remaining)
        schedule.append(current_slice)
        remaining -= current_slice
    return schedule


def get_summary_path(args: argparse.Namespace, log_dir: Path) -> Path:
    if args.summary_path:
        return Path(args.summary_path)
    tag_suffix = f"_{args.checkpoint_tag}" if args.checkpoint_tag else ""
    return log_dir / (
        f"dqn_slice_summary_{args.map_name}_ep_{args.total_episodes}_seed_{args.seed}"
        f"{tag_suffix}.csv"
    )


def get_slice_log_path(
    log_dir: Path,
    map_name: str,
    checkpoint_episodes: int,
    seed: int,
    checkpoint_tag: str,
) -> Path:
    tag_suffix = f"_{checkpoint_tag}" if checkpoint_tag else ""
    return log_dir / (
        f"dqn_slice_{map_name}_ep_{checkpoint_episodes}_seed_{seed}{tag_suffix}.log"
    )


def build_train_command(
    args: argparse.Namespace,
    slice_episodes: int,
    checkpoint_episodes: int,
    init_checkpoint: Path | None,
    starting_checkpoint_episodes: int,
) -> list[str]:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train_dqn.py"),
        "--map-name",
        args.map_name,
        "--episodes",
        str(slice_episodes),
        "--checkpoint-episodes",
        str(checkpoint_episodes),
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
        "--batch-size",
        str(args.batch_size),
        "--replay-capacity",
        str(args.replay_capacity),
        "--learning-starts",
        str(args.learning_starts),
        "--train-every",
        str(args.train_every),
        "--target-update-interval",
        str(args.target_update_interval),
        "--hidden-size",
        str(args.hidden_size),
        "--guided-exploration-ratio",
        str(args.guided_exploration_ratio),
        "--feature-version",
        str(args.feature_version),
        "--battery-profile",
        args.battery_profile,
        "--eval-episodes",
        str(args.eval_episodes),
        "--eval-every",
        str(args.eval_every),
        "--device",
        args.device,
    ]

    if args.checkpoint_tag:
        command.extend(["--checkpoint-tag", args.checkpoint_tag])
    if args.battery_capacity_override > 0:
        command.extend(
            ["--battery-capacity-override", str(args.battery_capacity_override)]
        )
    if init_checkpoint is not None:
        command.extend(
            [
                "--init-checkpoint",
                str(init_checkpoint),
                "--starting-checkpoint-episodes",
                str(starting_checkpoint_episodes),
            ]
        )
    if args.save_best_eval_checkpoint:
        command.append("--save-best-eval-checkpoint")

    return command


def run_command(command: list[str], log_path: Path) -> None:
    print("\n>>> Running:")
    print(" ".join(command))
    print(f"log: {display_path(log_path)}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"COMMAND: {' '.join(command)}\n\n")
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def write_summary(summary_path: Path, rows: list[dict[str, object]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    _, _, log_dir = ensure_dqn_output_dirs()
    summary_path = get_summary_path(args=args, log_dir=log_dir)
    cumulative_episodes = infer_dqn_checkpoint_episodes(
        path_str=args.init_checkpoint,
        explicit_value=args.starting_checkpoint_episodes,
    )
    slice_schedule = build_slice_schedule(
        total_episodes=args.total_episodes,
        slice_episodes=args.slice_episodes,
        starting_checkpoint_episodes=cumulative_episodes,
    )
    current_checkpoint = Path(args.init_checkpoint) if args.init_checkpoint else None
    summary_rows: list[dict[str, object]] = []

    print("=== SweepAgent DQN Slice Training ===")
    print(f"map_name: {args.map_name}")
    print(f"seed: {args.seed}")
    print(f"total_episodes: {args.total_episodes}")
    print(f"slice_episodes: {args.slice_episodes}")
    print(f"slice_schedule: {slice_schedule}")
    print(f"feature_version: {args.feature_version}")
    print(f"checkpoint_tag: {args.checkpoint_tag if args.checkpoint_tag else 'none'}")
    print(f"battery_profile: {args.battery_profile}")
    if args.battery_capacity_override > 0:
        print(f"battery_capacity_override: {args.battery_capacity_override}")
    print(f"eval_episodes: {args.eval_episodes}")
    print(f"eval_every: {args.eval_every}")
    print(f"save_best_eval_checkpoint: {args.save_best_eval_checkpoint}")
    print(f"starting_checkpoint: {current_checkpoint if current_checkpoint else 'none'}")
    print(f"starting_checkpoint_episodes: {cumulative_episodes}")
    print(f"summary_path: {display_path(summary_path)}")

    for slice_index, slice_episodes in enumerate(slice_schedule, start=1):
        starting_slice_episodes = cumulative_episodes
        cumulative_episodes += slice_episodes
        stage_checkpoint = get_dqn_checkpoint_path(
            map_name=args.map_name,
            episodes=cumulative_episodes,
            seed=args.seed,
            checkpoint_tag=args.checkpoint_tag,
        )
        log_path = get_slice_log_path(
            log_dir=log_dir,
            map_name=args.map_name,
            checkpoint_episodes=cumulative_episodes,
            seed=args.seed,
            checkpoint_tag=args.checkpoint_tag,
        )
        command = build_train_command(
            args=args,
            slice_episodes=slice_episodes,
            checkpoint_episodes=cumulative_episodes,
            init_checkpoint=current_checkpoint,
            starting_checkpoint_episodes=starting_slice_episodes,
        )

        print(
            f"\n--- Slice {slice_index}/{len(slice_schedule)} | "
            f"slice_episodes={slice_episodes} | "
            f"checkpoint_episodes={cumulative_episodes} ---"
        )
        run_command(command=command, log_path=log_path)

        metadata = read_dqn_checkpoint_metadata(stage_checkpoint)
        eval_result = metadata.get("eval_result", {})
        row = {
            "slice_index": slice_index,
            "slice_episodes": slice_episodes,
            "checkpoint_episodes": cumulative_episodes,
            "map_name": args.map_name,
            "seed": args.seed,
            "feature_version": metadata.get("feature_version", args.feature_version),
            "battery_profile": metadata.get("battery_profile", args.battery_profile),
            "eval_episodes": metadata.get("eval_episodes", args.eval_episodes),
            "avg_reward": round(float(eval_result.get("avg_reward", 0.0)), 4),
            "avg_steps": round(float(eval_result.get("avg_steps", 0.0)), 4),
            "avg_cleaned_ratio": round(float(eval_result.get("avg_cleaned_ratio", 0.0)), 4),
            "success_rate": round(float(eval_result.get("success_rate", 0.0)), 4),
            "checkpoint_path": display_path(stage_checkpoint),
            "log_path": display_path(log_path),
        }
        summary_rows.append(row)
        write_summary(summary_path=summary_path, rows=summary_rows)

        print(
            "slice_eval: "
            f"avg_reward={row['avg_reward']:.2f} | "
            f"avg_steps={row['avg_steps']:.2f} | "
            f"avg_cleaned_ratio={float(row['avg_cleaned_ratio']) * 100:.2f}% | "
            f"success_rate={float(row['success_rate']) * 100:.2f}%"
        )
        current_checkpoint = stage_checkpoint

    print("\n=== DQN Slice Training Complete ===")
    print(f"final_checkpoint: {display_path(current_checkpoint) if current_checkpoint else 'none'}")
    print(f"summary_path: {display_path(summary_path)}")

    if summary_rows:
        best_row = max(summary_rows, key=lambda row: float(row["avg_cleaned_ratio"]))
        first_zero_row = next(
            (
                row
                for row in summary_rows
                if float(row["avg_cleaned_ratio"]) == 0.0
            ),
            None,
        )
        first_regression_row = next(
            (
                row
                for row in summary_rows
                if int(row["checkpoint_episodes"]) > int(best_row["checkpoint_episodes"])
                and float(row["avg_cleaned_ratio"]) < float(best_row["avg_cleaned_ratio"])
            ),
            None,
        )

        print(
            "best_slice: "
            f"checkpoint_episodes={best_row['checkpoint_episodes']} | "
            f"avg_cleaned_ratio={float(best_row['avg_cleaned_ratio']) * 100:.2f}% | "
            f"success_rate={float(best_row['success_rate']) * 100:.2f}%"
        )
        if first_regression_row is not None:
            print(
                "first_regression_after_best: "
                f"checkpoint_episodes={first_regression_row['checkpoint_episodes']} | "
                f"avg_cleaned_ratio={float(first_regression_row['avg_cleaned_ratio']) * 100:.2f}% | "
                f"success_rate={float(first_regression_row['success_rate']) * 100:.2f}%"
            )
        if first_zero_row is not None:
            print(
                "first_zero_cleaned_slice: "
                f"checkpoint_episodes={first_zero_row['checkpoint_episodes']} | "
                f"success_rate={float(first_zero_row['success_rate']) * 100:.2f}%"
            )


if __name__ == "__main__":
    main()

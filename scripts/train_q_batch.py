from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from configs.map_presets import (  # noqa: E402
    DISCOUNT_FACTOR,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    LEARNING_RATE,
    MAP_PRESETS,
    PRINT_EVERY,
    TRAIN_EPISODES,
)
from utils.experiment_utils import get_checkpoint_path  # noqa: E402


@dataclass(frozen=True)
class TrainTask:
    map_name: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multiple SweepAgent Q-learning jobs in parallel.",
    )
    parser.add_argument(
        "--maps",
        nargs="+",
        required=True,
        help="One or more map preset names to train.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="One or more seeds to train.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=TRAIN_EPISODES,
        help="Training episodes per run.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=PRINT_EVERY,
        help="Progress print interval passed to each run.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="Q-learning learning rate.",
    )
    parser.add_argument(
        "--discount-factor",
        type=float,
        default=DISCOUNT_FACTOR,
        help="Q-learning discount factor.",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=EPSILON_START,
        help="Initial exploration rate.",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=EPSILON_DECAY,
        help="Per-episode epsilon decay.",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=EPSILON_MIN,
        help="Minimum epsilon floor.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Maximum parallel workers. Use 0 to auto-select.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Retrain even if the target checkpoint already exists.",
    )
    return parser.parse_args()


def validate_maps(map_names: Iterable[str]) -> None:
    invalid_maps = [map_name for map_name in map_names if map_name not in MAP_PRESETS]
    if invalid_maps:
        supported = ", ".join(MAP_PRESETS.keys())
        invalid = ", ".join(invalid_maps)
        raise ValueError(f"Unknown map name(s): {invalid}. Supported maps: {supported}")


def ensure_output_dirs() -> tuple[Path, Path]:
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return log_dir, checkpoint_dir


def build_tasks(args: argparse.Namespace) -> list[TrainTask]:
    tasks: list[TrainTask] = []
    for map_name in args.maps:
        for seed in args.seeds:
            tasks.append(TrainTask(map_name=map_name, seed=seed))
    return tasks


def resolve_max_workers(task_count: int, requested_workers: int) -> int:
    if requested_workers > 0:
        return min(requested_workers, task_count)

    logical_cpus = os.cpu_count() or 1
    suggested = max(1, logical_cpus // 2)
    return min(suggested, task_count)


def build_train_command(task: TrainTask, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-B",
        str(PROJECT_ROOT / "scripts" / "train_q_learning.py"),
        "--map-name",
        task.map_name,
        "--episodes",
        str(args.episodes),
        "--seed",
        str(task.seed),
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
    ]


def get_task_log_path(log_dir: Path, task: TrainTask, episodes: int) -> Path:
    return log_dir / (
        f"train_batch_{task.map_name}_ep_{episodes}_seed_{task.seed}.log"
    )


def run_task(task: TrainTask, args: argparse.Namespace, log_dir: Path) -> dict[str, str | int | float]:
    checkpoint_path = get_checkpoint_path(
        map_name=task.map_name,
        episodes=args.episodes,
        seed=task.seed,
    )
    log_path = get_task_log_path(log_dir=log_dir, task=task, episodes=args.episodes)

    if checkpoint_path.exists() and not args.overwrite_existing:
        return {
            "map_name": task.map_name,
            "seed": task.seed,
            "episodes": args.episodes,
            "status": "skipped_existing",
            "return_code": 0,
            "duration_seconds": 0.0,
            "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            "log_path": str(log_path.relative_to(PROJECT_ROOT)),
        }

    command = build_train_command(task=task, args=args)
    start_time = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"COMMAND: {' '.join(command)}\n\n")
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )

    duration_seconds = time.perf_counter() - start_time
    status = "completed" if completed.returncode == 0 else "failed"

    return {
        "map_name": task.map_name,
        "seed": task.seed,
        "episodes": args.episodes,
        "status": status,
        "return_code": completed.returncode,
        "duration_seconds": round(duration_seconds, 2),
        "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
    }


def save_summary_csv(results: list[dict[str, str | int | float]]) -> Path:
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    output_path = log_dir / "train_batch_summary.csv"
    fieldnames = [
        "map_name",
        "seed",
        "episodes",
        "status",
        "return_code",
        "duration_seconds",
        "checkpoint_path",
        "log_path",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return output_path


def main() -> None:
    args = parse_args()
    validate_maps(args.maps)
    log_dir, _ = ensure_output_dirs()
    tasks = build_tasks(args)
    max_workers = resolve_max_workers(
        task_count=len(tasks),
        requested_workers=args.max_workers,
    )

    print("=== Batch Training Configuration ===")
    print(f"maps: {', '.join(args.maps)}")
    print(f"seeds: {', '.join(str(seed) for seed in args.seeds)}")
    print(f"episodes: {args.episodes}")
    print(f"max_workers: {max_workers}")
    print(f"overwrite_existing: {args.overwrite_existing}")

    results: list[dict[str, str | int | float]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_task, task, args, log_dir): task
            for task in tasks
        }

        for future in as_completed(future_map):
            task = future_map[future]
            result = future.result()
            results.append(result)
            print(
                f"[{result['status']}] map={task.map_name} "
                f"seed={task.seed} "
                f"duration={result['duration_seconds']}s "
                f"log={result['log_path']}"
            )

    results.sort(key=lambda row: (str(row["map_name"]), int(row["seed"])))
    summary_path = save_summary_csv(results)

    completed = sum(1 for row in results if row["status"] == "completed")
    skipped = sum(1 for row in results if row["status"] == "skipped_existing")
    failed = sum(1 for row in results if row["status"] == "failed")

    print("\n=== Batch Training Summary ===")
    print(f"completed: {completed}")
    print(f"skipped_existing: {skipped}")
    print(f"failed: {failed}")
    print(f"summary_csv: {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

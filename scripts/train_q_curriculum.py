from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run medium-to-large curriculum training for SweepAgent."
    )
    parser.add_argument(
        "--stage1-map",
        type=str,
        default="charge_maze_medium",
        help="Curriculum stage 1 map.",
    )
    parser.add_argument(
        "--stage2-map",
        type=str,
        default="charge_maze_large",
        help="Curriculum stage 2 map.",
    )
    parser.add_argument(
        "--stage1-episodes",
        type=int,
        default=20000,
        help="Training episodes for stage 1.",
    )
    parser.add_argument(
        "--stage2-episodes",
        type=int,
        default=50000,
        help="Training episodes for stage 2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shared seed for both stages.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=100,
        help="Print one compact progress line every N episodes.",
    )

    parser.add_argument(
        "--stage1-learning-rate",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--stage1-discount-factor",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--stage1-epsilon-start",
        type=float,
        default=1.00,
    )
    parser.add_argument(
        "--stage1-epsilon-decay",
        type=float,
        default=0.997,
    )
    parser.add_argument(
        "--stage1-epsilon-min",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--stage2-learning-rate",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--stage2-discount-factor",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--stage2-epsilon-start",
        type=float,
        default=0.30,
        help="Reset epsilon for the harder stage after loading the stage 1 checkpoint.",
    )
    parser.add_argument(
        "--stage2-epsilon-decay",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--stage2-epsilon-min",
        type=float,
        default=0.10,
    )

    return parser.parse_args()


def get_checkpoint_path(map_name: str, episodes: int, seed: int) -> Path:
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"q_learning_agent_{map_name}_ep_{episodes}_seed_{seed}.json"


def run_command(command: list[str]) -> None:
    print("\\n>>> Running:")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    args = parse_args()

    stage1_checkpoint = get_checkpoint_path(
        map_name=args.stage1_map,
        episodes=args.stage1_episodes,
        seed=args.seed,
    )
    stage2_checkpoint = get_checkpoint_path(
        map_name=args.stage2_map,
        episodes=args.stage2_episodes,
        seed=args.seed,
    )

    stage1_command = [
        sys.executable,
        "scripts/train_q_learning.py",
        "--map-name",
        args.stage1_map,
        "--episodes",
        str(args.stage1_episodes),
        "--seed",
        str(args.seed),
        "--print-every",
        str(args.print_every),
        "--learning-rate",
        str(args.stage1_learning_rate),
        "--discount-factor",
        str(args.stage1_discount_factor),
        "--epsilon-start",
        str(args.stage1_epsilon_start) if False else str(args.stage1_epsilon_start),
        "--epsilon-decay",
        str(args.stage1_epsilon_decay),
        "--epsilon-min",
        str(args.stage1_epsilon_min),
    ]

    stage2_command = [
        sys.executable,
        "scripts/train_q_learning.py",
        "--map-name",
        args.stage2_map,
        "--episodes",
        str(args.stage2_episodes),
        "--seed",
        str(args.seed),
        "--print-every",
        str(args.print_every),
        "--learning-rate",
        str(args.stage2_learning_rate),
        "--discount-factor",
        str(args.stage2_discount_factor),
        "--epsilon-start",
        str(args.stage2_epsilon_start),
        "--epsilon-decay",
        str(args.stage2_epsilon_decay),
        "--epsilon-min",
        str(args.stage2_epsilon_min),
        "--init-checkpoint",
        str(stage1_checkpoint),
        "--reset-epsilon",
        "--override-loaded-hparams",
    ]

    print("=== SweepAgent Curriculum Training ===")
    print(f"stage1_map: {args.stage1_map}")
    print(f"stage1_episodes: {args.stage1_episodes}")
    print(f"stage2_map: {args.stage2_map}")
    print(f"stage2_episodes: {args.stage2_episodes}")
    print(f"seed: {args.seed}")
    print(f"stage1_checkpoint: {stage1_checkpoint}")
    print(f"stage2_checkpoint: {stage2_checkpoint}")

    run_command(stage1_command)
    run_command(stage2_command)

    print("\\n=== Curriculum Complete ===")
    print(f"stage1_checkpoint: {stage1_checkpoint}")
    print(f"stage2_checkpoint: {stage2_checkpoint}")


if __name__ == "__main__":
    main()
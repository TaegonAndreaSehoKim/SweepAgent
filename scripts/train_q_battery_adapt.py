from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


DEFAULT_STAGE2_EPISODES = 50000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SweepAgent with training-battery pretraining and evaluation-battery finetuning.",
    )
    parser.add_argument(
        "--map-name",
        type=str,
        required=True,
        help="Map preset name to train on.",
    )
    parser.add_argument(
        "--stage1-episodes",
        type=int,
        default=200000,
        help="Pretraining episodes using the training battery profile.",
    )
    parser.add_argument(
        "--stage2-episodes",
        type=int,
        default=DEFAULT_STAGE2_EPISODES,
        help="Finetuning episodes using the evaluation battery profile.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed shared by both stages.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1000,
        help="Progress print interval for both stages.",
    )
    parser.add_argument("--stage1-learning-rate", type=float, default=0.05)
    parser.add_argument("--stage1-discount-factor", type=float, default=0.99)
    parser.add_argument("--stage1-epsilon-start", type=float, default=1.0)
    parser.add_argument("--stage1-epsilon-decay", type=float, default=0.99999)
    parser.add_argument("--stage1-epsilon-min", type=float, default=0.20)
    parser.add_argument("--stage2-learning-rate", type=float, default=0.05)
    parser.add_argument("--stage2-discount-factor", type=float, default=0.99)
    parser.add_argument("--stage2-epsilon-start", type=float, default=0.30)
    parser.add_argument("--stage2-epsilon-decay", type=float, default=0.99998)
    parser.add_argument("--stage2-epsilon-min", type=float, default=0.08)
    parser.add_argument(
        "--state-abstraction-mode",
        type=str,
        choices=("identity", "safety_margin"),
        default="identity",
        help="Optional abstraction mode passed through to both training stages.",
    )
    parser.add_argument(
        "--safety-margin-bucket-size",
        type=int,
        default=5,
        help="Bucket size used by the safety_margin abstraction mode.",
    )
    return parser.parse_args()


def get_checkpoint_path(map_name: str, episodes: int, seed: int) -> Path:
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"q_learning_agent_{map_name}_ep_{episodes}_seed_{seed}.json"


def run_command(command: list[str]) -> None:
    print("\n>>> Running:")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    args = parse_args()
    final_checkpoint_episodes = args.stage1_episodes + args.stage2_episodes
    stage1_checkpoint = get_checkpoint_path(
        map_name=args.map_name,
        episodes=args.stage1_episodes,
        seed=args.seed,
    )
    final_checkpoint = get_checkpoint_path(
        map_name=args.map_name,
        episodes=final_checkpoint_episodes,
        seed=args.seed,
    )

    stage1_command = [
        sys.executable,
        "scripts/train_q_learning.py",
        "--map-name",
        args.map_name,
        "--episodes",
        str(args.stage1_episodes),
        "--seed",
        str(args.seed),
        "--print-every",
        str(args.print_every),
        "--battery-profile",
        "training",
        "--learning-rate",
        str(args.stage1_learning_rate),
        "--discount-factor",
        str(args.stage1_discount_factor),
        "--epsilon-start",
        str(args.stage1_epsilon_start),
        "--epsilon-decay",
        str(args.stage1_epsilon_decay),
        "--epsilon-min",
        str(args.stage1_epsilon_min),
        "--state-abstraction-mode",
        args.state_abstraction_mode,
        "--safety-margin-bucket-size",
        str(args.safety_margin_bucket_size),
    ]

    stage2_command = [
        sys.executable,
        "scripts/train_q_learning.py",
        "--map-name",
        args.map_name,
        "--episodes",
        str(args.stage2_episodes),
        "--checkpoint-episodes",
        str(final_checkpoint_episodes),
        "--seed",
        str(args.seed),
        "--print-every",
        str(args.print_every),
        "--battery-profile",
        "evaluation",
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
        "--state-abstraction-mode",
        args.state_abstraction_mode,
        "--safety-margin-bucket-size",
        str(args.safety_margin_bucket_size),
        "--init-checkpoint",
        str(stage1_checkpoint),
        "--reset-epsilon",
        "--override-loaded-hparams",
    ]

    print("=== SweepAgent Battery-Profile Adaptation Training ===")
    print(f"map_name: {args.map_name}")
    print(f"stage1_episodes: {args.stage1_episodes}")
    print(f"stage2_episodes: {args.stage2_episodes}")
    print(f"final_checkpoint_episodes: {final_checkpoint_episodes}")
    print(f"seed: {args.seed}")
    print(f"state_abstraction_mode: {args.state_abstraction_mode}")
    if args.state_abstraction_mode == "safety_margin":
        print(f"safety_margin_bucket_size: {args.safety_margin_bucket_size}")
    print(f"stage1_checkpoint: {stage1_checkpoint}")
    print(f"final_checkpoint: {final_checkpoint}")

    run_command(stage1_command)
    run_command(stage2_command)

    print("\n=== Battery-Profile Adaptation Complete ===")
    print(f"stage1_checkpoint: {stage1_checkpoint}")
    print(f"final_checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()

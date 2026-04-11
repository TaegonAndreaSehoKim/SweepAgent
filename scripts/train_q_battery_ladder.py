from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


CHECKPOINT_EPISODES_PATTERN = re.compile(r"_ep_(\d+)_seed_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train SweepAgent through a same-map battery ladder so the policy "
            "adapts to tighter battery budgets gradually."
        ),
    )
    parser.add_argument("--map-name", required=True, type=str)
    parser.add_argument("--battery-capacities", nargs="+", type=int, required=True)
    parser.add_argument(
        "--stage-episodes",
        nargs="+",
        type=int,
        required=True,
        help="Either one shared stage length or one length per battery stage.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help="Optional starting checkpoint to adapt from.",
    )
    parser.add_argument(
        "--starting-checkpoint-episodes",
        type=int,
        default=0,
        help=(
            "Optional explicit episode count for the initial checkpoint. "
            "Used only when it cannot be inferred from the filename."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=0.20)
    parser.add_argument("--epsilon-decay", type=float, default=0.99998)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    return parser.parse_args()


def normalize_stage_episodes(
    stage_episodes: list[int],
    stage_count: int,
) -> list[int]:
    if len(stage_episodes) == 1:
        return stage_episodes * stage_count
    if len(stage_episodes) != stage_count:
        raise ValueError(
            "stage episode count must be either 1 value or match the number of battery stages."
        )
    return stage_episodes


def infer_checkpoint_episodes(path_str: str, explicit_value: int) -> int:
    if explicit_value > 0:
        return explicit_value

    if not path_str:
        return 0

    match = CHECKPOINT_EPISODES_PATTERN.search(Path(path_str).name)
    if match:
        return int(match.group(1))

    raise ValueError(
        "Could not infer starting checkpoint episodes from filename. "
        "Pass --starting-checkpoint-episodes explicitly."
    )


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
    stage_episodes = normalize_stage_episodes(
        stage_episodes=args.stage_episodes,
        stage_count=len(args.battery_capacities),
    )
    cumulative_episodes = infer_checkpoint_episodes(
        path_str=args.init_checkpoint,
        explicit_value=args.starting_checkpoint_episodes,
    )
    current_checkpoint = Path(args.init_checkpoint) if args.init_checkpoint else None

    print("=== SweepAgent Battery Ladder Training ===")
    print(f"map_name: {args.map_name}")
    print(f"seed: {args.seed}")
    print(f"battery_capacities: {args.battery_capacities}")
    print(f"stage_episodes: {stage_episodes}")
    print(f"starting_checkpoint: {current_checkpoint if current_checkpoint else 'none'}")
    print(f"starting_checkpoint_episodes: {cumulative_episodes}")

    for stage_index, (capacity, episodes) in enumerate(
        zip(args.battery_capacities, stage_episodes),
        start=1,
    ):
        cumulative_episodes += episodes
        stage_checkpoint = get_checkpoint_path(
            map_name=args.map_name,
            episodes=cumulative_episodes,
            seed=args.seed,
        )

        command = [
            sys.executable,
            "scripts/train_q_learning.py",
            "--map-name",
            args.map_name,
            "--episodes",
            str(episodes),
            "--checkpoint-episodes",
            str(cumulative_episodes),
            "--seed",
            str(args.seed),
            "--print-every",
            str(args.print_every),
            "--battery-profile",
            "evaluation",
            "--battery-capacity-override",
            str(capacity),
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

        if current_checkpoint is not None:
            command.extend(
                [
                    "--init-checkpoint",
                    str(current_checkpoint),
                    "--reset-epsilon",
                    "--override-loaded-hparams",
                ]
            )

        print(
            f"\n--- Stage {stage_index}/{len(args.battery_capacities)} | "
            f"capacity={capacity} | stage_episodes={episodes} | "
            f"checkpoint_episodes={cumulative_episodes} ---"
        )
        run_command(command)
        current_checkpoint = stage_checkpoint

    print("\n=== Battery Ladder Training Complete ===")
    if current_checkpoint is not None:
        print(f"final_checkpoint: {current_checkpoint}")


if __name__ == "__main__":
    main()

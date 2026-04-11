from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


from configs.map_presets import MAP_PRESETS  # noqa: E402


CHECKPOINT_EPISODES_PATTERN = re.compile(r"_ep_(\d+)_seed_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train SweepAgent with a same-map dirty curriculum by pre-cleaning "
            "selected dirty tiles in earlier stages."
        ),
    )
    parser.add_argument("--map-name", required=True, type=str)
    parser.add_argument(
        "--stage-keep-dirty-indices",
        nargs="+",
        required=True,
        help=(
            "One token per stage. Use comma-separated dirty indices such as "
            "'1' or '0,1', or 'full' to keep all dirties active."
        ),
    )
    parser.add_argument(
        "--stage-episodes",
        nargs="+",
        type=int,
        required=True,
        help="Either one shared stage length or one length per curriculum stage.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=1000)
    parser.add_argument(
        "--battery-profile",
        choices=("training", "evaluation"),
        default="evaluation",
    )
    parser.add_argument("--battery-capacity-override", type=int, default=0)
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--starting-checkpoint-episodes", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=0.15)
    parser.add_argument("--epsilon-decay", type=float, default=0.99999)
    parser.add_argument("--epsilon-min", type=float, default=0.03)
    return parser.parse_args()


def normalize_stage_episodes(
    stage_episodes: list[int],
    stage_count: int,
) -> list[int]:
    if len(stage_episodes) == 1:
        return stage_episodes * stage_count
    if len(stage_episodes) != stage_count:
        raise ValueError(
            "stage episode count must be either 1 value or match the number of stages."
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


def parse_stage_keep_spec(
    keep_spec: str,
    total_dirty_tiles: int,
) -> list[int]:
    lowered = keep_spec.strip().lower()
    if lowered == "full":
        return list(range(total_dirty_tiles))

    keep_indices: list[int] = []
    for part in keep_spec.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        idx = int(stripped)
        if idx < 0 or idx >= total_dirty_tiles:
            raise ValueError(
                f"Dirty index {idx} is out of range for total_dirty_tiles={total_dirty_tiles}."
            )
        keep_indices.append(idx)

    if not keep_indices:
        raise ValueError(f"Invalid stage keep spec: '{keep_spec}'")
    return sorted(set(keep_indices))


def build_precleaned_positions(
    map_name: str,
    keep_indices: list[int],
) -> list[tuple[int, int]]:
    preset = MAP_PRESETS[map_name]
    dirty_positions: list[tuple[int, int]] = []
    for row_idx, row in enumerate(preset["grid_map"]):
        for col_idx, cell in enumerate(row):
            if cell == "D":
                dirty_positions.append((row_idx, col_idx))

    return [
        position
        for idx, position in enumerate(dirty_positions)
        if idx not in keep_indices
    ]


def run_command(command: list[str]) -> None:
    print("\n>>> Running:")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    args = parse_args()
    preset = MAP_PRESETS[args.map_name]
    total_dirty_tiles = sum(row.count("D") for row in preset["grid_map"])
    if total_dirty_tiles == 0:
        raise ValueError(f"Map '{args.map_name}' has no dirty tiles for curriculum.")

    stage_keep_indices = [
        parse_stage_keep_spec(spec, total_dirty_tiles)
        for spec in args.stage_keep_dirty_indices
    ]
    stage_episodes = normalize_stage_episodes(
        stage_episodes=args.stage_episodes,
        stage_count=len(stage_keep_indices),
    )
    cumulative_episodes = infer_checkpoint_episodes(
        path_str=args.init_checkpoint,
        explicit_value=args.starting_checkpoint_episodes,
    )
    current_checkpoint = Path(args.init_checkpoint) if args.init_checkpoint else None

    print("=== SweepAgent Dirty Curriculum Training ===")
    print(f"map_name: {args.map_name}")
    print(f"seed: {args.seed}")
    print(f"battery_profile: {args.battery_profile}")
    if args.battery_capacity_override > 0:
        print(f"battery_capacity_override: {args.battery_capacity_override}")
    print(f"stage_keep_dirty_indices: {stage_keep_indices}")
    print(f"stage_episodes: {stage_episodes}")
    print(f"starting_checkpoint: {current_checkpoint if current_checkpoint else 'none'}")
    print(f"starting_checkpoint_episodes: {cumulative_episodes}")

    for stage_index, (keep_indices, episodes) in enumerate(
        zip(stage_keep_indices, stage_episodes),
        start=1,
    ):
        cumulative_episodes += episodes
        precleaned_positions = build_precleaned_positions(
            map_name=args.map_name,
            keep_indices=keep_indices,
        )
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
            args.battery_profile,
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
            "--initial-cleaned-positions",
            ";".join(f"{row},{col}" for row, col in precleaned_positions),
        ]

        if args.battery_capacity_override > 0:
            command.extend(
                ["--battery-capacity-override", str(args.battery_capacity_override)]
            )

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
            f"\n--- Stage {stage_index}/{len(stage_keep_indices)} | "
            f"keep_dirty_indices={keep_indices} | stage_episodes={episodes} | "
            f"checkpoint_episodes={cumulative_episodes} ---"
        )
        run_command(command)
        current_checkpoint = stage_checkpoint

    print("\n=== Dirty Curriculum Training Complete ===")
    if current_checkpoint is not None:
        print(f"final_checkpoint: {current_checkpoint}")


if __name__ == "__main__":
    main()

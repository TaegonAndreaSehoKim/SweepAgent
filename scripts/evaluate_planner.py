from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.experiment_utils import build_env
from utils.planner_utils import compute_shortest_cleaning_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a deterministic shortest-route planner baseline."
    )
    parser.add_argument("--map-name", type=str, default="complex_charge_bastion")
    parser.add_argument(
        "--battery-profile",
        type=str,
        choices=("training", "evaluation"),
        default="evaluation",
    )
    parser.add_argument("--battery-capacity-override", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output-csv", type=str, default="")
    return parser.parse_args()


def resolve_output_path(path_str: str) -> Path:
    output_path = Path(path_str)
    if output_path.is_absolute():
        return output_path
    return PROJECT_ROOT / output_path


def save_csv(output_path: Path, row: dict[str, str | int | float | bool]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    env = build_env(
        map_name=args.map_name,
        battery_profile=args.battery_profile,
        battery_capacity_override=(
            args.battery_capacity_override
            if args.battery_capacity_override > 0
            else None
        ),
    )
    plan = compute_shortest_cleaning_plan(
        grid_map=env.raw_map,
        battery_capacity=env.battery_capacity,
    )

    print("=== Planner Baseline ===")
    print(f"map_name: {args.map_name}")
    print(f"battery_profile: {args.battery_profile}")
    print(f"battery_capacity: {env.battery_capacity}")
    print(f"solvable: {plan.solvable}")
    if not plan.solvable:
        print(f"reason: {plan.reason}")
        return

    state = env.reset()
    del state
    total_reward = 0.0
    final_info: dict[str, float | str] = {}
    done = False

    if args.render:
        env.render()

    for action in plan.actions:
        _, reward, done, final_info = env.step(action)
        total_reward += reward
        if args.render:
            print(f"\naction={action} ({env.ACTION_NAMES[action]})")
            print(f"reward={reward}")
            env.render()
        if done:
            break

    row = {
        "map_name": args.map_name,
        "battery_profile": args.battery_profile,
        "battery_capacity": env.battery_capacity if env.battery_capacity else -1,
        "solvable": plan.solvable,
        "planned_steps": plan.total_steps,
        "executed_steps": int(final_info.get("steps_taken", 0)),
        "total_reward": round(total_reward, 4),
        "cleaned_ratio": round(float(final_info.get("cleaned_ratio", 0.0)), 6),
        "success_rate": 1.0 if float(final_info.get("cleaned_ratio", 0.0)) == 1.0 else 0.0,
        "termination_reason": str(final_info.get("termination_reason", "not_run")),
        "special_route": " -> ".join(str(pos) for pos in plan.special_route),
    }

    print(f"planned_steps: {row['planned_steps']}")
    print(f"executed_steps: {row['executed_steps']}")
    print(f"total_reward: {row['total_reward']}")
    print(f"cleaned_ratio: {float(row['cleaned_ratio']) * 100:.2f}%")
    print(f"success_rate: {float(row['success_rate']) * 100:.2f}%")
    print(f"termination_reason: {row['termination_reason']}")
    print(f"special_route: {row['special_route']}")

    if args.output_csv:
        output_path = resolve_output_path(args.output_csv)
        save_csv(output_path=output_path, row=row)
        print(f"saved_csv: {output_path}")


if __name__ == "__main__":
    main()

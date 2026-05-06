from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.experiment_utils import build_env
from utils.planner_utils import compute_shortest_cleaning_plan
from utils.waypoint_controller import WaypointController, planner_route_to_waypoint_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the waypoint-level action interface using the deterministic "
            "planner route as the high-level policy."
        )
    )
    parser.add_argument("--map-name", type=str, default="complex_charge_bastion")
    parser.add_argument(
        "--battery-profile",
        type=str,
        choices=("training", "evaluation"),
        default="evaluation",
    )
    parser.add_argument("--battery-capacity-override", type=int, default=0)
    return parser.parse_args()


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
    print("=== Waypoint Policy Evaluation ===")
    print(f"map_name: {args.map_name}")
    print(f"battery_profile: {args.battery_profile}")
    print(f"planner_solvable: {plan.solvable}")
    if not plan.solvable:
        print(f"reason: {plan.reason}")
        return

    env.reset()
    controller = WaypointController(env)
    waypoint_actions = planner_route_to_waypoint_actions(env=env, plan=plan)
    total_reward = 0.0
    final_info: dict[str, float | str] = {}
    executed_waypoints: list[str] = []

    for waypoint_action in waypoint_actions:
        waypoint = controller.action_space.get(waypoint_action)
        result = controller.execute(waypoint_action)
        total_reward += result.total_reward
        final_info = result.info
        executed_waypoints.append(waypoint.label)
        if result.done:
            break

    print(f"waypoint_action_count: {len(waypoint_actions)}")
    print(f"waypoint_route: {' -> '.join(executed_waypoints)}")
    print(f"executed_steps: {int(final_info.get('steps_taken', 0))}")
    print(f"total_reward: {total_reward:.2f}")
    print(f"cleaned_ratio: {float(final_info.get('cleaned_ratio', 0.0)) * 100:.2f}%")
    print(
        "success_rate: "
        f"{100.0 if float(final_info.get('cleaned_ratio', 0.0)) == 1.0 else 0.0:.2f}%"
    )
    print(f"termination_reason: {final_info.get('termination_reason', 'not_run')}")


if __name__ == "__main__":
    main()

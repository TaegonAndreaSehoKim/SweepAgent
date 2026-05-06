from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from env.waypoint_graph_env import WaypointGraphEnv, planner_route_to_graph_actions
from utils.experiment_utils import build_env
from utils.planner_utils import compute_shortest_cleaning_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the special-node graph-state formulation."
    )
    parser.add_argument("--map-name", type=str, default="complex_charge_bastion")
    parser.add_argument(
        "--battery-profile",
        type=str,
        choices=("training", "evaluation"),
        default="evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    primitive_env = build_env(
        map_name=args.map_name,
        battery_profile=args.battery_profile,
    )
    graph_env = WaypointGraphEnv(
        grid_map=primitive_env.raw_map,
        battery_capacity=int(primitive_env.battery_capacity or 0),
    )
    plan = compute_shortest_cleaning_plan(
        grid_map=primitive_env.raw_map,
        battery_capacity=primitive_env.battery_capacity,
    )

    print("=== Graph-State Plan Evaluation ===")
    print(f"map_name: {args.map_name}")
    print(f"battery_profile: {args.battery_profile}")
    print(f"node_count: {len(graph_env.nodes)}")
    print(f"action_count: {graph_env.action_count()}")
    print(f"planner_solvable: {plan.solvable}")
    if not plan.solvable:
        print(f"reason: {plan.reason}")
        return

    graph_env.reset()
    graph_actions = planner_route_to_graph_actions(graph_env=graph_env, plan=plan)
    total_reward = 0.0
    final_info: dict[str, float | str] = {}
    done = False
    labels: list[str] = []

    for action in graph_actions:
        labels.append(graph_env.action_label(action))
        _, reward, done, final_info = graph_env.step(action)
        total_reward += reward
        if done:
            break

    print(f"graph_actions: {' -> '.join(labels)}")
    print(f"graph_action_count: {len(graph_actions)}")
    print(f"planned_primitive_steps: {plan.total_steps}")
    print(f"graph_reward: {total_reward:.2f}")
    print(f"cleaned_ratio: {float(final_info.get('cleaned_ratio', 0.0)) * 100:.2f}%")
    print(f"termination_reason: {final_info.get('termination_reason', 'not_run')}")


if __name__ == "__main__":
    main()

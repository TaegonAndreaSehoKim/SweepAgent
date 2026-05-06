from __future__ import annotations

from env.waypoint_graph_env import WaypointGraphEnv, planner_route_to_graph_actions
from utils.experiment_utils import build_env
from utils.planner_utils import compute_shortest_cleaning_plan


def test_graph_env_executes_planner_route() -> None:
    primitive_env = build_env(
        map_name="complex_charge_bastion",
        battery_profile="evaluation",
    )
    graph_env = WaypointGraphEnv(
        grid_map=primitive_env.raw_map,
        battery_capacity=int(primitive_env.battery_capacity or 0),
    )
    plan = compute_shortest_cleaning_plan(
        grid_map=primitive_env.raw_map,
        battery_capacity=primitive_env.battery_capacity,
    )
    assert plan.solvable

    graph_env.reset()
    graph_actions = planner_route_to_graph_actions(graph_env=graph_env, plan=plan)

    done = False
    final_info: dict[str, float | str] = {}
    for action in graph_actions:
        _, _, done, final_info = graph_env.step(action)
        if done:
            break

    assert done
    assert final_info["termination_reason"] == "all_cleaned"
    assert final_info["cleaned_ratio"] == 1.0
    assert len(graph_actions) < plan.total_steps

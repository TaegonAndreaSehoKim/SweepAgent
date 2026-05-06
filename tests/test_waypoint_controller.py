from __future__ import annotations

from utils.experiment_utils import build_env
from utils.planner_utils import compute_shortest_cleaning_plan
from utils.waypoint_controller import (
    WaypointController,
    planner_route_to_waypoint_actions,
)


def test_waypoint_controller_executes_planner_route() -> None:
    env = build_env(map_name="complex_charge_labyrinth", battery_profile="evaluation")
    plan = compute_shortest_cleaning_plan(
        grid_map=env.raw_map,
        battery_capacity=env.battery_capacity,
    )
    assert plan.solvable

    env.reset()
    controller = WaypointController(env)
    waypoint_actions = planner_route_to_waypoint_actions(env=env, plan=plan)
    assert len(waypoint_actions) < len(plan.actions)

    final_info: dict[str, float | str] = {}
    for waypoint_action in waypoint_actions:
        result = controller.execute(waypoint_action)
        final_info = result.info
        if result.done:
            break

    assert final_info["termination_reason"] == "all_cleaned"
    assert final_info["cleaned_ratio"] == 1.0
    assert final_info["steps_taken"] == plan.total_steps

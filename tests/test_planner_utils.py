from __future__ import annotations

from utils.experiment_utils import build_env
from utils.planner_utils import compute_shortest_cleaning_plan


def run_plan(map_name: str) -> dict[str, float | str]:
    env = build_env(map_name=map_name, battery_profile="evaluation")
    plan = compute_shortest_cleaning_plan(
        grid_map=env.raw_map,
        battery_capacity=env.battery_capacity,
    )
    assert plan.solvable

    final_info: dict[str, float | str] = {}
    done = False
    env.reset()
    for action in plan.actions:
        _, _, done, final_info = env.step(action)
        if done:
            break

    assert done
    return final_info


def test_planner_solves_default_map() -> None:
    final_info = run_plan("default")

    assert final_info["termination_reason"] == "all_cleaned"
    assert final_info["cleaned_ratio"] == 1.0


def test_planner_solves_complex_charge_switchback() -> None:
    final_info = run_plan("complex_charge_switchback")

    assert final_info["termination_reason"] == "all_cleaned"
    assert final_info["cleaned_ratio"] == 1.0

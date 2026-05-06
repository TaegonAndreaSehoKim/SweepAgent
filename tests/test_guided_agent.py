from __future__ import annotations

from agents.guided_agent import GuidedPolicyAgent
from utils.experiment_utils import build_env


def test_guided_policy_agent_solves_switchback() -> None:
    env = build_env(
        map_name="complex_charge_switchback",
        battery_profile="evaluation",
    )
    agent = GuidedPolicyAgent(
        map_name="complex_charge_switchback",
        battery_capacity=int(env.battery_capacity or 1),
    )

    state = env.reset()
    done = False
    final_info = {}

    while not done:
        state, _, done, final_info = env.step(agent.get_policy_action(state))

    assert final_info["termination_reason"] == "all_cleaned"
    assert final_info["cleaned_ratio"] == 1.0

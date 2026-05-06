from __future__ import annotations

from agents.dqn_agent import State, StateFeatureEncoder


class GuidedPolicyAgent:
    """Deterministic baseline that exposes the DQN guided action heuristic."""

    def __init__(
        self,
        map_name: str,
        battery_capacity: int,
        feature_version: int = StateFeatureEncoder.FEATURE_VERSION_ROUTE_VIA_CHARGER,
    ) -> None:
        self.encoder = StateFeatureEncoder(
            map_name=map_name,
            battery_capacity=battery_capacity,
            feature_version=feature_version,
        )

    def get_policy_action(self, state: State) -> int:
        action = self.encoder.guided_action(state)
        if action is None:
            return 0
        return int(action)

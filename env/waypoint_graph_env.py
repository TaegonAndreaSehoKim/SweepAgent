from __future__ import annotations

from dataclasses import dataclass

from utils.planner_utils import CleaningPlan, find_symbol_positions, _single_source_distances


@dataclass(frozen=True)
class GraphNode:
    index: int
    label: str
    position: tuple[int, int]
    kind: str


class WaypointGraphEnv:
    def __init__(
        self,
        grid_map: list[str],
        battery_capacity: int,
        reward_clean: float = 10.0,
        reward_finish: float = 50.0,
        reward_invalid: float = -50.0,
    ) -> None:
        self.grid_map = grid_map
        self.battery_capacity = battery_capacity
        self.reward_clean = reward_clean
        self.reward_finish = reward_finish
        self.reward_invalid = reward_invalid

        start_positions = find_symbol_positions(grid_map, "R")
        if len(start_positions) != 1:
            raise ValueError("grid_map must contain exactly one start position")

        self.start_position = start_positions[0]
        self.dirty_positions = find_symbol_positions(grid_map, "D")
        self.charger_positions = find_symbol_positions(grid_map, "C")
        self.nodes = self._build_nodes()
        self.action_targets = [
            node.index for node in self.nodes if node.kind in {"dirty", "charger"}
        ]
        self.distance_by_position = {
            node.position: _single_source_distances(grid_map, node.position)
            for node in self.nodes
        }
        self.distance_matrix = self._build_distance_matrix()
        self.full_mask = (1 << len(self.dirty_positions)) - 1
        self.current_node_idx = 0
        self.cleaned_mask = 0
        self.battery_remaining = battery_capacity

    def _build_nodes(self) -> list[GraphNode]:
        nodes = [
            GraphNode(
                index=0,
                label="start",
                position=self.start_position,
                kind="start",
            )
        ]
        for dirty_idx, position in enumerate(self.dirty_positions):
            nodes.append(
                GraphNode(
                    index=len(nodes),
                    label=f"dirty_{dirty_idx}",
                    position=position,
                    kind="dirty",
                )
            )
        for charger_idx, position in enumerate(self.charger_positions):
            nodes.append(
                GraphNode(
                    index=len(nodes),
                    label=f"charger_{charger_idx}",
                    position=position,
                    kind="charger",
                )
            )
        return nodes

    def _build_distance_matrix(self) -> list[list[int]]:
        matrix = [[-1 for _ in self.nodes] for _ in self.nodes]
        for source in self.nodes:
            distances = self.distance_by_position[source.position]
            for target in self.nodes:
                matrix[source.index][target.index] = distances.get(target.position, -1)
        return matrix

    def reset(self) -> tuple[int, int, int]:
        self.current_node_idx = 0
        self.cleaned_mask = 0
        self.battery_remaining = self.battery_capacity
        return self.state

    @property
    def state(self) -> tuple[int, int, int]:
        return self.current_node_idx, self.cleaned_mask, self.battery_remaining

    def action_count(self) -> int:
        return len(self.action_targets)

    def action_label(self, action_idx: int) -> str:
        return self.nodes[self.action_targets[action_idx]].label

    def action_for_position(self, position: tuple[int, int]) -> int | None:
        for action_idx, node_idx in enumerate(self.action_targets):
            if self.nodes[node_idx].position == position:
                return action_idx
        return None

    def step(
        self,
        action_idx: int,
    ) -> tuple[tuple[int, int, int], float, bool, dict[str, float | str]]:
        if action_idx < 0 or action_idx >= len(self.action_targets):
            raise ValueError(f"Invalid graph action: {action_idx}")

        target_node_idx = self.action_targets[action_idx]
        target_node = self.nodes[target_node_idx]
        distance = self.distance_matrix[self.current_node_idx][target_node_idx]

        if distance <= 0 or distance > self.battery_remaining:
            return (
                self.state,
                self.reward_invalid,
                False,
                {
                    "valid": 0.0,
                    "target": target_node.label,
                    "distance": distance,
                    "cleaned_ratio": self.cleaned_ratio,
                    "termination_reason": "invalid_transition",
                },
            )

        reward = -float(distance)
        self.current_node_idx = target_node_idx
        self.battery_remaining -= distance

        if target_node.kind == "dirty":
            dirty_idx = target_node_idx - 1
            dirty_bit = 1 << dirty_idx
            if not (self.cleaned_mask & dirty_bit):
                self.cleaned_mask |= dirty_bit
                reward += self.reward_clean
        elif target_node.kind == "charger":
            self.battery_remaining = self.battery_capacity

        done = self.cleaned_mask == self.full_mask
        termination_reason = "all_cleaned" if done else "ongoing"
        if done:
            reward += self.reward_finish

        return (
            self.state,
            reward,
            done,
            {
                "valid": 1.0,
                "target": target_node.label,
                "distance": distance,
                "cleaned_ratio": self.cleaned_ratio,
                "termination_reason": termination_reason,
            },
        )

    @property
    def cleaned_ratio(self) -> float:
        if not self.dirty_positions:
            return 1.0
        return self.cleaned_mask.bit_count() / len(self.dirty_positions)


def planner_route_to_graph_actions(
    graph_env: WaypointGraphEnv,
    plan: CleaningPlan,
) -> list[int]:
    actions: list[int] = []
    for position in plan.special_route[1:]:
        action_idx = graph_env.action_for_position(position)
        if action_idx is None:
            raise ValueError(f"Planner route position is not a graph action: {position}")
        actions.append(action_idx)
    return actions

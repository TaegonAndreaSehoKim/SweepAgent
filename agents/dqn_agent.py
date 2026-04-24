from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import random
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from configs.map_presets import MAP_PRESETS


State = tuple[int, int, int, int]


class Transition(NamedTuple):
    state: State
    action: int
    reward: float
    next_state: State
    done: bool


class TargetRouteProfile(NamedTuple):
    target_distance: int
    recovery_distance: int
    direct_route_cost: int
    direct_margin: float
    route_charger_id: int
    route_to_charger_distance: int
    charger_to_target_distance: int
    via_route_total_cost: int
    via_post_charge_margin: float
    direct_feasible: bool
    via_route_feasible: bool


@dataclass
class DQNConfig:
    map_name: str
    battery_capacity: int
    action_space_size: int = 4
    learning_rate: float = 0.001
    discount_factor: float = 0.99
    epsilon: float = 1.0
    epsilon_decay: float = 0.99995
    epsilon_min: float = 0.05
    batch_size: int = 128
    replay_capacity: int = 50000
    learning_starts: int = 1000
    train_every: int = 4
    target_update_interval: int = 1000
    hidden_size: int = 128
    guided_exploration_ratio: float = 0.0
    feature_version: int = 2
    seed: int | None = None


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int | None = None) -> None:
        self.capacity = capacity
        self.buffer: list[Transition] = []
        self.position = 0
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(self.buffer, batch_size)


class StateFeatureEncoder:
    FEATURE_VERSION_LEGACY = 1
    FEATURE_VERSION_ROUTE_VIA_CHARGER = 2
    ACTION_DELTAS = (
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    )

    def __init__(
        self,
        map_name: str,
        battery_capacity: int,
        feature_version: int = FEATURE_VERSION_ROUTE_VIA_CHARGER,
    ) -> None:
        if map_name not in MAP_PRESETS:
            supported_maps = ", ".join(MAP_PRESETS.keys())
            raise ValueError(
                f"Unknown map_name='{map_name}'. Supported maps: {supported_maps}"
            )
        if feature_version not in (
            self.FEATURE_VERSION_LEGACY,
            self.FEATURE_VERSION_ROUTE_VIA_CHARGER,
        ):
            raise ValueError(
                f"Unsupported feature_version={feature_version}. "
                "Supported values: 1, 2."
            )

        self.map_name = map_name
        self.feature_version = feature_version
        self.grid_map = MAP_PRESETS[map_name]["grid_map"]
        self.rows = len(self.grid_map)
        self.cols = len(self.grid_map[0]) if self.rows else 0
        self.battery_capacity = max(1, int(battery_capacity))
        self.battery_safety_reserve = max(
            8,
            math.ceil(self.battery_capacity * 0.15),
        )
        self.max_grid_distance = max(1, self.rows * self.cols)
        self.dirty_positions = self._find_positions("D")
        self.charger_positions = self._find_positions("C")
        self.dirty_distance_maps = [
            self._build_distance_map([position]) for position in self.dirty_positions
        ]
        self.charger_distance_maps = [
            self._build_distance_map([position]) for position in self.charger_positions
        ]
        self.dirty_anchor_charger_ids = self._build_dirty_anchor_charger_ids()
        self.base_feature_count = 6 + (len(self.dirty_positions) * 2) + len(
            self.charger_positions
        )
        self.action_lookahead_feature_count = len(self.ACTION_DELTAS) * 4
        self.target_context_feature_count = (
            8
            if self.feature_version == self.FEATURE_VERSION_LEGACY
            else 13
        )
        self.input_size = (
            self.base_feature_count
            + self.action_lookahead_feature_count
            + self.target_context_feature_count
        )

    def _find_positions(self, symbol: str) -> list[tuple[int, int]]:
        positions: list[tuple[int, int]] = []
        for row, row_text in enumerate(self.grid_map):
            for col, cell in enumerate(row_text):
                if cell == symbol:
                    positions.append((row, col))
        return positions

    def _build_distance_map(
        self,
        source_positions: Iterable[tuple[int, int]],
    ) -> list[list[int]]:
        distance_map = [[-1 for _ in range(self.cols)] for _ in range(self.rows)]
        queue: deque[tuple[int, int]] = deque()

        for row, col in source_positions:
            distance_map[row][col] = 0
            queue.append((row, col))

        while queue:
            row, col = queue.popleft()
            for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_row = row + delta_row
                next_col = col + delta_col
                if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
                    continue
                if self.grid_map[next_row][next_col] == "#":
                    continue
                if distance_map[next_row][next_col] != -1:
                    continue
                distance_map[next_row][next_col] = distance_map[row][col] + 1
                queue.append((next_row, next_col))

        return distance_map

    def _build_dirty_anchor_charger_ids(self) -> list[int]:
        anchor_ids: list[int] = []

        for dirty_row, dirty_col in self.dirty_positions:
            best_charger_id = -1
            best_distance = -1

            for charger_id, distance_map in enumerate(self.charger_distance_maps):
                distance = self._distance_from_map(distance_map, dirty_row, dirty_col)
                if distance < 0:
                    continue
                if best_distance == -1 or distance < best_distance:
                    best_distance = distance
                    best_charger_id = charger_id

            anchor_ids.append(best_charger_id)

        return anchor_ids

    def _normalized_distance(self, distance_map: list[list[int]], row: int, col: int) -> float:
        if row < 0 or col < 0 or row >= self.rows or col >= self.cols:
            return 1.0

        distance = distance_map[row][col]
        if distance < 0:
            return 1.0
        return min(1.0, distance / self.max_grid_distance)

    def _is_open_cell(self, row: int, col: int) -> bool:
        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
            and self.grid_map[row][col] != "#"
        )

    def valid_action_mask(self, state: State) -> list[bool]:
        row, col, _, _ = state
        return [
            self._is_open_cell(row + delta_row, col + delta_col)
            for delta_row, delta_col in self.ACTION_DELTAS
        ]

    def valid_action_masks(
        self,
        states: list[State],
        device: torch.device,
    ) -> torch.Tensor:
        return torch.tensor(
            [self.valid_action_mask(state) for state in states],
            dtype=torch.bool,
            device=device,
        )

    def _distance_from_map(self, distance_map: list[list[int]], row: int, col: int) -> int:
        if row < 0 or col < 0 or row >= self.rows or col >= self.cols:
            return -1
        return distance_map[row][col]

    def _nearest_remaining_dirty_distance(
        self,
        row: int,
        col: int,
        cleaned_mask: int,
    ) -> int:
        best_distance = -1
        for dirty_idx, distance_map in enumerate(self.dirty_distance_maps):
            if (cleaned_mask >> dirty_idx) & 1:
                continue
            distance = self._distance_from_map(distance_map, row, col)
            if distance < 0:
                continue
            if best_distance == -1 or distance < best_distance:
                best_distance = distance
        return best_distance

    def _nearest_remaining_dirty_index(
        self,
        row: int,
        col: int,
        cleaned_mask: int,
    ) -> int:
        best_dirty_idx = -1
        best_distance = -1
        for dirty_idx, distance_map in enumerate(self.dirty_distance_maps):
            if (cleaned_mask >> dirty_idx) & 1:
                continue
            distance = self._distance_from_map(distance_map, row, col)
            if distance < 0:
                continue
            if best_distance == -1 or distance < best_distance:
                best_distance = distance
                best_dirty_idx = dirty_idx
        return best_dirty_idx

    def _safe_remaining_dirty_distance(
        self,
        row: int,
        col: int,
        cleaned_mask: int,
        battery_remaining: int,
    ) -> int:
        best_distance = -1
        for dirty_idx, distance_map in enumerate(self.dirty_distance_maps):
            if (cleaned_mask >> dirty_idx) & 1:
                continue

            distance_to_dirty = self._distance_from_map(distance_map, row, col)
            if distance_to_dirty < 0:
                continue

            dirty_row, dirty_col = self.dirty_positions[dirty_idx]
            dirty_to_charger = self._nearest_charger_distance(dirty_row, dirty_col)
            route_cost = self._safe_route_cost(distance_to_dirty, dirty_to_charger)
            if route_cost < 0:
                continue
            if route_cost > battery_remaining:
                continue

            if best_distance == -1 or distance_to_dirty < best_distance:
                best_distance = distance_to_dirty

        return best_distance

    def _safe_remaining_dirty_index(
        self,
        row: int,
        col: int,
        cleaned_mask: int,
        battery_remaining: int,
    ) -> int:
        best_dirty_idx = -1
        best_distance = -1
        for dirty_idx, distance_map in enumerate(self.dirty_distance_maps):
            if (cleaned_mask >> dirty_idx) & 1:
                continue

            distance_to_dirty = self._distance_from_map(distance_map, row, col)
            if distance_to_dirty < 0:
                continue

            dirty_row, dirty_col = self.dirty_positions[dirty_idx]
            dirty_to_charger = self._nearest_charger_distance(dirty_row, dirty_col)
            route_cost = self._safe_route_cost(distance_to_dirty, dirty_to_charger)
            if route_cost < 0:
                continue
            if route_cost > battery_remaining:
                continue

            if best_distance == -1 or distance_to_dirty < best_distance:
                best_distance = distance_to_dirty
                best_dirty_idx = dirty_idx

        return best_dirty_idx

    def _nearest_charger_distance(self, row: int, col: int) -> int:
        best_distance = -1
        for distance_map in self.charger_distance_maps:
            distance = self._distance_from_map(distance_map, row, col)
            if distance < 0:
                continue
            if best_distance == -1 or distance < best_distance:
                best_distance = distance
        return best_distance

    def _nearest_charger_id(self, row: int, col: int) -> int:
        best_charger_id = -1
        best_distance = -1

        for charger_id, distance_map in enumerate(self.charger_distance_maps):
            distance = self._distance_from_map(distance_map, row, col)
            if distance < 0:
                continue
            if best_distance == -1 or distance < best_distance:
                best_distance = distance
                best_charger_id = charger_id

        return best_charger_id

    def _route_safety_reserve(self) -> int:
        if self.feature_version == self.FEATURE_VERSION_LEGACY:
            return 0
        return self.battery_safety_reserve

    def _emergency_charger_buffer(self) -> int:
        if self.feature_version == self.FEATURE_VERSION_LEGACY:
            return 2
        return self.battery_safety_reserve

    def _safe_route_cost(self, outbound_distance: int, recovery_distance: int) -> int:
        if outbound_distance < 0 or recovery_distance < 0:
            return -1
        return outbound_distance + recovery_distance + self._route_safety_reserve()

    def _normalized_route_margin(self, margin: int) -> float:
        return max(-1.0, min(1.0, margin / self.battery_capacity))

    def _normalized_route_cost(self, route_cost: int) -> float:
        if route_cost < 0:
            return 1.0
        max_route_cost = self.max_grid_distance + self._route_safety_reserve()
        return min(1.0, route_cost / max(1, max_route_cost))

    def _normalized_charger_id(self, charger_id: int) -> float:
        if charger_id < 0 or len(self.charger_positions) <= 1:
            return 0.0
        return charger_id / max(1, len(self.charger_positions) - 1)

    def _target_route_profile(
        self,
        row: int,
        col: int,
        target_dirty_idx: int,
        battery_remaining: int,
    ) -> TargetRouteProfile:
        target_row, target_col = self.dirty_positions[target_dirty_idx]
        target_distance = self._distance_from_map(
            self.dirty_distance_maps[target_dirty_idx],
            row,
            col,
        )
        recovery_distance = self._nearest_charger_distance(target_row, target_col)
        direct_route_cost = self._safe_route_cost(target_distance, recovery_distance)
        direct_margin = self._normalized_route_margin(
            battery_remaining - direct_route_cost
            if direct_route_cost >= 0
            else -self.battery_capacity
        )
        direct_feasible = direct_route_cost >= 0 and direct_route_cost <= battery_remaining

        route_charger_id = self._nearest_charger_id(row, col)
        route_to_charger_distance = -1
        charger_to_target_distance = -1
        via_route_total_cost = -1
        via_post_charge_margin = -1.0
        via_route_feasible = False

        if route_charger_id >= 0:
            route_to_charger_distance = self._distance_from_map(
                self.charger_distance_maps[route_charger_id],
                row,
                col,
            )
            charger_row, charger_col = self.charger_positions[route_charger_id]
            charger_to_target_distance = self._distance_from_map(
                self.dirty_distance_maps[target_dirty_idx],
                charger_row,
                charger_col,
            )
            via_post_charge_cost = self._safe_route_cost(
                charger_to_target_distance,
                recovery_distance,
            )
            via_route_total_cost = (
                route_to_charger_distance + via_post_charge_cost
                if route_to_charger_distance >= 0 and via_post_charge_cost >= 0
                else -1
            )
            via_post_charge_margin = self._normalized_route_margin(
                self.battery_capacity - via_post_charge_cost
                if via_post_charge_cost >= 0
                else -self.battery_capacity
            )
            via_route_feasible = (
                route_to_charger_distance >= 0
                and route_to_charger_distance <= battery_remaining
                and via_post_charge_cost >= 0
                and via_post_charge_cost <= self.battery_capacity
            )

        return TargetRouteProfile(
            target_distance=target_distance,
            recovery_distance=recovery_distance,
            direct_route_cost=direct_route_cost,
            direct_margin=direct_margin,
            route_charger_id=route_charger_id,
            route_to_charger_distance=route_to_charger_distance,
            charger_to_target_distance=charger_to_target_distance,
            via_route_total_cost=via_route_total_cost,
            via_post_charge_margin=via_post_charge_margin,
            direct_feasible=direct_feasible,
            via_route_feasible=via_route_feasible,
        )

    def _normalized_distance_delta(self, before: int, after: int) -> float:
        if before < 0 and after < 0:
            return 0.0
        if before < 0:
            return -1.0
        if after < 0:
            return 1.0
        return max(-1.0, min(1.0, (before - after) / self.max_grid_distance))

    def _safety_margin(self, battery_remaining: int, charger_distance: int) -> int:
        if charger_distance < 0:
            return -self.battery_capacity
        return battery_remaining - charger_distance

    def _encode_action_lookahead_features(
        self,
        row: int,
        col: int,
        cleaned_mask: int,
        battery_remaining: int,
    ) -> list[float]:
        features: list[float] = []
        current_dirty_distance = self._nearest_remaining_dirty_distance(
            row,
            col,
            cleaned_mask,
        )
        current_charger_distance = self._nearest_charger_distance(row, col)
        current_safety_margin = self._safety_margin(
            battery_remaining,
            current_charger_distance,
        )

        for delta_row, delta_col in self.ACTION_DELTAS:
            next_row = row + delta_row
            next_col = col + delta_col
            is_open = self._is_open_cell(next_row, next_col)

            if not is_open:
                features.extend([0.0, -1.0, -1.0, -1.0])
                continue

            next_battery = max(0, battery_remaining - 1)
            next_dirty_distance = self._nearest_remaining_dirty_distance(
                next_row,
                next_col,
                cleaned_mask,
            )
            next_charger_distance = self._nearest_charger_distance(next_row, next_col)
            next_safety_margin = self._safety_margin(
                next_battery,
                next_charger_distance,
            )

            dirty_delta = self._normalized_distance_delta(
                before=current_dirty_distance,
                after=next_dirty_distance,
            )
            charger_delta = self._normalized_distance_delta(
                before=current_charger_distance,
                after=next_charger_distance,
            )
            safety_delta = (next_safety_margin - current_safety_margin) / self.battery_capacity
            safety_delta = max(-1.0, min(1.0, safety_delta))

            features.extend([1.0, dirty_delta, charger_delta, safety_delta])

        return features

    def _encode_target_context_features(
        self,
        row: int,
        col: int,
        cleaned_mask: int,
        battery_remaining: int,
    ) -> list[float]:
        remaining_dirty_count = max(0, len(self.dirty_positions) - cleaned_mask.bit_count())
        dirty_count = max(1, len(self.dirty_positions))
        target_dirty_idx = self._safe_remaining_dirty_index(
            row,
            col,
            cleaned_mask,
            battery_remaining,
        )
        has_safe_target = 1.0
        if target_dirty_idx == -1:
            has_safe_target = 0.0
            target_dirty_idx = self._nearest_remaining_dirty_index(row, col, cleaned_mask)

        if target_dirty_idx == -1:
            target_context = [
                remaining_dirty_count / dirty_count,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ]
            if self.feature_version == self.FEATURE_VERSION_LEGACY:
                return target_context
            target_context.extend([0.0, 1.0, 0.0, 0.0, 0.0])
            return target_context

        target_row, target_col = self.dirty_positions[target_dirty_idx]
        target_anchor_id = self.dirty_anchor_charger_ids[target_dirty_idx]
        nearest_charger_distance = self._nearest_charger_distance(row, col)
        route_profile = self._target_route_profile(
            row=row,
            col=col,
            target_dirty_idx=target_dirty_idx,
            battery_remaining=battery_remaining,
        )
        target_context = [
            remaining_dirty_count / dirty_count,
            target_row / max(1, self.rows - 1),
            target_col / max(1, self.cols - 1),
            min(1.0, max(0, route_profile.target_distance) / self.max_grid_distance),
            self._normalized_charger_id(target_anchor_id),
            route_profile.direct_margin,
            has_safe_target,
            (
                1.0
                if nearest_charger_distance >= 0
                and battery_remaining
                <= nearest_charger_distance + self._emergency_charger_buffer()
                else 0.0
            ),
        ]

        if self.feature_version == self.FEATURE_VERSION_LEGACY:
            return target_context

        reroute_required = (
            1.0
            if not route_profile.direct_feasible and route_profile.via_route_feasible
            else 0.0
        )
        target_context.extend(
            [
                self._normalized_charger_id(route_profile.route_charger_id),
                self._normalized_route_cost(route_profile.via_route_total_cost),
                route_profile.via_post_charge_margin,
                1.0 if route_profile.via_route_feasible else 0.0,
                reroute_required,
            ]
        )
        return target_context

    def guided_action(self, state: State) -> int | None:
        row, col, cleaned_mask, battery_value = state
        battery_remaining = self.battery_capacity if battery_value < 0 else battery_value
        valid_actions = [
            action_idx
            for action_idx, is_valid in enumerate(self.valid_action_mask(state))
            if is_valid
        ]
        if not valid_actions:
            return None

        charger_distance = self._nearest_charger_distance(row, col)
        if (
            charger_distance >= 0
            and battery_remaining <= charger_distance + self._emergency_charger_buffer()
        ):
            target_kind = "charger"
        elif self._safe_remaining_dirty_distance(
            row,
            col,
            cleaned_mask,
            battery_remaining,
        ) >= 0:
            target_kind = "dirty"
        else:
            target_kind = "charger"

        best_actions: list[int] = []
        best_distance = -1

        for action_idx in valid_actions:
            delta_row, delta_col = self.ACTION_DELTAS[action_idx]
            next_row = row + delta_row
            next_col = col + delta_col

            if target_kind == "dirty":
                distance = self._safe_remaining_dirty_distance(
                    next_row,
                    next_col,
                    cleaned_mask,
                    max(0, battery_remaining - 1),
                )
            else:
                distance = self._nearest_charger_distance(next_row, next_col)

            if distance < 0:
                continue
            if best_distance == -1 or distance < best_distance:
                best_distance = distance
                best_actions = [action_idx]
            elif distance == best_distance:
                best_actions.append(action_idx)

        if best_actions:
            return min(best_actions)

        return min(valid_actions)

    def encode(self, state: State) -> list[float]:
        row, col, cleaned_mask, battery_value = state
        battery_remaining = self.battery_capacity if battery_value < 0 else battery_value
        battery_ratio = max(0.0, min(1.0, battery_remaining / self.battery_capacity))
        cleaned_count = cleaned_mask.bit_count()
        dirty_count = max(1, len(self.dirty_positions))
        nearest_charger_distance = self._nearest_charger_distance(row, col)
        charger_distance_norm = (
            1.0
            if nearest_charger_distance < 0
            else min(1.0, nearest_charger_distance / self.max_grid_distance)
        )
        safety_margin_norm = (
            battery_remaining - max(0, nearest_charger_distance)
        ) / self.battery_capacity
        safety_margin_norm = max(-1.0, min(1.0, safety_margin_norm))

        features = [
            row / max(1, self.rows - 1),
            col / max(1, self.cols - 1),
            battery_ratio,
            cleaned_count / dirty_count,
            charger_distance_norm,
            safety_margin_norm,
        ]

        for dirty_idx, distance_map in enumerate(self.dirty_distance_maps):
            is_uncleaned = 0.0 if ((cleaned_mask >> dirty_idx) & 1) else 1.0
            features.append(is_uncleaned)
            features.append(self._normalized_distance(distance_map, row, col))

        for distance_map in self.charger_distance_maps:
            features.append(self._normalized_distance(distance_map, row, col))

        features.extend(
            self._encode_target_context_features(
                row=row,
                col=col,
                cleaned_mask=cleaned_mask,
                battery_remaining=battery_remaining,
            )
        )

        features.extend(
            self._encode_action_lookahead_features(
                row=row,
                col=col,
                cleaned_mask=cleaned_mask,
                battery_remaining=battery_remaining,
            )
        )

        return features

    def encode_batch(self, states: list[State], device: torch.device) -> torch.Tensor:
        return torch.tensor(
            [self.encode(state) for state in states],
            dtype=torch.float32,
            device=device,
        )


class DQNNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    def __init__(
        self,
        config: DQNConfig,
        device: str = "auto",
    ) -> None:
        self.config = config
        self.action_space_size = config.action_space_size
        self.epsilon = config.epsilon
        self.rng = random.Random(config.seed)
        self.device = self._resolve_device(device)
        self.encoder = StateFeatureEncoder(
            map_name=config.map_name,
            battery_capacity=config.battery_capacity,
            feature_version=config.feature_version,
        )

        if config.seed is not None:
            torch.manual_seed(config.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(config.seed)

        self.policy_net = DQNNetwork(
            input_size=self.encoder.input_size,
            hidden_size=config.hidden_size,
            output_size=config.action_space_size,
        ).to(self.device)
        self.target_net = DQNNetwork(
            input_size=self.encoder.input_size,
            hidden_size=config.hidden_size,
            output_size=config.action_space_size,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(),
            lr=config.learning_rate,
            amsgrad=True,
        )
        self.replay_buffer = ReplayBuffer(
            capacity=config.replay_capacity,
            seed=config.seed,
        )
        self.training_steps = 0
        self.optimization_steps = 0

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")
        return resolved_device

    def select_action(self, state: State, training: bool = True) -> int:
        valid_actions = [
            action_idx
            for action_idx, is_valid in enumerate(self.encoder.valid_action_mask(state))
            if is_valid
        ]
        if not valid_actions:
            valid_actions = list(range(self.action_space_size))

        if training and self.rng.random() < self.epsilon:
            guided_ratio = max(0.0, min(1.0, self.config.guided_exploration_ratio))
            if guided_ratio > 0 and self.rng.random() < guided_ratio:
                guided_action = self.encoder.guided_action(state)
                if guided_action is not None:
                    return guided_action
            return self.rng.choice(valid_actions)

        with torch.no_grad():
            state_tensor = self.encoder.encode_batch([state], device=self.device)
            q_values = self.policy_net(state_tensor)[0]
            valid_mask = self.encoder.valid_action_masks([state], device=self.device)[0]
            masked_q_values = q_values.masked_fill(~valid_mask, -torch.inf)
            max_q = torch.max(masked_q_values).item()
            best_actions = [
                action_idx
                for action_idx, value in enumerate(masked_q_values.tolist())
                if value == max_q
            ]
            return self.rng.choice(best_actions)

    def remember(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        self.replay_buffer.push(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

    def optimize_if_ready(self) -> float | None:
        self.training_steps += 1

        if len(self.replay_buffer) < self.config.learning_starts:
            return None
        if self.training_steps % self.config.train_every != 0:
            return None
        if len(self.replay_buffer) < self.config.batch_size:
            return None

        batch = self.replay_buffer.sample(self.config.batch_size)
        states = self.encoder.encode_batch([transition.state for transition in batch], self.device)
        next_states = self.encoder.encode_batch(
            [transition.next_state for transition in batch],
            self.device,
        )
        actions = torch.tensor(
            [transition.action for transition in batch],
            dtype=torch.int64,
            device=self.device,
        ).unsqueeze(1)
        rewards = torch.tensor(
            [transition.reward for transition in batch],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.tensor(
            [transition.done for transition in batch],
            dtype=torch.float32,
            device=self.device,
        )

        predicted_q = self.policy_net(states).gather(1, actions).squeeze(1)

        with torch.no_grad():
            next_valid_masks = self.encoder.valid_action_masks(
                [transition.next_state for transition in batch],
                self.device,
            )
            next_policy_q = self.policy_net(next_states).masked_fill(
                ~next_valid_masks,
                -torch.inf,
            )
            next_actions = next_policy_q.argmax(dim=1, keepdim=True)
            next_target_q = self.target_net(next_states).masked_fill(
                ~next_valid_masks,
                -torch.inf,
            )
            next_q = next_target_q.gather(1, next_actions).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.config.discount_factor * next_q

        loss = F.smooth_l1_loss(predicted_q, target_q)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100.0)
        self.optimizer.step()

        self.optimization_steps += 1
        if self.optimization_steps % self.config.target_update_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)

    def get_policy_action(self, state: State) -> int:
        return self.select_action(state, training=False)

    def to_checkpoint(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "algorithm": "dqn",
            "config": asdict(self.config),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "optimization_steps": self.optimization_steps,
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": metadata or {},
        }

    def save(self, file_path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_checkpoint(metadata=metadata), output_path)

    @classmethod
    def load(
        cls,
        file_path: str | Path,
        device: str = "auto",
        training: bool = False,
    ) -> "DQNAgent":
        input_path = Path(file_path)
        resolved_device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        payload = torch.load(input_path, map_location=resolved_device)
        config_payload = dict(payload["config"])
        config_payload.setdefault(
            "feature_version",
            StateFeatureEncoder.FEATURE_VERSION_LEGACY,
        )
        config = DQNConfig(**config_payload)
        agent = cls(config=config, device=str(resolved_device))
        agent.epsilon = float(payload.get("epsilon", config.epsilon))
        agent.training_steps = int(payload.get("training_steps", 0))
        agent.optimization_steps = int(payload.get("optimization_steps", 0))
        agent.policy_net.load_state_dict(payload["policy_state_dict"])
        agent.target_net.load_state_dict(payload.get("target_state_dict", payload["policy_state_dict"]))
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if training:
            agent.policy_net.train()
        else:
            agent.policy_net.eval()
        agent.target_net.eval()
        return agent

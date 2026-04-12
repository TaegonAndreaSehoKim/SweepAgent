from __future__ import annotations

from collections import deque
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from configs.map_presets import MAP_PRESETS


State = Tuple[int, int, int, int]


class QLearningAgent:
    _SUPPORTED_ABSTRACTION_MODES = {"identity", "safety_margin", "charger_context"}
    _charger_distance_cache: Dict[str, List[List[int]]] = {}

    def __init__(
        self,
        action_space_size: int = 4,
        learning_rate: float = 0.05,
        discount_factor: float = 0.995,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.10,
        seed: int | None = None,
        state_abstraction_mode: str = "identity",
        abstraction_map_name: str = "",
        safety_margin_bucket_size: int = 5,
    ) -> None:
        # Store the core Q-learning hyperparameters.
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Store exploration settings for epsilon-greedy action selection.
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Keep the seed so the agent can be reconstructed later.
        self.seed = seed

        if state_abstraction_mode not in self._SUPPORTED_ABSTRACTION_MODES:
            supported = ", ".join(sorted(self._SUPPORTED_ABSTRACTION_MODES))
            raise ValueError(
                f"Unsupported state_abstraction_mode='{state_abstraction_mode}'. "
                f"Supported modes: {supported}"
            )

        # Keep optional state abstraction settings inside the agent so checkpoints
        # can be replayed without external wiring.
        self.state_abstraction_mode = state_abstraction_mode
        self.abstraction_map_name = abstraction_map_name
        self.safety_margin_bucket_size = max(1, int(safety_margin_bucket_size))
        self._charger_distance_map = self._get_charger_distance_map(abstraction_map_name)
        self._map_grid = MAP_PRESETS.get(abstraction_map_name, {}).get("grid_map", [])
        self._charger_positions = self._find_positions("C")
        self._dirty_positions = self._find_positions("D")
        self._charger_distance_maps = self._build_distance_maps(self._charger_positions)
        self._dirty_distance_maps = self._build_distance_maps(self._dirty_positions)
        self._dirty_anchor_charger_ids = self._build_dirty_anchor_charger_ids()

        # Use a local RNG so runs can be reproduced with the same seed.
        self.rng = random.Random(seed)

        # Map each state to one Q-value per action.
        self.q_table: Dict[State, List[float]] = {}

    @classmethod
    def _get_charger_distance_map(cls, map_name: str) -> List[List[int]] | None:
        if not map_name or map_name not in MAP_PRESETS:
            return None

        cached_map = cls._charger_distance_cache.get(map_name)
        if cached_map is not None:
            return cached_map

        grid_map = MAP_PRESETS[map_name]["grid_map"]
        rows = len(grid_map)
        cols = len(grid_map[0]) if rows else 0
        distance_map = [[-1 for _ in range(cols)] for _ in range(rows)]
        queue: deque[tuple[int, int]] = deque()

        for row in range(rows):
            for col in range(cols):
                if grid_map[row][col] == "C":
                    distance_map[row][col] = 0
                    queue.append((row, col))

        while queue:
            row, col = queue.popleft()

            for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_row = row + delta_row
                next_col = col + delta_col

                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    continue
                if grid_map[next_row][next_col] == "#":
                    continue
                if distance_map[next_row][next_col] != -1:
                    continue

                distance_map[next_row][next_col] = distance_map[row][col] + 1
                queue.append((next_row, next_col))

        cls._charger_distance_cache[map_name] = distance_map
        return distance_map

    def _find_positions(self, symbol: str) -> List[tuple[int, int]]:
        positions: List[tuple[int, int]] = []
        for row, row_text in enumerate(self._map_grid):
            for col, cell in enumerate(row_text):
                if cell == symbol:
                    positions.append((row, col))
        return positions

    def _build_distance_map(
        self,
        source_positions: List[tuple[int, int]],
    ) -> List[List[int]]:
        if not self._map_grid:
            return []

        rows = len(self._map_grid)
        cols = len(self._map_grid[0]) if rows else 0
        distance_map = [[-1 for _ in range(cols)] for _ in range(rows)]
        queue: deque[tuple[int, int]] = deque()

        for row, col in source_positions:
            distance_map[row][col] = 0
            queue.append((row, col))

        while queue:
            row, col = queue.popleft()

            for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                next_row = row + delta_row
                next_col = col + delta_col

                if not (0 <= next_row < rows and 0 <= next_col < cols):
                    continue
                if self._map_grid[next_row][next_col] == "#":
                    continue
                if distance_map[next_row][next_col] != -1:
                    continue

                distance_map[next_row][next_col] = distance_map[row][col] + 1
                queue.append((next_row, next_col))

        return distance_map

    def _build_distance_maps(
        self,
        positions: List[tuple[int, int]],
    ) -> List[List[List[int]]]:
        return [self._build_distance_map([position]) for position in positions]

    def _distance_from_map(
        self,
        distance_map: List[List[int]],
        row: int,
        col: int,
    ) -> int:
        if not distance_map:
            return -1
        if row < 0 or col < 0 or row >= len(distance_map) or col >= len(distance_map[row]):
            return -1
        return distance_map[row][col]

    def _build_dirty_anchor_charger_ids(self) -> List[int]:
        anchor_ids: List[int] = []

        for dirty_row, dirty_col in self._dirty_positions:
            best_charger_id = 0
            best_distance = -1

            for charger_id, charger_map in enumerate(self._charger_distance_maps):
                distance = self._distance_from_map(charger_map, dirty_row, dirty_col)
                if distance == -1:
                    continue
                if best_distance == -1 or distance < best_distance:
                    best_distance = distance
                    best_charger_id = charger_id

            anchor_ids.append(best_charger_id)

        return anchor_ids

    def _encode_safety_margin_feature(
        self,
        row: int,
        col: int,
        battery_value: int,
    ) -> int:
        if battery_value < 0:
            return -1

        if (
            self._charger_distance_map is None
            or row < 0
            or col < 0
            or row >= len(self._charger_distance_map)
            or col >= len(self._charger_distance_map[row])
        ):
            return battery_value // self.safety_margin_bucket_size

        charger_distance = self._charger_distance_map[row][col]
        if charger_distance < 0:
            return battery_value // self.safety_margin_bucket_size

        safety_margin = battery_value - charger_distance
        return safety_margin // self.safety_margin_bucket_size

    def _nearest_charger_id(self, row: int, col: int) -> int:
        best_charger_id = -1
        best_distance = -1

        for charger_id, charger_map in enumerate(self._charger_distance_maps):
            distance = self._distance_from_map(charger_map, row, col)
            if distance == -1:
                continue
            if best_distance == -1 or distance < best_distance:
                best_distance = distance
                best_charger_id = charger_id

        return best_charger_id

    def _remaining_dirty_indices(self, cleaned_mask: int) -> List[int]:
        return [
            dirty_idx
            for dirty_idx in range(len(self._dirty_positions))
            if ((cleaned_mask >> dirty_idx) & 1) == 0
        ]

    def _nearest_remaining_dirty_index(
        self,
        row: int,
        col: int,
        cleaned_mask: int,
    ) -> int:
        best_dirty_idx = -1
        best_distance = -1

        for dirty_idx in self._remaining_dirty_indices(cleaned_mask):
            distance = self._distance_from_map(
                self._dirty_distance_maps[dirty_idx],
                row,
                col,
            )
            if distance == -1:
                continue
            if best_distance == -1 or distance < best_distance:
                best_distance = distance
                best_dirty_idx = dirty_idx

        return best_dirty_idx

    def _count_safe_remaining_dirties(
        self,
        row: int,
        col: int,
        cleaned_mask: int,
        battery_value: int,
    ) -> int:
        if battery_value < 0:
            return 0

        safe_count = 0
        for dirty_idx in self._remaining_dirty_indices(cleaned_mask):
            distance_to_dirty = self._distance_from_map(
                self._dirty_distance_maps[dirty_idx],
                row,
                col,
            )
            dirty_row, dirty_col = self._dirty_positions[dirty_idx]
            distance_dirty_to_charger = self._distance_to_nearest_charger_from_position(
                dirty_row,
                dirty_col,
            )

            if distance_to_dirty == -1 or distance_dirty_to_charger == -1:
                continue
            if distance_to_dirty + distance_dirty_to_charger <= battery_value:
                safe_count += 1

        return safe_count

    def _distance_to_nearest_charger_from_position(self, row: int, col: int) -> int:
        best_distance = -1
        for charger_map in self._charger_distance_maps:
            distance = self._distance_from_map(charger_map, row, col)
            if distance == -1:
                continue
            if best_distance == -1 or distance < best_distance:
                best_distance = distance
        return best_distance

    def _encode_charger_context_feature(
        self,
        row: int,
        col: int,
        cleaned_mask: int,
        battery_value: int,
    ) -> int:
        if not self._map_grid:
            return self._encode_safety_margin_feature(row, col, battery_value)

        charger_count = max(1, len(self._charger_positions))
        none_region_id = charger_count

        current_region_id = self._nearest_charger_id(row, col)
        if current_region_id == -1:
            current_region_id = none_region_id

        nearest_dirty_idx = self._nearest_remaining_dirty_index(row, col, cleaned_mask)
        target_region_id = none_region_id
        if nearest_dirty_idx != -1 and nearest_dirty_idx < len(self._dirty_anchor_charger_ids):
            target_region_id = self._dirty_anchor_charger_ids[nearest_dirty_idx]

        remaining_count = len(self._remaining_dirty_indices(cleaned_mask))
        remaining_bucket = min(remaining_count, min(7, len(self._dirty_positions)))

        safe_dirty_count = self._count_safe_remaining_dirties(
            row,
            col,
            cleaned_mask,
            battery_value,
        )
        safe_dirty_bucket = min(safe_dirty_count, 3)

        margin_bucket = self._encode_safety_margin_feature(row, col, battery_value)
        shifted_margin_bucket = max(0, min(31, margin_bucket + 8))

        encoded_feature = shifted_margin_bucket
        encoded_feature = encoded_feature * 4 + safe_dirty_bucket
        encoded_feature = encoded_feature * (min(7, len(self._dirty_positions)) + 1) + remaining_bucket
        encoded_feature = encoded_feature * (charger_count + 1) + target_region_id
        encoded_feature = encoded_feature * (charger_count + 1) + current_region_id
        return encoded_feature

    def _encode_state(self, state: State) -> State:
        if self.state_abstraction_mode == "identity":
            return state

        row, col, cleaned_mask, battery_value = state

        if self.state_abstraction_mode == "safety_margin":
            return (
                row,
                col,
                cleaned_mask,
                self._encode_safety_margin_feature(row, col, battery_value),
            )

        if self.state_abstraction_mode == "charger_context":
            return (
                row,
                col,
                cleaned_mask,
                self._encode_charger_context_feature(
                    row,
                    col,
                    cleaned_mask,
                    battery_value,
                ),
            )

        raise ValueError(
            f"Unsupported state_abstraction_mode='{self.state_abstraction_mode}'."
        )

    def _ensure_state_exists(self, state: State) -> State:
        encoded_state = self._encode_state(state)

        # Lazily initialize unseen states with zero Q-values.
        if encoded_state not in self.q_table:
            self.q_table[encoded_state] = [
                0.0 for _ in range(self.action_space_size)
            ]
        return encoded_state

    def _state_to_key(self, state: State) -> str:
        # Convert a tuple state into a JSON-safe string key.
        return ",".join(str(value) for value in state)

    def _key_to_state(self, key: str) -> State:
        # Convert a JSON string key back into a tuple state.
        parts = [int(value) for value in key.split(",")]

        # Backward compatibility for older checkpoints that used 3-part states.
        if len(parts) == 3:
            row, col, cleaned_mask = parts
            return (row, col, cleaned_mask, -1)

        if len(parts) == 4:
            row, col, cleaned_mask, battery_value = parts
            return (row, col, cleaned_mask, battery_value)

        raise ValueError(f"Unexpected serialized state format: {key}")

    def get_q_values(self, state: State) -> List[float]:
        # Return the Q-values for the given state.
        encoded_state = self._ensure_state_exists(state)
        return self.q_table[encoded_state]

    def select_action(self, state: State, training: bool = True) -> int:
        # Make sure the current state exists in the table.
        encoded_state = self._ensure_state_exists(state)

        # Explore with probability epsilon during training.
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_space_size)

        # Otherwise choose the best action, breaking ties randomly.
        q_values = self.q_table[encoded_state]
        max_q = max(q_values)
        best_actions = [
            action for action, value in enumerate(q_values) if value == max_q
        ]
        return self.rng.choice(best_actions)

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        # Make sure both states exist before updating the table.
        encoded_state = self._ensure_state_exists(state)
        encoded_next_state = self._ensure_state_exists(next_state)

        current_q = self.q_table[encoded_state][action]

        # Use the terminal reward directly, or bootstrap from the next state.
        if done:
            target = reward
        else:
            next_max_q = max(self.q_table[encoded_next_state])
            target = reward + self.discount_factor * next_max_q

        # Apply the standard Q-learning update rule.
        new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table[encoded_state][action] = new_q

    def decay_epsilon(self) -> None:
        # Reduce exploration, but never go below the minimum value.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy_action(self, state: State) -> int:
        # Select the greedy action without exploration.
        return self.select_action(state, training=False)

    def to_dict(self) -> Dict[str, Any]:
        # Convert the full agent state into a JSON-serializable dictionary.
        return {
            "action_space_size": self.action_space_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "seed": self.seed,
            "state_abstraction_mode": self.state_abstraction_mode,
            "abstraction_map_name": self.abstraction_map_name,
            "safety_margin_bucket_size": self.safety_margin_bucket_size,
            "q_table": {
                self._state_to_key(state): q_values
                for state, q_values in self.q_table.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "QLearningAgent":
        # Rebuild an agent instance from a serialized dictionary.
        agent = cls(
            action_space_size=payload["action_space_size"],
            learning_rate=payload["learning_rate"],
            discount_factor=payload["discount_factor"],
            epsilon=payload["epsilon"],
            epsilon_decay=payload["epsilon_decay"],
            epsilon_min=payload["epsilon_min"],
            seed=payload.get("seed"),
            state_abstraction_mode=payload.get("state_abstraction_mode", "identity"),
            abstraction_map_name=payload.get("abstraction_map_name", ""),
            safety_margin_bucket_size=payload.get("safety_margin_bucket_size", 5),
        )

        agent.q_table = {
            agent._key_to_state(state_key): list(q_values)
            for state_key, q_values in payload["q_table"].items()
        }
        return agent

    def save(self, file_path: str | Path) -> None:
        # Save the agent checkpoint as a JSON file.
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)

    @classmethod
    def load(cls, file_path: str | Path) -> "QLearningAgent":
        # Load the agent checkpoint from a JSON file.
        input_path = Path(file_path)

        with input_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        return cls.from_dict(payload)

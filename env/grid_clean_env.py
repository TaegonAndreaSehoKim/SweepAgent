from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class StepResult:
    next_state: Tuple[int, int, int, int]
    reward: float
    done: bool
    info: Dict[str, float | str]


class GridCleanEnv:
    ACTIONS = {
        0: (-1, 0),   # up
        1: (1, 0),    # down
        2: (0, -1),   # left
        3: (0, 1),    # right
    }

    ACTION_NAMES = {
        0: "up",
        1: "down",
        2: "left",
        3: "right",
    }

    def __init__(
        self,
        grid_map: List[str] | None = None,
        max_steps: int = 100,
        reward_clean: float = 10,
        reward_move: float = -1,
        reward_revisit: float = -2,
        reward_invalid: float = -5,
        reward_finish: float = 50,
        battery_capacity: int | None = None,
        initial_cleaned_positions: List[Tuple[int, int]] | None = None,
        first_recharge_reward: float = 0,
        successful_recharge_completion_bonus: float = 10,
        low_battery_ratio: float = 0.3,
        reward_move_toward_charger: float = 0,
        penalty_move_away_from_charger: float = 0,
        reward_move_toward_safe_dirty: float = 0,
        penalty_move_away_from_safe_dirty: float = 0,
        reward_move_toward_relay_charger: float = 0,
        penalty_move_away_from_relay_charger: float = 0,
        battery_safety_reserve_min: int = 0,
        battery_safety_reserve_ratio: float = 0.0,
        low_battery_recharge_reward: float = 8,
        penalty_recharge_without_progress: float = 0,
        reward_final_dirty_bonus: float = 30,
        penalty_battery_depleted: float = -30,
        penalty_enter_unrecoverable_state: float = 0,
    ) -> None:
        if grid_map is None:
            grid_map = [
                "#######",
                "#R..D.#",
                "#..#..#",
                "#.D.D.#",
                "#######",
            ]

        self.raw_map = grid_map
        self.max_steps = max_steps

        self.reward_clean = reward_clean
        self.reward_move = reward_move
        self.reward_revisit = reward_revisit
        self.reward_invalid = reward_invalid
        self.reward_finish = reward_finish

        self.battery_capacity = battery_capacity
        self.battery_remaining = battery_capacity

        self.grid: List[List[str]] = [list(row) for row in grid_map]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

        self.start_pos = self._find_start_pos()
        start_row, start_col = self.start_pos
        self.grid[start_row][start_col] = "."

        self.charger_positions = self._find_charger_positions()
        self.charger_distance_maps = self._build_charger_distance_maps()
        self.charger_distance_map = self._build_charger_distance_map()

        self.dirty_positions = self._find_dirty_positions()
        self.dirty_distance_maps = self._build_dirty_distance_maps()
        self.dirty_index_map = {
            pos: idx for idx, pos in enumerate(self.dirty_positions)
        }
        self.total_dirty_tiles = len(self.dirty_positions)
        self.safe_dirty_distance_cache: dict[tuple[int, int, int, int], int] = {}
        self.initial_cleaned_mask = self._build_initial_cleaned_mask(
            initial_cleaned_positions or []
        )

        self.robot_pos: Tuple[int, int] = self.start_pos
        self.cleaned_mask: int = self.initial_cleaned_mask
        self.steps_taken: int = 0
        self.visited_positions: set[tuple[int, int]] = {self.start_pos}

        self.first_recharge_reward = first_recharge_reward
        self.successful_recharge_completion_bonus = successful_recharge_completion_bonus
        self.has_recharged_this_episode = False
        self.cleaned_tiles_at_last_recharge = 0

        self.low_battery_ratio = low_battery_ratio
        self.reward_move_toward_charger = reward_move_toward_charger
        self.penalty_move_away_from_charger = penalty_move_away_from_charger
        self.reward_move_toward_safe_dirty = reward_move_toward_safe_dirty
        self.penalty_move_away_from_safe_dirty = penalty_move_away_from_safe_dirty
        self.reward_move_toward_relay_charger = reward_move_toward_relay_charger
        self.penalty_move_away_from_relay_charger = (
            penalty_move_away_from_relay_charger
        )
        self.battery_safety_reserve_min = battery_safety_reserve_min
        self.battery_safety_reserve_ratio = battery_safety_reserve_ratio

        self.low_battery_recharge_reward = low_battery_recharge_reward
        self.penalty_recharge_without_progress = penalty_recharge_without_progress
        self.reward_final_dirty_bonus = reward_final_dirty_bonus
        self.penalty_battery_depleted = penalty_battery_depleted
        self.penalty_enter_unrecoverable_state = penalty_enter_unrecoverable_state

    def _find_start_pos(self) -> Tuple[int, int]:
        start_positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == "R":
                    start_positions.append((r, c))

        if len(start_positions) != 1:
            raise ValueError("grid_map must contain exactly one 'R' start position.")

        return start_positions[0]

    def _find_dirty_positions(self) -> List[Tuple[int, int]]:
        dirty_positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == "D":
                    dirty_positions.append((r, c))
        return dirty_positions

    def _find_charger_positions(self) -> List[Tuple[int, int]]:
        charger_positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == "C":
                    charger_positions.append((r, c))
        return charger_positions

    def _build_initial_cleaned_mask(
        self,
        initial_cleaned_positions: List[Tuple[int, int]],
    ) -> int:
        cleaned_mask = 0
        for pos in initial_cleaned_positions:
            if pos not in self.dirty_index_map:
                continue
            cleaned_mask |= 1 << self.dirty_index_map[pos]
        return cleaned_mask

    def _build_distance_map(
        self,
        source_positions: list[tuple[int, int]],
    ) -> list[list[int]]:
        distance_map = [[-1 for _ in range(self.cols)] for _ in range(self.rows)]

        if not source_positions:
            return distance_map

        queue = deque()

        for source_row, source_col in source_positions:
            distance_map[source_row][source_col] = 0
            queue.append((source_row, source_col))

        while queue:
            row, col = queue.popleft()

            for dr, dc in self.ACTIONS.values():
                next_row = row + dr
                next_col = col + dc

                if not (0 <= next_row < self.rows and 0 <= next_col < self.cols):
                    continue
                if self._is_wall(next_row, next_col):
                    continue
                if distance_map[next_row][next_col] != -1:
                    continue

                distance_map[next_row][next_col] = distance_map[row][col] + 1
                queue.append((next_row, next_col))

        return distance_map

    def _build_charger_distance_map(self) -> list[list[int]]:
        return self._build_distance_map(self.charger_positions)

    def _build_charger_distance_maps(self) -> Dict[Tuple[int, int], list[list[int]]]:
        return {
            charger_pos: self._build_distance_map([charger_pos])
            for charger_pos in self.charger_positions
        }

    def _build_dirty_distance_maps(self) -> Dict[Tuple[int, int], list[list[int]]]:
        return {
            dirty_pos: self._build_distance_map([dirty_pos])
            for dirty_pos in self.dirty_positions
        }

    def _is_wall(self, row: int, col: int) -> bool:
        return self.grid[row][col] == "#"

    def _is_dirty(self, row: int, col: int) -> bool:
        pos = (row, col)
        if pos not in self.dirty_index_map:
            return False

        idx = self.dirty_index_map[pos]
        return ((self.cleaned_mask >> idx) & 1) == 0

    def _is_charger(self, row: int, col: int) -> bool:
        return (row, col) in self.charger_positions

    def _distance_to_nearest_charger(self, row: int, col: int) -> int:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return -1
        return self.charger_distance_map[row][col]

    def _distance_to_charger(
        self,
        row: int,
        col: int,
        charger_pos: Tuple[int, int],
    ) -> int:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return -1
        distance_map = self.charger_distance_maps.get(charger_pos)
        if distance_map is None:
            return -1
        return distance_map[row][col]

    def _distance_to_dirty(self, row: int, col: int, dirty_pos: Tuple[int, int]) -> int:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return -1
        return self.dirty_distance_maps[dirty_pos][row][col]

    def _should_consider_charger(self) -> bool:
        return (
            self.battery_remaining is not None
            and self.battery_capacity is not None
            and len(self.charger_positions) > 0
        )

    def _battery_safety_reserve(self) -> int:
        if self.battery_capacity is None:
            return 0
        return max(
            self.battery_safety_reserve_min,
            math.ceil(self.battery_capacity * self.battery_safety_reserve_ratio),
        )

    def _dirty_route_cost(
        self,
        outbound_distance: int,
        recovery_distance: int,
    ) -> int:
        if outbound_distance < 0:
            return -1
        remaining_dirty_count = self.total_dirty_tiles - self._count_cleaned_tiles()
        if remaining_dirty_count <= 1:
            return outbound_distance
        if recovery_distance < 0:
            return -1
        return outbound_distance + recovery_distance + self._battery_safety_reserve()

    def _distance_to_nearest_safe_dirty(
        self,
        row: int,
        col: int,
        battery_remaining: int | None,
    ) -> int:
        if battery_remaining is None or not self._should_consider_charger():
            return -1

        cache_key = (row, col, self.cleaned_mask, battery_remaining)
        cached_distance = self.safe_dirty_distance_cache.get(cache_key)
        if cached_distance is not None:
            return cached_distance

        best_distance = -1

        for dirty_pos in self.dirty_positions:
            if not self._is_dirty(*dirty_pos):
                continue

            distance_to_dirty = self._distance_to_dirty(row, col, dirty_pos)
            charger_distance_from_dirty = self._distance_to_nearest_charger(*dirty_pos)

            if distance_to_dirty == -1 or charger_distance_from_dirty == -1:
                continue
            route_cost = self._dirty_route_cost(
                distance_to_dirty,
                charger_distance_from_dirty,
            )
            if route_cost < 0 or route_cost > battery_remaining:
                continue

            if best_distance == -1 or distance_to_dirty < best_distance:
                best_distance = distance_to_dirty

        self.safe_dirty_distance_cache[cache_key] = best_distance
        return best_distance

    def _best_relay_charger_position(
        self,
        row: int,
        col: int,
        battery_remaining: int | None,
    ) -> Tuple[int, int] | None:
        if battery_remaining is None or not self._should_consider_charger():
            return None

        best_charger_pos: Tuple[int, int] | None = None
        best_score: tuple[int, int, int, int, int, int] | None = None

        def dirty_unlock_score(charger_pos: Tuple[int, int]) -> tuple[int, int, int]:
            reachable_dirty_count = 0
            best_route_cost = -1
            best_target_distance = -1

            for dirty_pos in self.dirty_positions:
                if not self._is_dirty(*dirty_pos):
                    continue

                charger_to_dirty = self._distance_to_dirty(
                    charger_pos[0],
                    charger_pos[1],
                    dirty_pos,
                )
                dirty_to_charger = self._distance_to_nearest_charger(*dirty_pos)
                if charger_to_dirty < 0 or dirty_to_charger < 0:
                    continue

                route_cost = self._dirty_route_cost(charger_to_dirty, dirty_to_charger)
                if route_cost < 0 or route_cost > self.battery_capacity:
                    continue

                reachable_dirty_count += 1
                if best_route_cost == -1 or route_cost < best_route_cost:
                    best_route_cost = route_cost
                if best_target_distance == -1 or charger_to_dirty < best_target_distance:
                    best_target_distance = charger_to_dirty

            return reachable_dirty_count, best_route_cost, best_target_distance

        for charger_idx, charger_pos in enumerate(self.charger_positions):
            distance_to_charger = self._distance_to_charger(row, col, charger_pos)
            if distance_to_charger < 0 or distance_to_charger > battery_remaining:
                continue

            reachable_dirty_count, best_route_cost, best_target_distance = (
                dirty_unlock_score(charger_pos)
            )
            relay_hops = 0

            if reachable_dirty_count == 0:
                best_relay_score: tuple[int, int, int] | None = None

                for next_charger_pos in self.charger_positions:
                    if next_charger_pos == charger_pos:
                        continue

                    distance_to_next_charger = self._distance_to_charger(
                        charger_pos[0],
                        charger_pos[1],
                        next_charger_pos,
                    )
                    if (
                        distance_to_next_charger < 0
                        or distance_to_next_charger > self.battery_capacity
                    ):
                        continue

                    (
                        relay_dirty_count,
                        relay_route_cost,
                        relay_target_distance,
                    ) = dirty_unlock_score(next_charger_pos)
                    if relay_dirty_count == 0:
                        continue

                    relay_score = (
                        -relay_dirty_count,
                        distance_to_next_charger + relay_route_cost,
                        relay_target_distance,
                    )
                    if best_relay_score is None or relay_score < best_relay_score:
                        best_relay_score = relay_score

                if best_relay_score is not None:
                    reachable_dirty_count = -best_relay_score[0]
                    best_route_cost = best_relay_score[1]
                    best_target_distance = best_relay_score[2]
                    relay_hops = 1

            if reachable_dirty_count == 0:
                continue

            score = (
                -reachable_dirty_count,
                relay_hops,
                best_route_cost,
                distance_to_charger,
                best_target_distance,
                charger_idx,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_charger_pos = charger_pos

        return best_charger_pos

    def _clean_tile(self, row: int, col: int) -> None:
        pos = (row, col)
        if pos in self.dirty_index_map:
            idx = self.dirty_index_map[pos]
            self.cleaned_mask |= (1 << idx)

    def _count_cleaned_tiles(self) -> int:
        return self.cleaned_mask.bit_count()

    def _all_cleaned(self) -> bool:
        return self._count_cleaned_tiles() == self.total_dirty_tiles

    def _battery_depleted(self) -> bool:
        return self.battery_remaining is not None and self.battery_remaining <= 0

    def _get_battery_state_value(self) -> int:
        if self.battery_remaining is None:
            return -1
        return self.battery_remaining

    def _get_state(self) -> Tuple[int, int, int, int]:
        row, col = self.robot_pos
        return (
            row,
            col,
            self.cleaned_mask,
            self._get_battery_state_value(),
        )

    def _build_step_info(
        self,
        recharged: bool,
        termination_reason: str,
    ) -> Dict[str, float | str]:
        return {
            "steps_taken": self.steps_taken,
            "cleaned_tiles": self._count_cleaned_tiles(),
            "total_dirty_tiles": self.total_dirty_tiles,
            "cleaned_ratio": (
                self._count_cleaned_tiles() / self.total_dirty_tiles
                if self.total_dirty_tiles > 0
                else 1.0
            ),
            "battery_remaining": (
                self.battery_remaining if self.battery_remaining is not None else -1
            ),
            "battery_capacity": (
                self.battery_capacity if self.battery_capacity is not None else -1
            ),
            "battery_enabled": 1.0 if self.battery_capacity is not None else 0.0,
            "recharged": 1.0 if recharged else 0.0,
            "termination_reason": termination_reason,
        }

    def reset(self) -> Tuple[int, int, int, int]:
        self.robot_pos = self.start_pos
        self.cleaned_mask = self.initial_cleaned_mask
        self.steps_taken = 0
        self.battery_remaining = self.battery_capacity
        self.has_recharged_this_episode = False
        self.cleaned_tiles_at_last_recharge = self._count_cleaned_tiles()
        self.visited_positions = {self.start_pos}
        return self._get_state()

    def _step_core(self, action: int) -> tuple[Tuple[int, int, int, int], float, bool, bool, str]:
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        self.steps_taken += 1

        battery_before_action = self.battery_remaining

        if self.battery_remaining is not None:
            self.battery_remaining -= 1

        current_row, current_col = self.robot_pos
        dr, dc = self.ACTIONS[action]
        next_row = current_row + dr
        next_col = current_col + dc

        reward = 0.0
        recharged = False
        termination_reason = "ongoing"

        charger_distance_before = -1
        safe_dirty_distance_before = -1
        relay_charger_position: Tuple[int, int] | None = None
        relay_charger_distance_before = -1
        emergency_mode = False

        if self._should_consider_charger():
            safety_reserve = self._battery_safety_reserve()
            charger_distance_before = self._distance_to_nearest_charger(
                current_row,
                current_col,
            )

            if charger_distance_before != -1 and self.battery_remaining is not None:
                emergency_mode = (
                    self.battery_remaining <= charger_distance_before + safety_reserve
                )

                if not emergency_mode:
                    safe_dirty_distance_before = self._distance_to_nearest_safe_dirty(
                        current_row,
                        current_col,
                        self.battery_remaining,
                    )
                    if safe_dirty_distance_before == -1:
                        relay_charger_position = self._best_relay_charger_position(
                            current_row,
                            current_col,
                            self.battery_remaining,
                        )
                        if relay_charger_position is not None:
                            relay_charger_distance_before = self._distance_to_charger(
                                current_row,
                                current_col,
                                relay_charger_position,
                            )

        if self._is_wall(next_row, next_col):
            reward += self.reward_invalid
        else:
            self.robot_pos = (next_row, next_col)
            was_visited = (next_row, next_col) in self.visited_positions
            moved_onto_dirty = self._is_dirty(next_row, next_col)
            charger_distance_after = -1

            if moved_onto_dirty:
                self._clean_tile(next_row, next_col)
                reward += self.reward_clean

                if self._all_cleaned():
                    reward += self.reward_final_dirty_bonus
            else:
                if was_visited:
                    reward += self.reward_revisit
                else:
                    reward += self.reward_move

            if emergency_mode and charger_distance_before != -1:
                charger_distance_after = self._distance_to_nearest_charger(next_row, next_col)
                if charger_distance_after != -1:
                    if charger_distance_after < charger_distance_before:
                        reward += self.reward_move_toward_charger
                    elif charger_distance_after > charger_distance_before:
                        reward += self.penalty_move_away_from_charger

            if charger_distance_after == -1 and self._should_consider_charger():
                charger_distance_after = self._distance_to_nearest_charger(next_row, next_col)

            if (
                safe_dirty_distance_before != -1
                and not moved_onto_dirty
                and self.battery_remaining is not None
            ):
                safe_dirty_distance_after = self._distance_to_nearest_safe_dirty(
                    next_row,
                    next_col,
                    self.battery_remaining,
                )
                if safe_dirty_distance_after != -1:
                    if safe_dirty_distance_after < safe_dirty_distance_before:
                        reward += self.reward_move_toward_safe_dirty
                    elif safe_dirty_distance_after > safe_dirty_distance_before:
                        reward += self.penalty_move_away_from_safe_dirty

            if (
                relay_charger_position is not None
                and relay_charger_distance_before != -1
                and not moved_onto_dirty
            ):
                relay_charger_distance_after = self._distance_to_charger(
                    next_row,
                    next_col,
                    relay_charger_position,
                )
                if relay_charger_distance_after != -1:
                    if relay_charger_distance_after < relay_charger_distance_before:
                        reward += self.reward_move_toward_relay_charger
                    elif relay_charger_distance_after > relay_charger_distance_before:
                        reward += self.penalty_move_away_from_relay_charger

            if self._is_charger(next_row, next_col) and self.battery_capacity is not None:
                if self.battery_remaining < self.battery_capacity:
                    battery_before_recharge = self.battery_remaining
                    recovered_amount = self.battery_capacity - battery_before_recharge
                    cleaned_tiles_since_last_recharge = (
                        self._count_cleaned_tiles() - self.cleaned_tiles_at_last_recharge
                    )
                    low_battery_cutoff = max(
                        1,
                        int(self.battery_capacity * self.low_battery_ratio),
                    )

                    # Reward only meaningful low-battery recharge events.
                    if (
                        battery_before_recharge <= low_battery_cutoff
                        and recovered_amount >= 3
                    ):
                        reward += self.low_battery_recharge_reward

                        if not self.has_recharged_this_episode:
                            reward += self.first_recharge_reward
                    elif (
                        cleaned_tiles_since_last_recharge == 0
                        and not self._all_cleaned()
                    ):
                        reward += self.penalty_recharge_without_progress

                    self.battery_remaining = self.battery_capacity
                    self.has_recharged_this_episode = True
                    self.cleaned_tiles_at_last_recharge = self._count_cleaned_tiles()
                    recharged = True

            self.visited_positions.add((next_row, next_col))

            if (
                self._should_consider_charger()
                and not self._all_cleaned()
                and self.battery_remaining is not None
            ):
                if charger_distance_after == -1:
                    charger_distance_after = self._distance_to_nearest_charger(
                        next_row,
                        next_col,
                    )
                entered_unrecoverable_state = (
                    battery_before_action is not None
                    and charger_distance_before != -1
                    and charger_distance_after != -1
                    and battery_before_action
                    >= charger_distance_before + self._battery_safety_reserve()
                    and self.battery_remaining
                    < charger_distance_after + self._battery_safety_reserve()
                )
                if entered_unrecoverable_state:
                    reward += self.penalty_enter_unrecoverable_state

        done = False

        if self._all_cleaned():
            reward += self.reward_finish
            if self.has_recharged_this_episode:
                reward += self.successful_recharge_completion_bonus
            done = True
            termination_reason = "all_cleaned"
        elif self._battery_depleted():
            reward += self.penalty_battery_depleted
            done = True
            termination_reason = "battery_depleted"
        elif self.steps_taken >= self.max_steps:
            done = True
            termination_reason = "step_limit"

        return self._get_state(), reward, done, recharged, termination_reason

    def step(self, action: int) -> Tuple[Tuple[int, int, int, int], float, bool, Dict[str, float | str]]:
        next_state, reward, done, recharged, termination_reason = self._step_core(action)
        return (
            next_state,
            reward,
            done,
            self._build_step_info(
                recharged=recharged,
                termination_reason=termination_reason,
            ),
        )

    def step_training(self, action: int) -> tuple[Tuple[int, int, int, int], float, bool, str]:
        next_state, reward, done, _, termination_reason = self._step_core(action)
        return next_state, reward, done, termination_reason

    def get_episode_info(self, termination_reason: str) -> Dict[str, float | str]:
        return self._build_step_info(
            recharged=False,
            termination_reason=termination_reason,
        )

    def render(self) -> None:
        display_grid = [row[:] for row in self.grid]

        for (r, c), idx in self.dirty_index_map.items():
            if ((self.cleaned_mask >> idx) & 1) == 1:
                if self._is_charger(r, c):
                    display_grid[r][c] = "C"
                else:
                    display_grid[r][c] = "."

        robot_r, robot_c = self.robot_pos
        display_grid[robot_r][robot_c] = "R"

        print()
        for row in display_grid:
            print("".join(row))

        if self.battery_capacity is None:
            print(
                f"steps={self.steps_taken} | "
                f"cleaned={self._count_cleaned_tiles()}/{self.total_dirty_tiles}"
            )
        else:
            print(
                f"steps={self.steps_taken} | "
                f"cleaned={self._count_cleaned_tiles()}/{self.total_dirty_tiles} | "
                f"battery={self.battery_remaining}/{self.battery_capacity}"
            )

    def sample_action(self) -> int:
        return random.choice(list(self.ACTIONS.keys()))

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class StepResult:
    next_state: Tuple[int, int, int, int]
    reward: int
    done: bool
    info: Dict[str, float]


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
        reward_clean: int = 10,
        reward_move: int = -1,
        reward_revisit: int = -2,
        reward_invalid: int = -5,
        reward_finish: int = 50,
        battery_capacity: int | None = None,
    ) -> None:
        # Load a small default room if no custom map is provided.
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

        # Store the reward configuration.
        self.reward_clean = reward_clean
        self.reward_move = reward_move
        self.reward_revisit = reward_revisit
        self.reward_invalid = reward_invalid
        self.reward_finish = reward_finish

        # Optional battery constraint for harder environment settings.
        self.battery_capacity = battery_capacity
        self.battery_remaining = battery_capacity

        # Build the internal grid representation.
        self.grid: List[List[str]] = [list(row) for row in grid_map]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

        # Find and clear the start tile in the static map.
        self.start_pos = self._find_start_pos()
        start_row, start_col = self.start_pos
        self.grid[start_row][start_col] = "."

        # Index dirty tiles so their cleaned status can be tracked by bitmask.
        self.dirty_positions = self._find_dirty_positions()
        self.dirty_index_map = {
            pos: idx for idx, pos in enumerate(self.dirty_positions)
        }
        self.total_dirty_tiles = len(self.dirty_positions)

        # Initialize the dynamic episode state.
        self.robot_pos: Tuple[int, int] = self.start_pos
        self.cleaned_mask: int = 0
        self.steps_taken: int = 0

    def _find_start_pos(self) -> Tuple[int, int]:
        # Collect all start tiles and validate that there is exactly one.
        start_positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == "R":
                    start_positions.append((r, c))

        if len(start_positions) != 1:
            raise ValueError("grid_map must contain exactly one 'R' start position.")

        return start_positions[0]

    def _find_dirty_positions(self) -> List[Tuple[int, int]]:
        # Record every dirty tile in the map.
        dirty_positions = []
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == "D":
                    dirty_positions.append((r, c))
        return dirty_positions

    def _is_wall(self, row: int, col: int) -> bool:
        # Check whether the target tile is blocked.
        return self.grid[row][col] == "#"

    def _is_dirty(self, row: int, col: int) -> bool:
        # Return True only if the tile exists and has not been cleaned yet.
        pos = (row, col)
        if pos not in self.dirty_index_map:
            return False

        idx = self.dirty_index_map[pos]
        return ((self.cleaned_mask >> idx) & 1) == 0

    def _clean_tile(self, row: int, col: int) -> None:
        # Mark the tile as cleaned in the bitmask.
        pos = (row, col)
        if pos in self.dirty_index_map:
            idx = self.dirty_index_map[pos]
            self.cleaned_mask |= (1 << idx)

    def _count_cleaned_tiles(self) -> int:
        # Count how many dirty tiles have been cleaned so far.
        return self.cleaned_mask.bit_count()

    def _all_cleaned(self) -> bool:
        # Check whether the episode goal has been reached.
        return self._count_cleaned_tiles() == self.total_dirty_tiles

    def _battery_depleted(self) -> bool:
        # Return True only when battery mode is enabled and no charge remains.
        return self.battery_remaining is not None and self.battery_remaining <= 0

    def _get_battery_state_value(self) -> int:
        # Return a stable battery value for the agent state.
        if self.battery_remaining is None:
            return -1
        return self.battery_remaining

    def _get_state(self) -> Tuple[int, int, int, int]:
        # Return the current environment state including battery information.
        row, col = self.robot_pos
        return (
            row,
            col,
            self.cleaned_mask,
            self._get_battery_state_value(),
        )

    def reset(self) -> Tuple[int, int, int, int]:
        # Reset the robot position, cleaning progress, step count, and battery.
        self.robot_pos = self.start_pos
        self.cleaned_mask = 0
        self.steps_taken = 0
        self.battery_remaining = self.battery_capacity
        return self._get_state()

    def step(self, action: int) -> Tuple[Tuple[int, int, int, int], int, bool, Dict[str, float]]:
        # Reject invalid action ids early.
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        self.steps_taken += 1

        # Spend one unit of battery per action when battery mode is enabled.
        if self.battery_remaining is not None:
            self.battery_remaining -= 1

        current_row, current_col = self.robot_pos
        dr, dc = self.ACTIONS[action]
        next_row = current_row + dr
        next_col = current_col + dc

        reward = 0

        # Penalize invalid moves, otherwise update the robot state.
        if self._is_wall(next_row, next_col):
            reward += self.reward_invalid
        else:
            self.robot_pos = (next_row, next_col)

            # Reward new cleaning, penalize dirty-tile revisits, or apply move cost.
            if self._is_dirty(next_row, next_col):
                self._clean_tile(next_row, next_col)
                reward += self.reward_clean
            else:
                if (next_row, next_col) in self.dirty_index_map:
                    reward += self.reward_revisit
                else:
                    reward += self.reward_move

        done = False

        # End when all tiles are cleaned, battery runs out, or step limit is reached.
        if self._all_cleaned():
            reward += self.reward_finish
            done = True
        elif self._battery_depleted():
            done = True
        elif self.steps_taken >= self.max_steps:
            done = True

        info = {
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
        }

        return self._get_state(), reward, done, info

    def render(self) -> None:
        # Copy the static map so the current episode state can be overlaid.
        display_grid = [row[:] for row in self.grid]

        # Replace cleaned dirty tiles with normal floor tiles.
        for (r, c), idx in self.dirty_index_map.items():
            if ((self.cleaned_mask >> idx) & 1) == 1:
                display_grid[r][c] = "."

        # Draw the robot on top of the room.
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
        # Sample a random action from the environment action space.
        return random.choice(list(self.ACTIONS.keys()))
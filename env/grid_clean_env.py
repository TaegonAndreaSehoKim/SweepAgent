from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class StepResult:
    next_state: Tuple[int, int, int]
    reward: int
    done: bool
    info: Dict[str, float]


class GridCleanEnv:
    ACTIONS = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
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

        self.grid: List[List[str]] = [list(row) for row in grid_map]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

        self.start_pos = self._find_start_pos()
        start_row, start_col = self.start_pos
        self.grid[start_row][start_col] = "."

        self.dirty_positions = self._find_dirty_positions()
        self.dirty_index_map = {
            pos: idx for idx, pos in enumerate(self.dirty_positions)
        }
        self.total_dirty_tiles = len(self.dirty_positions)

        self.robot_pos: Tuple[int, int] = self.start_pos
        self.cleaned_mask: int = 0
        self.steps_taken: int = 0

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

    def _is_wall(self, row: int, col: int) -> bool:
        return self.grid[row][col] == "#"

    def _is_dirty(self, row: int, col: int) -> bool:
        pos = (row, col)
        if pos not in self.dirty_index_map:
            return False

        idx = self.dirty_index_map[pos]
        return ((self.cleaned_mask >> idx) & 1) == 0

    def _clean_tile(self, row: int, col: int) -> None:
        pos = (row, col)
        if pos in self.dirty_index_map:
            idx = self.dirty_index_map[pos]
            self.cleaned_mask |= (1 << idx)

    def _count_cleaned_tiles(self) -> int:
        return self.cleaned_mask.bit_count()

    def _all_cleaned(self) -> bool:
        return self._count_cleaned_tiles() == self.total_dirty_tiles

    def _get_state(self) -> Tuple[int, int, int]:
        row, col = self.robot_pos
        return (row, col, self.cleaned_mask)

    def reset(self) -> Tuple[int, int, int]:
        self.robot_pos = self.start_pos
        self.cleaned_mask = 0
        self.steps_taken = 0
        return self._get_state()

    def step(self, action: int) -> Tuple[Tuple[int, int, int], int, bool, Dict[str, float]]:
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        self.steps_taken += 1

        current_row, current_col = self.robot_pos
        dr, dc = self.ACTIONS[action]
        next_row = current_row + dr
        next_col = current_col + dc

        reward = 0

        if self._is_wall(next_row, next_col):
            reward += self.reward_invalid
        else:
            self.robot_pos = (next_row, next_col)

            if self._is_dirty(next_row, next_col):
                self._clean_tile(next_row, next_col)
                reward += self.reward_clean
            else:
                if (next_row, next_col) in self.dirty_index_map:
                    reward += self.reward_revisit
                else:
                    reward += self.reward_move

        done = False
        if self._all_cleaned():
            reward += self.reward_finish
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
        }

        return self._get_state(), reward, done, info

    def render(self) -> None:
        display_grid = [row[:] for row in self.grid]

        for (r, c), idx in self.dirty_index_map.items():
            if ((self.cleaned_mask >> idx) & 1) == 1:
                display_grid[r][c] = "."

        robot_r, robot_c = self.robot_pos
        display_grid[robot_r][robot_c] = "R"

        print()
        for row in display_grid:
            print("".join(row))
        print(
            f"steps={self.steps_taken} | "
            f"cleaned={self._count_cleaned_tiles()}/{self.total_dirty_tiles}"
        )

    def sample_action(self) -> int:
        import random
        return random.choice(list(self.ACTIONS.keys()))
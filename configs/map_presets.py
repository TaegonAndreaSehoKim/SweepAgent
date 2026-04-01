from __future__ import annotations

from collections import deque

# Shared reward settings.
REWARD_CLEAN = 10
REWARD_MOVE = -1
REWARD_REVISIT = -2
REWARD_INVALID = -5
REWARD_FINISH = 50

# Shared training settings.
TRAIN_EPISODES = 1000
TRAIN_SEED = 42
PRINT_EVERY = 100

# Q-learning hyperparameters.
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON_START = 1.0
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.05


def _find_symbol_positions(grid_map: list[str], symbol: str) -> list[tuple[int, int]]:
    """Find all coordinates for a given symbol in the map."""
    positions: list[tuple[int, int]] = []
    for row_idx, row in enumerate(grid_map):
        for col_idx, cell in enumerate(row):
            if cell == symbol:
                positions.append((row_idx, col_idx))
    return positions


def _single_source_shortest_paths(
    grid_map: list[str],
    start: tuple[int, int],
) -> list[list[int]]:
    """
    Compute shortest path distances from one source to every reachable cell.
    Walls are treated as blocked cells.
    """
    rows = len(grid_map)
    cols = len(grid_map[0])
    distance = [[-1 for _ in range(cols)] for _ in range(rows)]

    queue: deque[tuple[int, int]] = deque([start])
    distance[start[0]][start[1]] = 0

    while queue:
        row, col = queue.popleft()

        for d_row, d_col in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            next_row = row + d_row
            next_col = col + d_col

            if not (0 <= next_row < rows and 0 <= next_col < cols):
                continue
            if grid_map[next_row][next_col] == "#":
                continue
            if distance[next_row][next_col] != -1:
                continue

            distance[next_row][next_col] = distance[row][col] + 1
            queue.append((next_row, next_col))

    return distance


def _min_required_battery_via_state_search(grid_map: list[str]) -> int | None:
    """
    Compute the minimum battery capacity that still makes the map solvable.

    We compress the problem to special nodes:
    - start
    - dirty tiles
    - chargers

    Then we search over:
    - current special node
    - cleaned dirty mask
    - battery remaining

    Visiting a charger resets battery to full capacity.
    """
    start_positions = _find_symbol_positions(grid_map, "R")
    if len(start_positions) != 1:
        raise ValueError("grid_map must contain exactly one 'R' start position.")

    start_pos = start_positions[0]
    dirty_positions = _find_symbol_positions(grid_map, "D")
    charger_positions = _find_symbol_positions(grid_map, "C")

    if not dirty_positions:
        return None

    special_points = [start_pos, *dirty_positions, *charger_positions]
    dirty_count = len(dirty_positions)
    dirty_start_idx = 1
    charger_start_idx = 1 + dirty_count

    pairwise_distances = [
        [0 for _ in range(len(special_points))]
        for _ in range(len(special_points))
    ]

    for src_idx, src_point in enumerate(special_points):
        dist_map = _single_source_shortest_paths(grid_map, src_point)
        for dst_idx, dst_point in enumerate(special_points):
            path_len = dist_map[dst_point[0]][dst_point[1]]
            if path_len == -1:
                raise ValueError(
                    f"Map has unreachable special points: {src_point} -> {dst_point}"
                )
            pairwise_distances[src_idx][dst_idx] = path_len

    open_cells = sum(cell != "#" for row in grid_map for cell in row)

    def _can_solve_with_capacity(capacity: int) -> bool:
        full_mask = (1 << dirty_count) - 1
        start_state = (0, 0, capacity)  # (current_special_idx, cleaned_mask, battery_left)

        queue: deque[tuple[int, int, int]] = deque([start_state])
        visited = {start_state}

        while queue:
            current_idx, cleaned_mask, battery_left = queue.popleft()

            if cleaned_mask == full_mask:
                return True

            # Move to an uncleaned dirty tile.
            for dirty_offset in range(dirty_count):
                dirty_bit = 1 << dirty_offset
                if cleaned_mask & dirty_bit:
                    continue

                next_idx = dirty_start_idx + dirty_offset
                distance = pairwise_distances[current_idx][next_idx]

                if 0 < distance <= battery_left:
                    next_state = (
                        next_idx,
                        cleaned_mask | dirty_bit,
                        battery_left - distance,
                    )
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append(next_state)

            # Move to a charger and refill to full.
            for charger_offset in range(len(charger_positions)):
                next_idx = charger_start_idx + charger_offset
                distance = pairwise_distances[current_idx][next_idx]

                if 0 < distance <= battery_left:
                    next_state = (
                        next_idx,
                        cleaned_mask,
                        capacity,
                    )
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append(next_state)

        return False

    low = 1
    high = open_cells * 2
    answer: int | None = None

    while low <= high:
        mid = (low + high) // 2
        if _can_solve_with_capacity(mid):
            answer = mid
            high = mid - 1
        else:
            low = mid + 1

    return answer


def _battery_capacity_with_margin(grid_map: list[str], margin: int = 5) -> int | None:
    """Use the minimum required solvable battery and add a small design margin."""
    minimum_capacity = _min_required_battery_via_state_search(grid_map)
    if minimum_capacity is None:
        return None
    return minimum_capacity + margin


def _recommended_max_steps(grid_map: list[str], multiplier: int = 3) -> int:
    """Set a roomy step limit for larger maze-style maps."""
    open_cells = sum(cell != "#" for row in grid_map for cell in row)
    return open_cells * multiplier


# Shared map presets for benchmarking and visualization.
MAP_PRESETS = {
    "default": {
        "grid_map": [
            "#######",
            "#R..D.#",
            "#..#..#",
            "#.D.D.#",
            "#######",
        ],
        "max_steps": 100,
    },
    "harder": {
        "grid_map": [
            "#########",
            "#R..#..D#",
            "#.#.#.#.#",
            "#...D...#",
            "#.#...#.#",
            "#D..#..D#",
            "#########",
        ],
        "max_steps": 120,
    },
    "wide_room": {
        "grid_map": [
            "##########",
            "#R..D....#",
            "#..##..#.#",
            "#....D...#",
            "#.#..##..#",
            "#..D...D.#",
            "##########",
        ],
        "max_steps": 150,
    },
    "corridor": {
        "grid_map": [
            "###########",
            "#R..#....D#",
            "#.#.#.###.#",
            "#.#...#...#",
            "#.###.#.#.#",
            "#D....#..D#",
            "###########",
        ],
        "max_steps": 170,
    },
    "charge_required_v2": {
        "grid_map": [
            "###########",
            "#R..D.C...#",
            "#.###.###.#",
            "#.........#",
            "#.###.###.#",
            "#..D....D.#",
            "###########",
        ],
        "max_steps": _recommended_max_steps(
            [
                "###########",
                "#R..D.C...#",
                "#.###.###.#",
                "#.........#",
                "#.###.###.#",
                "#..D....D.#",
                "###########",
            ]
        ),
    },
    "charge_maze_medium": {
        "grid_map": [
            "###############",
            "#R..#...#....D#",
            "#.#.#.#.#.###.#",
            "#.#...#...#...#",
            "#.#####.###.#.#",
            "#...C...#...#.#",
            "###.#####.###.#",
            "#D..#.....#..D#",
            "###############",
        ],
        "max_steps": _recommended_max_steps(
            [
                "###############",
                "#R..#...#....D#",
                "#.#.#.#.#.###.#",
                "#.#...#...#...#",
                "#.#####.###.#.#",
                "#...C...#...#.#",
                "###.#####.###.#",
                "#D..#.....#..D#",
                "###############",
            ]
        ),
    },
    "charge_maze_large": {
        "grid_map": [
            "#####################",
            "#R..#.....#......D..#",
            "#.#.#.###.#.#######.#",
            "#.#...#...#.....#...#",
            "#.#####.#######.#.#.#",
            "#.....#...C...#.#.#.#",
            "###.#.###.###.#.#.#.#",
            "#...#...#...#.#...#.#",
            "#.#####.#.#.#.###.#.#",
            "#.#...#.#.#.#...#.#.#",
            "#.#.#.#.#.#.###.#.#.#",
            "#D..#...#...#...#..D#",
            "#.#########.#.#####.#",
            "#.....C.....#...D...#",
            "#####################",
        ],
        "max_steps": _recommended_max_steps(
            [
                "#####################",
                "#R..#.....#......D..#",
                "#.#.#.###.#.#######.#",
                "#.#...#...#.....#...#",
                "#.#####.#######.#.#.#",
                "#.....#...C...#.#.#.#",
                "###.#.###.###.#.#.#.#",
                "#...#...#...#.#...#.#",
                "#.#####.#.#.#.###.#.#",
                "#.#...#.#.#.#...#.#.#",
                "#.#.#.#.#.#.###.#.#.#",
                "#D..#...#...#...#..D#",
                "#.#########.#.#####.#",
                "#.....C.....#...D...#",
                "#####################",
            ]
        ),
    },
    "multi_charge_detour": {
        "grid_map": [
            "###################",
            "#R..#....C....#..D#",
            "#.#.#.#######.#.#.#",
            "#.#...#.....#...#.#",
            "#.#####.###.#####.#",
            "#.....#...#.....#.#",
            "###.#.###.###.#.#.#",
            "#D..#...C...#.#..D#",
            "###################",
        ],
        "max_steps": _recommended_max_steps(
            [
                "###################",
                "#R..#....C....#..D#",
                "#.#.#.#######.#.#.#",
                "#.#...#.....#...#.#",
                "#.#####.###.#####.#",
                "#.....#...#.....#.#",
                "###.#.###.###.#.#.#",
                "#D..#...C...#.#..D#",
                "###################",
            ]
        ),
    },
    "complex_charge_labyrinth": {
        "grid_map": [
            "#####################",
            "#R..#.....#......D..#",
            "#.#.#.###.#.#######.#",
            "#.#...#...#.....#...#",
            "#.#####.#######.#.#.#",
            "#.....#...C...#.#.#.#",
            "###.#.###.###.#.#.#.#",
            "#...#...#...#.#...#.#",
            "#.#####.#.#.#.###.#.#",
            "#.#...#.#.#.#...#.#.#",
            "#.#.#.#.#.#.###.#.#.#",
            "#D..#...#...#...#..D#",
            "#.#########.#.#####.#",
            "#.....C.....#...D...#",
            "#####################",
        ],
        "max_steps": _recommended_max_steps(
            [
                "#####################",
                "#R..#.....#......D..#",
                "#.#.#.###.#.#######.#",
                "#.#...#...#.....#...#",
                "#.#####.#######.#.#.#",
                "#.....#...C...#.#.#.#",
                "###.#.###.###.#.#.#.#",
                "#...#...#...#.#...#.#",
                "#.#####.#.#.#.###.#.#",
                "#.#...#.#.#.#...#.#.#",
                "#.#.#.#.#.#.###.#.#.#",
                "#D..#...#...#...#..D#",
                "#.#########.#.#####.#",
                "#.....C.....#...D...#",
                "#####################",
            ]
        ),
    },
    "complex_charge_bastion": {
        "grid_map": [
            "#########################",
            "#R..#........#......D...#",
            "#.#.#.######.#.#######.##",
            "#.#...#....#.#.....#....#",
            "#.#####.##.#.#####.#.##.#",
            "#.....#.#..#...C...#.#..#",
            "#####.#.#.#######.##.#.##",
            "#D..#.#.#.....#...#.....#",
            "#.#.#.#.#####.#.###.###.#",
            "#.#.#.#.....#.#...#...#.#",
            "#.#.#.#####.#.###.###.#.#",
            "#...#.....#.#...#.....#.#",
            "###.#####.#.###.#####.#.#",
            "#...#...#.#...#.#...#.#.#",
            "#.###.#.#.###.#.#.#.#.#.#",
            "#.....#...C...#...#..D..#",
            "#########################",
        ],
        "max_steps": _recommended_max_steps(
            [
                "#########################",
                "#R..#........#......D...#",
                "#.#.#.######.#.#######.##",
                "#.#...#....#.#.....#....#",
                "#.#####.##.#.#####.#.##.#",
                "#.....#.#..#...C...#.#..#",
                "#####.#.#.#######.##.#.##",
                "#D..#.#.#.....#...#.....#",
                "#.#.#.#.#####.#.###.###.#",
                "#.#.#.#.....#.#...#...#.#",
                "#.#.#.#####.#.###.###.#.#",
                "#...#.....#.#...#.....#.#",
                "###.#####.#.###.#####.#.#",
                "#...#...#.#...#.#...#.#.#",
                "#.###.#.#.###.#.#.#.#.#.#",
                "#.....#...C...#...#..D..#",
                "#########################",
            ]
        ),
    },
    "complex_charge_switchback": {
        "grid_map": [
            "#######################",
            "#R..#.....#.....#...D.#",
            "#.#.#.###.#.###.#.#.#.#",
            "#.#...#...#...#...#.#.#",
            "#.#####.#####.#####.#.#",
            "#.....#...C...#.....#.#",
            "###.#.###.###.#.#####.#",
            "#...#...#...#.#.#.....#",
            "#.#####.#.#.#.#.#.###.#",
            "#.#.....#.#.#...#.#...#",
            "#.#.#####.#.#####.#.#.#",
            "#D#.....#.#.....#.#.#.#",
            "#.#####.#.#####.#.#.#.#",
            "#...C...#.....#...#..D#",
            "#######################",
        ],
        "max_steps": _recommended_max_steps(
            [
                "#######################",
                "#R..#.....#.....#...D.#",
                "#.#.#.###.#.###.#.#.#.#",
                "#.#...#...#...#...#.#.#",
                "#.#####.#####.#####.#.#",
                "#.....#...C...#.....#.#",
                "###.#.###.###.#.#####.#",
                "#...#...#...#.#.#.....#",
                "#.#####.#.#.#.#.#.###.#",
                "#.#.....#.#.#...#.#...#",
                "#.#.#####.#.#####.#.#.#",
                "#D#.....#.#.....#.#.#.#",
                "#.#####.#.#####.#.#.#.#",
                "#...C...#.....#...#..D#",
                "#######################",
            ]
        ),
    },
}


for preset in MAP_PRESETS.values():
    preset["battery_capacity"] = _battery_capacity_with_margin(preset["grid_map"], margin=5)
from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import Iterable


ACTION_DELTAS = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}


@dataclass(frozen=True)
class CleaningPlan:
    solvable: bool
    total_steps: int
    special_route: list[tuple[int, int]]
    actions: list[int]
    reason: str = ""


def find_symbol_positions(grid_map: list[str], symbol: str) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for row_idx, row in enumerate(grid_map):
        for col_idx, cell in enumerate(row):
            if cell == symbol:
                positions.append((row_idx, col_idx))
    return positions


def _neighbors(grid_map: list[str], row: int, col: int) -> Iterable[tuple[int, int, int]]:
    rows = len(grid_map)
    cols = len(grid_map[0])
    for action, (delta_row, delta_col) in ACTION_DELTAS.items():
        next_row = row + delta_row
        next_col = col + delta_col
        if not (0 <= next_row < rows and 0 <= next_col < cols):
            continue
        if grid_map[next_row][next_col] == "#":
            continue
        yield action, next_row, next_col


def shortest_path_actions(
    grid_map: list[str],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[int] | None:
    if start == goal:
        return []

    queue: deque[tuple[int, int]] = deque([start])
    previous: dict[tuple[int, int], tuple[tuple[int, int], int]] = {}
    visited = {start}

    while queue:
        row, col = queue.popleft()
        for action, next_row, next_col in _neighbors(grid_map, row, col):
            next_pos = (next_row, next_col)
            if next_pos in visited:
                continue
            visited.add(next_pos)
            previous[next_pos] = ((row, col), action)
            if next_pos == goal:
                actions: list[int] = []
                cursor = goal
                while cursor != start:
                    prev_pos, prev_action = previous[cursor]
                    actions.append(prev_action)
                    cursor = prev_pos
                actions.reverse()
                return actions
            queue.append(next_pos)

    return None


def _single_source_distances(
    grid_map: list[str],
    start: tuple[int, int],
) -> dict[tuple[int, int], int]:
    distances = {start: 0}
    queue: deque[tuple[int, int]] = deque([start])

    while queue:
        row, col = queue.popleft()
        for _, next_row, next_col in _neighbors(grid_map, row, col):
            next_pos = (next_row, next_col)
            if next_pos in distances:
                continue
            distances[next_pos] = distances[(row, col)] + 1
            queue.append(next_pos)

    return distances


def _expand_special_route(
    grid_map: list[str],
    special_route: list[tuple[int, int]],
) -> list[int] | None:
    actions: list[int] = []
    for start, goal in zip(special_route, special_route[1:]):
        segment_actions = shortest_path_actions(grid_map, start, goal)
        if segment_actions is None:
            return None
        actions.extend(segment_actions)
    return actions


def compute_shortest_cleaning_plan(
    grid_map: list[str],
    battery_capacity: int | None,
    initial_cleaned_positions: list[tuple[int, int]] | None = None,
) -> CleaningPlan:
    start_positions = find_symbol_positions(grid_map, "R")
    if len(start_positions) != 1:
        return CleaningPlan(
            solvable=False,
            total_steps=0,
            special_route=[],
            actions=[],
            reason="grid_map must contain exactly one start position",
        )

    start_pos = start_positions[0]
    dirty_positions = find_symbol_positions(grid_map, "D")
    charger_positions = find_symbol_positions(grid_map, "C")
    dirty_index = {position: idx for idx, position in enumerate(dirty_positions)}
    full_mask = (1 << len(dirty_positions)) - 1

    initial_cleaned_mask = 0
    for position in initial_cleaned_positions or []:
        if position in dirty_index:
            initial_cleaned_mask |= 1 << dirty_index[position]

    if initial_cleaned_mask == full_mask:
        return CleaningPlan(
            solvable=True,
            total_steps=0,
            special_route=[start_pos],
            actions=[],
        )

    special_points = [start_pos, *dirty_positions, *charger_positions]
    distance_maps = {
        point: _single_source_distances(grid_map=grid_map, start=point)
        for point in special_points
    }

    for src in special_points:
        for dst in special_points:
            if dst not in distance_maps[src]:
                return CleaningPlan(
                    solvable=False,
                    total_steps=0,
                    special_route=[],
                    actions=[],
                    reason=f"unreachable special point: {src} -> {dst}",
                )

    effective_capacity = battery_capacity
    if effective_capacity is None:
        effective_capacity = sum(cell != "#" for row in grid_map for cell in row) * 4

    start_state = (start_pos, initial_cleaned_mask, effective_capacity)
    queue: list[tuple[int, int, tuple[tuple[int, int], int, int]]] = []
    heapq.heappush(queue, (0, 0, start_state))
    best_cost = {start_state: 0}
    previous: dict[
        tuple[tuple[int, int], int, int],
        tuple[tuple[tuple[int, int], int, int], tuple[int, int]],
    ] = {}
    sequence_id = 0
    goal_state: tuple[tuple[int, int], int, int] | None = None

    while queue:
        cost, _, state = heapq.heappop(queue)
        current_pos, cleaned_mask, battery_remaining = state
        if cost != best_cost[state]:
            continue
        if cleaned_mask == full_mask:
            goal_state = state
            break

        candidates: list[tuple[tuple[int, int], int]] = []
        for dirty_pos in dirty_positions:
            dirty_bit = 1 << dirty_index[dirty_pos]
            if cleaned_mask & dirty_bit:
                continue
            candidates.append((dirty_pos, cleaned_mask | dirty_bit))
        for charger_pos in charger_positions:
            candidates.append((charger_pos, cleaned_mask))

        for next_pos, next_cleaned_mask in candidates:
            distance = distance_maps[current_pos][next_pos]
            if distance <= 0 or distance > battery_remaining:
                continue

            next_battery = battery_remaining - distance
            if next_pos in charger_positions:
                next_battery = effective_capacity

            next_state = (next_pos, next_cleaned_mask, next_battery)
            next_cost = cost + distance
            if next_cost >= best_cost.get(next_state, 10**12):
                continue

            sequence_id += 1
            best_cost[next_state] = next_cost
            previous[next_state] = (state, next_pos)
            heapq.heappush(queue, (next_cost, sequence_id, next_state))

    if goal_state is None:
        return CleaningPlan(
            solvable=False,
            total_steps=0,
            special_route=[],
            actions=[],
            reason="no feasible dirty/charger route within battery capacity",
        )

    route_reversed = [goal_state[0]]
    cursor = goal_state
    while cursor != start_state:
        prev_state, reached_pos = previous[cursor]
        route_reversed.append(prev_state[0])
        if reached_pos != cursor[0]:
            raise RuntimeError("invalid planner predecessor chain")
        cursor = prev_state
    special_route = list(reversed(route_reversed))
    actions = _expand_special_route(grid_map=grid_map, special_route=special_route)
    if actions is None:
        return CleaningPlan(
            solvable=False,
            total_steps=0,
            special_route=special_route,
            actions=[],
            reason="failed to expand special route to primitive actions",
        )

    return CleaningPlan(
        solvable=True,
        total_steps=len(actions),
        special_route=special_route,
        actions=actions,
    )

from __future__ import annotations

# Shared map presets for benchmarking, training, and UI visualization.
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

    # Small battery-awareness demo map.
    "charging_easy": {
        "grid_map": [
            "#########",
            "#R.D.C.D#",
            "#.......#",
            "#########",
        ],
        "max_steps": 80,
        "battery_capacity": 10,
    },

    # Small map used for short charging visualization demos.
    "charging_demo": {
        "grid_map": [
            "#########",
            "#R.D.C..#",
            "#.....D.#",
            "#########",
        ],
        "max_steps": 90,
        "battery_capacity": 10,
    },

    # Current main charge-required map from the project.
    "charge_required_v2": {
        "grid_map": [
            "###########",
            "#R.D.C...D#",
            "#.###.###.#",
            "#.........#",
            "#.###.###.#",
            "#..D......#",
            "###########",
        ],
        "max_steps": 120,
        "battery_capacity": 10,
    },

    # New medium-sized maze map:
    # - multiple dirty clusters
    # - one charger in the middle corridor
    # - requires navigating around walls instead of just straight sweeping
    "charge_maze_medium": {
        "grid_map": [
            "#############",
            "#R..#...D...#",
            "#.#.#.###.#.#",
            "#.#...C...#.#",
            "#.###.#.###.#",
            "#...D.#...D.#",
            "#.#.###.#.#.#",
            "#...#.....#.#",
            "#############",
        ],
        "max_steps": 180,
        "battery_capacity": 12,
    },

    # New larger map:
    # - longer travel distances
    # - more detours
    # - central charger that is useful but not always on the shortest path
    # - multiple distant dirty tiles
    "charge_maze_large": {
        "grid_map": [
            "#################",
            "#R....#....D....#",
            "#.###.#.#####.#.#",
            "#...#.#.....#.#.#",
            "###.#.###.#.#.#.#",
            "#...#...C.#...#.#",
            "#.#####.#.#####.#",
            "#.....#.#.....#.#",
            "#.###.#.###.#.#.#",
            "#D..#.....#...D.#",
            "#################",
        ],
        "max_steps": 260,
        "battery_capacity": 40,
    },

    # New multi-charge detour map:
    # - two chargers
    # - several dirty tiles in different regions
    # - encourages future algorithm comparison on charger choice behavior
    "multi_charge_detour": {
        "grid_map": [
            "#################",
            "#R..D....#....C.#",
            "#.###.##.#.##.#.#",
            "#...#....#....#.#",
            "###.#.######.#.#.",
            "#...#..D..C..#..#",
            "#.#####.#######.#",
            "#.....#.....#...#",
            "#.###.#.###.#.#.#",
            "#D....#...D...#.#",
            "#################",
        ],
        "max_steps": 280,
        "battery_capacity": 40,
    },
}

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
from __future__ import annotations

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
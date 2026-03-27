from __future__ import annotations

# Default map used in the early experiments.
DEFAULT_GRID_MAP = [
    "#######",
    "#R..D.#",
    "#..#..#",
    "#.D.D.#",
    "#######",
]

# A slightly harder map for later comparison.
HARDER_GRID_MAP = [
    "#########",
    "#R..#..D#",
    "#.#.#.#.#",
    "#...D...#",
    "#.#...#.#",
    "#D..#..D#",
    "#########",
]

# Shared environment settings.
DEFAULT_MAX_STEPS = 100
HARDER_MAX_STEPS = 120

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
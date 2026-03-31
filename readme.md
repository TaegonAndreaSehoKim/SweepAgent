# SweepAgent

SweepAgent is a reinforcement learning project for a grid-based cleaning robot.
The agent learns how to clean dirty tiles efficiently while handling walls, limited step budgets, and charge-aware navigation on larger maps.

## Current Project Status

SweepAgent now includes:

- a custom `GridCleanEnv` environment with:
  - walls
  - dirty tiles
  - optional battery constraints
  - optional charging stations
- a random baseline agent
- a tabular Q-learning agent
- reusable training and checkpoint loading utilities
- map presets ranging from small rooms to larger charge-aware maze layouts
- comparison scripts for random vs learned policy evaluation
- GIF and policy rendering utilities
- a pygame-based training and playback UI
- episode-aware checkpoint naming
- map-specific hyperparameter tuning for harder charge-aware maps

Recent development focused on making the project more demo-friendly and more stable on larger maps.

## Key Features

### 1. Custom Environment
The environment supports:

- wall collisions
- dirty tile cleaning
- revisit penalties
- step limits
- optional battery capacity
- optional charger tiles
- charge-aware termination and reward handling

### 2. Agents
Implemented agents:

- `RandomAgent`
- `QLearningAgent`

The learned policy can be trained once and reused through JSON checkpoints.

### 3. Shared Experiment Utilities
The project uses shared utilities so that training, comparison, rendering, and UI playback all follow the same rules for:

- map preset loading
- reward configuration
- checkpoint naming
- checkpoint reuse

### 4. Visualization and Demo Tools
SweepAgent includes:

- learned policy GIF rendering
- side-by-side comparison GIF rendering
- benchmark plotting
- an interactive pygame training app

The pygame UI now supports:

- map selection
- algorithm selection
- result view selection
- episode count selection
- train seed selection
- playback seed selection
- Q-learning hyperparameter selection from the menu

### 5. Charge-Aware Training
The project now includes charge-aware maps such as:

- `charge_required_v2`
- `charge_maze_medium`
- `charge_maze_large`
- `multi_charge_detour`

These maps make the environment meaningfully harder than the original small-room tasks.

## Recent Improvements

### Charge-aware environment and map expansion
The project moved beyond simple room layouts and added larger maps that require longer-horizon planning under battery constraints.

### UI refactor
The training app was split into more maintainable modules and updated so training control and playback control are easier to extend.

### Episode-aware checkpoints
Checkpoint file names now include the episode count so that different training runs do not silently reuse the wrong model.

Example format:

`q_learning_agent_<map_name>_ep_<episodes>_seed_<seed>.json`

### Map-specific hyperparameter tuning
A single fixed hyperparameter set was not reliable across all maps.
The current direction is to use easier settings for smaller maps and slower exploration decay for larger charge-aware maps.

A strong combination for the larger charge-aware map was:

- `learning_rate = 0.05`
- `discount_factor = 0.99`
- `epsilon_decay = 0.999`
- `epsilon_min = 0.10`

This combination gave more stable learning behavior than the earlier default settings.

## Suggested Hyperparameter Presets

### Small / medium charge-aware maps
Recommended starting point:

- `learning_rate = 0.05`
- `discount_factor = 0.99`
- `epsilon_decay = 0.997`
- `epsilon_min = 0.05`

### Large charge-aware maps
Recommended starting point:

- `learning_rate = 0.05`
- `discount_factor = 0.99`
- `epsilon_decay = 0.999`
- `epsilon_min = 0.10`

## Project Structure

```text
SweepAgent/
├── README.md
├── requirements.txt
├── agents/
│   ├── q_learning_agent.py
│   └── random_agent.py
├── configs/
│   └── map_presets.py
├── env/
│   └── grid_clean_env.py
├── scripts/
│   ├── benchmark_maps.py
│   ├── compare_agents.py
│   ├── render_comparison_gif.py
│   ├── render_policy_gif.py
│   ├── run_training_app.py
│   └── train_q_learning.py
├── ui/
│   ├── training_app_core.py
│   ├── training_app_handlers.py
│   ├── training_app_state.py
│   └── training_app_views.py
├── utils/
│   ├── experiment_utils.py
│   └── ui_utils.py
├── outputs/
│   ├── checkpoints/
│   ├── gifs/
│   ├── logs/
│   └── plots/
└── docs/
    └── devlog/
        └── week1.md
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train a Q-learning agent

```bash
python scripts/train_q_learning.py --map-name charge_required_v2 --episodes 5000 --seed 42
```

### 3. Compare random vs learned policy

```bash
python scripts/compare_agents.py --map-name charge_required_v2 --episodes 5000 --seed 42 --eval-episodes 100
```

### 4. Launch the training UI

```bash
python scripts/run_training_app.py
```

## Current Development Notes

At this stage, the project is no longer just a single-map RL demo.
It now supports:

- multi-map experimentation
- charge-aware planning
- reusable checkpoints
- a modular training UI
- harder large-map tuning workflows

The next major step is planned to be curriculum learning, but that work has not been started yet.

## Devlog

Weekly development notes are recorded in:

`docs/devlog/week1.md`

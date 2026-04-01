# SweepAgent

SweepAgent is a reinforcement learning project for a grid-based cleaning robot.
The agent learns cleaning policies under walls, dirty tiles, step limits, and battery-constrained charge-aware navigation.

## Current Project Status

SweepAgent now includes:

- a custom `GridCleanEnv` environment with:
  - walls
  - dirty tiles
  - one robot start position
  - step limits
  - optional battery constraints
  - optional charger tiles
- a random baseline agent
- a tabular Q-learning agent
- JSON checkpoint save/load support
- shared experiment utilities for map construction and checkpoint reuse
- multiple benchmark maps
- larger charge-aware maze maps
- curriculum training support
- a pygame-based training and playback UI

The project has moved beyond the original small-room example and now supports harder charge-aware maps, map-specific checkpoint reuse, UI-driven experiments, and curriculum-based training flows.

## Main Features

### 1. Environment

The environment supports:

- wall collisions
- dirty tile cleaning
- revisit penalties
- step limits
- optional battery capacity
- optional charging stations

The base map symbols are:

- `#` wall
- `.` floor
- `D` dirty tile
- `R` robot start
- `C` charging station

### 2. Agents

Implemented agents:

- `RandomAgent`
- `QLearningAgent`

The learned policy is stored as a JSON checkpoint and can be reused across training, evaluation, comparison, GIF rendering, and UI playback.

### 3. Shared Experiment Utilities

Common utilities are centralized so multiple scripts follow the same rules for:

- map preset loading
- reward settings
- checkpoint path generation
- checkpoint reuse

### 4. Training UI

The pygame UI supports:

- map selection
- algorithm selection
- result view selection
- episode selection
- train seed selection
- playback seed selection
- Q-learning hyperparameter selection
- curriculum-learning selection from the algorithm menu

Recent UI changes:

- dropdown layout fixes so controls fit within the window
- safer dropdown click handling to avoid lower controls stealing clicks
- removal of the mini rollout preview from the training screen to reduce unused space
- cleaner training layout with metrics, logs, graphs, and cancel action only

### 5. Curriculum Learning

The UI now supports a curriculum-learning path through the algorithm selector.

Current curriculum behavior:

- stage 1 uses `charge_maze_medium`
- stage 2 uses the map selected in the UI

This allows the same curriculum entry point to be reused for different harder stage-2 maps rather than being locked to one fixed final map.

## Maps

SweepAgent now includes:

### Earlier maps

- `default`
- `harder`
- `wide_room`
- `corridor`
- `charge_required_v2`
- `charge_maze_medium`
- `charge_maze_large`
- `multi_charge_detour`

### New complex charge-aware maps

- `complex_charge_labyrinth`
- `complex_charge_bastion`
- `complex_charge_switchback`

These newer maps are intended for longer-horizon navigation and harder battery-aware planning.

## Battery Capacity Design

Battery capacity is now set using the same rule across maps.

For each preset:

- compute the minimum solvable battery requirement using shortest-path search over the important locations
- add a margin of `+5`

This makes battery sizing more systematic than manual tuning and keeps the maps challenging while still solvable.

## Checkpoints

Checkpoint naming is episode-aware.

Format:

`q_learning_agent_<map_name>_ep_<episodes>_seed_<seed>.json`

Examples from the recent work include:

- `q_learning_agent_charge_maze_medium_ep_50000_seed_42.json`
- `q_learning_agent_complex_charge_labyrinth_ep_50000_seed_42.json`
- `q_learning_agent_complex_charge_bastion_ep_50000_seed_42.json`
- `q_learning_agent_complex_charge_switchback_ep_50000_seed_42.json`

## Recent Training Notes

Observed from the latest uploaded checkpoints:

- `charge_maze_medium` at 50k episodes appears substantially more stable than the new complex maps
- the three new complex maps still look harder and likely need additional curriculum tuning, more training, or further reward/structure refinement

That matches the current project direction: the curriculum setup is in place, but the hardest maps are still active tuning targets.

## Project Structure

```text
SweepAgent/
├── README.md
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
│   ├── evaluate_agents.py
│   ├── render_comparison_gif.py
│   ├── render_policy_gif.py
│   ├── run_training_app.py
│   ├── train_q_curriculum.py
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
        ├── week1.md
        └── week2.md
```

## How to Run

### Train Q-learning on a selected map

```bash
python scripts/train_q_learning.py --map-name charge_required_v2 --episodes 5000 --seed 42
```

### Run comparison

```bash
python scripts/compare_agents.py --map-name charge_required_v2 --episodes 5000 --seed 42 --eval-episodes 50
```

### Launch the training UI

```bash
python scripts/run_training_app.py
```

### Curriculum training

```bash
python scripts/train_q_curriculum.py --stage1-map charge_maze_medium --stage2-map complex_charge_labyrinth --seed 42
```

## Current Direction

The current focus is no longer just “can Q-learning solve a small room.”
It is now:

- battery-aware planning on larger maps
- curriculum learning for harder charge-aware layouts
- UI-driven experimentation
- systematic map design and checkpoint reuse

The next likely improvement areas are:

- stronger curriculum schedules for the hardest maps
- more stable training on the three newest complex maps
- better evaluation summaries in docs and plots

# SweepAgent

SweepAgent is a reinforcement learning project where a grid-based vacuum agent learns efficient cleaning policies in a room with walls, dirty tiles, battery limits, charging stations, and an interactive program-based UI.

## Project Goal

The goal of SweepAgent is to train a self-learning agent that can clean a grid-based room more efficiently than a random policy, and to present the learned behavior through reproducible evaluation, visualization, and an interactive UI.

The project started with a simple environment and baseline agent, and now includes:
- Q-learning
- GIF-based policy visualization
- multi-map benchmarking
- reusable experiment checkpoints
- battery-constrained environments
- charging-station-aware behavior
- an interactive pygame-based training and playback UI

## Current Status

Completed:
- project structure setup
- grid cleaning environment
- reward and terminal logic
- text-based environment rendering
- random baseline agent
- Q-learning agent
- training loop with epsilon-greedy exploration
- training plots for reward, cleaned ratio, success trend, and epsilon decay
- random vs learned policy comparison
- GIF rendering for learned policy playback
- side-by-side comparison GIF rendering
- shared map presets
- multi-map benchmarking across several layouts
- shared experiment utilities for environment creation and checkpoint reuse
- map-specific checkpoint saving and loading
- battery-constrained environment variants
- battery-aware Q-learning state
- charging station tiles and recharge logic
- charger-aware reward shaping
- charger behavior learning on `charging_demo`
- charge-required behavior validation on `charge_required_v2`
- multi-seed validation for charger-aware behavior
- first program-based UI prototype using pygame
- menu-based map / model / playback selection
- training screen with live logs and mini rollout preview
- single playback and compare playback UI modes
- UI module split into core / state / handlers / views

Next ideas:
- stabilize the refactored training app further
- improve compare playback summary UX
- add richer training trend charts
- add dynamic obstacles
- try larger and more difficult room layouts
- test additional RL methods beyond tabular Q-learning

## Environment Overview

The environment is a grid-based room made of:
- walls (`#`)
- empty tiles (`.`)
- dirty tiles (`D`)
- charging stations (`C`)
- one robot start position (`R`)

Example map:

```text
#######
#R..D.#
#..#..#
#.D.D.#
#######
```

The agent can take four actions:
- up
- down
- left
- right

Some maps also include a battery budget. In those maps, every action consumes one unit of battery, and stepping onto a charging station (`C`) restores the battery to full.

## Reward Design

The current reward setting is:

- clean a new dirty tile: `+10`
- normal move: `-1`
- revisit an already cleaned dirty tile: `-2`
- invalid move / wall collision: `-5`
- clean all dirty tiles: `+50`

For charging maps, the project also uses charger-aware shaping:

- first recharge in an episode: `+1`
- successful completion after using recharge: `+10`
- when the battery is low, a small shaping signal encourages moves toward the nearest charger and discourages moves away from it

An episode ends when:
- all dirty tiles are cleaned, or
- the battery is depleted, or
- the maximum number of steps is reached

## Rendering and Visualization

The project now supports text rendering, GIF rendering, and an interactive pygame-based UI.

Available visual outputs include:
- step-by-step terminal rendering for debugging
- learned greedy policy GIF playback
- side-by-side comparison GIFs between random and learned agents
- interactive menu / training / playback screens through `run_training_app.py`

Generated GIF outputs include:
- `outputs/gifs/learned_policy_harder.gif`
- `outputs/gifs/comparison_harder.gif`
- `outputs/gifs/learned_policy_charging_demo.gif`
- `outputs/gifs/comparison_charging_demo.gif`
- `outputs/gifs/learned_policy_charge_required_v2.gif`
- `outputs/gifs/comparison_charge_required_v2.gif`

## Project Structure

```bash
SweepAgent/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── default_config.py
│   └── map_presets.py
├── env/
│   └── grid_clean_env.py
├── agents/
│   ├── random_agent.py
│   └── q_learning_agent.py
├── utils/
│   ├── experiment_utils.py
│   └── ui_utils.py
├── ui/
│   ├── training_app_core.py
│   ├── training_app_state.py
│   ├── training_app_handlers.py
│   └── training_app_views.py
├── scripts/
│   ├── evaluate_agents.py
│   ├── train_q_learning.py
│   ├── compare_agents.py
│   ├── render_policy_gif.py
│   ├── render_comparison_gif.py
│   ├── benchmark_maps.py
│   └── run_training_app.py
├── outputs/
│   ├── checkpoints/
│   ├── plots/
│   ├── gifs/
│   └── logs/
└── docs/
    └── devlog/
        └── week1.md
```

## Current Experiment Workflow

The project uses shared experiment utilities and map-specific checkpoints so that
training results can be reused across evaluation and visualization scripts.

### Shared utilities

Common environment creation and checkpoint handling are centralized in:

- `utils/experiment_utils.py`

This utility module provides:
- `build_env(map_name)`
- `get_checkpoint_path(map_name, seed)`
- `train_q_learning_agent(...)`
- `load_or_train_q_agent(...)`

### Checkpoint naming

Trained Q-learning agents are saved per map:

- `outputs/checkpoints/q_learning_agent_default_seed_42.json`
- `outputs/checkpoints/q_learning_agent_harder_seed_42.json`
- `outputs/checkpoints/q_learning_agent_wide_room_seed_42.json`
- `outputs/checkpoints/q_learning_agent_corridor_seed_42.json`
- `outputs/checkpoints/q_learning_agent_battery_harder_seed_42.json`
- `outputs/checkpoints/q_learning_agent_charging_demo_seed_42.json`
- `outputs/checkpoints/q_learning_agent_charge_required_v2_seed_42.json`

This prevents policies trained on different maps from being mixed together.

## Training App UI Architecture

The interactive training app was refactored into smaller modules for maintainability.

### Module layout
- `scripts/run_training_app.py`
  - entry point
  - pygame loop
  - screen switching
  - high-level event routing

- `ui/training_app_state.py`
  - dataclasses for menu, training, mini-preview, single playback, and compare playback state

- `ui/training_app_handlers.py`
  - training start/cancel handlers
  - subprocess output parsing
  - mini-preview stepping
  - single/compare playback stepping

- `ui/training_app_core.py`
  - shared constants
  - `TrainingRunner`
  - `PreviewPolicy`
  - helper utilities

- `ui/training_app_views.py`
  - rendering logic for menu, training panels, and playback screens

### Why this refactor helps
- reduces the size of `run_training_app.py`
- separates state, logic, and rendering concerns
- makes future UI iteration safer and easier
- improves readability when debugging training/playback behavior

## How to Run

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd SweepAgent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the random baseline evaluation

```bash
python scripts/evaluate_agents.py
```

### 4. Train a Q-learning agent on the default map

```bash
python scripts/train_q_learning.py
```

By default, this saves:
- checkpoint: `outputs/checkpoints/q_learning_agent_default_seed_42.json`
- plots:
  - `outputs/plots/training_reward_default.png`
  - `outputs/plots/training_cleaned_ratio_default.png`
  - `outputs/plots/training_success_default.png`
  - `outputs/plots/training_epsilon_default.png`

### 5. Compare the random baseline and learned Q-learning policy

```bash
python scripts/compare_agents.py
```

### 6. Render the learned greedy policy as a GIF

```bash
python scripts/render_policy_gif.py
```

### 7. Render a side-by-side comparison GIF

```bash
python scripts/render_comparison_gif.py
```

### 8. Run the multi-map benchmark

```bash
python scripts/benchmark_maps.py
```

### 9. Train and evaluate charger-aware maps

```bash
python scripts/train_q_learning.py --map-name charging_demo --episodes 3000 --seed 42 --print-every 100
python scripts/train_q_learning.py --map-name charge_required_v2 --episodes 5000 --seed 42 --print-every 100
python scripts/compare_agents.py --map-name charge_required_v2 --episodes 5000 --seed 42 --eval-episodes 100
```

### 10. Launch the interactive training app

```bash
python scripts/run_training_app.py
```

Current app capabilities:
- choose map / model / result-view mode from menu
- run q-learning training through the UI
- inspect live logs during training
- watch a mini rollout preview during training
- switch to single playback or compare playback after training
- control playback speed and restart rollouts

## Current Results

### Default Map

Random agent evaluation over 100 episodes:
- Average reward: `-94.03`
- Average steps: `64.58`
- Average cleaned ratio: `91.00%`
- Success rate: `79.00%`

Learned greedy policy over 100 episodes:
- Average reward: `76.00`
- Average steps: `7.00`
- Average cleaned ratio: `100.00%`
- Success rate: `100.00%`

### Harder Map

Random agent evaluation over 100 episodes:
- Average reward: `-300.97`
- Average steps: `119.86`
- Average cleaned ratio: `49.25%`
- Success rate: `2.00%`

Learned greedy policy over 100 episodes:
- Average reward: `76.00`
- Average steps: `18.00`
- Average cleaned ratio: `100.00%`
- Success rate: `100.00%`

### Charge-Required V2

Evaluation over 100 episodes:

| Seed | Avg Reward | Avg Steps | Avg Cleaned Ratio | Success Rate |
|---|---:|---:|---:|---:|
| 11 | 81.00 | 20.00 | 100.00% | 100.00% |
| 22 | 81.00 | 20.00 | 100.00% | 100.00% |
| 33 | 81.00 | 20.00 | 100.00% | 100.00% |

The random baseline remained at 0% success rate on the same map.

## Multi-Map Benchmark

The learned policy achieved 100% success rate on all four shared map presets:
- `default`
- `harder`
- `wide_room`
- `corridor`

The random baseline degraded significantly as map complexity increased.

## Roadmap

- [x] Build grid cleaning environment
- [x] Add random baseline agent
- [x] Add baseline evaluation script
- [x] Implement Q-learning agent
- [x] Train and evaluate learned policy
- [x] Visualize reward and cleaning progress
- [x] Add step-by-step movement visualization
- [x] Compare random vs learned agent
- [x] Add a harder map evaluation
- [x] Add more room layouts
- [x] Add multi-map benchmarking
- [x] Reuse checkpoints across scripts
- [x] Add battery constraints
- [x] Add charging stations
- [x] Train charger-aware behavior
- [x] Validate charge-required behavior across multiple seeds
- [x] Add initial program-based UI visualization
- [x] Refactor UI into state / handlers / core / views modules
- [ ] Stabilize refactored training app
- [ ] Add dynamic obstacles
- [ ] Test additional RL methods

## Future Extensions

Possible future improvements include:
- larger and more complex room layouts
- dynamic obstacles
- priority-based cleaning objectives
- multiple cleaning agents
- return-to-dock behavior after cleaning
- richer training charts in the UI
- deep reinforcement learning methods such as DQN

## Devlog

Development notes are recorded in:

```bash
docs/devlog/week1.md
```

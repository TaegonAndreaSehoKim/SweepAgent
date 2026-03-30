# SweepAgent

SweepAgent is a reinforcement learning project where a grid-based vacuum agent learns efficient cleaning policies in a room with walls and dirty tiles.

## Project Goal

The goal of SweepAgent is to train a self-learning agent that can clean a grid-based room more efficiently than a random policy.

The project started with a simple environment and baseline agent, and now includes Q-learning, GIF-based policy visualization, multi-map benchmarking, and reusable experiment checkpoints.

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

Next ideas:
- add battery constraints and charging stations
- add dynamic obstacles
- try larger and more difficult room layouts
- test additional RL methods beyond tabular Q-learning

## Environment Overview

The environment is a grid-based room made of:
- walls (`#`)
- empty tiles (`.`)
- dirty tiles (`D`)
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

## Reward Design

The current reward setting is:

- clean a new dirty tile: `+10`
- normal move: `-1`
- revisit an already cleaned dirty tile: `-2`
- invalid move / wall collision: `-5`
- clean all dirty tiles: `+50`

An episode ends when:
- all dirty tiles are cleaned, or
- the maximum number of steps is reached

## Rendering and Visualization

The project now supports both text-based rendering and GIF-based visualization.

Available visual outputs include:
- step-by-step terminal rendering for debugging
- learned greedy policy GIF playback
- side-by-side comparison GIFs between random and learned agents

Generated GIF outputs:
- `outputs/gifs/learned_policy_harder.gif`
- `outputs/gifs/comparison_harder.gif`

These visualizations make it much easier to inspect the learned behavior and compare it directly against the random baseline.

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
│   └── experiment_utils.py
├── scripts/
│   ├── evaluate_agents.py
│   ├── train_q_learning.py
│   ├── compare_agents.py
│   ├── render_policy_gif.py
│   ├── render_comparison_gif.py
│   └── benchmark_maps.py
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

The project now uses shared experiment utilities and map-specific checkpoints so that
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

This prevents policies trained on different maps from being mixed together.

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

This script reuses a saved checkpoint when available instead of retraining.

### 6. Render the learned greedy policy as a GIF

```bash
python scripts/render_policy_gif.py
```

This script also reuses the saved checkpoint for the selected map.

### 7. Render a side-by-side comparison GIF

```bash
python scripts/render_comparison_gif.py
```

This generates a side-by-side rollout comparing:
- random agent
- learned greedy Q-learning agent

### 8. Run the multi-map benchmark

```bash
python scripts/benchmark_maps.py
```

This generates:
- `outputs/logs/map_benchmark_results.csv`
- `outputs/plots/map_benchmark_success_rate.png`
- `outputs/plots/map_benchmark_reward.png`
- `outputs/plots/map_benchmark_steps.png`
- `outputs/plots/map_benchmark_cleaned_ratio.png`

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

Improvement over the random baseline:
- Reward gain: `170.03`
- Step reduction: `57.58`
- Cleaned ratio gain: `9.00 percentage points`
- Success rate gain: `21.00 percentage points`

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

These results show that the learned policy is not only better than the random baseline on the default map, but also remains highly effective on a more difficult room layout.

## Multi-Map Benchmark

To test whether the learned policy remains effective beyond a single room layout, SweepAgent was evaluated on four map presets:
- `default`
- `harder`
- `wide_room`
- `corridor`

The benchmark compares a random baseline against a learned greedy Q-learning policy.

### Benchmark Summary

| Map | Agent | Avg Reward | Avg Steps | Avg Cleaned Ratio | Success Rate |
|---|---|---:|---:|---:|---:|
| default | Random | -94.03 | 64.58 | 91.00% | 79.00% |
| default | Learned | 76.00 | 7.00 | 100.00% | 100.00% |
| harder | Random | -300.97 | 119.86 | 49.25% | 2.00% |
| harder | Learned | 76.00 | 18.00 | 100.00% | 100.00% |
| wide_room | Random | -307.48 | 143.36 | 70.50% | 18.00% |
| wide_room | Learned | 80.00 | 14.00 | 100.00% | 100.00% |
| corridor | Random | -478.95 | 167.84 | 35.67% | 5.00% |
| corridor | Learned | 63.00 | 20.00 | 100.00% | 100.00% |

### Benchmark Figures

Generated benchmark plots:
- `outputs/plots/map_benchmark_success_rate.png`
- `outputs/plots/map_benchmark_reward.png`
- `outputs/plots/map_benchmark_steps.png`
- `outputs/plots/map_benchmark_cleaned_ratio.png`

These figures highlight three important trends:
1. The learned policy solves every tested map consistently.
2. The learned policy uses far fewer steps than the random baseline.
3. The learned policy maintains full cleaning performance even when the room layout becomes more difficult.

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
- [ ] Explore more difficulty settings
- [ ] Test additional RL methods

## Future Extensions

Possible future improvements include:
- larger and more complex room layouts
- battery constraints and charging stations
- dynamic obstacles
- multiple cleaning agents
- deep reinforcement learning methods such as DQN

## Devlog

Development notes are recorded in:

```bash
docs/devlog/week1.md
```

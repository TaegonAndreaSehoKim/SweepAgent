# SweepAgent

SweepAgent is a grid-based reinforcement learning project for a cleaning robot.
The agent learns to clean dirty tiles while handling walls, step limits, battery constraints, and charger navigation.

## Current Status

SweepAgent currently includes:

- a custom `GridCleanEnv`
- a random baseline agent
- a tabular `QLearningAgent`
- JSON checkpoint save/load support
- shared experiment builders and checkpoint helpers
- benchmark, evaluation, comparison, and rendering scripts
- a pygame training/playback UI
- curriculum training support
- complex charge-aware maps

The current project focus is no longer the original small-room task.
It is now centered on longer-horizon charge-aware planning, harder map design, and repeatable experiment workflows.

## Main Features

### Environment

`GridCleanEnv` supports:

- wall collisions
- dirty tile cleaning
- revisit penalties
- step limits
- optional battery capacity
- charger tiles
- reward shaping for charger-aware and safety-aware navigation

Map symbols:

- `#` wall
- `.` floor
- `D` dirty tile
- `R` robot start
- `C` charging station

### Agents

Implemented agents:

- `RandomAgent`
- `QLearningAgent`

Checkpoints are reused across training, evaluation, comparison, GIF rendering, and UI playback.

### Training UI

The pygame UI supports:

- map selection
- algorithm selection
- result view selection
- direct numeric input for:
  - episodes
  - train seed
  - playback seed
  - learning rate
  - discount factor
  - epsilon start
  - epsilon decay
  - epsilon min
  - playback delay
- curriculum mode selection from the algorithm menu

The recent UI direction is to keep only high-value controls and make hyperparameters directly editable instead of limiting them to dropdown presets.

### Curriculum Support

A curriculum workflow is still available:

- stage 1 uses `charge_maze_medium`
- stage 2 uses the map selected by the user

In practice, the latest tuning work has focused more on direct single-map training for the hardest maps than on cross-map checkpoint transfer.

## Maps

Available presets include:

- `default`
- `harder`
- `wide_room`
- `corridor`
- `charge_required_v2`
- `charge_maze_medium`
- `charge_maze_large`
- `multi_charge_detour`
- `complex_charge_labyrinth`
- `complex_charge_bastion`
- `complex_charge_switchback`

The three `complex_charge_*` maps are the current hard battery-planning targets.

## Battery and Reward Design

Battery sizing is now generated systematically from the map structure.

For each preset, the project:

1. computes the minimum solvable battery requirement through shortest-path state search
2. adds a slack margin based on:
   - a minimum margin
   - a percentage of the minimum requirement
   - rounding up to a fixed unit
3. allows per-map overrides when a harder map needs more training slack

Battery capacity is now split by purpose:

- `battery_capacity_training`
- `battery_capacity_evaluation`

This allows a hard map to train with more slack while still being evaluated under a stricter setting.
For backward compatibility, the legacy `battery_capacity` field now points to the evaluation capacity.
The training scripts use the training profile, while evaluation, comparison, rendering, and playback flows use the evaluation profile.

Recent charge-aware reward changes also added:

- weaker charger-only shaping to avoid recharge loops
- revisit penalties for general path loops
- safe-dirty progress shaping
- penalties for entering unrecoverable low-battery states
- battery safety reserves for deciding when to return to a charger

These changes were introduced to fix failure modes on the hardest maps, especially charger loops and late battery-depletion mistakes.

## Current Training Defaults

The shared Q-learning defaults are now tuned for harder charge-aware runs:

- `episodes = 200000`
- `learning_rate = 0.05`
- `discount_factor = 0.99`
- `epsilon_start = 1.0`
- `epsilon_decay = 0.99999`
- `epsilon_min = 0.20`

Harder maps may still use longer runs such as `500000` episodes.

## Checkpoints

Checkpoint format:

`q_learning_agent_<map_name>_ep_<episodes>_seed_<seed>.json`

Examples:

- `q_learning_agent_complex_charge_labyrinth_ep_200000_seed_42.json`
- `q_learning_agent_complex_charge_switchback_ep_200000_seed_42.json`
- `q_learning_agent_complex_charge_bastion_ep_500000_seed_42.json`

## Recent Results

The recent tuning cycle materially improved the hardest maps.

Current read:

- `complex_charge_labyrinth` is solved under the latest direct-training setup
- `complex_charge_switchback` is solved after safety-aware reward shaping
- `complex_charge_bastion` is solved with stronger safety-aware shaping, slower epsilon decay, and a larger battery override during training

This project is now past the stage where all three newest maps are unresolved.
The current direction is improving training efficiency and documenting the learned behavior more clearly.

## Performance Notes

Recent performance work focused on CPU-side optimization rather than GPU usage.

That is intentional:

- the current algorithm is tabular Q-learning, not a neural-network method
- most of the cost is Python environment stepping and Q-table updates
- GPU acceleration is not useful without changing the learning algorithm itself

Recent optimizations include:

- caching safe-dirty lookup results
- a leaner `step_training()` path that avoids building full `info` payloads on every training step

## Project Structure

```text
SweepAgent/
|-- README.md
|-- agents/
|   |-- q_learning_agent.py
|   `-- random_agent.py
|-- configs/
|   `-- map_presets.py
|-- docs/
|   `-- devlog/
|       |-- week1.md
|       |-- week2.md
|       `-- week3.md
|-- env/
|   `-- grid_clean_env.py
|-- outputs/
|   |-- checkpoints/
|   |-- gifs/
|   |-- logs/
|   `-- plots/
|-- scripts/
|   |-- benchmark_maps.py
|   |-- compare_agents.py
|   |-- evaluate_agents.py
|   |-- render_comparison_gif.py
|   |-- render_policy_gif.py
|   |-- run_training_app.py
|   |-- train_q_curriculum.py
|   `-- train_q_learning.py
|-- ui/
|   |-- training_app_core.py
|   |-- training_app_handlers.py
|   |-- training_app_state.py
|   `-- training_app_views.py
`-- utils/
    |-- experiment_utils.py
    `-- ui_utils.py
```

## How to Run

### Train Q-learning

```bash
python scripts/train_q_learning.py --map-name complex_charge_switchback --episodes 200000 --seed 42
```

### Run batch training in parallel

```bash
python scripts/train_q_batch.py --maps complex_charge_labyrinth complex_charge_switchback --seeds 41 42 43 --episodes 200000 --max-workers 3
```

This workflow launches multiple `train_q_learning.py` runs in parallel subprocesses, writes one log per run under `outputs/logs/`, and saves a batch summary CSV.

### Evaluate checkpoints across maps and seeds

```bash
python scripts/evaluate_q_batch.py --maps complex_charge_labyrinth complex_charge_switchback --seeds 41 42 43 --episodes 200000 --eval-episodes 100
```

This workflow evaluates saved checkpoints under the evaluation battery profile and writes both per-run and per-map CSV summaries under `outputs/logs/`.

### Compare random vs learned agent

```bash
python scripts/compare_agents.py --map-name complex_charge_switchback --episodes 200000 --seed 42 --eval-episodes 50
```

### Evaluate a trained checkpoint

```bash
python scripts/evaluate_agents.py --map-name complex_charge_switchback --episodes 200000 --seed 42 --eval-episodes 50
```

### Launch the training UI

```bash
python scripts/run_training_app.py
```

### Run curriculum training

```bash
python scripts/train_q_curriculum.py --stage1-map charge_maze_medium --stage2-map complex_charge_labyrinth --seed 42
```

## Current Direction

The current improvement areas are:

- reducing wall-clock training time on hard maps
- comparing training efficiency across seeds
- deciding how much of the current success comes from reward design vs battery slack
- documenting hard-map training recipes more clearly

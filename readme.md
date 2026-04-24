# SweepAgent

SweepAgent is a grid-based reinforcement learning project for a cleaning robot.
The agent learns to clean dirty tiles while handling walls, step limits, battery constraints, and charger navigation.

## Current Status

SweepAgent currently includes:

- a custom `GridCleanEnv`
- a random baseline agent
- a tabular `QLearningAgent`
- a PyTorch-based `DQNAgent`
- JSON checkpoint save/load support
- DQN `.pt` checkpoint save/load support
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
- `DQNAgent`

Checkpoints are reused across training, evaluation, comparison, GIF rendering, and UI playback.
The DQN workflow is currently script-driven and is not yet integrated into the pygame UI.

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

Recent representation work also added optional internal state abstraction modes for tabular Q-learning:

- `identity`
- `safety_margin`
- `charger_context`

These modes do not change the environment state itself.
They only change how the Q-table keys are formed inside the agent.

## Current Training Defaults

The shared Q-learning defaults are now tuned for harder charge-aware runs:

- `episodes = 200000`
- `learning_rate = 0.05`
- `discount_factor = 0.99`
- `epsilon_start = 1.0`
- `epsilon_decay = 0.99999`
- `epsilon_min = 0.20`

Harder maps may still use longer runs such as `500000` episodes.

## Development Setup

The repository is easiest to use from PowerShell with the local `.venv`.

Dependency files:

- `requirements.txt`: runtime dependencies
- `requirements-dev.txt`: runtime dependencies plus test tooling
- `requirements-cuda.txt`: CUDA-specific PyTorch override

Recommended Windows setup:

```powershell
.\scripts\setup_dev.ps1
```

CUDA setup:

```powershell
.\scripts\setup_dev.ps1 -Cuda
```

If you want the setup script to install `ripgrep` through `winget`:

```powershell
.\scripts\setup_dev.ps1 -InstallRipgrep
```

Useful wrappers after setup:

```powershell
.\scripts\test.ps1
.\scripts\train_dqn.ps1 --map-name default --episodes 10
```

## Checkpoints

Q-learning checkpoint format:

`q_learning_agent_<map_name>_ep_<episodes>_seed_<seed>.json`

Examples:

- `q_learning_agent_complex_charge_labyrinth_ep_200000_seed_42.json`
- `q_learning_agent_complex_charge_switchback_ep_200000_seed_42.json`
- `q_learning_agent_complex_charge_bastion_ep_500000_seed_42.json`

DQN checkpoint format:

`dqn_agent_<map_name>_ep_<episodes>_seed_<seed>.pt`

Examples:

- `dqn_agent_complex_charge_bastion_ep_5000_seed_416.pt`
- `dqn_agent_complex_charge_bastion_ep_5000_seed_417.pt`

## Recent Results

The recent tuning cycle materially improved the hardest maps.

Current read:

- `complex_charge_labyrinth` is solved under both training and evaluation battery settings
- `complex_charge_switchback` is solved after safety-aware reward shaping
- `complex_charge_bastion` is solved on the training battery profile with abstracted Q-learning, but still plateaus at `2/3` cleaned on the stricter evaluation profile

This means the project has working hard-map training recipes, but `bastion` still exposes a real generalization limit for the current tabular approach.

Recent DQN experiments changed that picture slightly:

- a naive DQN reached `0/3` cleaned on `complex_charge_bastion`
- action masking removed invalid wall-action loops
- guided exploration made dirty-reaching trajectories appear in replay memory
- target dirty/charger context features improved greedy evaluation from `1/3` cleaned to `2/3` cleaned
- a later route-via-charger feature version regressed under longer training and fell back to `0/3` cleaned at `10000` episodes on seed `417`

The best current DQN result on `complex_charge_bastion` also plateaus at `2/3` cleaned under the evaluation battery profile.
That mirrors the best tabular result, but reaches it through a different mechanism.

## Performance And CUDA Notes

The tabular Q-learning path is still CPU-bound.

For Q-learning, GPU acceleration is not useful because:

- the current algorithm is tabular Q-learning, not a neural-network method
- most of the cost is Python environment stepping and Q-table updates

Recent optimizations include:

- caching safe-dirty lookup results
- a leaner `step_training()` path that avoids building full `info` payloads on every training step

The DQN path uses PyTorch and can use CUDA when a CUDA-enabled PyTorch wheel is installed.
On the current development machine, CUDA was verified with:

- NVIDIA GeForce RTX 3080 Ti
- PyTorch `2.11.0+cu128`
- `torch.cuda.is_available() == True`

For a CUDA-capable environment, install the normal requirements first, then install the CUDA-specific PyTorch wheel:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-cuda.txt
```

## Project Structure

```text
SweepAgent/
|-- README.md
|-- agents/
|   |-- dqn_agent.py
|   |-- q_learning_agent.py
|   `-- random_agent.py
|-- configs/
|   `-- map_presets.py
|-- docs/
|   `-- devlog/
|       |-- week1.md
|       |-- week2.md
|       |-- week3.md
|       `-- week4.md
|-- env/
|   `-- grid_clean_env.py
|-- outputs/
|   |-- checkpoints/
|   |-- gifs/
|   |-- logs/
|   `-- plots/
|-- requirements-cuda.txt
|-- requirements-dev.txt
|-- requirements.txt
|-- scripts/
|   |-- benchmark_maps.py
|   |-- compare_agents.py
|   |-- evaluate_agents.py
|   |-- render_comparison_gif.py
|   |-- render_policy_gif.py
|   |-- run_training_app.py
|   |-- setup_dev.ps1
|   |-- test.ps1
|   |-- train_dqn.py
|   |-- train_dqn.ps1
|   |-- train_q_curriculum.py
|   `-- train_q_learning.py
|-- tests/
|   `-- test_dqn_agent.py
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

PowerShell with the local virtual environment:

```powershell
.\.venv\Scripts\python.exe scripts\train_q_learning.py --map-name complex_charge_switchback --episodes 200000 --seed 42
```

### Train abstracted Q-learning

```bash
python scripts/train_q_learning.py --map-name complex_charge_bastion --episodes 200000 --seed 42 --state-abstraction-mode charger_context --safety-margin-bucket-size 5
```

This mode keeps the raw environment unchanged and only changes the internal Q-table key. The current abstraction options are:

- `safety_margin`: buckets battery as `(battery_remaining - nearest_charger_distance)`
- `charger_context`: extends that idea with charger-region and remaining-dirty context

### Train with battery-profile adaptation

```bash
python scripts/train_q_battery_adapt.py --map-name complex_charge_bastion --stage1-episodes 200000 --stage2-episodes 150000 --seed 42 --stage2-learning-rate 0.015 --stage2-epsilon-start 0.15 --stage2-epsilon-decay 0.999985 --stage2-epsilon-min 0.04 --state-abstraction-mode safety_margin
```

This workflow first trains on the training battery profile, then fine-tunes on the evaluation battery profile with a smaller exploration schedule. The UI exposes both stage lengths directly, and the final checkpoint is saved under the total combined episode count.

For `complex_charge_bastion`, the best recent tabular result used:

- abstracted Q-learning
- `stage1 = 200000`
- `stage2 = 150000`
- conservative-but-not-too-conservative stage2 exploration

That recipe preserved training-profile success and raised evaluation behavior to a stable `2/3` cleaned, but it still did not reach full evaluation success.

### Train DQN

```bash
python scripts/train_dqn.py --map-name complex_charge_bastion --battery-profile evaluation --episodes 5000 --seed 417 --print-every 500 --learning-rate 0.0005 --epsilon-decay 0.99995 --epsilon-min 0.10 --learning-starts 1000 --batch-size 128 --replay-capacity 100000 --train-every 8 --target-update-interval 1000 --hidden-size 256 --guided-exploration-ratio 0.6 --eval-episodes 20
```

PowerShell wrapper:

```powershell
.\scripts\train_dqn.ps1 --map-name complex_charge_bastion --battery-profile evaluation --episodes 5000 --seed 417 --print-every 500 --learning-rate 0.0005 --epsilon-decay 0.99995 --epsilon-min 0.10 --learning-starts 1000 --batch-size 128 --replay-capacity 100000 --train-every 8 --target-update-interval 1000 --hidden-size 256 --guided-exploration-ratio 0.6 --eval-episodes 20 --feature-version 2
```

To keep the best greedy evaluation checkpoint instead of only the final checkpoint, add:

```bash
python scripts/train_dqn.py --map-name complex_charge_bastion --battery-profile evaluation --episodes 5000 --seed 417 --print-every 500 --learning-rate 0.0005 --epsilon-decay 0.99995 --epsilon-min 0.10 --learning-starts 1000 --batch-size 128 --replay-capacity 100000 --train-every 8 --target-update-interval 1000 --hidden-size 256 --guided-exploration-ratio 0.6 --eval-episodes 50 --eval-every 500 --feature-version 2 --save-best-eval-checkpoint
```

This writes the usual final checkpoint and also keeps a separate best-evaluation checkpoint such as
`dqn_agent_complex_charge_bastion_best_eval_seed_417.pt`.

The DQN implementation currently includes:

- replay buffer training
- Double-DQN target selection
- hard target-network updates
- action masking for wall moves
- optional guided exploration
- map-derived state features for dirty, charger, battery, and route context
- feature-versioned state encoding so older checkpoints can still load

The best current DQN result on `complex_charge_bastion` reached `2/3` cleaned under the evaluation battery profile.
Higher guided exploration ratios produced more dirty-reaching training trajectories but did not improve greedy evaluation, so the current best setting keeps guidance at `0.6`.
The newer `feature-version 2` route-via-charger experiment did not beat that baseline in long training and currently serves as an experimental branch rather than the default best recipe.

### Slice DQN checkpoints by episode

```bash
python scripts/train_dqn_sliced.py --map-name complex_charge_bastion --battery-profile evaluation --total-episodes 10000 --slice-episodes 500 --seed 417 --print-every 500 --learning-rate 0.0005 --epsilon-decay 0.99995 --epsilon-min 0.10 --learning-starts 1000 --batch-size 128 --replay-capacity 100000 --train-every 8 --target-update-interval 1000 --hidden-size 256 --guided-exploration-ratio 0.6 --eval-episodes 50 --feature-version 2 --device cuda --checkpoint-tag v2slice
```

This workflow resumes DQN training across episode slices, saves one checkpoint per cumulative slice, and writes a CSV summary under `outputs/logs/` so the collapse window can be located without manually rerunning each cutoff.
Use `--checkpoint-tag` for experimental branches such as `feature-version 2` so older DQN checkpoints with the same map, episode count, and seed are not overwritten.

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

### Run tests

```powershell
.\scripts\test.ps1
```

### Run curriculum training

```bash
python scripts/train_q_curriculum.py --stage1-map charge_maze_medium --stage2-map complex_charge_labyrinth --seed 42
```

## Current Direction

The current improvement areas are:

- improving the final `complex_charge_bastion` transition from `2/3` cleaned to full success
- stabilizing route-via-charger DQN training so richer features do not collapse greedy evaluation
- comparing DQN behavior across seeds instead of relying on one promising run
- deciding whether DQN should be integrated into the pygame UI
- keeping generated checkpoints, plots, logs, and GIFs out of version control

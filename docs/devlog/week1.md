# Week 1 Devlog

## Day 1 - Environment Setup and Random Baseline

Today I built the first working version of the SweepAgent project environment and confirmed that the basic simulation loop runs correctly.

### What I completed
- Initialized the SweepAgent project structure
- Implemented the core grid cleaning environment in `env/grid_clean_env.py`
- Added environment logic for:
  - `reset()`
  - `step(action)`
  - reward calculation
  - terminal condition checking
  - simple text-based `render()`
- Represented the environment state as:
  - robot position `(row, col)`
  - cleaned tile status using a bitmask
- Added a random baseline agent in `agents/random_agent.py`
- Added an evaluation script in `scripts/evaluate_agents.py`

### Environment design
The current environment is a small grid-based room with:
- walls (`#`)
- empty tiles (`.`)
- dirty tiles (`D`)
- one robot start position (`R`)

The agent can take four actions:
- up
- down
- left
- right

Reward design:
- clean a new dirty tile: `+10`
- normal move: `-1`
- revisit an already cleaned dirty tile: `-2`
- invalid move / wall collision: `-5`
- clean all dirty tiles: `+50`

An episode ends when:
- all dirty tiles are cleaned, or
- the agent reaches the maximum step limit

### Bug fixed
During testing, I found a rendering issue where the robot appeared twice on the grid after moving.  
This happened because the original start position remained marked as `R` in the base grid.

I fixed it by converting the start tile to `.` after reading the initial robot position, and now the current robot position is rendered correctly.

### Baseline result
I ran 100 evaluation episodes using the random agent.

Results:
- Average total reward: `-94.03`
- Average steps taken: `64.58`
- Average cleaned tiles: `2.73 / 3`
- Average cleaned ratio: `91.00%`
- Success rate: `79.00%`

### Takeaway
The environment is now fully playable and produces stable episode results.  
The random baseline performs better than expected because the current map is small and the step budget is generous. This means the environment is working, but I may need to increase difficulty later so that improvements from Q-learning are more visible.

## Day 2 - Q-Learning Training, Evaluation, and Visualization

Today I implemented the first learning-based agent for SweepAgent using tabular Q-learning and confirmed that it significantly outperforms the random baseline.

### What I completed
- Implemented `QLearningAgent` in `agents/q_learning_agent.py`
- Added epsilon-greedy exploration
- Added Q-table update logic
- Added a training script in `scripts/train_q_learning.py`
- Saved training plots for:
  - reward
  - cleaned ratio
  - success trend
  - epsilon decay
- Added shared experiment settings in `configs/default_config.py`
- Added `scripts/compare_agents.py` to compare the random baseline and the learned greedy policy
- Added a harder map preset for more meaningful evaluation
- Added GIF rendering for a learned policy rollout
- Added a side-by-side comparison GIF for the random and learned policies

### Training result on the default map
After training for 1000 episodes:
- Final epsilon: `0.0500`
- Learned Q-table states: `64`
- Last 100 average reward: `75.02`
- Last 100 average cleaned ratio: `100.00%`
- Last 100 success rate: `100.00%`

The learned greedy policy cleaned the full room in 7 steps with total reward `76`.

### Comparison on the default map
Random baseline over 100 episodes:
- Average reward: `-94.03`
- Average steps: `64.58`
- Average cleaned ratio: `91.00%`
- Success rate: `79.00%`

Learned greedy policy over 100 episodes:
- Average reward: `76.00`
- Average steps: `7.00`
- Average cleaned ratio: `100.00%`
- Success rate: `100.00%`

### Comparison on the harder map
Random baseline over 100 episodes:
- Average reward: `-300.97`
- Average steps: `119.86`
- Average cleaned ratio: `49.25%`
- Success rate: `2.00%`

Learned greedy policy over 100 episodes:
- Average reward: `76.00`
- Average steps: `18.00`
- Average cleaned ratio: `100.00%`
- Success rate: `100.00%`

### Visualization output
The project now includes two GIF-based visual outputs:
- `outputs/gifs/learned_policy_harder.gif`
- `outputs/gifs/comparison_harder.gif`

### Takeaway
The harder map made the difference even clearer: the random policy almost always failed, while the learned policy solved the task consistently.

## Day 3 - Multi-Map Benchmarking and Result Visualization

Today I expanded SweepAgent from single-map evaluation to a multi-map benchmark.

### What I completed
- Added shared map presets in `configs/map_presets.py`
- Added a benchmarking script in `scripts/benchmark_maps.py`
- Evaluated both the random baseline and the learned greedy policy across multiple room layouts
- Saved benchmark results as CSV
- Saved benchmark comparison plots for:
  - success rate
  - average reward
  - average steps
  - average cleaned ratio

### Maps included
The benchmark currently includes four map presets:
- `default`
- `harder`
- `wide_room`
- `corridor`

### Key results
The learned policy achieved 100% success rate on all four maps.

The random baseline performed much worse, especially as map difficulty increased:
- `default`: 79.00%
- `harder`: 2.00%
- `wide_room`: 18.00%
- `corridor`: 5.00%

### Takeaway
This stage made SweepAgent look much more like a real RL project rather than a single-environment demo.

## Day 4 - Experiment Pipeline Refactor and Checkpoint Reuse

Today I refactored the experiment workflow so that training, comparison, rendering, and benchmarking all follow the same map and checkpoint rules.

### What I completed
- Added checkpoint serialization in `agents/q_learning_agent.py`
- Added JSON-based save/load support for the Q-learning agent
- Added shared utilities in `utils/experiment_utils.py`
- Standardized map-specific checkpoint naming
- Added backward compatibility for the older default-map checkpoint name
- Updated `scripts/compare_agents.py` to reuse saved checkpoints
- Updated `scripts/render_policy_gif.py` to reuse saved checkpoints
- Updated `scripts/render_comparison_gif.py` to reuse saved checkpoints
- Updated `scripts/benchmark_maps.py` to use shared experiment utilities
- Updated `scripts/train_q_learning.py` to save map-specific checkpoints and map-specific training plots

### Takeaway
The project now has a reusable and scalable experiment pipeline. The codebase is cleaner, the scripts are faster to rerun, and the benchmark evidence is much easier to present.

## Day 5 - Battery Constraints, Charging Stations, and Multi-Seed Validation

Today I extended SweepAgent beyond simple cleaning and added battery-aware, charger-aware behavior.

### What I completed
- Added optional battery constraints to `env/grid_clean_env.py`
- Extended the environment state to include battery level
- Added charging station tiles (`C`)
- Added recharge logic so the battery is restored to full on charger tiles
- Added charger-aware reward shaping for more learnable recharge behavior
- Added `battery_harder` as a battery-constrained environment
- Added `charging_demo` as a charger behavior learning map
- Added `charge_required_v2` as a harder charger-dependent map
- Updated charger-aware GIF rendering for learned policy playback and side-by-side comparison
- Validated charger-aware behavior across multiple random seeds

### Charge-required validation
Evaluation over 100 episodes showed:

- Seed 11:
  - Avg reward: `81.00`
  - Avg steps: `20.00`
  - Avg cleaned ratio: `100.00%`
  - Success rate: `100.00%`

- Seed 22:
  - Avg reward: `81.00`
  - Avg steps: `20.00`
  - Avg cleaned ratio: `100.00%`
  - Success rate: `100.00%`

- Seed 33:
  - Avg reward: `81.00`
  - Avg steps: `20.00`
  - Avg cleaned ratio: `100.00%`
  - Success rate: `100.00%`

The random baseline remained at 0% success rate on the same map.

### Takeaway
This was the first stage where SweepAgent clearly demonstrated charger-aware reinforcement learning behavior.

## Day 6 - Program-Based UI Visualization and App Refactor

Today I moved SweepAgent beyond offline plots and GIFs and started building a program-based interactive UI.

### What I completed
- Added an initial pygame-based UI runner in `scripts/run_training_app.py`
- Added menu-based map, model, episode, and playback-delay selection
- Added training screen with:
  - live log panel
  - training snapshot bars
  - mini rollout preview
- Added playback screens for:
  - single playback
  - side-by-side compare playback
- Improved playback layout so headers, controls, and info panels no longer overlap
- Added responsive behavior so the training screen uses a taller window while playback screens still size to content
- Refactored the UI into separate modules:
  - `ui/training_app_core.py`
  - `ui/training_app_state.py`
  - `ui/training_app_handlers.py`
  - `ui/training_app_views.py`

### Why this mattered
Before this stage, SweepAgent mainly relied on terminal output, saved plots, and GIF rendering.

With the program-based UI:
- the user can choose experiment settings without editing scripts
- training progress can be monitored live
- playback can be controlled interactively
- the project now feels closer to a real interactive ML demo

### Current UI status
The UI already supports:
- menu selection for map / model / result mode
- q-learning training launch from the app
- live training log display
- mini preview during training
- single playback mode
- compare playback mode
- pause / resume / restart / slower / faster controls

The main remaining work is stabilization and cleanup after the refactor, not initial feature creation.

### Takeaway
This was an important presentation milestone.  
SweepAgent is no longer just a script-based RL project. It now has the foundation of an interactive application layer, which makes the learned behavior much easier to inspect and demonstrate.

### Next steps
For the next stage, I plan to:
- stabilize the refactored training app
- improve state management further
- polish compare playback UX
- add richer in-app training charts

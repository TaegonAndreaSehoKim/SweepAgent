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

### Next steps
For Day 2, I planned to:
- implement a Q-learning agent
- train it with epsilon-greedy exploration
- compare Q-learning against the random baseline
- visualize learning progress with reward and cleaning performance curves

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

Improvement over the random baseline:
- Reward gain: `170.03`
- Step reduction: `57.58`
- Cleaned ratio gain: `9.00 percentage points`
- Success rate gain: `21.00 percentage points`

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

These visualizations make it much easier to inspect the learned behavior and compare it directly against the random baseline.

### Takeaway
The default map already showed that Q-learning learns a much more efficient cleaning strategy than a random walk.

The harder map made the difference even clearer: the random policy almost always failed, while the learned policy solved the task consistently. At this stage, SweepAgent is now demonstrating clear reinforcement learning behavior, meaningful baseline improvement, and visual policy playback.

### Next steps
For the next stage, I planned to:
- test more room layouts
- organize experiment outputs more cleanly
- improve README documentation with figures and GIF previews
- explore additional environment difficulty settings and evaluation scenarios

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

The learned policy also remained much more efficient:
- `default`: 7.00 average steps
- `harder`: 18.00 average steps
- `wide_room`: 14.00 average steps
- `corridor`: 20.00 average steps

In contrast, the random baseline required far more steps and often failed to complete the task.

### Output files
Saved benchmark outputs:
- `outputs/logs/map_benchmark_results.csv`
- `outputs/plots/map_benchmark_success_rate.png`
- `outputs/plots/map_benchmark_reward.png`
- `outputs/plots/map_benchmark_steps.png`
- `outputs/plots/map_benchmark_cleaned_ratio.png`

### Takeaway
This stage makes SweepAgent look much more like a real RL project rather than a single-environment demo.

The learned agent no longer just performs well on one small room. It now shows consistent performance across multiple layouts, with clear advantages over the random baseline in completion, reward, and efficiency.

### Next steps
For the next stage, I planned to:
- improve README presentation with benchmark figures
- organize outputs more cleanly
- try additional room layouts
- explore more challenging environment settings

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

### Why this matters
Previously, several scripts retrained the Q-learning agent every time they were run.  
This made experiments slower and introduced unnecessary duplication across scripts.

With the new structure:
- trained agents are reused automatically
- map-specific policies are stored separately
- all experiment scripts follow the same environment and checkpoint rules
- adding new maps is easier because the workflow is now driven by `MAP_PRESETS`

### Checkpoint outputs
The project now saves separate checkpoints for each map:
- `outputs/checkpoints/q_learning_agent_default_seed_42.json`
- `outputs/checkpoints/q_learning_agent_harder_seed_42.json`
- `outputs/checkpoints/q_learning_agent_wide_room_seed_42.json`
- `outputs/checkpoints/q_learning_agent_corridor_seed_42.json`

### Updated benchmark observations
The random baseline degraded quickly as map complexity increased, while the learned greedy Q-learning agent consistently solved every shared map preset.

Observed success rates:
- Random agent:
  - `default`: 79%
  - `harder`: 2%
  - `wide_room`: 18%
  - `corridor`: 5%
- Learned greedy agent:
  - `default`: 100%
  - `harder`: 100%
  - `wide_room`: 100%
  - `corridor`: 100%

### Generated artifacts
- Checkpoints:
  - `outputs/checkpoints/q_learning_agent_default_seed_42.json`
  - `outputs/checkpoints/q_learning_agent_harder_seed_42.json`
  - `outputs/checkpoints/q_learning_agent_wide_room_seed_42.json`
  - `outputs/checkpoints/q_learning_agent_corridor_seed_42.json`
- GIFs:
  - `outputs/gifs/learned_policy_harder.gif`
  - `outputs/gifs/comparison_harder.gif`
- Benchmark outputs:
  - `outputs/logs/map_benchmark_results.csv`
  - `outputs/plots/map_benchmark_success_rate.png`
  - `outputs/plots/map_benchmark_reward.png`
  - `outputs/plots/map_benchmark_steps.png`
  - `outputs/plots/map_benchmark_cleaned_ratio.png`

### Takeaway
The project now has a reusable and scalable experiment pipeline. The codebase is cleaner, the scripts are faster to rerun, and the benchmark evidence is much easier to present.

### Next steps
For the next stage, I plan to:
- expose script settings through CLI arguments
- improve README presentation with embedded benchmark figures
- try additional environment variants
- explore richer RL extensions such as battery constraints or obstacles

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

### Battery milestone
The battery-constrained environment successfully increased difficulty while preserving learnability.

On `battery_harder`, the learned policy was able to clean all tiles consistently, while the random baseline performed very poorly. This confirmed that battery-aware state tracking was working correctly and that the learned agent could adapt its policy to a tighter movement budget.

### Charging behavior milestone
The first goal for charging was not to make charging absolutely mandatory, but to make the learned policy actually use a charger during rollout.

This was achieved with `charging_demo`. The learned policy:
- moved to a charger during the episode
- restored its battery
- continued cleaning after recharge
- finished the full cleaning task successfully

This was the first stage where SweepAgent learned behavior that goes beyond direct cleaning and starts to resemble resource-aware planning.

### Charge-required validation
After confirming charger usage on `charging_demo`, I moved to a harder map called `charge_required_v2`.

This map required more careful route planning under battery limits. Early versions were unstable, so I refined:
- map layout
- battery capacity
- battery-aware shaping
- recharge-related reward shaping

The final version was validated with multiple seeds.

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

### Generated artifacts
New charger-aware outputs include:
- `outputs/gifs/learned_policy_charging_demo.gif`
- `outputs/gifs/comparison_charging_demo.gif`
- `outputs/gifs/learned_policy_charge_required_v2.gif`
- `outputs/gifs/comparison_charge_required_v2.gif`
- `outputs/checkpoints/q_learning_agent_battery_harder_seed_42.json`
- `outputs/checkpoints/q_learning_agent_charging_demo_seed_42.json`
- `outputs/checkpoints/q_learning_agent_charge_required_v2_seed_42.json`

### Takeaway
This was the first stage where SweepAgent clearly demonstrated charger-aware reinforcement learning behavior.

The project now goes beyond basic path efficiency:
- it can manage battery limits
- it can use charging stations mid-episode
- it can solve a charger-dependent map reliably across multiple seeds

At this point, the next natural step is no longer environment basics. The project is ready to move toward richer interaction and presentation, such as program-based UI visualization or dynamic obstacle scenarios.

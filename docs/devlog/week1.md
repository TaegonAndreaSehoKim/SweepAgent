# SweepAgent - Week 1 Devlog

## Day 1 - Environment and Random Baseline

Today I built the first playable version of SweepAgent.

### What I completed
- Built the first version of `GridCleanEnv`
- Added support for:
  - walls
  - dirty tiles
  - robot start position
  - step limits
- Added a `RandomAgent`
- Added a baseline evaluation script
- Measured random-policy performance on the default map

### Initial result
The random baseline worked better than expected on the small default map because the map was compact and the step budget was generous.

### Takeaway
This gave me a useful baseline, but it also made it clear that I would need harder maps and stronger evaluation settings later.

---

## Day 2 - Q-learning and First Learned Policy

Today I implemented Q-learning and verified that a learned policy clearly outperformed the random baseline.

### What I completed
- Added `QLearningAgent`
- Added training support
- Evaluated the learned greedy policy against the random baseline
- Added first visualization output for policy behavior

### Key comparison
On the default map, the learned agent solved the task consistently and used far fewer steps than the random baseline.

On the harder map, the difference became even clearer:
- the random policy almost always failed
- the learned policy solved the task consistently

### Takeaway
SweepAgent was no longer just a random-walk toy example.
It became a proper RL learning demo with clear performance gains.

---

## Day 3 - Multi-Map Benchmarking and Result Visualization

Today I expanded SweepAgent from a single-map experiment into a multi-map benchmark.

### What I completed
- Added shared map presets in `configs/map_presets.py`
- Added `scripts/benchmark_maps.py`
- Evaluated random vs learned policy across multiple room layouts
- Saved benchmark logs and plots

### Benchmark maps
The benchmark set included:
- `default`
- `harder`
- `wide_room`
- `corridor`

### Takeaway
This made the project look much more like a real RL project rather than a single-room demonstration.

---

## Day 4 - Experiment Pipeline Refactor and Checkpoint Reuse

Today I refactored the experiment workflow so that training, comparison, rendering, and benchmarking all follow the same map and checkpoint rules.

### What I completed
- Added JSON checkpoint save/load support to `QLearningAgent`
- Added shared helpers in `utils/experiment_utils.py`
- Standardized checkpoint reuse across scripts
- Updated training and rendering scripts to share the same experiment utilities

### Why this mattered
Previously, several scripts retrained models unnecessarily.
After this refactor, training outputs became easier to reuse and compare.

---

## Day 5 - UI Direction and Playback Visualization

Today I started pushing SweepAgent toward a more interactive demo experience.

### What I completed
- Improved visualization layout
- Added playback-oriented displays
- Iterated on panel layout and hover information
- Improved readability of the grid display and status overlays

### Takeaway
This was the point where the project started to feel like a product demo rather than just a script collection.

---

## Day 6 - Pygame Training App Refactor

Today I focused on making the UI codebase easier to maintain and extend.

### What I completed
- Broke the large training app into smaller modules:
  - `training_app_core.py`
  - `training_app_state.py`
  - `training_app_views.py`
  - `training_app_handlers.py`
- Fixed multiple import and integration issues during the refactor
- Improved layout behavior for training logs, graph panels, and playback regions
- Added better controls for playback and menu interaction

### Result
The training app became much easier to extend with new controls and future algorithm options.

---

## Day 7 - Charge-Aware Maps, Hyperparameter Controls, and Large-Map Tuning

Today I moved into the hardest part of Week 1: charge-aware planning and larger maps.

### What I completed
- Added larger and more complex charge-aware maps:
  - `charge_required_v2`
  - `charge_maze_medium`
  - `charge_maze_large`
  - `multi_charge_detour`
- Updated the environment to support charge-aware runs more cleanly
- Added episode-aware checkpoint naming
- Added menu controls in the UI for:
  - episodes
  - train seed
  - playback seed
  - learning rate
  - discount factor
  - epsilon start
  - epsilon decay
  - epsilon min
- Expanded episode presets so longer runs could be launched directly from the UI
- Tuned large-map hyperparameters and found that map-specific settings worked much better than one shared global preset

### Important lesson
A single fixed Q-learning configuration was not good enough for both small maps and large charge-aware maps.

The larger map responded much better to:
- `learning_rate = 0.05`
- `discount_factor = 0.99`
- `epsilon_decay = 0.999`
- `epsilon_min = 0.10`

### Problems encountered
- Several reward-shaping attempts made behavior worse rather than better
- Charge-aware loops near the charger were difficult to tune
- Some environment changes caused regressions on maps that had previously worked
- Increasing episode count eventually helped, but it was slower than desired

### Final state for the day
By the end of the day:
- the tuned parameter combination worked better
- large-map training became more stable
- the project had a much stronger experimental control surface
- the next logical step became curriculum learning

### Planned next step
Curriculum learning is the next major feature I plan to add, but it will start in the next session rather than today.

---

## Week 1 Summary

By the end of Week 1, SweepAgent had grown from a simple random-baseline environment into a much more complete RL project.

### Week 1 outcomes
- built a custom grid-cleaning environment
- implemented a random baseline
- implemented tabular Q-learning
- added benchmarking across multiple layouts
- added checkpoint reuse
- built visualization and playback outputs
- built a modular pygame training UI
- added charge-aware maps and controls
- stabilized larger-map training through map-specific hyperparameter tuning

### Final takeaway
The most important Week 1 insight was that harder charge-aware maps need different training behavior from the original small maps.
That finding directly motivates the next stage: curriculum learning.

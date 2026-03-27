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
For Day 2, I plan to:
- implement a Q-learning agent
- train it with epsilon-greedy exploration
- compare Q-learning against the random baseline
- visualize learning progress with reward and cleaning performance curves

## Day 2 - Q-Learning Training and Agent Comparison

Today I implemented the first learning-based agent for SweepAgent using tabular Q-learning and verified that it significantly outperforms the random baseline.

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
- Added a harder map preset for a more meaningful evaluation

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

Improvement:
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

### Takeaway
The default map already showed that Q-learning learns a much more efficient cleaning strategy than a random walk.

The harder map made the difference even clearer: the random policy almost always failed, while the learned policy solved the task consistently. This confirms that SweepAgent is now demonstrating meaningful reinforcement learning behavior rather than just environment interaction.

### Next steps
For the next stage, I plan to:
- add step-by-step visual playback of agent movement
- save learned trajectories as GIFs or animations
- test more room layouts
- prepare cleaner result summaries for the README

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
For the next stage, I plan to:
- test more room layouts
- organize experiment outputs more cleanly
- improve README documentation with figures and GIF previews
- explore additional environment difficulty settings and evaluation scenarios
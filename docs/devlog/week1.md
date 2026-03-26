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
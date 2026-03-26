# SweepAgent

SweepAgent is a reinforcement learning project where a grid-based vacuum agent learns efficient cleaning policies in a room with walls and dirty tiles.

## Project Goal

The goal of SweepAgent is to train a self-learning agent that can clean a grid-based room more efficiently than a random policy.

The project starts with a simple environment and baseline agent, then gradually expands to more advanced reinforcement learning settings.

## Current Status

Completed:
- project structure setup
- grid cleaning environment
- reward and terminal logic
- text-based environment rendering
- random baseline agent
- evaluation script for baseline performance

Planned next:
- Q-learning agent
- training loop with epsilon-greedy exploration
- reward and cleaning performance plots
- policy comparison between random and learned agents
- step-by-step visual playback of the agent's movement using a dedicated visualization tool

## Environment Overview

The current environment is a small grid-based room made of:
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

At the current stage, the environment is rendered using a simple text-based ASCII view in the terminal for debugging and quick inspection.

In later stages, I plan to add a dedicated visualization tool so that the full movement process of the agent can be viewed more clearly. This will likely include:
- step-by-step animation of the agent moving through the room
- visual indication of cleaned and remaining dirty tiles
- saved episode playback as GIF or video
- easier comparison between random and learned policies

Possible future visualization options include:
- matplotlib animation
- GIF generation with imageio
- an interactive 2D interface using pygame

## Project Structure

```bash
SweepAgent/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default_config.py
├── env/
│   └── grid_clean_env.py
├── agents/
│   └── random_agent.py
├── scripts/
│   └── evaluate_agents.py
├── outputs/
│   ├── plots/
│   ├── gifs/
│   └── logs/
└── docs/
    └── devlog/
        └── week1.md
```

## Current Baseline Result

Random agent evaluation over 100 episodes:

- Average total reward: `-94.03`
- Average steps taken: `64.58`
- Average cleaned tiles: `2.73 / 3`
- Average cleaned ratio: `91.00%`
- Success rate: `79.00%`

The random baseline performs better than expected because the current map is small and the step budget is relatively generous. This will make it useful to later adjust environment difficulty and compare performance against a learned agent.

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

## Roadmap

- [x] Build grid cleaning environment
- [x] Add random baseline agent
- [x] Add baseline evaluation script
- [ ] Implement Q-learning agent
- [ ] Train and evaluate learned policy
- [ ] Visualize reward and cleaning progress
- [ ] Add step-by-step movement visualization
- [ ] Compare random vs learned agent
- [ ] Add harder maps and extended environment rules

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

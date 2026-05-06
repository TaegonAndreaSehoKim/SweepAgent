# SweepAgent

SweepAgent is a grid-based reinforcement learning project for a cleaning robot with walls, dirty tiles, battery limits, and charging stations. The project started as a small gridworld exercise, but the final focus is long-horizon, charge-aware route learning on maps that behave more like resource-constrained planning problems.

The main conclusion from the final experiments is that the hardest maps are feasible, but primitive grid-action reinforcement learning has a poor representation for them. Planner, waypoint, and graph-state baselines show that the real decision problem is much smaller than the raw step-by-step environment suggests.

## Project Status

The project is in a report-ready state. It includes:

- a custom `GridCleanEnv`
- battery-aware map presets
- random, Q-learning, SARSA, DQN, PPO, and guided-policy agents
- planner, waypoint-controller, and graph-state baselines
- checkpoint save/load support for JSON and PyTorch checkpoints
- training, evaluation, comparison, and rendering scripts
- a pygame UI for training/playback inspection
- final experiment reports under `docs/`

The generated experiment artifacts live under `outputs/` and are intentionally kept out of version control.

## Reports

The main writeups are:

- [Experiment Report Draft](docs/report_draft.md)
- [Improvement Experiment Report](docs/improvement_experiment_report.md)
- [Development Log](docs/devlog/)

The improvement report summarizes the final follow-up experiments:

- structural planner/waypoint/graph baselines
- guided-policy heuristic checks
- Q-learning seed sweep
- SARSA ablation
- current `complex_charge_bastion` reference comparison

## Environment

Map symbols:

| Symbol | Meaning |
| --- | --- |
| `#` | wall |
| `.` | floor |
| `R` | robot start |
| `D` | dirty tile |
| `C` | charging station |

The environment state is:

```text
(row, col, cleaned_mask, battery_value)
```

Battery capacities are generated from each map's shortest feasible route requirement, with separate training and evaluation profiles where needed.

## Maps

Important presets:

- `default`
- `charge_required_v2`
- `charge_maze_medium`
- `charge_maze_large`
- `multi_charge_detour`
- `complex_charge_labyrinth`
- `complex_charge_bastion`
- `complex_charge_switchback`

The `complex_charge_*` maps are the final hard benchmarks. `complex_charge_bastion` is the strongest stress test because many policies plateau at partial completion.

## Agents And Baselines

Implemented learning agents:

- `RandomAgent`
- `QLearningAgent`
- `SarsaAgent`
- `DQNAgent`
- `PPOAgent`
- `GuidedPolicyAgent`

Implemented structural baselines:

- deterministic shortest-route planner
- waypoint-level controller
- special-node graph environment

The planner and graph baselines are important because they separate map feasibility from learning difficulty. For example, `complex_charge_bastion` needs about 150 primitive moves but only 6 graph-level actions.

## Key Results

Final experiment summary:

- Q-learning solves `charge_required_v2`, but fails the complex maps in the 100k episode seed sweep.
- SARSA with charger-context abstraction solves `complex_charge_labyrinth` and `complex_charge_switchback`.
- `complex_charge_bastion` requires the combined guided exploration plus relay reward shaping SARSA variant in the final ablation.
- Existing DQN, PPO, SARSA, and guided references solve `complex_charge_bastion`.
- Reward shaping alone is not the root solution; representation and exploration structure matter more.

Current `complex_charge_bastion` reference comparison, 200 evaluation episodes:

| Algorithm | Cleaned | Success | Avg Steps |
| --- | ---: | ---: | ---: |
| Q-learning reference | 66.67% | 0% | 87 |
| DQN reference | 100% | 100% | 168 |
| PPO reference | 100% | 100% | 150 |
| SARSA reference | 100% | 100% | 155 |
| Guided policy | 100% | 100% | 150 |

See [docs/improvement_experiment_report.md](docs/improvement_experiment_report.md) for the full interpretation.

## Setup

Recommended Windows setup:

```powershell
.\scripts\setup_dev.ps1
```

CUDA setup:

```powershell
.\scripts\setup_dev.ps1 -Cuda
```

Manual install:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

CUDA-specific PyTorch packages are listed in `requirements-cuda.txt`.

## Common Commands

Run tests:

```powershell
.\scripts\test.ps1
```

Train Q-learning:

```powershell
.\.venv\Scripts\python.exe scripts\train_q_learning.py --map-name complex_charge_switchback --episodes 200000 --seed 42
```

Train SARSA:

```powershell
.\.venv\Scripts\python.exe scripts\train_sarsa.py --map-name complex_charge_bastion --episodes 100000 --seed 42 --state-abstraction-mode charger_context --guided-exploration-ratio 0.9 --reward-move-toward-relay-charger 0.5 --penalty-move-away-from-relay-charger -0.75 --eval-every 20000 --save-best-eval-checkpoint
```

Compare SARSA variants:

```powershell
.\.venv\Scripts\python.exe scripts\compare_sarsa_variants.py --map-name complex_charge_bastion --episodes 100000 --seed 42
```

Evaluate current algorithm references:

```powershell
.\.venv\Scripts\python.exe scripts\compare_all_algorithms.py --map-name complex_charge_bastion --include all --eval-episodes 200
```

Evaluate planner baseline:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_planner.py --map-name complex_charge_bastion
```

Evaluate graph-state route:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_graph_plan.py --map-name complex_charge_bastion
```

Launch the pygame UI:

```powershell
.\.venv\Scripts\python.exe scripts\run_training_app.py
```

## Project Structure

```text
SweepAgent/
|-- agents/      agent implementations and checkpoint logic
|-- configs/     reward defaults and map presets
|-- docs/        reports, devlog, and session notes
|-- env/         grid and graph environments
|-- outputs/     generated checkpoints, logs, plots, and GIFs
|-- scripts/     training, evaluation, comparison, rendering, and setup entry points
|-- tests/       pytest coverage
|-- ui/          pygame training/playback UI
`-- utils/       shared builders, planner helpers, and controllers
```

## Generated Files

`outputs/` is for generated artifacts:

- checkpoints
- logs
- plots
- GIFs

These files are useful for local analysis, but they are not committed by default.

## Final Takeaway

The final project claim is:

> The complex charger maps are feasible, but primitive tabular RL struggles because the state/action formulation creates a long-horizon routing problem with sparse completion feedback. Planner and graph baselines show that the effective decision problem is much smaller. Reward shaping improves local behavior, but robust completion on `complex_charge_bastion` requires better exploration and representation structure.

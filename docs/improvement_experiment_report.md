# SweepAgent Improvement Experiment Report

Date: 2026-05-06

This document summarizes the follow-up experiments run after adding planner, waypoint, graph-state, guided-policy, feasibility, and evaluation-protocol improvements. The goal is not to present a formal paper, but to clarify what the experiments say about the current design limits and which improvement direction is most defensible.

## Summary

The main result is that the difficult maps are not simply "unsolvable by reinforcement learning." They are solvable, and in several cases learned agents can solve them. The issue is that primitive grid-action learning has a poor search structure for maps that require long-range routing, charger timing, and ordered subgoals.

The strongest interpretation is:

- Plain Q-learning over primitive actions does not scale to the complex charger maps in the current setup.
- A graph/waypoint formulation makes the same task dramatically smaller.
- Reward shaping alone is not enough for `complex_charge_bastion`.
- `complex_charge_bastion` becomes learnable when guided exploration and relay-aware shaping are combined.
- A deterministic guided heuristic is useful as a reference, but it is not a universal solver.

## Experiment Artifacts

Generated outputs are under `outputs/logs/`:

- `improvement_structural_baselines.csv`
- `improvement_guided_<map>_<profile>.csv`
- `improvement_q_batch_train_100k_summary.csv`
- `improvement_q_batch_eval_100k_results.csv`
- `improvement_q_batch_eval_100k_summary.csv`
- `improvement_sarsa_ablation_all_complex_100k.csv`
- `improvement_bastion_all_algorithms_200.csv`
- `improvement_bastion_all_algorithms_200.md`
- `improvement_experiment_summary.md`

## Experimental Setup

### Structural Baselines

These experiments compare the original primitive state/action formulation with deterministic or compressed alternatives:

- Primitive state upper bound: grid position x cleaned mask x battery value.
- Graph state upper bound: special node x cleaned mask x battery value.
- Planner baseline: shortest feasible route over special points.
- Guided policy: deterministic relay-aware local policy using existing DQN feature logic.

### Q-learning Seed Sweep

Q-learning was trained from scratch for 100,000 episodes on four maps:

- `charge_required_v2`
- `complex_charge_labyrinth`
- `complex_charge_bastion`
- `complex_charge_switchback`

Seeds: `111`, `222`, `333`

Each checkpoint was evaluated greedily for 200 episodes.

### SARSA Ablation

SARSA was trained for 100,000 episodes on the three complex charger maps:

- `complex_charge_labyrinth`
- `complex_charge_bastion`
- `complex_charge_switchback`

Variants:

- `plain`: charger-context state abstraction, no guided exploration, no relay shaping.
- `shaping`: relay reward shaping only.
- `guided`: guided exploration only.
- `guided_shaping`: guided exploration plus relay reward shaping.

Each best checkpoint was re-evaluated for 200 episodes.

## Structural Results

| Map | Primitive State Upper Bound | Graph State Upper Bound | Planner Steps | Graph Actions | Guided Success | Guided Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `complex_charge_labyrinth` | 105,984 | 5,152 | 72 | 5 | 0% | 432 |
| `complex_charge_bastion` | 103,488 | 3,168 | 150 | 6 | 100% | 150 |
| `complex_charge_switchback` | 76,616 | 2,928 | 84 | 4 | 100% | 84 |

The graph-state formulation reduces the state upper bound by roughly one to two orders of magnitude on the complex maps. More importantly, the number of high-level decisions is tiny: 5-6 graph actions are enough for the complex maps where primitive agents may need 72-150 low-level moves.

This supports the view that the core difficulty is not only reward design. The primitive policy must discover a long ordered route through sparse, delayed feedback. A waypoint or graph-level policy faces a much shorter planning horizon.

## Guided Heuristic Results

The guided policy is useful but not sufficient by itself.

| Map | Planner Steps | Guided Success | Guided Steps |
| --- | ---: | ---: | ---: |
| `charge_required_v2` | 15 | 100% | 15 |
| `charge_maze_medium` | 34 | 100% | 34 |
| `charge_maze_large` | 72 | 0% | 432 |
| `multi_charge_detour` | 48 | 100% | 70 |
| `complex_charge_labyrinth` | 72 | 0% | 432 |
| `complex_charge_bastion` | 150 | 100% | 150 |
| `complex_charge_switchback` | 84 | 100% | 84 |

The guided heuristic exactly matches the planner on `bastion` and `switchback`, but fails on `charge_maze_large` and `labyrinth`, where it stalls at partial coverage. This means the heuristic should be treated as a strong baseline and exploration prior, not as a replacement for learning or planning.

## Q-learning Seed Sweep

100k training episodes, 3 seeds, 200 evaluation episodes per checkpoint:

| Map | Runs | Mean Success | Mean Cleaned | Mean Steps | Mean Recharges |
| --- | ---: | ---: | ---: | ---: | ---: |
| `charge_required_v2` | 3 | 100% | 100% | 18.67 | 0.00 |
| `complex_charge_labyrinth` | 3 | 0% | 25.00% | 69.67 | 0.67 |
| `complex_charge_bastion` | 3 | 0% | 11.11% | 588.00 | 19.61 |
| `complex_charge_switchback` | 3 | 0% | 66.67% | 144.00 | 2.00 |

Q-learning handles the smaller charger map, but fails across all three complex maps. The failure modes differ:

- `labyrinth`: reaches only one quarter of the dirt on average.
- `switchback`: reaches two thirds of the dirt, but cannot complete the full route.
- `bastion`: performs worst, with very low cleaned ratio and many recharge events.

This result supports the claim that primitive tabular Q-learning is not a good fit for the complex maps under the current state/action design.

## SARSA Ablation

100k training episodes, seed 42, best checkpoint re-evaluated for 200 episodes:

| Map | Variant | Best Episode | Eval Steps | Eval Cleaned | Eval Success |
| --- | --- | ---: | ---: | ---: | ---: |
| `complex_charge_labyrinth` | plain | 80,000 | 72 | 100% | 100% |
| `complex_charge_labyrinth` | shaping | 60,000 | 72 | 100% | 100% |
| `complex_charge_labyrinth` | guided | 60,000 | 88 | 100% | 100% |
| `complex_charge_labyrinth` | guided_shaping | 40,000 | 78 | 100% | 100% |
| `complex_charge_switchback` | plain | 80,000 | 85 | 100% | 100% |
| `complex_charge_switchback` | shaping | 60,000 | 85 | 100% | 100% |
| `complex_charge_switchback` | guided | 40,000 | 84 | 100% | 100% |
| `complex_charge_switchback` | guided_shaping | 20,000 | 84 | 100% | 100% |
| `complex_charge_bastion` | plain | 40,000 | 143 | 66.67% | 0% |
| `complex_charge_bastion` | shaping | 100,000 | 87 | 66.67% | 0% |
| `complex_charge_bastion` | guided | 100,000 | 87 | 66.67% | 0% |
| `complex_charge_bastion` | guided_shaping | 100,000 | 156 | 100% | 100% |

The ablation gives the clearest improvement story.

`labyrinth` and `switchback` are learnable with SARSA once charger-context abstraction is used. Reward shaping and guided exploration mainly affect convergence speed or route length, not final solvability.

`bastion` is different. Neither shaping alone nor guided exploration alone solves it. Both get stuck at 66.67% cleaned. The combined `guided_shaping` variant reaches 100% success. This suggests that `bastion` requires both a better exploration prior and reward signals that stabilize the relay-charger behavior.

## Bastion Reference Comparison

200 evaluation episodes under the shaped reward profile:

| Algorithm | Reference | Avg Steps | Cleaned | Success | Avg Reward |
| --- | --- | ---: | ---: | ---: | ---: |
| Q-learning | `q_learning_battery_adapt_400k_seed505` | 87 | 66.67% | 0% | -100.50 |
| DQN | `dqn_seed418_v2relay_shape_finalroute` | 168 | 100% | 100% | -83.50 |
| PPO | `ppo_finalrelay_curriculum6500` | 150 | 100% | 100% | -53.00 |
| SARSA | `sarsa_guided09_relay100k` | 155 | 100% | 100% | -43.50 |
| Guided | `relay_guided_policy` | 150 | 100% | 100% | -53.00 |

The best existing `bastion` references show that the map is solvable by learned or guided policies. However, the Q-learning reference still fails at 66.67% coverage, which is consistent with the new seed sweep and ablation results.

## Interpretation

### 1. The main bottleneck is representation and horizon length.

The primitive environment forces the agent to choose every grid move. On `bastion`, the successful route is around 150 primitive steps, but only 6 graph actions. The graph formulation exposes the real decision structure: which dirty tile or charger to target next.

This is why the planner and graph baseline are so informative. They show that the map itself is feasible, but the primitive action formulation creates an unnecessarily long credit-assignment problem.

### 2. Reward shaping helps, but it is not the root solution.

The `bastion` SARSA ablation is the key evidence:

- `plain`: fails.
- `shaping`: still fails.
- `guided`: still fails.
- `guided_shaping`: succeeds.

If reward shaping alone were enough, the shaping variant would solve `bastion`. It does not. The successful variant combines guided exploration with relay-aware reward shaping, so the improvement is better understood as an exploration-and-representation fix.

### 3. The guided heuristic is strong but brittle.

The guided policy solves `bastion` and `switchback`, but fails on `labyrinth` despite `labyrinth` being easy for SARSA plain. That means the heuristic encodes useful domain knowledge but can choose poor local decisions on some layouts.

For future work, guided logic should be used as one of:

- an exploration prior,
- an imitation target,
- a fallback baseline,
- or a planner-generated curriculum signal.

It should not be treated as the final policy architecture.

### 4. Bastion is the best stress test.

`labyrinth` and `switchback` are complex but learnable with the current SARSA abstraction. `bastion` is harder because partial policies can reach two of three dirty tiles and still fail the full sequence. This creates a misleading plateau at 66.67% cleaned.

That plateau is useful for evaluation. It separates policies that can make local progress from policies that can complete the full recharge-aware route.

## Recommended Improvement Direction

The most defensible next design is a hierarchical or graph-level RL setup:

1. Keep the primitive environment for execution and visualization.
2. Train a high-level policy over special nodes: dirty tiles and chargers.
3. Use deterministic shortest-path control to execute each high-level action.
4. Evaluate against planner, guided heuristic, and primitive RL baselines.

This directly attacks the main issue found by the experiments: the long primitive horizon. It also gives cleaner state features and more interpretable failures.

For a smaller next step, SARSA with charger-context abstraction plus guided+relay shaping is the best current learned baseline for `bastion`.

## Limitations

These experiments are sufficient for a project report, but not for a final statistical claim.

- SARSA ablation used one seed: seed 42.
- Q-learning seed sweep used three seeds, but only 100k episodes.
- DQN and PPO were compared using existing reference checkpoints, not retrained in this experiment batch.
- Generated outputs under `outputs/` are experiment artifacts and are not intended to be committed by default.

## Report Claim

A concise claim for the final report:

> The complex charger maps are feasible, but primitive tabular RL struggles because the state/action formulation creates a long-horizon routing problem with sparse completion feedback. Planner and graph baselines show that the effective decision problem is much smaller. Reward shaping improves local behavior, but `bastion` requires a combination of guided exploration and relay-aware shaping to consistently complete the map. Therefore, future improvements should focus on hierarchical or graph-level representations rather than only tuning primitive-action rewards.

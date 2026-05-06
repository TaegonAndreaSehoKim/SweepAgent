# SweepAgent Experiment Report Draft

## 1. Project Goal

This project studies reinforcement learning agents in a grid-based cleaning
environment with battery constraints and charging stations.

The agent must:

- move through a wall-bounded grid
- clean all dirty tiles
- manage limited battery
- recharge at charger tiles
- avoid running out of battery before completing the task

At first, this looks like a small gridworld problem.
However, the harder maps show that the task is closer to a resource-constrained
planning problem than a simple navigation task.

The main question became:

> Can common reinforcement learning methods learn long-horizon cleaning routes
> that require charger sequencing, or do they need stronger planning guidance?

The experiments compare four algorithm families:

- Q-learning
- SARSA
- DQN
- PPO

The main focus is not only whether an agent eventually succeeds, but also how
stable the success is across maps, seeds, and training recipes.

---

## 2. Environment Design

The environment is a grid map with the following symbols:

| symbol | meaning |
| --- | --- |
| `#` | wall |
| `.` | open cell |
| `R` | robot start |
| `D` | dirty tile |
| `C` | charger |

The environment state is:

```text
(row, col, cleaned_mask, battery_value)
```

The task terminates when:

- all dirty tiles are cleaned
- the battery is depleted
- the step limit is reached

The hard maps are difficult because successful behavior requires more than
moving toward the closest dirty tile. The agent often has to visit chargers in a
specific order before the final dirty tile becomes reachable.

This creates a long-horizon credit assignment problem: a bad charger choice may
only cause failure many steps later.

---

## 3. Algorithms Compared

### Q-learning

Q-learning is the off-policy tabular baseline.
It learns state-action values and updates toward the best next action.

The strongest Q-learning variants use charger-aware state abstraction so that
the Q-table can generalize across battery and charger contexts more effectively.

### SARSA

SARSA is the on-policy tabular comparison.
It updates using the next action selected by the current behavior policy.

The best SARSA results use:

- `charger_context` state abstraction
- guided exploration
- relay-aware reward shaping

SARSA became one of the most stable methods on the hard charge maps.

### DQN

DQN uses a neural network to approximate action values from map-derived state
features.

The implementation includes:

- action masking
- feature encoding for battery, dirty tiles, chargers, and route context
- guided exploration
- replay buffer
- target network
- best-eval checkpoint tracking

DQN can solve some hard maps, but the results are sensitive to exploration
schedule and seed.

### PPO

PPO is the policy-gradient baseline.

The implementation includes:

- actor-critic networks
- clipped PPO objective
- generalized advantage estimation
- action masking
- curriculum stages
- guided behavior-cloning warm-start
- DAgger-style guided relabeling

PPO can solve some hard maps when heavily guided, but it often struggles to keep
a stable full-map greedy policy on the hardest charger-sequencing tasks.

---

## 4. Why The Hard Maps Matter

The hard charge maps are not difficult only because they are larger.
They are difficult because they require a correct sequence of:

```text
dirty tile -> charger -> dirty tile -> charger -> charger -> final dirty tile
```

In other words, the agent must solve both:

- where to go next
- when a charger is not just a recharge point, but a relay point for future reachability

This matters because standard model-free RL does not explicitly plan over future
waypoints.
It must discover useful long-horizon routes through reward feedback.

The project therefore exposes a useful limitation:

> When the map requires explicit resource planning, pure or weakly guided
> model-free RL becomes sample-inefficient and unstable.

This is not necessarily a failure of the project.
It is one of the main findings.

---

## 5. Bastion Reference Comparison

`complex_charge_bastion` was the first major hard-map benchmark.
It requires charger relay behavior that the early agents did not learn reliably.

The current shared comparison evaluates Q-learning, DQN, PPO, and SARSA under
the same evaluation profile for `200` greedy episodes.

| algorithm | reference | cleaned ratio | success rate | average steps | average reward |
| --- | --- | ---: | ---: | ---: | ---: |
| Q-learning | battery-adapt 400k seed 505 | 66.67% | 0.00% | 87 | -100.50 |
| DQN | seed 418 relay-shape final route | 100.00% | 100.00% | 168 | -83.50 |
| PPO | final-relay curriculum 6500 | 100.00% | 100.00% | 150 | -53.00 |
| SARSA | guided09 relay 100k | 100.00% | 100.00% | 155 | -43.50 |

### Interpretation

The important point is not only that some methods solved `bastion`.
The important point is what was required to solve it.

DQN, PPO, and SARSA all needed some combination of:

- guided exploration
- charger relay awareness
- reward shaping
- curriculum or best-eval checkpoint selection

This suggests that `bastion` is not a simple gridworld task for generic
model-free RL.
It is a planning-heavy task where the agent needs help discovering the correct
charger sequence.

PPO found the shortest route among the compared learned agents at `150` steps.
SARSA was slightly longer at `155` steps but had the best average reward under
the current reward accounting.
DQN solved the map but used a longer route at `168` steps.

Q-learning remained stuck at partial completion under the stricter evaluation
battery profile.

---

## 6. Cross-map Fresh Baseline

After the `bastion` result, the next question was whether the observed
difficulty was specific to one map.

The non-`bastion` complex charge maps were rerun from fresh checkpoints:

- `complex_charge_labyrinth`
- `complex_charge_switchback`

The first fresh comparison used seed `604`.

| map | Q-learning | DQN | PPO | SARSA |
| --- | ---: | ---: | ---: | ---: |
| `complex_charge_labyrinth` | 100%, 80 steps | 25%, failed | 50%, failed | 100%, 72 steps |
| `complex_charge_switchback` | 100%, 97 steps | 100%, 84 steps | 100%, 84 steps | 100%, 84 steps |

### Interpretation

This result separated the maps clearly.

`switchback` was solved by all four algorithm families.
It appears to be a hard but stable benchmark.

`labyrinth` was solved by the tabular methods but not by the neural methods.
This suggested that `labyrinth` exposes a harder long-horizon sequencing problem
for DQN and PPO.

---

## 7. Seed Check

To test whether the fresh seed `604` result was just a seed accident, the same
comparison was repeated with seeds `605`, `606`, and `607`.

Each result below is from `200` greedy evaluation episodes.

| map | algorithm | seed 605 | seed 606 | seed 607 |
| --- | --- | ---: | ---: | ---: |
| `complex_charge_labyrinth` | Q-learning | 100%, 80 steps | 100%, 74 steps | 100%, 80 steps |
| `complex_charge_labyrinth` | DQN | 75%, 0% success | 75%, 0% success | 50%, 0% success |
| `complex_charge_labyrinth` | PPO | 75%, 0% success | 100%, 78 steps | 75%, 0% success |
| `complex_charge_labyrinth` | SARSA | 100%, 72 steps | 100%, 72 steps | 100%, 72 steps |
| `complex_charge_switchback` | Q-learning | 100%, 88 steps | 100%, 88 steps | 100%, 89 steps |
| `complex_charge_switchback` | DQN | 100%, 84 steps | 100%, 84 steps | 100%, 86 steps |
| `complex_charge_switchback` | PPO | 100%, 84 steps | 100%, 86 steps | 100%, 84 steps |
| `complex_charge_switchback` | SARSA | 100%, 84 steps | 100%, 84 steps | 100%, 84 steps |

### Interpretation

The seed check supports the map-structure hypothesis.

`switchback` is consistently solved across all checked seeds and algorithms.

`labyrinth` is consistently solved by Q-learning and SARSA, but not consistently
solved by DQN and PPO.

The most stable `labyrinth` result is SARSA:

- `100%` success on every checked seed
- stable `72`-step route

PPO solved `labyrinth` on seed `606`, but failed on seeds `605` and `607`.
DQN did not solve `labyrinth` in any of the seed-check runs.

This means the neural methods are not fundamentally incapable, but they are not
stable under the current recipes.

---

## 8. Labyrinth Adjustment Experiments

The next experiment tested whether DQN/PPO failures on `labyrinth` could be
fixed by changing the suspected weak points.

The hypothesis was:

- DQN might need longer high-exploration training
- PPO might need more full-map training or stronger behavior cloning

Adjustment runs used `labyrinth` only, with seeds `604` and `605`.

| variant | seed 604 | seed 605 |
| --- | ---: | ---: |
| DQN, guided `0.9`, 8000 episodes | 25%, 0% success | 25%, 0% success |
| DQN, slower epsilon, guided `0.8`, 8000 episodes | 100%, 76 steps | 50%, 0% success |
| PPO, longer full stage, 9500 episodes | 50%, 0% success | 75%, 0% success |
| PPO, stronger BC, 9500 episodes | 75%, 0% success | 75%, 0% success |

### Interpretation

The DQN slower-epsilon run is important.
It solved `labyrinth` on seed `604` with a `76`-step route.

This proves that DQN can solve the map with the current feature representation.
However, the same recipe failed on seed `605`, so the method is not yet stable.

The PPO adjustments improved partial completion but did not produce a successful
full-map greedy policy.
The best PPO adjustment reached `75%` cleaned, but still had `0%` success.

The adjustment experiments support this conclusion:

> Neural success on `labyrinth` is tunable, but not robust.

---

## 9. Main Findings

### Finding 1: The hard maps are planning-heavy

The hardest maps are not difficult only because of grid size.
They require long-horizon charger sequencing.

This makes them hard for model-free RL because the useful reward signal often
arrives long after the important decision.

### Finding 2: Switchback is stable

`complex_charge_switchback` is solved by all current algorithm families across
the checked seeds.

It is a useful hard-map success case.

### Finding 3: Labyrinth exposes neural instability

`complex_charge_labyrinth` is reliably solved by Q-learning and SARSA, but DQN
and PPO are unstable.

This suggests that the current tabular abstraction captures the charger context
more directly than the current neural recipes.

### Finding 4: Bastion requires planning priors

The `bastion` map can be solved, but only after adding relay-aware guidance,
reward shaping, curriculum, or best-eval checkpoint selection.

This is evidence that the environment formulation is challenging for generic
model-free RL.

### Finding 5: More tuning is not the only next step

The results suggest that the next meaningful improvement is not just more
hyperparameter tuning.

A more principled direction would be a hybrid approach:

- RL for policy learning
- graph search for route feasibility
- waypoint or option-level actions
- explicit charger/dirty planning features
- hierarchical control over dirty and charger targets

---

## 10. Design Reflection

A reasonable concern is:

> If reinforcement learning struggles to clear a map like `bastion`, is this a
> basic design problem?

The answer is partly yes.

If the goal is to train a practical cleaning agent with generic model-free RL,
then the current hard maps reveal a task-formulation problem.
The agent is being asked to infer a resource-constrained route plan from sparse
trial-and-error feedback.

That is inefficient.

However, if the goal is to compare how RL methods behave under long-horizon
charger constraints, then the hard maps are useful.
They expose exactly where the methods break:

- weak exploration
- unstable greedy policies
- failure to learn charger relay behavior
- partial-route learning without full-route completion

So the project should not claim that plain RL easily solves this problem.
The more honest conclusion is:

> These hard maps reveal the limits of model-free RL in a small but planning-heavy
> environment. The successful agents require planning-like guidance, and the
> remaining failures suggest that future work should combine RL with explicit
> route planning.

---

## 11. Recommended Report Framing

The final report should avoid presenting the project as a simple leaderboard.
The more interesting story is the progression of the diagnosis.

Recommended structure:

1. Start with the environment and battery-constrained cleaning task.
2. Explain why charger maps create long-horizon planning requirements.
3. Present the algorithm families.
4. Show `bastion` as the first hard relay benchmark.
5. Show `switchback` as the stable solved hard map.
6. Show `labyrinth` as the neural-stability stress test.
7. Discuss why guidance, shaping, and curriculum were necessary.
8. Conclude that future work should move toward RL + planning hybrids.

The report should be direct about limitations.
The strongest claim is not that one algorithm is universally best.

The strongest claim is:

> Map structure strongly affects whether the current RL recipes succeed. Tabular
> charger-context methods are surprisingly stable on some hard maps, while neural
> methods need more guidance and remain seed-sensitive on long-horizon charger
> sequencing tasks.

---

## 12. Limitations

These experiments still have several limitations.

- The checked seed count is small.
- Many results depend on best-eval checkpoint selection.
- DQN and PPO use guided components, so they are not pure baselines.
- Generated logs/checkpoints are local artifacts and not committed.
- The reward shaping is hand-designed for charger behavior.
- The current state/action interface still forces low-level step-by-step actions
  instead of higher-level waypoint choices.

These limitations do not invalidate the results.
They clarify the scope of the conclusion.

The current conclusion is about practical learnability in this environment, not
about universal algorithm ranking.

---

## 13. Next Work

The most promising next direction is a hybrid design.

Possible improvements:

- add waypoint-level actions such as "go to dirty tile" or "go to charger"
- use shortest-path planning for route feasibility
- train a high-level policy over dirty and charger targets
- keep low-level movement deterministic with BFS/A* path following
- compare model-free RL against a planner baseline
- evaluate whether reward shaping is still needed when planning information is explicit

This would better match the actual structure of the problem.

The hard maps show that SweepAgent is less about local movement and more about
choosing the right sequence of resource-constrained goals.

That is a planning problem, and the next version should treat it that way.

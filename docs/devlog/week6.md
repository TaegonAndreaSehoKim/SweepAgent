# Week 6 Devlog

## Summary

This phase moved from the single-map `complex_charge_bastion` comparison to a
cross-map hypothesis check.

The working question was:

- are the observed algorithm differences mostly seed noise
- or do the hard maps expose different planning failure modes

The useful result is that the difference is structural enough to document.

`complex_charge_switchback` is stable across the current algorithm families.
`complex_charge_labyrinth` is the sharper stress test: tabular methods solve it
reliably, while DQN and PPO are seed-sensitive and often stop at partial
completion.

---

## 1. Fresh Cross-map Baseline

The first step was to rerun the non-`bastion` complex charge maps from fresh
checkpoints rather than reusing older artifacts.

Scope:

- maps: `complex_charge_labyrinth`, `complex_charge_switchback`
- seed: `604`
- algorithms: Q-learning, DQN, PPO, SARSA
- evaluation profile: `evaluation`
- evaluation episodes: `200`

Result:

| map | Q-learning | DQN | PPO | SARSA |
| --- | ---: | ---: | ---: | ---: |
| `complex_charge_labyrinth` | 100%, 80 steps | 25%, failed | 50%, failed | 100%, 72 steps |
| `complex_charge_switchback` | 100%, 97 steps | 100%, 84 steps | 100%, 84 steps | 100%, 84 steps |

This immediately separated the two maps.

`switchback` was solved by all four algorithm families.
`labyrinth` was solved by the tabular methods but not by the neural methods.

---

## 2. Hypothesis

The working hypothesis became:

- `switchback` has a hard but learnable charger order that the current guided
  recipes expose reliably
- `labyrinth` requires a longer or more fragile charger/dirty sequencing pattern
- the current neural recipes can learn partial routes on `labyrinth`, but often
  fail to connect the full route into a stable greedy policy
- tabular `charger_context` policies are currently more reliable for this map

That hypothesis is narrower than "DQN/PPO are weak."
DQN and PPO both solve `switchback`, and PPO solves one `labyrinth` seed.
The issue is the interaction between `labyrinth` and the current neural
training schedules.

---

## 3. Seed Check

The first validation step repeated the same comparison with seeds `605`, `606`,
and `607`.

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

This supports the original read:

- `switchback` is consistently solved
- `labyrinth` is consistently solved by Q-learning and SARSA
- `labyrinth` remains unstable for DQN and PPO
- SARSA is the strongest `labyrinth` reference, with a stable 72-step route

---

## 4. Labyrinth Adjustment Experiments

The next step tested whether the `labyrinth` neural failures could be fixed by
changing the hypothesis-specific knobs:

- DQN: increase guided exploration or keep exploration high longer
- PPO: lengthen the final full-map stage or strengthen behavior cloning

Adjustment runs used `labyrinth` only, with seeds `604` and `605`.

| variant | seed 604 | seed 605 |
| --- | ---: | ---: |
| DQN, guided `0.9`, 8000 episodes | 25%, 0% success | 25%, 0% success |
| DQN, slower epsilon, guided `0.8`, 8000 episodes | 100%, 76 steps | 50%, 0% success |
| PPO, longer full stage, 9500 episodes | 50%, 0% success | 75%, 0% success |
| PPO, stronger BC, 9500 episodes | 75%, 0% success | 75%, 0% success |

The DQN slow-epsilon run is important because it proves that neural success on
`labyrinth` is possible under the current feature representation.
But the same recipe did not reproduce on seed `605`, so it is not yet stable.

The PPO adjustments improved partial completion but did not produce a successful
full-map greedy policy.
The strongest PPO adjustment reached `75%` cleaned but still had `0%` success.

---

## 5. Interpretation For The Report

The report should frame the result as a map-structure effect.

`switchback` is a good "solved hard map" benchmark because all current
algorithm families solve it across the checked seeds.

`labyrinth` is a better stress test for long-horizon stability.
It exposes a clear split:

- tabular `charger_context` methods: stable full solutions
- DQN: can solve with one adjusted seed, but is not stable
- PPO: can learn partial or staged behavior, but often fails full-map greedy
  completion

This supports a defensible claim:

> The performance gap is not explained only by algorithm class or map size.
> It depends on how each map exposes the charger sequencing problem. `switchback`
> presents a sequence that all current methods can learn, while `labyrinth`
> requires a longer-horizon policy that is currently handled most reliably by
> tabular charger-context learning.

---

## 6. Generated Outputs

The experiment launcher and generated logs/checkpoints were kept under
`outputs/`, which remains ignored by git.

Key local outputs:

- `outputs/logs/hypothesis_seedcheck_with_comparisons.csv`
- `outputs/logs/hypothesis_adjustment_with_comparisons.csv`
- `outputs/logs/algorithm_comparison_*seedcheck*eval_200.md`
- `outputs/logs/algorithm_comparison_complex_charge_labyrinth_lab_*eval_200.md`

These files are useful local evidence for the report but are not committed.

---

## Next Steps

The next practical work is report writing.

The most useful report structure is:

- introduce `switchback` as the stable solved hard-map case
- introduce `labyrinth` as the harder stability case
- use the seed check table to show the pattern is not a single-seed accident
- use the adjustment table to show DQN/PPO failures are tunable but not stable
- keep `bastion` as the separate relay-map reference comparison

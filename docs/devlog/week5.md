# Week 5 Devlog

## Summary

This phase closed the main `complex_charge_bastion` DQN failure that remained at the end of week 4.

The important result was not another reward tweak.
It was a narrower diagnosis:

- the learned policy could already clean the first one or two dirty tiles
- it could already recharge repeatedly without dying
- it still failed the final charger-region handoff needed to reach the last left-side dirty tile

That diagnosis led to one focused change:

- relay-aware charger guidance during guided exploration

The first probe run after that change reached deterministic greedy success on the evaluation battery profile.

---

## 1. Starting Point

The previous DQN work had already added:

- action masking
- guided exploration
- target dirty and charger context
- route-via-charger feature version 2
- periodic evaluation and best-eval checkpoint saving

Those changes were enough to improve `complex_charge_bastion` from:

- `0/3` cleaned

to:

- `2/3` cleaned

but they still did not solve the last dirty tile under the stricter evaluation battery.

The most recent long run also showed that the final checkpoint could regress badly even when a better checkpoint appeared earlier in training.

---

## 2. Resume And Best-Eval Tracking

Before changing behavior again, the long `v2best` run was cleaned up and resumed correctly.

The important practical pieces were:

- keep a separate best-eval checkpoint
- resume from that checkpoint instead of restarting from scratch
- pass `--starting-checkpoint-episodes` explicitly when the best-eval filename does not encode episode count
- use a detached launcher so the training process is not killed by an interactive console close event

That made the experiment flow much more reliable and made later comparison possible.

---

## 3. What The Failed Policies Were Actually Doing

The first useful correction in this phase was analytical.

At first glance, the old `bastion` DQN checkpoints looked like they were failing because they did not recharge correctly.
That was wrong.

Step-by-step rollout inspection showed:

- the old `best_eval` checkpoint recharged `10` times
- the old final `ep_10000` checkpoint recharged `11` times
- both policies still terminated by `step_limit`

So the problem was not "the agent ignores chargers."
The real problem was:

- the agent kept reusing the upper charger region
- it never committed to the relay transition needed to reach the lower charger and then the last left-side dirty tile

That changed the engineering target from vague "better route planning" to a much more specific failure mode.

---

## 4. State-Search Sanity Check

To make sure the evaluation battery was not the real blocker, the map was checked with a direct state search.

For `complex_charge_bastion`:

- evaluation battery: `65`
- minimum solvable battery: `45`

and a feasible route exists at battery `65`:

`R -> D(top-right) -> C(upper) -> D(bottom-right) -> C(upper) -> C(lower) -> D(left)`

This mattered because it proved the project was not stuck on an impossible target.
The missing behavior was specifically the charger-to-charger relay.

---

## 5. Why Guided Exploration Was Still Blind

The earlier guided exploration logic only had two main ideas:

- go toward a currently safe dirty tile
- otherwise go toward a charger

That was not enough for `bastion`.

When no remaining dirty tile was directly battery-safe from the current position, the policy often reduced distance to the nearest charger.
On `bastion`, that made it easy to keep orbiting the same upper charger instead of transferring to the lower one.

In other words, the guidance logic knew how to:

- chase a safe dirty tile
- retreat to a charger

but it did **not** know how to:

- choose a charger that unlocks the next safe dirty tile after recharging

That missing concept was exactly the relay behavior the map required.

---

## 6. Relay-Aware Charger Guidance

The DQN encoder was updated so guided exploration can now evaluate candidate chargers by future usefulness.

The new helper:

- checks which chargers are reachable with the current battery
- asks how many remaining dirty tiles become battery-safe from each charger after recharge
- prefers chargers that unlock more reachable remaining dirty tiles
- breaks ties with route cost and charger distance

`guided_action()` now uses that relay charger target whenever:

- no remaining dirty tile is directly safe from the current state

instead of blindly walking toward the nearest charger.

This is still guided exploration, not a hardcoded evaluation route.
Greedy evaluation remains learned policy behavior.

---

## 7. Probe Result

Rather than launching another long run immediately, the change was tested with a narrow resume probe:

```bash
python scripts/train_dqn.py --map-name complex_charge_bastion --battery-profile evaluation --episodes 1000 --seed 417 --print-every 500 --init-checkpoint outputs/checkpoints/dqn_agent_complex_charge_bastion_best_eval_seed_417_v2best.pt --starting-checkpoint-episodes 1000 --learning-rate 0.0005 --epsilon-decay 0.99995 --epsilon-min 0.10 --learning-starts 1000 --batch-size 128 --replay-capacity 100000 --train-every 8 --target-update-interval 1000 --hidden-size 256 --guided-exploration-ratio 0.6 --eval-episodes 50 --eval-every 500 --feature-version 2 --device cuda --checkpoint-tag v2relayprobe --save-best-eval-checkpoint
```

This resumed from the old 1000-episode `v2best` best-eval checkpoint and trained only `1000` more episodes.

### Result

At cumulative episode `2000`, the new best-eval checkpoint reached:

- `avg_reward = -180.00`
- `avg_steps = 222`
- `avg_cleaned_ratio = 100.00%`
- `success_rate = 100.00%`

The resulting checkpoint was:

- `dqn_agent_complex_charge_bastion_best_eval_seed_417_v2relayprobe.pt`

and the final cumulative checkpoint was:

- `dqn_agent_complex_charge_bastion_ep_2000_seed_417_v2relayprobe.pt`

---

## 8. Repro Check

The probe result was not left as a single logged number.

The new best checkpoint was re-evaluated on CPU with fixed RNG:

- `50` evaluation episodes: `100%` success
- `200` evaluation episodes: `100%` success

That strongly suggests the improvement is real for this map and seed, not a one-off evaluation fluctuation.

---

## 9. What The Solved Rollout Looks Like

A traced greedy rollout of the new relay-aware checkpoint showed the intended pattern clearly.

Clean events:

- step `39`: first dirty tile
- step `100`: second dirty tile
- step `222`: final left-side dirty tile

Recharge sequence:

- upper charger
- upper charger
- upper charger
- lower charger

Termination:

- `all_cleaned`

That is exactly the relay behavior the older checkpoints were missing.

---

## 10. Practical Conclusion

This phase changed the `bastion` story in a useful way.

### Before

- best tabular result: `2/3` cleaned on evaluation
- best DQN result: `2/3` cleaned on evaluation
- unclear whether the remaining gap was reward, representation, or route instability

### After

- the blocker was narrowed to charger-to-charger relay choice
- one targeted guided-exploration change was enough to produce full evaluation success

This does **not** prove the whole DQN recipe is now completely stable across seeds or very long runs.
It does prove that the previous plateau was not fundamental.

The project now has a concrete mechanism that can unlock the final `bastion` transition.

---

## 11. PPO Baseline

After the DQN route-planning work, PPO was added as the next learning-family baseline.

The PPO implementation uses:

- the same map-derived feature encoder used by DQN
- action masking for invalid moves
- actor-critic networks
- generalized advantage estimation
- clipped PPO policy updates
- entropy regularization
- `.pt` checkpoint save/load
- periodic evaluation and best-eval checkpoint selection

The first useful result was on the small `default` map.
A 3000-episode PPO run solved it reliably:

- best eval at episode `1000`
- `avg_reward = 95.00`
- `avg_steps = 13`
- `avg_cleaned_ratio = 100.00%`
- `success_rate = 100.00%`

That confirmed the implementation was capable of learning a simple cleaning task end to end.

The harder charge maps were different.

For `complex_charge_bastion`, a 5000-episode plain PPO run with evaluation battery never cleaned any dirty tile:

- best eval at episode `2500`
- `avg_reward = -156.50`
- `avg_steps = 65`
- `avg_cleaned_ratio = 0.00%`
- `success_rate = 0.00%`

For the intermediate `charge_maze_medium` map, plain PPO did better but still did not solve the task:

- best eval at episode `3500`
- `avg_reward = -81.00`
- `avg_steps = 29`
- `avg_cleaned_ratio = 33.33%`
- `success_rate = 0.00%`

The read is that PPO can learn the easy map and can find the first useful dirty tile on a medium charge map, but plain on-policy exploration is not enough for multi-dirty charger planning.

---

## 12. PPO Curriculum And Guided Warm-Start

Two PPO support mechanisms were then added:

- curriculum stages that mark selected dirty tiles as already cleaned
- guided behavior-cloning warm-start from the existing encoder guidance

The curriculum interface accepts dirty-tile index sets such as:

- `2`
- `1,2`
- `full`

and converts each stage into an environment reset configuration with the other dirty tiles precleaned.

The guided warm-start collects state/action pairs from `guided_action()` and trains the PPO actor with a supervised negative log-probability objective before PPO rollouts begin.
It can run once at the beginning or once per curriculum stage.

A combined 5000-episode test on `charge_maze_medium` used:

- stages: `2 -> 1,2 -> full`
- stage episodes: `1500 / 1500 / 2000`
- guided warm-start: `20` episodes per stage
- PPO rollout steps: `2048`
- update epochs: `4`
- minibatch size: `256`
- hidden size: `256`

This did not improve the final result over plain PPO.

Full-map greedy eval reached `1/3` cleaned at episodes `2500` and `3000`, but the final full-map stage regressed:

- best eval at episode `2500`
- `avg_reward = -83.00`
- `avg_steps = 29`
- `avg_cleaned_ratio = 33.33%`
- `success_rate = 0.00%`
- final eval at episode `5000`: `0/3` cleaned

The practical conclusion is that one-time warm-start is too weak.
The next PPO attempt should probably mix imitation loss into PPO updates, increase full-map guided data, or add a stage-transition safeguard that prevents losing the best full-map behavior.

---

## 13. Guided PPO Final-Relay Curriculum

The next PPO iteration added three things on top of curriculum and warm-start:

- mixed guided imitation loss during PPO updates
- DAgger-style labels collected from states visited by the current PPO policy
- periodic behavior-cloning updates on the accumulated DAgger buffer

For `complex_charge_bastion`, the useful curriculum was:

- keep dirty tile `2`
- keep dirty tiles `0,2`
- keep dirty tiles `1,2`
- full map

with stage episode counts:

- `800`
- `1200`
- `2000`
- `2500`

The successful run used:

- seed `42`
- guided warm-start: `20` episodes per stage
- guided imitation coefficient: `0.2`
- DAgger coefficient: `0.75`
- DAgger buffer capacity: `50000`
- periodic BC every `100` episodes
- BC epochs: `2`

The best-eval checkpoint solved the map:

- `avg_reward = -53.00`
- `avg_steps = 150`
- `avg_cleaned_ratio = 100.00%`
- `success_rate = 100.00%`

The result reproduced over `20`, `50`, and `200` CPU evaluation episodes.
The final checkpoint from the same run regressed to `2/3`, so comparisons should use the saved best-eval checkpoint:

- `ppo_agent_complex_charge_bastion_best_eval_seed_42_ppo_finalrelay_curriculum6500.pt`

This makes guided PPO the current policy-gradient reference for `bastion`.
Plain PPO remains a useful negative baseline because it still fails on the same map.

---

## 14. UI Training And Playback Integration

The pygame UI was extended beyond tabular Q-learning.

It now supports:

- `dqn` script-backed training
- `ppo` script-backed training
- `ppo_guided` training with the staged guided-imitation recipe
- `dqn_best` playback for saved DQN best-eval checkpoints
- `ppo_best` playback for saved PPO best-eval checkpoints

For `complex_charge_bastion`, the UI loader prefers the current reference checkpoints:

- DQN seed `418`, tag `v2relay_shape_finalroute`
- PPO seed `42`, tag `ppo_finalrelay_curriculum6500`

The UI training subprocess now uses unbuffered output, and DQN/PPO progress is reported every `250` episodes by default.
This avoids the earlier issue where a long DQN run looked stuck because no metrics appeared before episode `1000`.

A fresh UI DQN test was run on `complex_charge_bastion`:

- seed `421`
- episodes `7000`
- learning rate `0.0007`
- discount factor `0.995`
- epsilon decay `0.999965`
- epsilon minimum `0.12`
- guided exploration ratio `0.6`

That run did not solve the map.
The best-eval checkpoint at episode `4000` reached only:

- `avg_reward = -132.00`
- `avg_steps = 87`
- `avg_cleaned_ratio = 33.33%`
- `success_rate = 0.00%`

The final checkpoint regressed to `0/3` cleaned.
The read is that this specific DQN configuration still failed to learn the charger-relay chain after the first dirty tile.
The next DQN attempt should increase guided exploration exposure and maintain exploration longer, rather than treating this run as a solved setting.

---

## 15. SARSA Baseline And Guided Improvement

SARSA was added as the on-policy tabular comparison point.
The implementation reuses the existing Q-table serialization and state abstraction machinery, but updates with the selected next action:

`Q(s,a) <- Q(s,a) + alpha * (r + gamma * Q(s',a') - Q(s,a))`

Initial tests showed the baseline is sound:

- `default`, 10000 episodes: greedy eval `avg_reward=106.00`, `avg_steps=7`, `success_rate=100%`
- `charge_maze_medium`, 50000 episodes with `charger_context`: greedy eval `avg_reward=59.50`, `avg_steps=34`, `success_rate=100%`

Plain SARSA on `complex_charge_bastion` did not solve the hard relay map:

- 50000 episodes
- evaluation battery
- `charger_context`
- greedy eval `avg_reward=-166.50`, `avg_steps=87`, `avg_cleaned_ratio=33.33%`, `success_rate=0%`

The same hard-map pattern appeared again: the policy could reach part of the task, but did not learn the full charger relay chain.

SARSA was then extended with:

- relay-aware guided exploration via the shared `StateFeatureEncoder.guided_action()`
- relay charger reward shaping flags
- periodic greedy evaluation
- best-eval checkpoint saving

The successful `complex_charge_bastion` run used:

- seed `42`
- episodes `100000`
- learning rate `0.05`
- gamma `0.99`
- epsilon decay `0.99995`
- epsilon minimum `0.10`
- guided exploration ratio `0.6`
- relay progress reward `0.5`
- relay-away penalty `-0.75`
- checkpoint tag `guided_relay100k`

Greedy eval first reached full success at episode `70000`:

- `avg_reward = -44.00`
- `avg_steps = 156`
- `avg_cleaned_ratio = 100.00%`
- `success_rate = 100.00%`

Both the best-eval checkpoint and the final 100000-episode checkpoint reproduced the same result over 200 CPU evaluation episodes:

- `200/200` all-cleaned rollouts
- `avg_reward = -44.00`
- `avg_steps = 156`
- `success_rate = 100.00%`

This makes guided SARSA the current on-policy tabular reference for `bastion`.
It also confirms that the relay guidance and shaping are not only useful for DQN/PPO; they can unlock the hard map for a tabular on-policy method too.

Follow-up numeric tuning kept the same seed and schedule but changed only the guided exploration ratio.
`guided_exploration_ratio=0.9` improved the best greedy route slightly:

- checkpoint tag `guided09_relay100k`
- best eval at episode `80000`
- `avg_reward = -43.50`
- `avg_steps = 155`
- `avg_cleaned_ratio = 100.00%`
- `success_rate = 100.00%`

The best checkpoint reproduced the same `155`-step all-cleaned rollout over 200 CPU evaluation episodes.
Two nearby variants were worse:

- fine-tuning the `guided_relay100k` best checkpoint with `gamma=0.995`, lower epsilon, and `guided=0.8` regressed to `2/3` at best
- `guided=0.95` solved the map but returned to a `156`-step route
- `guided=0.9` with learning rate `0.03` failed to reach full greedy success

To make SARSA comparison repeatable, `scripts/compare_sarsa_variants.py` now runs a controlled ablation matrix:

- `plain`: no guided exploration and no relay shaping
- `shaping`: relay reward shaping only
- `guided`: relay-aware guided exploration only
- `guided_shaping`: guided exploration plus relay reward shaping

The default command is:

```bash
python scripts/compare_sarsa_variants.py --map-name complex_charge_bastion --episodes 100000 --seed 42
```

It keeps the same SARSA schedule used by the current best guided run, saves best-eval checkpoints for each variant, reevaluates the saved best checkpoint, and writes a CSV summary under `outputs/logs/`.

The 100000-episode `complex_charge_bastion` run with seed `42` produced:

| variant | best episode | cleaned ratio | success rate | average steps |
| --- | ---: | ---: | ---: | ---: |
| `plain` | 100000 | 66.67% | 0.00% | 87 |
| `shaping` | 100000 | 100.00% | 100.00% | 192 |
| `guided` | 90000 | 33.33% | 0.00% | 87 |
| `guided_shaping` | 80000 | 100.00% | 100.00% | 155 |

This isolates the important behavior:

- reward shaping alone can unlock the charger relay pattern, but with a longer route
- guided exploration alone does not solve `bastion`
- guided exploration plus relay shaping remains the best SARSA recipe and reproduces the 155-step successful route

## 16. Cross-Algorithm Bastion Comparison

`scripts/compare_all_algorithms.py` now evaluates the current reference checkpoints for Q-learning, DQN, PPO, and SARSA under one shared evaluation loop.
It writes both CSV and Markdown outputs so the same run can feed plotting, spreadsheet checks, and report text.

Default command:

```bash
python scripts/compare_all_algorithms.py --map-name complex_charge_bastion --eval-episodes 200
```

The current `complex_charge_bastion` reference comparison is:

| algorithm | reference | cleaned ratio | success rate | average steps | average reward |
| --- | --- | ---: | ---: | ---: | ---: |
| Q-learning | battery-adapt 400k seed 505 | 66.67% | 0.00% | 87 | -100.50 |
| DQN | seed 418 relay-shape final route | 100.00% | 100.00% | 168 | -83.50 |
| PPO | final-relay curriculum 6500 | 100.00% | 100.00% | 150 | -53.00 |
| SARSA | guided09 relay 100k | 100.00% | 100.00% | 155 | -43.50 |

This table is the first apples-to-apples comparison across the current algorithm families.
The strongest route length is still PPO at 150 steps, matching the known guided shortest route.
SARSA is slightly longer at 155 steps but has the best reward under the current reward accounting.
DQN solves the map but remains longer at 168 steps.
Q-learning remains capped at `2/3` under the evaluation battery profile.

Generated outputs:

- `outputs/logs/algorithm_comparison_complex_charge_bastion_eval_200.csv`
- `outputs/logs/algorithm_comparison_complex_charge_bastion_eval_200.md`

---

## Next Steps

The most useful next steps from here are:

- treat the relay-aware DQN seed `418` result as the current strongest `bastion` reference
- treat the guided PPO final-relay best-eval checkpoint as the current policy-gradient `bastion` reference
- treat the guided SARSA seed `42` tag `guided09_relay100k` checkpoint as the current on-policy tabular `bastion` reference
- use the SARSA ablation CSV when discussing whether guidance or reward shaping drove the hard-map improvement
- extend the Q-learning vs DQN vs PPO vs SARSA comparison to more seeds or maps if needed
- use the UI for quick visual checks of DQN, PPO, guided PPO, and saved best-eval checkpoints
- keep generated checkpoints, plots, logs, and GIFs out of version control

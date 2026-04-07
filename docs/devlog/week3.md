# Week 3 Devlog

## Summary

This phase focused on turning the three hardest charge-aware maps from unstable experiments into solvable training targets.
The main themes were reward redesign, battery slack redesign, direct-training experiments, and training-speed optimization.

---

## 1. Re-evaluating the Hard Maps

The first pass on the new hard maps showed that the project had not really solved them yet.

### What the checkpoints showed

- `complex_charge_labyrinth` was the closest to stable
- `complex_charge_switchback` often cleaned two of three dirty tiles, then died on battery
- `complex_charge_bastion` was the worst case and often failed after cleaning only one dirty tile

The main insight was that the earlier reward shaping had simply changed the failure mode.
Instead of dying immediately, the agent often learned recharge loops or weak local policies.

---

## 2. Reward Redesign for Charge-Aware Planning

Several reward changes were made to stop the agent from exploiting chargers and to push it toward actually completing maps.

### Main changes

- reduced charger-direction shaping
- reduced low-battery recharge reward
- added revisit penalties for general loop behavior
- added safe-dirty progress shaping
- added penalties for entering unrecoverable low-battery states

### Why this mattered

The earlier charger shaping made it too easy for the agent to treat charger movement itself as the goal.
The new shaping instead tries to answer a harder question:

"Can the agent move toward a dirty tile only when that move is still battery-safe?"

That change was the turning point for `complex_charge_switchback`.

---

## 3. Battery Capacity Rule Update

Battery sizing was revised again.

### Previous rule

- minimum solvable battery
- fixed `+5` margin

### New rule

- minimum solvable battery
- slack based on `max(min_margin, minimum_capacity * margin_ratio)`
- slack rounded up to the next fixed unit
- per-map override support when needed

This made large maps less dependent on fragile hand tuning and gave the hardest maps more realistic training slack.

---

## 4. Safety Reserve Logic

One of the biggest functional improvements was adding a battery safety reserve to charger decisions.

### New behavior

- safe-dirty movement now requires enough battery for:
  - current position to dirty tile
  - dirty tile to charger
  - extra safety reserve
- emergency return-to-charger mode now activates earlier
- unrecoverable-state penalties also use the same reserve logic

### Result

This reduced the number of cases where the agent committed to a dirty tile too late and then died one corridor short of recovery.

---

## 5. Hard Map Outcomes

By the end of this phase, the project moved from "interesting but unstable" to "actually solved under documented settings."

### `complex_charge_labyrinth`

- solved under the latest direct single-map training setup

### `complex_charge_switchback`

- solved after the safety-aware reward redesign
- greedy evaluation reached full cleaning consistently

### `complex_charge_bastion`

- remained the hardest map
- required:
  - slower epsilon decay
  - safety-aware shaping
  - larger battery slack during training
- finally solved in a `500000`-episode run with battery override `100`

This map ended up being the clearest example that harder charge-aware tasks needed both better reward signals and more training slack.

---

## 6. Epsilon Schedule Reassessment

Another important finding was that the previous exploration schedule was too aggressive for hard maps.

### Problem

With the older decay, epsilon hit its floor much earlier than the point where the hardest maps started to show meaningful success.

### Change

The shared defaults were updated to keep exploration alive longer:

- `epsilon_decay = 0.99999`
- `epsilon_min = 0.20`

This did not solve the maps by itself, but it prevented exploration from collapsing before the useful behaviors had a chance to emerge.

---

## 7. UI and Workflow Updates

The training UI was updated so experimentation became less constrained.

### Changes

- numeric hyperparameters moved from dropdowns to direct text input
- defaults were aligned with the long-run hard-map settings

This made it much easier to try larger episode counts and small hyperparameter adjustments without changing code.

---

## 8. Performance Optimization

After the hard maps became solvable, the next issue was wall-clock speed.

### Observed behavior

- CPU utilization looked low in Windows Task Manager
- GPU stayed unused

### Diagnosis

This was expected because the project still uses single-process tabular Q-learning.
The bottleneck is Python environment stepping, not tensor math.

### Optimizations added

- caching for safe-dirty lookup results
- a lean `step_training()` path that skips building full `info` payloads during each training step

These changes do not alter the algorithm, but they reduce per-step overhead in long training runs.

---

## 9. Main Takeaways

This phase changed the project in three important ways.

### First

The hardest maps are no longer just stress tests.
They now have working training recipes.

### Second

The biggest improvements did not come from changing the learning algorithm.
They came from:

- better reward semantics
- better battery slack rules
- better exploration schedules

### Third

Once the hard maps became solvable, runtime efficiency became the next practical bottleneck.

---

## 10. Follow-up Runtime Work

After the first optimization pass, the project moved from "reduce per-step overhead" to "use more of the machine during experiment runs."

### Additional optimization work

- added caching for safe-dirty lookup results inside the environment
- added a lighter training-only environment step path so training does not build a full `info` payload on every step

These changes did not radically change wall-clock time by themselves, but they cleaned up the hottest path and prepared the codebase for larger experiment batches.

### Parallel batch training

A new batch training script was added for practical CPU utilization:

- `scripts/train_q_batch.py`

This workflow:

- launches multiple `train_q_learning.py` runs as parallel subprocesses
- allows one command to sweep multiple maps and seeds
- writes one log file per run
- writes a CSV summary for the whole batch

### Why this matters

The current learning algorithm is still single-process tabular Q-learning.
That means one training run will not naturally saturate a modern multi-core machine.

The more practical solution was not to force one Q-learning run to use every core.
It was to make many independent experiment runs easy to launch at the same time.

That gives the project a better way to use available CPU resources for:

- seed sweeps
- map comparisons
- training recipe comparisons

---

## Next Steps

The most natural next steps from here are:

- measure actual wall-clock speed gains from the new environment optimizations
- benchmark the new batch runner on larger seed sweeps
- decide whether `complex_charge_bastion` should keep its larger battery override or receive a stricter evaluation preset
- compare multiple seeds systematically on the three hard maps
- decide whether to keep investing in tabular Q-learning or move to a neural approximation method for better scaling

---

## 11. Training vs Evaluation Battery Profiles

Once `complex_charge_bastion` became solvable with a larger training battery, it became important to separate "can the agent learn the route?" from "can the same policy survive a tighter battery budget?"

### Change

- split map battery settings into:
  - `battery_capacity_training`
  - `battery_capacity_evaluation`
- kept training scripts on the training profile
- kept playback and evaluation flows on the evaluation profile by default

### Why this mattered

This exposed an important gap:

- some checkpoints that looked fully solved under training conditions still failed completely under evaluation conditions

That made later tuning much more honest.

---

## 12. Batch Evaluation and Stability Checking

After adding batch training, the next gap was evaluation consistency.

### New workflow

- added `scripts/evaluate_q_batch.py`
- evaluates `map x seed x episode-count` checkpoint sets
- writes detailed and summary CSV files under `outputs/logs/`

### Why this mattered

This moved the project away from single-run interpretation.
It became easier to answer questions like:

- does a recipe work reliably across seeds?
- does it only solve under training battery settings?
- does a checkpoint fail by `step_limit` looping or by `battery_depleted`?

That evaluation pass confirmed that several hard-map checkpoints were still brittle once the stricter evaluation battery profile was applied.

---

## 13. Battery Adaptation Training

Because some policies learned well under the easier training battery but collapsed under the tighter evaluation battery, a two-stage adaptation workflow was added.

### New workflow

- pretrain on the training battery profile
- load that checkpoint
- finetune on the evaluation battery profile
- save the final checkpoint under the total combined episode count

This was added as:

- `scripts/train_q_battery_adapt.py`

and wired into the UI.

### UI updates

- added `battery_adapt_q_learning` as a selectable training mode
- kept high-precision numeric input for values like `0.999999`
- later updated the UI again so both `Stage1 Episodes` and `Stage2 Episodes` can be edited directly

### Why this mattered

This created a cleaner way to test whether the policy could first learn the route with slack, then adapt to the tighter deployment condition.

---

## 14. What Battery Adaptation Actually Improved

The battery-adapt workflow did not instantly solve `complex_charge_bastion`, but it did produce a meaningful behavioral change.

### Observed pattern

- pure `500000`-episode Q-learning often solved `bastion` under training battery settings
- those same checkpoints often failed completely under evaluation battery settings
- battery-adapt checkpoints still failed on evaluation in many runs, but some of them improved from:
  - cleaning `0/3` dirty tiles
  - to cleaning `1/3` or `2/3` dirty tiles before failure

### Interpretation

The two-stage process was meaningful, but not sufficient by itself.

It improved transfer to the stricter battery setting, especially in runs where pure Q-learning showed zero progress under evaluation.
But it also showed that `bastion` still needs better charger handoff and final-return planning under tight battery constraints.

---

## 15. Bastion-Specific Finding

The latest `bastion` experiments clarified the real bottleneck.

### What works

- under the larger training battery, the agent can learn a full cleaning route
- with good seeds, greedy evaluation reaches full success on the training profile

### What still fails

- under the evaluation battery, the agent often:
  - cleans one or two dirty tiles
  - reaches a charger correctly
  - then fails to execute the next long-range transition safely

### Practical conclusion

The remaining problem is no longer just "find any route."
It is "switch charger regions and time the return correctly under a much tighter battery budget."

That is a much narrower and more useful failure mode than the earlier recharge-loop behavior.

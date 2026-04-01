# Week 2 Devlog

## Summary

This stage focused on moving SweepAgent from a working charge-aware RL demo into a more flexible experiment platform.
The biggest themes were curriculum training, harder map design, UI iteration, and systematic battery sizing.

---

## Curriculum Learning

A dedicated curriculum flow was added so that the project no longer has to rely only on direct single-map training.

### What changed

- Added a curriculum training script
- Added resume-style training support through checkpoints
- Connected curriculum mode to the training UI
- Updated the curriculum flow so that:
  - stage 1 uses `charge_maze_medium`
  - stage 2 uses the map currently selected in the UI

### Why this matters

Earlier, curriculum training was effectively tied to one fixed stage-2 map.
Changing it so stage 2 follows the user-selected map makes curriculum mode much more reusable for experiments on the newer hard maps.

---

## New Complex Charge-Aware Maps

Three new maps were added to push the project beyond the earlier charge-aware layouts.

### Added maps

- `complex_charge_labyrinth`
- `complex_charge_bastion`
- `complex_charge_switchback`

These are larger and more constrained than the earlier examples.
They are intended to test whether the agent can plan over longer routes, use chargers more intelligently, and avoid getting trapped in low-value local behavior.

---

## Battery Capacity Rule Update

Battery sizing was changed from manual assignment to a systematic rule.

### New rule

For each map:

1. compute the minimum battery needed to make the map solvable
2. add a margin of `+5`

### Result

This rule now applies across both older and newer maps.
That makes preset design more consistent and makes future map creation easier.

---

## UI Improvements

The training UI received several fixes and cleanup passes.

### Improvements made

- dropdown placement was adjusted so controls stay inside the window more reliably
- dropdown click handling was fixed so option selection is not stolen by lower controls
- the curriculum option was integrated into the algorithm selector
- the training screen layout was simplified by removing the mini rollout preview

### Why the preview was removed

The mini rollout preview took a lot of space but usually did not add much value because training often advanced or finished before the panel became useful.
Removing it made the training page cleaner and let the important information stand out more clearly.

---

## Checkpoints and Current Results

Recent uploaded checkpoints include:

- `q_learning_agent_charge_maze_medium_ep_50000_seed_42.json`
- `q_learning_agent_complex_charge_labyrinth_ep_50000_seed_42.json`
- `q_learning_agent_complex_charge_bastion_ep_50000_seed_42.json`
- `q_learning_agent_complex_charge_switchback_ep_50000_seed_42.json`

### Current read on those checkpoints

- `charge_maze_medium` looks much healthier and more settled
- the three new complex maps are still substantially harder
- the hardest maps likely need more tuning, better curriculum staging, or more training to become consistently strong

This is still a useful outcome because it confirms that the new maps are genuinely challenging and gives the project a clear next target.

---

## Development Takeaways

The key insight from this phase is that harder charge-aware maps do not just need more episodes.
They also need:

- better curriculum structure
- careful battery design
- flexible UI control for experiments
- cleaner ways to compare checkpoints and map difficulty

That means the project is now in a stronger experimental state even though the hardest maps are not fully solved yet.

---

## Next Steps

The most natural next steps are:

- evaluate curriculum vs direct training on each new complex map
- document benchmark results for the three new maps
- improve the curriculum schedule for the hardest stage-2 maps
- add clearer experiment summaries so checkpoint quality is easier to compare

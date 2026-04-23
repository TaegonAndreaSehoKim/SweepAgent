# Week 4 Devlog

## Summary

This phase resumed after a one-week development pause caused by school exams.
The main technical shift was moving beyond tabular Q-learning and adding a PyTorch DQN baseline for the hardest charge-aware map, `complex_charge_bastion`.

The week did not fully solve `bastion`, but it made the next bottleneck much clearer:

- naive DQN could not reach any dirty tile reliably
- action masking removed invalid wall-action loops
- guided exploration made dirty-reaching trajectories appear during training
- richer target context improved greedy evaluation from `1/3` cleaned to `2/3` cleaned
- the final failure is now the last long-range return/transition to the remaining dirty tile

---

## 1. One-Week Development Pause

There was a one-week gap in active development because of school exams.

This mattered because the next task was not a small tuning pass.
The previous phase had reached a clear tabular Q-learning plateau, so the project needed a larger design step rather than more incremental reward shaping.

After the pause, the next direction was to test whether function approximation could break the `complex_charge_bastion` evaluation plateau.

---

## 2. Why DQN Was Added

The previous tabular work had a consistent pattern:

- `complex_charge_labyrinth` could be solved
- `complex_charge_switchback` could be solved
- `complex_charge_bastion` could be solved under the training battery profile
- `complex_charge_bastion` still plateaued at `2/3` cleaned under the stricter evaluation battery profile

That made the problem less about one bad seed or one bad reward value.
The tabular representation was probably not enough for the final charger-region handoff and long-range return planning.

So the project added a DQN path instead of continuing to add `bastion`-specific reward tweaks.

---

## 3. DQN Baseline Implementation

A new PyTorch agent was added:

- `agents/dqn_agent.py`

and a new training entry point:

- `scripts/train_dqn.py`

The DQN implementation includes:

- MLP Q-network
- target network
- Double-DQN target action selection
- replay buffer
- epsilon-greedy exploration
- `.pt` checkpoint save/load
- CPU/CUDA device selection
- training plots for reward, cleaned ratio, success, epsilon, and loss

The DQN checkpoint naming format is:

```text
dqn_agent_<map_name>_ep_<episodes>_seed_<seed>.pt
```

This keeps DQN checkpoints separate from the existing Q-learning JSON checkpoints.

---

## 4. CUDA Setup

The local machine had an NVIDIA GPU available:

- NVIDIA GeForce RTX 3080 Ti

The initial PyTorch install was CPU-only:

- `torch 2.11.0+cpu`
- `torch.cuda.is_available() == False`

That was replaced with a CUDA wheel:

- `torch 2.11.0+cu128`
- `torch.cuda.is_available() == True`

A CUDA-specific requirements file was added:

- `requirements-cuda.txt`

This lets the normal project dependencies remain simple while still documenting the CUDA install path for machines that support it.

---

## 5. First DQN Result: Naive DQN Failed

The first DQN runs on `complex_charge_bastion` did not clean any dirty tiles.

### Result

- training battery run: `0/3` cleaned in evaluation
- evaluation battery run: `0/3` cleaned in evaluation

### Observed behavior

The greedy policy often stayed near the start or moved into local loops.

This showed that simply replacing the Q-table with a neural network was not enough.
The network needed better state features and better exploration.

---

## 6. Action Lookahead Features

The first representation improvement added action-level lookahead features.

For each of the four actions, the encoder now exposes:

- whether the move is valid
- whether the move gets closer to remaining dirty
- whether the move gets closer to a charger
- whether the move improves battery safety margin

This is not a map-specific route.
It is a general feature that lets the network see whether an action improves local navigation context.

### Result

This helped the policy move farther through the map, but it still did not solve the sparse-reward problem.

---

## 7. Action Masking

The next issue was invalid wall actions.

Even when the network had lookahead features, it could still select actions that moved into walls.
Those actions are always invalid under the environment rules, so DQN now masks them out.

Masking is applied to:

- random exploration
- greedy action selection
- target Q calculation

### Result

Invalid wall-action loops disappeared.

The failure mode changed from:

- repeatedly trying to walk into walls

to:

- moving through valid corridors but eventually entering route loops or battery depletion

That was a useful improvement because it removed a low-value failure mode without hardcoding a cleaning route.

---

## 8. Guided Exploration

The biggest breakthrough came from guided exploration.

The problem was that random exploration almost never produced useful dirty-reaching trajectories on `bastion`.
Without those transitions in replay memory, the DQN had little useful signal to learn from.

The new guided exploration option:

- only affects training-time epsilon exploration
- does not affect greedy evaluation
- chooses some exploratory actions using shortest-path dirty/charger guidance
- keeps the learned policy responsible for evaluation behavior

The control flag is:

```bash
--guided-exploration-ratio
```

### Result

With `--guided-exploration-ratio 0.6`, training cleaned ratio rose immediately.

In the best 5000-episode guided run before target-context features, greedy evaluation reached:

- `1/3` cleaned
- `33.33%` cleaned ratio

This was the first DQN run that showed real `bastion` progress under the evaluation battery profile.

---

## 9. Target Dirty and Charger Context

After guided exploration, the policy could clean the first dirty tile but often failed to switch to the next target.

To address that, the state encoder was extended with target context:

- remaining dirty ratio
- target dirty row and column
- distance to target dirty
- target dirty's anchor charger region
- battery margin for target dirty plus charger recovery
- whether any safe target exists
- whether emergency recharge is needed

### Result

This improved greedy evaluation from:

- `1/3` cleaned

to:

- `2/3` cleaned

on `complex_charge_bastion` with:

```bash
python scripts/train_dqn.py --map-name complex_charge_bastion --battery-profile evaluation --episodes 5000 --seed 417 --print-every 500 --learning-rate 0.0005 --epsilon-decay 0.99995 --epsilon-min 0.10 --learning-starts 1000 --batch-size 128 --replay-capacity 100000 --train-every 8 --target-update-interval 1000 --hidden-size 256 --guided-exploration-ratio 0.6 --eval-episodes 20
```

The greedy policy cleaned:

- right-top dirty
- right-bottom dirty

and then failed to reach the final left-side dirty tile before battery depletion.

---

## 10. Guided Ratio Experiment

A more aggressive exploration setting was tested:

- `guided_exploration_ratio = 0.8`
- `epsilon_decay = 0.99997`

The 5000-episode run exceeded the local time limit, so a 2000-episode version was completed.

### Result

The training cleaned ratio stayed high, but greedy evaluation regressed to:

- `1/3` cleaned

### Interpretation

More guided experience is not automatically better.
At high guidance and slow epsilon decay, the training episodes contain many useful guided trajectories, but the greedy policy does not absorb them as well.

The better current setting is still:

- `guided_exploration_ratio = 0.6`
- `epsilon_decay = 0.99995`

---

## 11. Current Best DQN Result

The best current DQN checkpoint for `complex_charge_bastion` evaluation behavior is:

- `episodes = 5000`
- `seed = 417`
- `battery_profile = evaluation`
- `guided_exploration_ratio = 0.6`
- `epsilon_decay = 0.99995`

### Result

- greedy evaluation cleaned ratio: `66.67%`
- success rate: `0%`
- final failure: `battery_depleted`

This matches the best tabular result in terms of cleaned tiles, but it reached that point through DQN features and guided replay rather than tabular state abstraction.

---

## 12. Main Takeaways

The DQN work changed the project in three ways.

### First

CUDA is now available for neural experiments.

The original Q-learning path remains CPU-bound, but the DQN path can use GPU acceleration.

### Second

The hard-map bottleneck is no longer just "the agent cannot find dirty tiles."

With guided exploration, the agent can see useful trajectories.
With target context, it can also switch from the first dirty to the second dirty.

### Third

The remaining blocker is now narrower:

`complex_charge_bastion` still needs better route planning for the final long-range transition to the last remaining dirty tile under the evaluation battery budget.

---

## Next Steps

The most useful next steps are:

- add route-via-charger context features
- test whether the final dirty requires an explicit charger-first target decision
- compare the best DQN setup across multiple seeds
- decide whether DQN should be exposed in the pygame training UI
- keep generated DQN checkpoints and plots out of version control unless explicitly needed

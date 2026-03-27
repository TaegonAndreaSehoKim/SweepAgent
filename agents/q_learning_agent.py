from __future__ import annotations

import random
from typing import Dict, List, Tuple


State = Tuple[int, int, int]


class QLearningAgent:
    def __init__(
        self,
        action_space_size: int = 4,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        seed: int | None = None,
    ) -> None:
        # Store the core Q-learning hyperparameters.
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        # Store exploration settings for epsilon-greedy action selection.
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Use a local RNG so runs can be reproduced with the same seed.
        self.rng = random.Random(seed)

        # Map each state to one Q-value per action.
        self.q_table: Dict[State, List[float]] = {}

    def _ensure_state_exists(self, state: State) -> None:
        # Lazily initialize unseen states with zero Q-values.
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(self.action_space_size)]

    def get_q_values(self, state: State) -> List[float]:
        # Return the Q-values for the given state.
        self._ensure_state_exists(state)
        return self.q_table[state]

    def select_action(self, state: State, training: bool = True) -> int:
        # Make sure the current state exists in the table.
        self._ensure_state_exists(state)

        # Explore with probability epsilon during training.
        if training and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_space_size)

        # Otherwise choose the best action, breaking ties randomly.
        q_values = self.q_table[state]
        max_q = max(q_values)
        best_actions = [
            action for action, value in enumerate(q_values) if value == max_q
        ]
        return self.rng.choice(best_actions)

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
    ) -> None:
        # Make sure both states exist before updating the table.
        self._ensure_state_exists(state)
        self._ensure_state_exists(next_state)

        current_q = self.q_table[state][action]

        # Use the terminal reward directly, or bootstrap from the next state.
        if done:
            target = reward
        else:
            next_max_q = max(self.q_table[next_state])
            target = reward + self.discount_factor * next_max_q

        # Apply the standard Q-learning update rule.
        new_q = current_q + self.learning_rate * (target - current_q)
        self.q_table[state][action] = new_q

    def decay_epsilon(self) -> None:
        # Reduce exploration, but never go below the minimum value.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy_action(self, state: State) -> int:
        # Select the greedy action without exploration.
        return self.select_action(state, training=False)

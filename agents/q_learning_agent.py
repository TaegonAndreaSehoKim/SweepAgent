from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


State = Tuple[int, int, int, int]


class QLearningAgent:
    def __init__(
        self,
        action_space_size: int = 4,
        learning_rate: float = 0.05,
        discount_factor: float = 0.995,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.999,
        epsilon_min: float = 0.10,
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

        # Keep the seed so the agent can be reconstructed later.
        self.seed = seed

        # Use a local RNG so runs can be reproduced with the same seed.
        self.rng = random.Random(seed)

        # Map each state to one Q-value per action.
        self.q_table: Dict[State, List[float]] = {}

    def _ensure_state_exists(self, state: State) -> None:
        # Lazily initialize unseen states with zero Q-values.
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in range(self.action_space_size)]

    def _state_to_key(self, state: State) -> str:
        # Convert a tuple state into a JSON-safe string key.
        return ",".join(str(value) for value in state)

    def _key_to_state(self, key: str) -> State:
        # Convert a JSON string key back into a tuple state.
        parts = [int(value) for value in key.split(",")]

        # Backward compatibility for older checkpoints that used 3-part states.
        if len(parts) == 3:
            row, col, cleaned_mask = parts
            return (row, col, cleaned_mask, -1)

        if len(parts) == 4:
            row, col, cleaned_mask, battery_value = parts
            return (row, col, cleaned_mask, battery_value)

        raise ValueError(f"Unexpected serialized state format: {key}")

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

    def to_dict(self) -> Dict[str, Any]:
        # Convert the full agent state into a JSON-serializable dictionary.
        return {
            "action_space_size": self.action_space_size,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "epsilon_min": self.epsilon_min,
            "seed": self.seed,
            "q_table": {
                self._state_to_key(state): q_values
                for state, q_values in self.q_table.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "QLearningAgent":
        # Rebuild an agent instance from a serialized dictionary.
        agent = cls(
            action_space_size=payload["action_space_size"],
            learning_rate=payload["learning_rate"],
            discount_factor=payload["discount_factor"],
            epsilon=payload["epsilon"],
            epsilon_decay=payload["epsilon_decay"],
            epsilon_min=payload["epsilon_min"],
            seed=payload.get("seed"),
        )

        agent.q_table = {
            agent._key_to_state(state_key): list(q_values)
            for state_key, q_values in payload["q_table"].items()
        }
        return agent

    def save(self, file_path: str | Path) -> None:
        # Save the agent checkpoint as a JSON file.
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)

    @classmethod
    def load(cls, file_path: str | Path) -> "QLearningAgent":
        # Load the agent checkpoint from a JSON file.
        input_path = Path(file_path)

        with input_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        return cls.from_dict(payload)
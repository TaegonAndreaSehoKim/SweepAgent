from __future__ import annotations

from typing import Tuple

from agents.q_learning_agent import QLearningAgent


State = Tuple[int, int, int, int]


class SarsaAgent(QLearningAgent):
    """Tabular on-policy SARSA agent using the QLearningAgent table machinery."""

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        next_action: int,
        done: bool,
    ) -> None:
        encoded_state = self._ensure_state_exists(state)
        encoded_next_state = self._ensure_state_exists(next_state)

        current_q = self.q_table[encoded_state][action]
        if done:
            target = reward
        else:
            target = (
                reward
                + self.discount_factor
                * self.q_table[encoded_next_state][next_action]
            )

        self.q_table[encoded_state][action] = (
            current_q + self.learning_rate * (target - current_q)
        )

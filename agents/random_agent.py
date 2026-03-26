from __future__ import annotations

import random
from typing import Tuple


class RandomAgent:
    def __init__(self, action_space_size: int = 4, seed: int | None = None) -> None:
        self.action_space_size = action_space_size
        self.rng = random.Random(seed)

    def select_action(self, state: Tuple[int, int, int]) -> int:
        del state
        return self.rng.randrange(self.action_space_size)
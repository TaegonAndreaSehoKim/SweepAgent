from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from ui.training_app_core import PreviewPolicy, TrainingRunner


@dataclass
class MenuSelection:
    """Current selections from the menu screen."""
    map_name: str
    model_name: str = "q_learning"
    result_view: str = "single_playback"
    episodes: int = 5000
    step_delay: float = 0.5
    open_dropdown: str | None = None


@dataclass
class TrainingState:
    """Training subprocess state and parsed log/metric outputs."""
    runner: TrainingRunner = field(default_factory=TrainingRunner)
    latest_metrics: dict[str, str] = field(default_factory=dict)
    log_lines: list[str] = field(default_factory=list)
    log_scroll_offset: int = 0


@dataclass
class PreviewState:
    """Mini rollout preview shown during training."""
    env: Any | None = None
    state: Any | None = None
    done: bool = False
    total_reward: float = 0.0
    step_idx: int = 0
    path_history: list[tuple[int, int]] = field(default_factory=list)
    visit_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    last_step_time: float = field(default_factory=time.time)
    policy: PreviewPolicy | None = None
    mode_label: str = "random fallback"


@dataclass
class SinglePlaybackState:
    """Single rollout playback state."""
    env: Any | None = None
    agent: Any | None = None
    model_name: str | None = None
    state: Any | None = None
    done: bool = False
    total_reward: float = 0.0
    step_idx: int = 0
    path_history: list[tuple[int, int]] = field(default_factory=list)
    visit_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    is_paused: bool = False
    last_step_time: float = field(default_factory=time.time)
    rng: random.Random = field(default_factory=lambda: random.Random(42))


@dataclass
class ComparePlaybackState:
    """Side-by-side compare playback state."""
    random_env: Any | None = None
    learned_env: Any | None = None
    learned_agent: Any | None = None

    random_state: Any | None = None
    learned_state: Any | None = None

    random_done: bool = False
    learned_done: bool = False

    random_reward: float = 0.0
    learned_reward: float = 0.0

    random_step: int = 0
    learned_step: int = 0

    random_path: list[tuple[int, int]] = field(default_factory=list)
    learned_path: list[tuple[int, int]] = field(default_factory=list)

    random_visits: dict[tuple[int, int], int] = field(default_factory=dict)
    learned_visits: dict[tuple[int, int], int] = field(default_factory=dict)

    is_paused: bool = False
    last_step_time: float = field(default_factory=time.time)
    rng: random.Random = field(default_factory=lambda: random.Random(42))
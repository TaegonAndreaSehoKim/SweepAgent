from __future__ import annotations

import copy
import json
import queue
import random
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_OPTIONS = [
    "q_learning",
    "random_baseline",
]

RESULT_VIEW_OPTIONS = [
    "single_playback",
    "compare_playback",
]

EPISODE_OPTIONS = [500, 1000, 2000, 3000, 5000]
STEP_DELAY_OPTIONS = [0.1, 0.2, 0.3, 0.5, 0.8]

TRAINING_LINE_PATTERN = re.compile(
    r"Episode\s+(\d+)/(\d+)\s+\|\s+avg_reward=([-\d.]+)\s+\|\s+avg_cleaned_ratio=([\d.]+)%\s+\|\s+success_rate=([\d.]+)%\s+\|\s+epsilon=([\d.]+)"
)

MENU_WIDTH = 1040
MENU_HEIGHT = 820

PLAYBACK_TITLE_AREA_HEIGHT = 72
PLAYBACK_CONTROL_BAR_HEIGHT = 72
PLAYBACK_TOP_RESERVED = PLAYBACK_TITLE_AREA_HEIGHT + PLAYBACK_CONTROL_BAR_HEIGHT

TRAINING_METRICS_HEIGHT = 180
TRAINING_LOG_HEIGHT = 220
TRAINING_GRAPH_HEIGHT = 180
TRAINING_PREVIEW_HEIGHT = 360

TRAINING_TOP_Y = 96
TRAINING_GAP = 18
TRAINING_BOTTOM_ACTION_HEIGHT = 72
TRAINING_BOTTOM_MARGIN = 24

PREVIEW_CHECKPOINT_REFRESH_SEC = 2.0
PREVIEW_STEP_INTERVAL_SEC = 0.35

BUTTON_BG = (255, 255, 255)
BUTTON_HOVER_BG = (240, 244, 248)
BUTTON_ACTIVE_BG = (230, 239, 255)
PRIMARY_BUTTON_BG = (58, 134, 255)
PRIMARY_BUTTON_HOVER_BG = (47, 117, 223)
PRIMARY_BUTTON_TEXT = (255, 255, 255)

CONTROL_BAR_BG = (250, 251, 253)
DROPDOWN_BG = (255, 255, 255)
DROPDOWN_HOVER_BG = (242, 246, 252)

PROGRESS_BG = (230, 235, 241)
PROGRESS_FILL = (58, 134, 255)
SUCCESS_FILL = (25, 135, 84)
CLEANED_FILL = (255, 193, 7)
EPSILON_FILL = (111, 66, 193)

SCROLL_TRACK = (236, 240, 244)
SCROLL_THUMB = (173, 181, 189)

ALGORITHM_DEFAULT_PARAMS: dict[str, dict[str, float]] = {
    "q_learning": {
        "learning_rate": 0.10,
        "discount_factor": 0.95,
        "epsilon_start": 1.00,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.05,
    },
    "random_baseline": {},
}


@dataclass
class Button:
    """Simple clickable button model."""
    rect: object
    label: str
    on_click: Callable[[], None]
    primary: bool = False
    enabled: bool = True


class TrainingRunner:
    """
    Run the external training script in a subprocess and stream stdout lines
    back into the UI through a queue.
    """

    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.reader_thread: threading.Thread | None = None
        self.finished = False
        self.return_code: int | None = None

    def start(self, command: list[str]) -> None:
        self.process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.finished = False
        self.return_code = None

        def _reader() -> None:
            assert self.process is not None
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.output_queue.put(line.rstrip("\n"))
            self.process.wait()
            self.return_code = self.process.returncode
            self.finished = True

        self.reader_thread = threading.Thread(target=_reader, daemon=True)
        self.reader_thread.start()

    def poll_lines(self) -> list[str]:
        lines: list[str] = []
        while True:
            try:
                lines.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return lines

    def terminate(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()


class PreviewPolicy:
    """
    Load the latest checkpoint file and provide greedy actions for the
    training mini-preview.
    """

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.q_table: dict[str, list[float]] = {}
        self.last_mtime: float | None = None
        self.last_reload_time = 0.0

    def maybe_reload(self) -> bool:
        now = time.time()
        if now - self.last_reload_time < PREVIEW_CHECKPOINT_REFRESH_SEC:
            return False
        self.last_reload_time = now

        if not self.checkpoint_path.exists():
            return False

        try:
            mtime = self.checkpoint_path.stat().st_mtime
        except OSError:
            return False

        if self.last_mtime is not None and mtime <= self.last_mtime:
            return False

        try:
            with self.checkpoint_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return False

        raw_q_table = payload.get("q_table", {})
        if not isinstance(raw_q_table, dict):
            return False

        parsed_q_table: dict[str, list[float]] = {}
        for key, value in raw_q_table.items():
            if isinstance(key, str) and isinstance(value, list):
                parsed_q_table[key] = value

        self.q_table = parsed_q_table
        self.last_mtime = mtime
        return True

    def get_greedy_action(self, state: tuple[int, int, int, int], action_count: int) -> int:
        state_key = str(tuple(state))
        q_values = self.q_table.get(state_key)

        if not q_values or len(q_values) != action_count:
            return random.randrange(action_count)

        best_action = 0
        best_value = q_values[0]
        for idx in range(1, len(q_values)):
            if q_values[idx] > best_value:
                best_value = q_values[idx]
                best_action = idx
        return best_action


def get_default_algorithm_params(algorithm_name: str) -> dict[str, float]:
    """Return a copy of default hyperparameters for the selected algorithm."""
    return copy.deepcopy(ALGORITHM_DEFAULT_PARAMS.get(algorithm_name, {}))


def is_trainable_algorithm(algorithm_name: str) -> bool:
    """Return whether this selection launches a training subprocess."""
    return algorithm_name != "random_baseline"


def build_training_command(
    algorithm_name: str,
    map_name: str,
    episodes: int,
    seed: int,
    algorithm_params: dict[str, float] | None = None,
) -> list[str]:
    """
    Build the subprocess command for the selected training algorithm.

    For now, keep the runtime path stable and do not forward UI hyperparameters
    until train_q_learning.py explicitly supports those CLI options.
    """
    if algorithm_name == "q_learning":
        return [
            sys.executable,
            "scripts/train_q_learning.py",
            "--map-name",
            map_name,
            "--episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--print-every",
            "100",
        ]

    raise ValueError(f"Unsupported training algorithm: {algorithm_name}")


def clamp_step_delay(value: float, min_step_delay: float, max_step_delay: float) -> float:
    """Clamp playback delay to the allowed UI range."""
    return max(min_step_delay, min(max_step_delay, value))


def compute_training_window_height() -> int:
    """Compute the taller window height used only during training."""
    return (
        TRAINING_TOP_Y
        + TRAINING_METRICS_HEIGHT
        + TRAINING_GAP
        + TRAINING_LOG_HEIGHT
        + TRAINING_GAP
        + TRAINING_GRAPH_HEIGHT
        + TRAINING_GAP
        + TRAINING_PREVIEW_HEIGHT
        + TRAINING_GAP
        + TRAINING_BOTTOM_ACTION_HEIGHT
        + TRAINING_BOTTOM_MARGIN
    )


def get_checkpoint_path(map_name: str, seed: int) -> Path:
    """Return the q-learning checkpoint file path used by the training script."""
    return PROJECT_ROOT / "outputs" / "checkpoints" / f"q_learning_agent_{map_name}_seed_{seed}.json"
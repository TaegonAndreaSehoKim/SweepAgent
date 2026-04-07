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
    "battery_adapt_q_learning",
    "curriculum_q_learning",
    "random_baseline",
]

RESULT_VIEW_OPTIONS = [
    "single_playback",
    "compare_playback",
]

EPISODE_OPTIONS = [1000, 3000, 5000, 10000, 20000, 50000]
STEP_DELAY_OPTIONS = [0.1, 0.2, 0.3, 0.5, 0.8]
TRAIN_SEED_OPTIONS = [11, 22, 33, 42, 43, 99]
PLAYBACK_SEED_OPTIONS = [11, 22, 33, 42, 43, 99]

LEARNING_RATE_OPTIONS = [0.03, 0.05, 0.10, 0.20]
DISCOUNT_FACTOR_OPTIONS = [0.95, 0.99, 0.995]
EPSILON_START_OPTIONS = [0.30, 0.80, 1.00]
EPSILON_DECAY_OPTIONS = [0.995, 0.997, 0.999]
EPSILON_MIN_OPTIONS = [0.03, 0.05, 0.10]

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
TRAINING_PREVIEW_HEIGHT = 0

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

CURRICULUM_STAGE1_MAP = "charge_maze_medium"
CURRICULUM_STAGE1_MIN_EPISODES = 20000
CURRICULUM_STAGE2_MIN_EPISODES = 50000
DEFAULT_BATTERY_ADAPT_STAGE2_EPISODES = 50000
DEFAULT_PRINT_EVERY = 1000

MENU_NUMERIC_FIELDS = (
    "episodes",
    "stage2_episodes",
    "train_seed",
    "playback_seed",
    "learning_rate",
    "discount_factor",
    "epsilon_start",
    "epsilon_decay",
    "epsilon_min",
    "delay",
)

ALGORITHM_DEFAULT_PARAMS: dict[str, dict[str, float]] = {
    "q_learning": {
        "learning_rate": 0.05,
        "discount_factor": 0.99,
        "epsilon_start": 1.00,
        "epsilon_decay": 0.99999,
        "epsilon_min": 0.20,
    },
    "battery_adapt_q_learning": {
        "stage1_learning_rate": 0.05,
        "stage1_discount_factor": 0.99,
        "stage1_epsilon_start": 1.00,
        "stage1_epsilon_decay": 0.99999,
        "stage1_epsilon_min": 0.20,
        "stage2_episodes": DEFAULT_BATTERY_ADAPT_STAGE2_EPISODES,
        "stage2_learning_rate": 0.05,
        "stage2_discount_factor": 0.99,
        "stage2_epsilon_start": 0.30,
        "stage2_epsilon_decay": 0.99998,
        "stage2_epsilon_min": 0.08,
    },
    "curriculum_q_learning": {
        "stage1_learning_rate": 0.05,
        "stage1_discount_factor": 0.99,
        "stage1_epsilon_start": 1.00,
        "stage1_epsilon_decay": 0.997,
        "stage1_epsilon_min": 0.05,
        "stage2_learning_rate": 0.05,
        "stage2_discount_factor": 0.99,
        "stage2_epsilon_start": 1.00,
        "stage2_epsilon_decay": 0.99995,
        "stage2_epsilon_min": 0.15,
    },
    "random_baseline": {},
}


@dataclass
class Button:
    rect: object
    label: str
    on_click: Callable[[], None]
    primary: bool = False
    enabled: bool = True


class TrainingRunner:
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
    return copy.deepcopy(ALGORITHM_DEFAULT_PARAMS.get(algorithm_name, {}))


def is_trainable_algorithm(algorithm_name: str) -> bool:
    return algorithm_name != "random_baseline"


def get_preview_map_name(algorithm_name: str, selected_map_name: str) -> str:
    if algorithm_name == "curriculum_q_learning":
        return selected_map_name
    return selected_map_name


def get_playback_target_map_name(algorithm_name: str, selected_map_name: str) -> str:
    if algorithm_name == "curriculum_q_learning":
        return selected_map_name
    return selected_map_name


def get_battery_adapt_stage2_episodes(algorithm_params: dict[str, float] | None = None) -> int:
    algorithm_params = algorithm_params or {}
    raw_value = algorithm_params.get(
        "stage2_episodes",
        DEFAULT_BATTERY_ADAPT_STAGE2_EPISODES,
    )
    return max(1, int(raw_value))


def get_playback_target_episodes(
    algorithm_name: str,
    selected_episodes: int,
    algorithm_params: dict[str, float] | None = None,
) -> int:
    if algorithm_name == "curriculum_q_learning":
        return max(CURRICULUM_STAGE2_MIN_EPISODES, selected_episodes)
    if algorithm_name == "battery_adapt_q_learning":
        return selected_episodes + get_battery_adapt_stage2_episodes(algorithm_params)
    return selected_episodes


def get_preview_checkpoint_path(
    algorithm_name: str,
    selected_map_name: str,
    selected_episodes: int,
    seed: int,
    algorithm_params: dict[str, float] | None = None,
) -> Path:
    preview_map_name = get_preview_map_name(algorithm_name, selected_map_name)
    preview_episodes = get_playback_target_episodes(
        algorithm_name,
        selected_episodes,
        algorithm_params=algorithm_params,
    )
    return get_checkpoint_path(preview_map_name, preview_episodes, seed)


def get_display_hyperparams(algorithm_name: str, algorithm_params: dict[str, float]) -> dict[str, float]:
    if algorithm_name in {"curriculum_q_learning", "battery_adapt_q_learning"}:
        return {
            "learning_rate": algorithm_params.get("stage2_learning_rate", 0.05),
            "discount_factor": algorithm_params.get("stage2_discount_factor", 0.99),
            "epsilon_start": algorithm_params.get("stage2_epsilon_start", 0.30),
            "epsilon_decay": algorithm_params.get("stage2_epsilon_decay", 0.99998),
            "epsilon_min": algorithm_params.get("stage2_epsilon_min", 0.08),
        }

    return {
        "learning_rate": algorithm_params.get("learning_rate", 0.0),
        "discount_factor": algorithm_params.get("discount_factor", 0.0),
        "epsilon_start": algorithm_params.get("epsilon_start", 0.0),
        "epsilon_decay": algorithm_params.get("epsilon_decay", 0.0),
        "epsilon_min": algorithm_params.get("epsilon_min", 0.0),
    }


def update_menu_hyperparam(menu, param_name: str, value: float) -> None:
    if menu.algorithm_name in {"curriculum_q_learning", "battery_adapt_q_learning"}:
        stage2_key = f"stage2_{param_name}"
        if stage2_key in menu.algorithm_params:
            menu.algorithm_params[stage2_key] = value
        return

    menu.algorithm_params[param_name] = value


def get_menu_numeric_value(menu, field_name: str) -> int | float:
    if field_name == "episodes":
        return int(menu.episodes)
    if field_name == "stage2_episodes":
        return get_battery_adapt_stage2_episodes(menu.algorithm_params)
    if field_name == "train_seed":
        return int(menu.train_seed)
    if field_name == "playback_seed":
        return int(menu.playback_seed)
    if field_name == "delay":
        return float(menu.step_delay)
    if field_name in {
        "learning_rate",
        "discount_factor",
        "epsilon_start",
        "epsilon_decay",
        "epsilon_min",
    }:
        display_params = get_display_hyperparams(menu.algorithm_name, menu.algorithm_params)
        return float(display_params.get(field_name, 0.0))
    raise KeyError(f"Unsupported numeric field: {field_name}")


def format_menu_numeric_value(field_name: str, value: int | float) -> str:
    precision_map = {
        "learning_rate": 8,
        "discount_factor": 8,
        "epsilon_start": 8,
        "epsilon_decay": 8,
        "epsilon_min": 8,
        "delay": 4,
    }
    if field_name in {"episodes", "stage2_episodes", "train_seed", "playback_seed"}:
        return str(int(value))
    if field_name in precision_map:
        return f"{float(value):.{precision_map[field_name]}f}".rstrip("0").rstrip(".")
    raise KeyError(f"Unsupported numeric field: {field_name}")


def sync_menu_numeric_inputs(menu) -> None:
    menu.input_values = {
        field_name: format_menu_numeric_value(
            field_name,
            get_menu_numeric_value(menu, field_name),
        )
        for field_name in MENU_NUMERIC_FIELDS
    }


def commit_menu_numeric_input(menu, field_name: str) -> bool:
    raw_value = menu.input_values.get(field_name, "").strip()
    previous_value = get_menu_numeric_value(menu, field_name)

    try:
        if field_name in {"episodes", "stage2_episodes", "train_seed", "playback_seed"}:
            parsed_value = int(raw_value)
        else:
            parsed_value = float(raw_value)
    except ValueError:
        menu.input_values[field_name] = format_menu_numeric_value(field_name, previous_value)
        return False

    valid = True
    if field_name == "episodes":
        valid = parsed_value >= 1
        if valid:
            menu.episodes = int(parsed_value)
    elif field_name == "stage2_episodes":
        valid = parsed_value >= 1
        if valid and menu.algorithm_name == "battery_adapt_q_learning":
            menu.algorithm_params["stage2_episodes"] = float(parsed_value)
    elif field_name == "train_seed":
        menu.train_seed = int(parsed_value)
    elif field_name == "playback_seed":
        menu.playback_seed = int(parsed_value)
    elif field_name == "delay":
        valid = parsed_value > 0
        if valid:
            menu.step_delay = float(parsed_value)
    elif field_name == "learning_rate":
        valid = parsed_value > 0
        if valid:
            update_menu_hyperparam(menu, field_name, float(parsed_value))
    elif field_name == "discount_factor":
        valid = 0 < parsed_value <= 1
        if valid:
            update_menu_hyperparam(menu, field_name, float(parsed_value))
    elif field_name in {"epsilon_start", "epsilon_min"}:
        valid = 0 <= parsed_value <= 1
        if valid:
            update_menu_hyperparam(menu, field_name, float(parsed_value))
    elif field_name == "epsilon_decay":
        valid = 0 < parsed_value <= 1
        if valid:
            update_menu_hyperparam(menu, field_name, float(parsed_value))

    current_value = get_menu_numeric_value(menu, field_name)
    menu.input_values[field_name] = format_menu_numeric_value(field_name, current_value)
    return valid and current_value != previous_value


def build_training_command(
    algorithm_name: str,
    map_name: str,
    episodes: int,
    seed: int,
    algorithm_params: dict[str, float] | None = None,
) -> list[str]:
    algorithm_params = algorithm_params or {}

    if algorithm_name == "q_learning":
        command = [
            sys.executable,
            "scripts/train_q_learning.py",
            "--map-name",
            map_name,
            "--episodes",
            str(episodes),
            "--seed",
            str(seed),
            "--print-every",
            str(DEFAULT_PRINT_EVERY),
        ]

        if "learning_rate" in algorithm_params:
            command.extend(["--learning-rate", str(algorithm_params["learning_rate"])])
        if "discount_factor" in algorithm_params:
            command.extend(["--discount-factor", str(algorithm_params["discount_factor"])])
        if "epsilon_start" in algorithm_params:
            command.extend(["--epsilon-start", str(algorithm_params["epsilon_start"])])
        if "epsilon_decay" in algorithm_params:
            command.extend(["--epsilon-decay", str(algorithm_params["epsilon_decay"])])
        if "epsilon_min" in algorithm_params:
            command.extend(["--epsilon-min", str(algorithm_params["epsilon_min"])])

        return command

    if algorithm_name == "battery_adapt_q_learning":
        stage2_episodes = get_battery_adapt_stage2_episodes(algorithm_params)
        command = [
            sys.executable,
            "scripts/train_q_battery_adapt.py",
            "--map-name",
            map_name,
            "--stage1-episodes",
            str(episodes),
            "--stage2-episodes",
            str(stage2_episodes),
            "--seed",
            str(seed),
            "--print-every",
            str(DEFAULT_PRINT_EVERY),
        ]

        if "stage1_learning_rate" in algorithm_params:
            command.extend(["--stage1-learning-rate", str(algorithm_params["stage1_learning_rate"])])
        if "stage1_discount_factor" in algorithm_params:
            command.extend(["--stage1-discount-factor", str(algorithm_params["stage1_discount_factor"])])
        if "stage1_epsilon_start" in algorithm_params:
            command.extend(["--stage1-epsilon-start", str(algorithm_params["stage1_epsilon_start"])])
        if "stage1_epsilon_decay" in algorithm_params:
            command.extend(["--stage1-epsilon-decay", str(algorithm_params["stage1_epsilon_decay"])])
        if "stage1_epsilon_min" in algorithm_params:
            command.extend(["--stage1-epsilon-min", str(algorithm_params["stage1_epsilon_min"])])

        if "stage2_learning_rate" in algorithm_params:
            command.extend(["--stage2-learning-rate", str(algorithm_params["stage2_learning_rate"])])
        if "stage2_discount_factor" in algorithm_params:
            command.extend(["--stage2-discount-factor", str(algorithm_params["stage2_discount_factor"])])
        if "stage2_epsilon_start" in algorithm_params:
            command.extend(["--stage2-epsilon-start", str(algorithm_params["stage2_epsilon_start"])])
        if "stage2_epsilon_decay" in algorithm_params:
            command.extend(["--stage2-epsilon-decay", str(algorithm_params["stage2_epsilon_decay"])])
        if "stage2_epsilon_min" in algorithm_params:
            command.extend(["--stage2-epsilon-min", str(algorithm_params["stage2_epsilon_min"])])

        return command

    if algorithm_name == "curriculum_q_learning":
        command = [
            sys.executable,
            "scripts/train_q_curriculum.py",
            "--stage1-map",
            CURRICULUM_STAGE1_MAP,
            "--stage2-map",
            map_name,
            "--stage1-episodes",
            str(max(CURRICULUM_STAGE1_MIN_EPISODES, episodes)),
            "--stage2-episodes",
            str(max(CURRICULUM_STAGE2_MIN_EPISODES, episodes)),
            "--seed",
            str(seed),
            "--print-every",
            str(DEFAULT_PRINT_EVERY),
        ]

        if "stage1_learning_rate" in algorithm_params:
            command.extend(["--stage1-learning-rate", str(algorithm_params["stage1_learning_rate"])])
        if "stage1_discount_factor" in algorithm_params:
            command.extend(["--stage1-discount-factor", str(algorithm_params["stage1_discount_factor"])])
        if "stage1_epsilon_start" in algorithm_params:
            command.extend(["--stage1-epsilon-start", str(algorithm_params["stage1_epsilon_start"])])
        if "stage1_epsilon_decay" in algorithm_params:
            command.extend(["--stage1-epsilon-decay", str(algorithm_params["stage1_epsilon_decay"])])
        if "stage1_epsilon_min" in algorithm_params:
            command.extend(["--stage1-epsilon-min", str(algorithm_params["stage1_epsilon_min"])])

        if "stage2_learning_rate" in algorithm_params:
            command.extend(["--stage2-learning-rate", str(algorithm_params["stage2_learning_rate"])])
        if "stage2_discount_factor" in algorithm_params:
            command.extend(["--stage2-discount-factor", str(algorithm_params["stage2_discount_factor"])])
        if "stage2_epsilon_start" in algorithm_params:
            command.extend(["--stage2-epsilon-start", str(algorithm_params["stage2_epsilon_start"])])
        if "stage2_epsilon_decay" in algorithm_params:
            command.extend(["--stage2-epsilon-decay", str(algorithm_params["stage2_epsilon_decay"])])
        if "stage2_epsilon_min" in algorithm_params:
            command.extend(["--stage2-epsilon-min", str(algorithm_params["stage2_epsilon_min"])])

        return command

    raise ValueError(f"Unsupported training algorithm: {algorithm_name}")


def clamp_step_delay(value: float, min_step_delay: float, max_step_delay: float) -> float:
    return max(min_step_delay, min(max_step_delay, value))


def compute_training_window_height() -> int:
    return (
        TRAINING_TOP_Y
        + TRAINING_METRICS_HEIGHT
        + TRAINING_GAP
        + TRAINING_LOG_HEIGHT
        + TRAINING_GAP
        + TRAINING_GRAPH_HEIGHT
        + TRAINING_GAP
        + TRAINING_BOTTOM_ACTION_HEIGHT
        + TRAINING_BOTTOM_MARGIN
    )


def get_checkpoint_path(map_name: str, episodes: int, seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / f"q_learning_agent_{map_name}_ep_{episodes}_seed_{seed}.json"
    )

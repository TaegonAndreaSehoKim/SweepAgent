from __future__ import annotations

import re
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from agents.ppo_agent import PPOAgent
from utils.experiment_utils import build_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_EPISODES_PATTERN = re.compile(r"_ep_(\d+)_seed_")


def ensure_ppo_output_dirs() -> tuple[Path, Path, Path]:
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    plot_dir = PROJECT_ROOT / "outputs" / "plots"
    log_dir = PROJECT_ROOT / "outputs" / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, plot_dir, log_dir


def get_ppo_checkpoint_path(
    map_name: str,
    episodes: int,
    seed: int,
    checkpoint_tag: str = "",
) -> Path:
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{checkpoint_tag}" if checkpoint_tag else ""
    return checkpoint_dir / f"ppo_agent_{map_name}_ep_{episodes}_seed_{seed}{tag_suffix}.pt"


def get_ppo_best_checkpoint_path(
    map_name: str,
    seed: int,
    checkpoint_tag: str = "",
) -> Path:
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{checkpoint_tag}" if checkpoint_tag else ""
    return checkpoint_dir / f"ppo_agent_{map_name}_best_eval_seed_{seed}{tag_suffix}.pt"


def infer_ppo_checkpoint_episodes(path_str: str, explicit_value: int) -> int:
    if explicit_value > 0:
        return explicit_value
    if not path_str:
        return 0
    match = CHECKPOINT_EPISODES_PATTERN.search(Path(path_str).name)
    if match:
        return int(match.group(1))
    raise ValueError(
        "Could not infer starting checkpoint episodes from filename. "
        "Pass --starting-checkpoint-episodes explicitly."
    )


def evaluate_ppo_agent(
    map_name: str,
    eval_episodes: int,
    agent: PPOAgent,
    battery_capacity_override: int | None = None,
) -> dict[str, float]:
    if eval_episodes <= 0:
        return {
            "avg_reward": 0.0,
            "avg_steps": 0.0,
            "avg_cleaned_ratio": 0.0,
            "success_rate": 0.0,
        }

    env = build_env(
        map_name=map_name,
        battery_profile="evaluation",
        battery_capacity_override=battery_capacity_override,
    )
    rewards: list[float] = []
    steps: list[float] = []
    cleaned_ratios: list[float] = []
    successes: list[float] = []
    was_training = agent.network.training
    agent.network.eval()

    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        final_info: dict[str, float | str] = {}

        while not done:
            action = agent.get_policy_action(state)
            state, reward, done, final_info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        steps.append(float(final_info["steps_taken"]))
        cleaned_ratios.append(float(final_info["cleaned_ratio"]))
        successes.append(1.0 if float(final_info["cleaned_ratio"]) == 1.0 else 0.0)

    if was_training:
        agent.network.train()

    return {
        "avg_reward": mean(rewards),
        "avg_steps": mean(steps),
        "avg_cleaned_ratio": mean(cleaned_ratios),
        "success_rate": mean(successes),
    }


def format_ppo_eval_result(eval_result: dict[str, float]) -> str:
    return (
        f"avg_reward={eval_result['avg_reward']:.2f} | "
        f"avg_steps={eval_result['avg_steps']:.2f} | "
        f"avg_cleaned_ratio={eval_result['avg_cleaned_ratio'] * 100:.2f}% | "
        f"success_rate={eval_result['success_rate'] * 100:.2f}%"
    )


def get_ppo_eval_sort_key(eval_result: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(eval_result["success_rate"]),
        float(eval_result["avg_cleaned_ratio"]),
        float(eval_result["avg_reward"]),
        -float(eval_result["avg_steps"]),
    )


def is_better_ppo_eval_result(
    candidate: dict[str, float],
    incumbent: dict[str, float] | None,
) -> bool:
    if incumbent is None:
        return True
    return get_ppo_eval_sort_key(candidate) > get_ppo_eval_sort_key(incumbent)


def read_ppo_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    metadata = payload.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}

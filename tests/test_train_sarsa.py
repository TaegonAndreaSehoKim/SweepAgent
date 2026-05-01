from __future__ import annotations

from agents.sarsa_agent import SarsaAgent
from scripts.train_sarsa import get_sarsa_checkpoint_path, train_one_episode
from utils.experiment_utils import build_env


def test_sarsa_checkpoint_path_uses_sarsa_prefix() -> None:
    path = get_sarsa_checkpoint_path("default", 10, 42)

    assert path.name == "sarsa_agent_default_ep_10_seed_42.json"


def test_train_one_episode_updates_sarsa_table() -> None:
    env = build_env(map_name="default")
    agent = SarsaAgent(
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=1.0,
        seed=42,
        abstraction_map_name="default",
    )

    result = train_one_episode(env=env, agent=agent)

    assert result["steps_taken"] > 0
    assert agent.q_table

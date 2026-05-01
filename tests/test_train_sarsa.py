from __future__ import annotations

from agents.sarsa_agent import SarsaAgent
from agents.dqn_agent import StateFeatureEncoder
from scripts.train_sarsa import (
    get_sarsa_best_checkpoint_path,
    get_sarsa_checkpoint_path,
    is_better_sarsa_eval_result,
    select_training_action,
    train_one_episode,
)
from utils.experiment_utils import build_env


def test_sarsa_checkpoint_path_uses_sarsa_prefix() -> None:
    path = get_sarsa_checkpoint_path("default", 10, 42)
    tagged_path = get_sarsa_checkpoint_path("default", 10, 42, "guided")
    best_path = get_sarsa_best_checkpoint_path("default", 42, "guided")

    assert path.name == "sarsa_agent_default_ep_10_seed_42.json"
    assert tagged_path.name == "sarsa_agent_default_ep_10_seed_42_guided.json"
    assert best_path.name == "sarsa_agent_default_best_eval_seed_42_guided.json"


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


def test_sarsa_guided_training_action_can_use_encoder_guidance() -> None:
    agent = SarsaAgent(epsilon=1.0, seed=42, abstraction_map_name="default")
    encoder = StateFeatureEncoder(map_name="default", battery_capacity=17)
    state = (1, 1, 0, 17)

    action = select_training_action(
        agent=agent,
        state=state,
        guided_encoder=encoder,
        guided_exploration_ratio=1.0,
    )

    assert action == encoder.guided_action(state)


def test_sarsa_eval_sort_prioritizes_success_and_cleaned_ratio() -> None:
    incumbent = {
        "avg_reward": 10.0,
        "avg_steps": 10.0,
        "avg_cleaned_ratio": 0.5,
        "success_rate": 0.0,
    }
    candidate = {
        "avg_reward": -10.0,
        "avg_steps": 20.0,
        "avg_cleaned_ratio": 1.0,
        "success_rate": 1.0,
    }

    assert is_better_sarsa_eval_result(candidate, incumbent)

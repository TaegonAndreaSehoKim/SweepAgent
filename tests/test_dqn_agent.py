from __future__ import annotations

import torch

from agents.dqn_agent import DQNAgent, DQNConfig, ReplayBuffer, StateFeatureEncoder
from utils.dqn_experiment_utils import (
    get_dqn_best_checkpoint_path,
    get_dqn_checkpoint_path,
    infer_dqn_checkpoint_episodes,
    is_better_dqn_eval_result,
)


def test_state_feature_encoder_uses_map_specific_feature_size() -> None:
    encoder = StateFeatureEncoder(map_name="default", battery_capacity=17)

    features = encoder.encode((1, 1, 0, 17))

    assert len(features) == encoder.input_size
    assert encoder.target_context_feature_count == 13
    assert encoder.input_size == 6 + (3 * 2) + 13 + (4 * 4)
    assert all(-1.0 <= value <= 1.0 for value in features)


def test_state_feature_encoder_legacy_feature_version_preserves_old_size() -> None:
    encoder = StateFeatureEncoder(
        map_name="default",
        battery_capacity=17,
        feature_version=1,
    )

    features = encoder.encode((1, 1, 0, 17))

    assert len(features) == encoder.input_size
    assert encoder.target_context_feature_count == 8
    assert encoder.input_size == 6 + (3 * 2) + 8 + (4 * 4)


def test_state_feature_encoder_marks_invalid_action_lookahead() -> None:
    encoder = StateFeatureEncoder(map_name="default", battery_capacity=17)

    features = encoder.encode((1, 1, 0, 17))
    action_lookahead = features[-16:]

    assert action_lookahead[0:4] == [0.0, -1.0, -1.0, -1.0]
    assert action_lookahead[4] == 1.0
    assert action_lookahead[12] == 1.0


def test_state_feature_encoder_updates_target_context_after_cleaning() -> None:
    encoder = StateFeatureEncoder(map_name="complex_charge_bastion", battery_capacity=65)

    before_cleaning = encoder.encode((1, 20, 0, 48))
    after_cleaning = encoder.encode((1, 20, 1, 48))
    target_context_start = -(
        encoder.action_lookahead_feature_count + encoder.target_context_feature_count
    )
    before_target_context = before_cleaning[
        target_context_start : -encoder.action_lookahead_feature_count
    ]
    after_target_context = after_cleaning[
        target_context_start : -encoder.action_lookahead_feature_count
    ]

    assert before_target_context != after_target_context
    assert before_target_context[0] == 1.0
    assert after_target_context[0] == 2 / 3


def test_state_feature_encoder_allows_final_dirty_without_recovery_route() -> None:
    encoder = StateFeatureEncoder(map_name="complex_charge_bastion", battery_capacity=65)

    features = encoder.encode((15, 10, 5, 30))
    target_context_start = -(
        encoder.action_lookahead_feature_count + encoder.target_context_feature_count
    )
    target_context = features[
        target_context_start : -encoder.action_lookahead_feature_count
    ]

    assert target_context[5] > 0.0
    assert target_context[6] == 1.0
    assert target_context[12] == 0.0


def test_state_feature_encoder_guides_multi_charger_relay_to_final_dirty() -> None:
    encoder = StateFeatureEncoder(map_name="complex_charge_bastion", battery_capacity=65)

    assert encoder._best_relay_charger_id(15, 21, 5, 45) == 0
    assert encoder.guided_action((15, 21, 5, 45)) == 0
    assert encoder._best_relay_charger_id(5, 15, 5, 65) == 1
    assert encoder._safe_remaining_dirty_distance(15, 10, 5, 65) >= 0


def test_dqn_agent_masks_invalid_actions_for_random_exploration() -> None:
    config = DQNConfig(
        map_name="default",
        battery_capacity=17,
        hidden_size=16,
        epsilon=1.0,
        seed=42,
    )
    agent = DQNAgent(config=config, device="cpu")

    selected_actions = {
        agent.select_action((1, 1, 0, 17), training=True)
        for _ in range(100)
    }

    assert selected_actions <= {1, 3}
    assert selected_actions == {1, 3}


def test_dqn_agent_masks_invalid_actions_for_greedy_selection() -> None:
    config = DQNConfig(
        map_name="default",
        battery_capacity=17,
        hidden_size=16,
        epsilon=0.0,
        seed=42,
    )
    agent = DQNAgent(config=config, device="cpu")

    selected_actions = {
        agent.select_action((1, 1, 0, 17), training=False)
        for _ in range(20)
    }

    assert selected_actions <= {1, 3}


def test_state_feature_encoder_guides_toward_safe_dirty() -> None:
    encoder = StateFeatureEncoder(map_name="default", battery_capacity=17)

    action = encoder.guided_action((1, 1, 0, 17))

    assert action == 1


def test_dqn_agent_can_use_guided_exploration() -> None:
    config = DQNConfig(
        map_name="default",
        battery_capacity=17,
        hidden_size=16,
        epsilon=1.0,
        guided_exploration_ratio=1.0,
        seed=42,
    )
    agent = DQNAgent(config=config, device="cpu")

    selected_actions = {
        agent.select_action((1, 1, 0, 17), training=True)
        for _ in range(20)
    }

    assert selected_actions == {1}


def test_replay_buffer_respects_capacity() -> None:
    replay_buffer = ReplayBuffer(capacity=3, seed=42)

    for idx in range(5):
        replay_buffer.push(
            state=(1, 1, 0, 17),
            action=idx % 4,
            reward=float(idx),
            next_state=(1, 2, 0, 16),
            done=False,
        )

    sample = replay_buffer.sample(batch_size=3)

    assert len(replay_buffer) == 3
    assert len(sample) == 3
    assert {transition.reward for transition in sample} == {2.0, 3.0, 4.0}


def test_dqn_checkpoint_round_trip(tmp_path) -> None:
    config = DQNConfig(
        map_name="default",
        battery_capacity=17,
        hidden_size=16,
        batch_size=2,
        replay_capacity=10,
        learning_starts=2,
        seed=42,
    )
    agent = DQNAgent(config=config, device="cpu")
    agent.remember(
        state=(1, 1, 0, 17),
        action=1,
        reward=1.0,
        next_state=(2, 1, 1, 16),
        done=False,
    )
    agent.remember(
        state=(2, 1, 1, 16),
        action=3,
        reward=2.5,
        next_state=(2, 2, 1, 15),
        done=True,
    )
    agent.epsilon = 0.25
    agent.training_steps = 7
    agent.optimization_steps = 3
    checkpoint_path = tmp_path / "dqn.pt"

    agent.save(checkpoint_path, metadata={"test": True})
    loaded_agent = DQNAgent.load(checkpoint_path, device="cpu")
    original_sample = agent.replay_buffer.sample(batch_size=2)
    loaded_sample = loaded_agent.replay_buffer.sample(batch_size=2)

    assert loaded_agent.config.map_name == "default"
    assert loaded_agent.encoder.input_size == agent.encoder.input_size
    assert loaded_agent.epsilon == agent.epsilon
    assert loaded_agent.training_steps == agent.training_steps
    assert loaded_agent.optimization_steps == agent.optimization_steps
    assert len(loaded_agent.replay_buffer) == len(agent.replay_buffer)
    assert loaded_agent.replay_buffer.position == agent.replay_buffer.position
    assert [transition.reward for transition in loaded_sample] == [
        transition.reward for transition in original_sample
    ]
    assert loaded_agent.select_action((1, 1, 0, 17), training=False) in {0, 1, 2, 3}


def test_dqn_checkpoint_loads_legacy_feature_version_when_missing(tmp_path) -> None:
    config = DQNConfig(
        map_name="default",
        battery_capacity=17,
        hidden_size=16,
        feature_version=1,
        seed=42,
    )
    agent = DQNAgent(config=config, device="cpu")
    checkpoint_path = tmp_path / "legacy_dqn.pt"
    payload = agent.to_checkpoint(metadata={"legacy": True})
    del payload["config"]["feature_version"]
    torch.save(payload, checkpoint_path)

    loaded_agent = DQNAgent.load(checkpoint_path, device="cpu")

    assert loaded_agent.config.feature_version == 1
    assert loaded_agent.encoder.target_context_feature_count == 8
    assert loaded_agent.encoder.input_size == agent.encoder.input_size


def test_dqn_checkpoint_utils_support_tagged_paths() -> None:
    checkpoint_path = get_dqn_checkpoint_path(
        map_name="default",
        episodes=123,
        seed=77,
        checkpoint_tag="v2slice",
    )

    assert checkpoint_path.name == "dqn_agent_default_ep_123_seed_77_v2slice.pt"
    assert infer_dqn_checkpoint_episodes(str(checkpoint_path), explicit_value=0) == 123

    best_checkpoint_path = get_dqn_best_checkpoint_path(
        map_name="default",
        seed=77,
        checkpoint_tag="v2slice",
    )

    assert best_checkpoint_path.name == "dqn_agent_default_best_eval_seed_77_v2slice.pt"


def test_dqn_eval_result_comparison_prefers_task_outcomes_first() -> None:
    baseline = {
        "avg_reward": -100.0,
        "avg_steps": 90.0,
        "avg_cleaned_ratio": 1 / 3,
        "success_rate": 0.0,
    }
    better_cleaned = {
        "avg_reward": -300.0,
        "avg_steps": 500.0,
        "avg_cleaned_ratio": 2 / 3,
        "success_rate": 0.0,
    }
    better_success = {
        "avg_reward": -400.0,
        "avg_steps": 550.0,
        "avg_cleaned_ratio": 1.0,
        "success_rate": 0.2,
    }
    reward_only = {
        "avg_reward": -50.0,
        "avg_steps": 300.0,
        "avg_cleaned_ratio": 1 / 3,
        "success_rate": 0.0,
    }

    assert is_better_dqn_eval_result(better_cleaned, baseline)
    assert is_better_dqn_eval_result(better_success, better_cleaned)
    assert is_better_dqn_eval_result(reward_only, baseline)
    assert not is_better_dqn_eval_result(baseline, reward_only)

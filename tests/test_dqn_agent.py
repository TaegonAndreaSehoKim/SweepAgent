from __future__ import annotations

from agents.dqn_agent import DQNAgent, DQNConfig, ReplayBuffer, StateFeatureEncoder


def test_state_feature_encoder_uses_map_specific_feature_size() -> None:
    encoder = StateFeatureEncoder(map_name="default", battery_capacity=17)

    features = encoder.encode((1, 1, 0, 17))

    assert len(features) == encoder.input_size
    assert encoder.input_size == 6 + (3 * 2) + 8 + (4 * 4)
    assert all(-1.0 <= value <= 1.0 for value in features)


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
    before_target_context = before_cleaning[-24:-16]
    after_target_context = after_cleaning[-24:-16]

    assert before_target_context != after_target_context
    assert before_target_context[0] == 1.0
    assert after_target_context[0] == 2 / 3


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
    checkpoint_path = tmp_path / "dqn.pt"

    agent.save(checkpoint_path, metadata={"test": True})
    loaded_agent = DQNAgent.load(checkpoint_path, device="cpu")

    assert loaded_agent.config.map_name == "default"
    assert loaded_agent.encoder.input_size == agent.encoder.input_size
    assert loaded_agent.select_action((1, 1, 0, 17), training=False) in {0, 1, 2, 3}

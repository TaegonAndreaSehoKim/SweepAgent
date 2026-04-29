from __future__ import annotations

from agents.ppo_agent import PPOAgent, PPOConfig, PPORolloutStep


def test_ppo_agent_masks_invalid_actions_for_sampling_and_greedy() -> None:
    agent = PPOAgent(
        config=PPOConfig(
            map_name="default",
            battery_capacity=17,
            hidden_size=16,
            seed=42,
        ),
        device="cpu",
    )

    sampled_actions = {
        agent.select_action((1, 1, 0, 17), training=True)[0]
        for _ in range(50)
    }
    greedy_actions = {
        agent.get_policy_action((1, 1, 0, 17))
        for _ in range(10)
    }

    assert sampled_actions <= {1, 3}
    assert greedy_actions <= {1, 3}


def test_ppo_update_changes_optimization_step_count() -> None:
    agent = PPOAgent(
        config=PPOConfig(
            map_name="default",
            battery_capacity=17,
            hidden_size=16,
            rollout_steps=4,
            minibatch_size=2,
            update_epochs=2,
            seed=42,
        ),
        device="cpu",
    )
    rollout = []
    state = (1, 1, 0, 17)

    for _ in range(4):
        action, log_prob, value = agent.select_action(state, training=True)
        next_state = (2, 1, 0, 16)
        rollout.append(
            PPORolloutStep(
                state=state,
                action=action,
                reward=1.0,
                done=False,
                log_prob=log_prob,
                value=value,
            )
        )
        state = next_state

    metrics = agent.update(rollout=rollout, last_value=0.0)

    assert agent.optimization_steps == 4
    assert agent.training_steps == 4
    assert metrics["loss"] != 0.0


def test_ppo_behavior_clone_update_changes_optimization_step_count() -> None:
    agent = PPOAgent(
        config=PPOConfig(
            map_name="default",
            battery_capacity=17,
            hidden_size=16,
            seed=42,
        ),
        device="cpu",
    )

    metrics = agent.behavior_clone_update(
        states=[(1, 1, 0, 17), (2, 1, 0, 16)],
        actions=[1, 3],
        epochs=2,
        minibatch_size=1,
    )

    assert agent.optimization_steps == 4
    assert metrics["bc_loss"] > 0.0


def test_ppo_checkpoint_round_trip(tmp_path) -> None:
    agent = PPOAgent(
        config=PPOConfig(
            map_name="default",
            battery_capacity=17,
            hidden_size=16,
            seed=42,
        ),
        device="cpu",
    )
    checkpoint_path = tmp_path / "ppo.pt"

    agent.save(checkpoint_path, metadata={"test": True})
    loaded_agent = PPOAgent.load(checkpoint_path, device="cpu")

    assert loaded_agent.config.map_name == "default"
    assert loaded_agent.encoder.input_size == agent.encoder.input_size
    assert loaded_agent.get_policy_action((1, 1, 0, 17)) in {1, 3}

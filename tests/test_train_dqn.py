from __future__ import annotations

from argparse import Namespace

from agents.dqn_agent import DQNAgent, DQNConfig
from scripts.train_dqn import apply_resume_training_overrides


def test_resume_training_overrides_can_reset_epsilon_for_finetuning() -> None:
    agent = DQNAgent(
        config=DQNConfig(
            map_name="default",
            battery_capacity=17,
            hidden_size=16,
            epsilon=0.9,
            epsilon_min=0.2,
            seed=42,
        ),
        device="cpu",
    )

    apply_resume_training_overrides(
        agent=agent,
        args=Namespace(
            learning_rate=0.0005,
            discount_factor=0.98,
            epsilon_decay=0.999,
            epsilon_min=0.1,
            resume_epsilon=0.15,
            batch_size=64,
            learning_starts=500,
            train_every=2,
            target_update_interval=250,
            guided_exploration_ratio=0.4,
        ),
    )

    assert agent.epsilon == 0.15
    assert agent.config.epsilon_min == 0.1
    assert agent.config.learning_rate == 0.0005
    assert agent.optimizer.param_groups[0]["lr"] == 0.0005


def test_resume_training_overrides_clamps_resume_epsilon_to_floor() -> None:
    agent = DQNAgent(
        config=DQNConfig(
            map_name="default",
            battery_capacity=17,
            hidden_size=16,
            epsilon=0.9,
            epsilon_min=0.2,
            seed=42,
        ),
        device="cpu",
    )

    apply_resume_training_overrides(
        agent=agent,
        args=Namespace(
            learning_rate=0.001,
            discount_factor=0.99,
            epsilon_decay=0.999,
            epsilon_min=0.2,
            resume_epsilon=0.05,
            batch_size=64,
            learning_starts=500,
            train_every=2,
            target_update_interval=250,
            guided_exploration_ratio=0.4,
        ),
    )

    assert agent.epsilon == 0.2

from __future__ import annotations

from agents.sarsa_agent import SarsaAgent


def test_sarsa_update_uses_selected_next_action_not_max() -> None:
    agent = SarsaAgent(
        action_space_size=2,
        learning_rate=1.0,
        discount_factor=0.5,
        epsilon=0.0,
        seed=42,
    )
    state = (1, 1, 0, 17)
    next_state = (1, 2, 0, 16)
    agent.get_q_values(next_state)[0] = 100.0
    agent.get_q_values(next_state)[1] = 5.0

    agent.update(
        state=state,
        action=0,
        reward=1.0,
        next_state=next_state,
        next_action=1,
        done=False,
    )

    assert agent.get_q_values(state)[0] == 3.5


def test_sarsa_checkpoint_round_trip(tmp_path) -> None:
    checkpoint_path = tmp_path / "sarsa.json"
    agent = SarsaAgent(
        learning_rate=0.1,
        discount_factor=0.9,
        epsilon=0.2,
        seed=7,
        state_abstraction_mode="charger_context",
        abstraction_map_name="default",
    )
    agent.update(
        state=(1, 1, 0, 17),
        action=2,
        reward=4.0,
        next_state=(1, 2, 0, 16),
        next_action=1,
        done=True,
    )

    agent.save(checkpoint_path)
    loaded_agent = SarsaAgent.load(checkpoint_path)

    assert isinstance(loaded_agent, SarsaAgent)
    assert loaded_agent.seed == 7
    assert loaded_agent.state_abstraction_mode == "charger_context"
    assert loaded_agent.get_q_values((1, 1, 0, 17))[2] == 0.4

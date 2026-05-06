from __future__ import annotations

import pytest

from utils.experiment_utils import build_env


def test_simple_reward_profile_disables_shaping_rewards() -> None:
    env = build_env(
        map_name="complex_charge_bastion",
        battery_profile="evaluation",
        reward_profile="simple",
        reward_move_toward_relay_charger=0.5,
        penalty_move_away_from_relay_charger=-0.75,
    )

    assert env.reward_move_toward_charger == 0.0
    assert env.penalty_move_away_from_charger == 0.0
    assert env.reward_move_toward_safe_dirty == 0.0
    assert env.penalty_move_away_from_safe_dirty == 0.0
    assert env.reward_move_toward_relay_charger == 0.0
    assert env.penalty_move_away_from_relay_charger == 0.0
    assert env.low_battery_recharge_reward == 0.0
    assert env.penalty_recharge_without_progress == 0.0
    assert env.reward_final_dirty_bonus == 0.0
    assert env.penalty_enter_unrecoverable_state == 0.0
    assert env.successful_recharge_completion_bonus == 0.0


def test_build_env_rejects_unknown_reward_profile() -> None:
    with pytest.raises(ValueError, match="reward_profile"):
        build_env(reward_profile="dense")  # type: ignore[arg-type]

from __future__ import annotations

from utils.experiment_utils import build_env


def test_relay_charger_shaping_rewards_progress_toward_target_charger() -> None:
    env = build_env(
        map_name="complex_charge_bastion",
        battery_profile="evaluation",
        reward_move_toward_relay_charger=0.5,
        penalty_move_away_from_relay_charger=-0.75,
    )
    env.reset()
    env.robot_pos = (15, 21)
    env.cleaned_mask = 4
    env.battery_remaining = 45
    env.visited_positions = {(15, 21)}

    _, reward, _, _ = env.step(0)

    assert reward == -0.5


def test_relay_charger_shaping_penalizes_moving_away_from_target_charger() -> None:
    env = build_env(
        map_name="complex_charge_bastion",
        battery_profile="evaluation",
        reward_move_toward_relay_charger=0.5,
        penalty_move_away_from_relay_charger=-0.75,
    )
    env.reset()
    env.robot_pos = (15, 21)
    env.cleaned_mask = 4
    env.battery_remaining = 45
    env.visited_positions = {(15, 21)}

    _, reward, _, _ = env.step(3)

    assert reward == -1.75


def test_relay_charger_shaping_handles_final_dirty_without_recovery_route() -> None:
    env = build_env(
        map_name="complex_charge_bastion",
        battery_profile="evaluation",
        reward_move_toward_relay_charger=0.5,
        penalty_move_away_from_relay_charger=-0.75,
    )
    env.reset()
    env.robot_pos = (15, 21)
    env.cleaned_mask = 5
    env.battery_remaining = 45
    env.visited_positions = {(15, 21)}

    _, reward, _, _ = env.step(0)

    assert reward == -0.5

from __future__ import annotations

from argparse import Namespace

from scripts.train_ppo import (
    build_precleaned_positions,
    build_training_stages,
    normalize_stage_episodes,
    parse_initial_cleaned_positions,
    parse_stage_keep_spec,
)


def test_parse_initial_cleaned_positions() -> None:
    assert parse_initial_cleaned_positions("1,2; 3,4") == [(1, 2), (3, 4)]
    assert parse_initial_cleaned_positions("") == []


def test_parse_stage_keep_spec() -> None:
    assert parse_stage_keep_spec("2,0,2", total_dirty_tiles=3) == [0, 2]
    assert parse_stage_keep_spec("full", total_dirty_tiles=3) == [0, 1, 2]


def test_normalize_stage_episodes() -> None:
    assert normalize_stage_episodes([100], stage_count=3) == [100, 100, 100]
    assert normalize_stage_episodes([100, 200], stage_count=2) == [100, 200]


def test_build_precleaned_positions_for_kept_dirty_indices() -> None:
    assert build_precleaned_positions("default", keep_indices=[2]) == [(1, 4), (3, 2)]


def test_build_training_stages_for_curriculum() -> None:
    args = Namespace(
        map_name="default",
        episodes=10,
        initial_cleaned_positions="",
        curriculum_stage_keep_dirty_indices=["2", "full"],
        curriculum_stage_episodes=[5],
    )

    stages = build_training_stages(args)

    assert stages == [
        ("stage_1_keep_2", 5, [(1, 4), (3, 2)]),
        ("stage_2_keep_0,1,2", 5, []),
    ]

from __future__ import annotations

from dataclasses import dataclass

from env.grid_clean_env import GridCleanEnv
from utils.planner_utils import CleaningPlan, shortest_path_actions


@dataclass(frozen=True)
class WaypointAction:
    index: int
    label: str
    position: tuple[int, int]
    kind: str


@dataclass(frozen=True)
class WaypointExecutionResult:
    primitive_steps: int
    total_reward: float
    done: bool
    info: dict[str, float | str]
    primitive_actions: list[int]


class WaypointActionSpace:
    def __init__(
        self,
        dirty_positions: list[tuple[int, int]],
        charger_positions: list[tuple[int, int]],
    ) -> None:
        actions: list[WaypointAction] = []
        for dirty_idx, position in enumerate(dirty_positions):
            actions.append(
                WaypointAction(
                    index=len(actions),
                    label=f"dirty_{dirty_idx}",
                    position=position,
                    kind="dirty",
                )
            )
        for charger_idx, position in enumerate(charger_positions):
            actions.append(
                WaypointAction(
                    index=len(actions),
                    label=f"charger_{charger_idx}",
                    position=position,
                    kind="charger",
                )
            )
        self.actions = actions
        self._position_to_action = {action.position: action for action in actions}

    @classmethod
    def from_env(cls, env: GridCleanEnv) -> WaypointActionSpace:
        return cls(
            dirty_positions=list(env.dirty_positions),
            charger_positions=list(env.charger_positions),
        )

    def __len__(self) -> int:
        return len(self.actions)

    def get(self, action_idx: int) -> WaypointAction:
        if action_idx < 0 or action_idx >= len(self.actions):
            raise ValueError(f"Invalid waypoint action: {action_idx}")
        return self.actions[action_idx]

    def action_for_position(self, position: tuple[int, int]) -> WaypointAction | None:
        return self._position_to_action.get(position)


class WaypointController:
    def __init__(self, env: GridCleanEnv) -> None:
        self.env = env
        self.action_space = WaypointActionSpace.from_env(env)

    def execute(self, action_idx: int) -> WaypointExecutionResult:
        waypoint = self.action_space.get(action_idx)
        primitive_actions = shortest_path_actions(
            grid_map=self.env.raw_map,
            start=self.env.robot_pos,
            goal=waypoint.position,
        )
        if primitive_actions is None:
            return WaypointExecutionResult(
                primitive_steps=0,
                total_reward=0.0,
                done=False,
                info=self.env.get_episode_info(termination_reason="unreachable_waypoint"),
                primitive_actions=[],
            )

        total_reward = 0.0
        final_info: dict[str, float | str] = {}
        done = False

        for primitive_action in primitive_actions:
            _, reward, done, final_info = self.env.step(primitive_action)
            total_reward += reward
            if done:
                break

        if not final_info:
            final_info = self.env.get_episode_info(termination_reason="same_waypoint")

        return WaypointExecutionResult(
            primitive_steps=len(primitive_actions),
            total_reward=total_reward,
            done=done,
            info=final_info,
            primitive_actions=primitive_actions,
        )


def planner_route_to_waypoint_actions(
    env: GridCleanEnv,
    plan: CleaningPlan,
) -> list[int]:
    action_space = WaypointActionSpace.from_env(env)
    waypoint_actions: list[int] = []

    for position in plan.special_route[1:]:
        action = action_space.action_for_position(position)
        if action is None:
            raise ValueError(f"Planner route position is not a waypoint: {position}")
        waypoint_actions.append(action.index)

    return waypoint_actions

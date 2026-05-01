from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.sarsa_agent import SarsaAgent
from agents.dqn_agent import StateFeatureEncoder
from configs.map_presets import (
    DISCOUNT_FACTOR,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    LEARNING_RATE,
    PENALTY_MOVE_AWAY_FROM_RELAY_CHARGER,
    PRINT_EVERY,
    REWARD_MOVE_TOWARD_RELAY_CHARGER,
    TRAIN_EPISODES,
)
from scripts.train_q_learning import (
    apply_hparam_overrides,
    ensure_output_dirs,
    get_current_epsilon,
    maybe_decay_epsilon,
    maybe_get_q_table,
    moving_average,
    parse_initial_cleaned_positions,
    save_line_plot,
)
from utils.experiment_utils import build_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tabular SARSA agent for SweepAgent."
    )
    parser.add_argument("--map-name", type=str, default="default")
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--discount-factor", type=float, default=DISCOUNT_FACTOR)
    parser.add_argument("--epsilon-start", type=float, default=EPSILON_START)
    parser.add_argument("--epsilon-decay", type=float, default=EPSILON_DECAY)
    parser.add_argument("--epsilon-min", type=float, default=EPSILON_MIN)
    parser.add_argument(
        "--state-abstraction-mode",
        type=str,
        choices=("identity", "safety_margin", "charger_context"),
        default="identity",
    )
    parser.add_argument("--safety-margin-bucket-size", type=int, default=5)
    parser.add_argument(
        "--battery-profile",
        type=str,
        choices=("training", "evaluation"),
        default="training",
    )
    parser.add_argument("--battery-capacity-override", type=int, default=0)
    parser.add_argument(
        "--initial-cleaned-positions",
        type=str,
        default="",
        help=(
            "Optional semicolon-separated row,col positions to mark as already "
            "cleaned at reset time. Example: '1,5;7,3'"
        ),
    )
    parser.add_argument(
        "--checkpoint-episodes",
        type=int,
        default=0,
        help="Optional episode count to embed in the checkpoint filename.",
    )
    parser.add_argument(
        "--checkpoint-tag",
        type=str,
        default="",
        help="Optional suffix appended to the checkpoint filename.",
    )
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--reset-epsilon", action="store_true")
    parser.add_argument("--override-loaded-hparams", action="store_true")
    parser.add_argument(
        "--guided-exploration-ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of epsilon exploration actions chosen by the shared "
            "relay-aware guided policy instead of uniform random sampling."
        ),
    )
    parser.add_argument(
        "--reward-move-toward-relay-charger",
        type=float,
        default=REWARD_MOVE_TOWARD_RELAY_CHARGER,
        help="Training reward for moving toward a selected relay charger.",
    )
    parser.add_argument(
        "--penalty-move-away-from-relay-charger",
        type=float,
        default=PENALTY_MOVE_AWAY_FROM_RELAY_CHARGER,
        help="Training penalty for moving away from a selected relay charger.",
    )
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--save-best-eval-checkpoint", action="store_true")
    return parser.parse_args()


def get_sarsa_checkpoint_path(
    map_name: str,
    episodes: int,
    seed: int,
    checkpoint_tag: str = "",
) -> Path:
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{checkpoint_tag}" if checkpoint_tag else ""
    return checkpoint_dir / f"sarsa_agent_{map_name}_ep_{episodes}_seed_{seed}{tag_suffix}.json"


def get_sarsa_best_checkpoint_path(
    map_name: str,
    seed: int,
    checkpoint_tag: str = "",
) -> Path:
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = f"_{checkpoint_tag}" if checkpoint_tag else ""
    return checkpoint_dir / f"sarsa_agent_{map_name}_best_eval_seed_{seed}{tag_suffix}.json"


def build_fresh_agent(args: argparse.Namespace) -> SarsaAgent:
    return SarsaAgent(
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon_start,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        seed=args.seed,
        state_abstraction_mode=args.state_abstraction_mode,
        abstraction_map_name=args.map_name,
        safety_margin_bucket_size=args.safety_margin_bucket_size,
    )


def build_agent(args: argparse.Namespace) -> SarsaAgent:
    if not args.init_checkpoint:
        return build_fresh_agent(args)

    checkpoint_path = Path(args.init_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")

    agent = SarsaAgent.load(checkpoint_path)
    if args.override_loaded_hparams or args.reset_epsilon:
        apply_hparam_overrides(
            agent=agent,
            args=args,
            reset_epsilon=args.reset_epsilon,
        )
    return agent


def select_training_action(
    agent: SarsaAgent,
    state,
    guided_encoder: StateFeatureEncoder | None = None,
    guided_exploration_ratio: float = 0.0,
) -> int:
    encoded_state = agent._ensure_state_exists(state)

    if agent.rng.random() < agent.epsilon:
        if guided_encoder is not None and agent.rng.random() < guided_exploration_ratio:
            guided_action = guided_encoder.guided_action(state)
            if guided_action is not None:
                return int(guided_action)
        return agent.rng.randrange(agent.action_space_size)

    q_values = agent.q_table[encoded_state]
    max_q = max(q_values)
    best_actions = [
        action for action, value in enumerate(q_values) if value == max_q
    ]
    return agent.rng.choice(best_actions)


def evaluate_sarsa_agent(
    map_name: str,
    eval_episodes: int,
    agent: SarsaAgent,
    battery_capacity_override: int | None = None,
) -> dict[str, float]:
    if eval_episodes <= 0:
        return {
            "avg_reward": 0.0,
            "avg_steps": 0.0,
            "avg_cleaned_ratio": 0.0,
            "success_rate": 0.0,
        }

    env = build_env(
        map_name=map_name,
        battery_profile="evaluation",
        battery_capacity_override=battery_capacity_override,
    )
    rewards: list[float] = []
    steps: list[float] = []
    cleaned_ratios: list[float] = []
    successes: list[float] = []

    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        final_info: dict[str, float | str] = {}

        while not done:
            action = agent.get_policy_action(state)
            state, reward, done, final_info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        steps.append(float(final_info["steps_taken"]))
        cleaned_ratios.append(float(final_info["cleaned_ratio"]))
        successes.append(1.0 if float(final_info["cleaned_ratio"]) == 1.0 else 0.0)

    return {
        "avg_reward": mean(rewards),
        "avg_steps": mean(steps),
        "avg_cleaned_ratio": mean(cleaned_ratios),
        "success_rate": mean(successes),
    }


def format_sarsa_eval_result(eval_result: dict[str, float]) -> str:
    return (
        f"avg_reward={eval_result['avg_reward']:.2f} | "
        f"avg_steps={eval_result['avg_steps']:.2f} | "
        f"avg_cleaned_ratio={eval_result['avg_cleaned_ratio'] * 100:.2f}% | "
        f"success_rate={eval_result['success_rate'] * 100:.2f}%"
    )


def get_sarsa_eval_sort_key(eval_result: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        float(eval_result["success_rate"]),
        float(eval_result["avg_cleaned_ratio"]),
        float(eval_result["avg_reward"]),
        -float(eval_result["avg_steps"]),
    )


def is_better_sarsa_eval_result(
    candidate: dict[str, float],
    incumbent: dict[str, float] | None,
) -> bool:
    if incumbent is None:
        return True
    return get_sarsa_eval_sort_key(candidate) > get_sarsa_eval_sort_key(incumbent)


def build_checkpoint_metadata(
    args: argparse.Namespace,
    checkpoint_episodes: int,
    eval_result: dict[str, float] | None = None,
    best_eval_result: dict[str, float] | None = None,
    best_checkpoint_episodes: int = 0,
    best_checkpoint_path: Path | None = None,
    is_best_eval_checkpoint: bool = False,
) -> dict[str, Any]:
    return {
        "algorithm": "sarsa",
        "map_name": args.map_name,
        "episodes": args.episodes,
        "checkpoint_episodes": checkpoint_episodes,
        "seed": args.seed,
        "checkpoint_tag": args.checkpoint_tag,
        "init_checkpoint": args.init_checkpoint,
        "battery_profile": args.battery_profile,
        "battery_capacity_override": args.battery_capacity_override,
        "initial_cleaned_positions": args.initial_cleaned_positions,
        "state_abstraction_mode": args.state_abstraction_mode,
        "safety_margin_bucket_size": args.safety_margin_bucket_size,
        "guided_exploration_ratio": args.guided_exploration_ratio,
        "reward_move_toward_relay_charger": args.reward_move_toward_relay_charger,
        "penalty_move_away_from_relay_charger": args.penalty_move_away_from_relay_charger,
        "eval_episodes": args.eval_episodes,
        "eval_every": args.eval_every,
        "eval_result": eval_result or {},
        "best_eval_result": best_eval_result or {},
        "best_checkpoint_episodes": best_checkpoint_episodes,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else "",
        "is_best_eval_checkpoint": is_best_eval_checkpoint,
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "discount_factor": args.discount_factor,
            "epsilon_start": args.epsilon_start,
            "epsilon_decay": args.epsilon_decay,
            "epsilon_min": args.epsilon_min,
            "reset_epsilon": args.reset_epsilon,
            "override_loaded_hparams": args.override_loaded_hparams,
        },
    }


def save_sarsa_checkpoint(
    agent: SarsaAgent,
    checkpoint_path: Path,
    metadata: dict[str, Any],
) -> Path:
    payload = agent.to_dict()
    payload["metadata"] = metadata
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    return checkpoint_path


def train_one_episode(
    env,
    agent: SarsaAgent,
    guided_encoder: StateFeatureEncoder | None = None,
    guided_exploration_ratio: float = 0.0,
) -> dict[str, float]:
    state = env.reset()
    action = select_training_action(
        agent=agent,
        state=state,
        guided_encoder=guided_encoder,
        guided_exploration_ratio=guided_exploration_ratio,
    )
    done = False
    total_reward = 0.0
    termination_reason = "ongoing"

    while not done:
        next_state, reward, done, termination_reason = env.step_training(action)
        next_action = (
            select_training_action(
                agent=agent,
                state=next_state,
                guided_encoder=guided_encoder,
                guided_exploration_ratio=guided_exploration_ratio,
            )
            if not done
            else 0
        )

        agent.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            next_action=next_action,
            done=done,
        )

        state = next_state
        action = next_action
        total_reward += reward

    final_info = env.get_episode_info(termination_reason=termination_reason)
    cleaned_ratio = float(final_info["cleaned_ratio"])
    return {
        "total_reward": total_reward,
        "steps_taken": float(final_info["steps_taken"]),
        "cleaned_ratio": cleaned_ratio,
        "success": 1.0 if cleaned_ratio == 1.0 else 0.0,
    }


def main() -> None:
    args = parse_args()
    checkpoint_episodes = (
        args.checkpoint_episodes if args.checkpoint_episodes > 0 else args.episodes
    )
    env = build_env(
        map_name=args.map_name,
        battery_profile=args.battery_profile,
        battery_capacity_override=(
            args.battery_capacity_override
            if args.battery_capacity_override > 0
            else None
        ),
        initial_cleaned_positions=parse_initial_cleaned_positions(
            args.initial_cleaned_positions
        ),
        reward_move_toward_relay_charger=args.reward_move_toward_relay_charger,
        penalty_move_away_from_relay_charger=(
            args.penalty_move_away_from_relay_charger
        ),
    )
    agent = build_agent(args)
    guided_encoder = (
        StateFeatureEncoder(
            map_name=args.map_name,
            battery_capacity=env.battery_capacity,
            feature_version=2,
        )
        if args.guided_exploration_ratio > 0.0
        else None
    )

    _, plot_dir = ensure_output_dirs()
    checkpoint_path = get_sarsa_checkpoint_path(
        map_name=args.map_name,
        episodes=checkpoint_episodes,
        seed=args.seed,
        checkpoint_tag=args.checkpoint_tag,
    )
    best_checkpoint_path = (
        get_sarsa_best_checkpoint_path(
            map_name=args.map_name,
            seed=args.seed,
            checkpoint_tag=args.checkpoint_tag,
        )
        if args.save_best_eval_checkpoint
        else None
    )
    best_eval_result: dict[str, float] | None = None
    best_checkpoint_episodes = 0
    last_eval_result: dict[str, float] | None = None
    last_eval_checkpoint_episodes = 0

    rewards: list[float] = []
    cleaned_ratios: list[float] = []
    successes: list[float] = []
    epsilons: list[float] = []

    print(f"Training SARSA agent on map='{args.map_name}'...")
    print(f"episodes: {args.episodes}")
    print(f"seed: {args.seed}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"discount_factor: {args.discount_factor}")
    print(f"epsilon_start: {args.epsilon_start}")
    print(f"epsilon_decay: {args.epsilon_decay}")
    print(f"epsilon_min: {args.epsilon_min}")
    print(f"state_abstraction_mode: {agent.state_abstraction_mode}")
    print(f"battery_profile: {args.battery_profile}")
    if args.battery_capacity_override > 0:
        print(f"battery_capacity_override: {args.battery_capacity_override}")
    if args.initial_cleaned_positions:
        print(f"initial_cleaned_positions: {args.initial_cleaned_positions}")
    print(f"checkpoint_episodes: {checkpoint_episodes}")
    print(f"checkpoint_tag: {args.checkpoint_tag if args.checkpoint_tag else 'none'}")
    print(f"init_checkpoint: {args.init_checkpoint if args.init_checkpoint else 'none'}")
    print(f"reset_epsilon: {args.reset_epsilon}")
    print(f"override_loaded_hparams: {args.override_loaded_hparams}")
    print(f"guided_exploration_ratio: {args.guided_exploration_ratio}")
    print(
        "reward_move_toward_relay_charger: "
        f"{args.reward_move_toward_relay_charger}"
    )
    print(
        "penalty_move_away_from_relay_charger: "
        f"{args.penalty_move_away_from_relay_charger}"
    )
    print(f"eval_episodes: {args.eval_episodes}")
    print(f"eval_every: {args.eval_every}")
    print(f"save_best_eval_checkpoint: {args.save_best_eval_checkpoint}")
    if best_checkpoint_path is not None:
        print(f"best_checkpoint_path: {best_checkpoint_path}")

    for episode_idx in range(1, args.episodes + 1):
        episode_result = train_one_episode(
            env=env,
            agent=agent,
            guided_encoder=guided_encoder,
            guided_exploration_ratio=args.guided_exploration_ratio,
        )

        rewards.append(episode_result["total_reward"])
        cleaned_ratios.append(episode_result["cleaned_ratio"])
        successes.append(episode_result["success"])
        epsilons.append(get_current_epsilon(agent))

        maybe_decay_epsilon(agent)

        if episode_idx % args.print_every == 0 or episode_idx == args.episodes:
            start_idx = max(0, len(rewards) - args.print_every)
            print(
                f"Episode {episode_idx}/{args.episodes} | "
                f"avg_reward={mean(rewards[start_idx:]):.2f} | "
                f"avg_cleaned_ratio={mean(cleaned_ratios[start_idx:]) * 100:.2f}% | "
                f"success_rate={mean(successes[start_idx:]) * 100:.2f}% | "
                f"epsilon={get_current_epsilon(agent):.4f}"
            )

        if (
            args.eval_episodes > 0
            and args.eval_every > 0
            and episode_idx % args.eval_every == 0
        ):
            current_eval_checkpoint_episodes = episode_idx
            eval_result = evaluate_sarsa_agent(
                map_name=args.map_name,
                eval_episodes=args.eval_episodes,
                agent=agent,
                battery_capacity_override=(
                    args.battery_capacity_override
                    if args.battery_capacity_override > 0
                    else None
                ),
            )
            last_eval_result = eval_result
            last_eval_checkpoint_episodes = current_eval_checkpoint_episodes
            print(
                f"eval@{current_eval_checkpoint_episodes}: "
                f"{format_sarsa_eval_result(eval_result)}"
            )

            if is_better_sarsa_eval_result(eval_result, best_eval_result):
                best_eval_result = eval_result
                best_checkpoint_episodes = current_eval_checkpoint_episodes
                print(
                    "best_eval_updated: "
                    f"checkpoint_episodes={best_checkpoint_episodes} | "
                    f"{format_sarsa_eval_result(best_eval_result)}"
                )

                if best_checkpoint_path is not None:
                    save_sarsa_checkpoint(
                        agent=agent,
                        checkpoint_path=best_checkpoint_path,
                        metadata=build_checkpoint_metadata(
                            args=args,
                            checkpoint_episodes=best_checkpoint_episodes,
                            eval_result=eval_result,
                            best_eval_result=best_eval_result,
                            best_checkpoint_episodes=best_checkpoint_episodes,
                            best_checkpoint_path=best_checkpoint_path,
                            is_best_eval_checkpoint=True,
                        ),
                    )

    reward_plot_path = save_line_plot(
        values=moving_average(rewards, window=100),
        title=f"SweepAgent SARSA Training Reward ({args.map_name})",
        y_label="Reward (100-episode moving average)",
        output_path=plot_dir / f"sarsa_training_reward_{args.map_name}.png",
    )
    cleaned_plot_path = save_line_plot(
        values=moving_average(cleaned_ratios, window=100),
        title=f"SweepAgent SARSA Training Cleaned Ratio ({args.map_name})",
        y_label="Cleaned Ratio (100-episode moving average)",
        output_path=plot_dir / f"sarsa_training_cleaned_ratio_{args.map_name}.png",
    )
    success_plot_path = save_line_plot(
        values=moving_average(successes, window=100),
        title=f"SweepAgent SARSA Training Success Rate ({args.map_name})",
        y_label="Success Rate (100-episode moving average)",
        output_path=plot_dir / f"sarsa_training_success_{args.map_name}.png",
    )
    epsilon_plot_path = save_line_plot(
        values=epsilons,
        title=f"SweepAgent SARSA Epsilon Decay ({args.map_name})",
        y_label="Epsilon",
        output_path=plot_dir / f"sarsa_training_epsilon_{args.map_name}.png",
    )

    final_eval_result = evaluate_sarsa_agent(
        map_name=args.map_name,
        eval_episodes=args.eval_episodes,
        agent=agent,
        battery_capacity_override=(
            args.battery_capacity_override
            if args.battery_capacity_override > 0
            else None
        ),
    )
    if is_better_sarsa_eval_result(final_eval_result, best_eval_result):
        best_eval_result = final_eval_result
        best_checkpoint_episodes = checkpoint_episodes
        if best_checkpoint_path is not None:
            save_sarsa_checkpoint(
                agent=agent,
                checkpoint_path=best_checkpoint_path,
                metadata=build_checkpoint_metadata(
                    args=args,
                    checkpoint_episodes=best_checkpoint_episodes,
                    eval_result=final_eval_result,
                    best_eval_result=best_eval_result,
                    best_checkpoint_episodes=best_checkpoint_episodes,
                    best_checkpoint_path=best_checkpoint_path,
                    is_best_eval_checkpoint=True,
                ),
            )

    save_sarsa_checkpoint(
        agent=agent,
        checkpoint_path=checkpoint_path,
        metadata=build_checkpoint_metadata(
            args=args,
            checkpoint_episodes=checkpoint_episodes,
            eval_result=final_eval_result,
            best_eval_result=best_eval_result,
            best_checkpoint_episodes=best_checkpoint_episodes,
            best_checkpoint_path=best_checkpoint_path,
        ),
    )
    q_table = maybe_get_q_table(agent)
    final_window = min(100, len(rewards))

    print("\n=== Training Complete ===")
    print(f"checkpoint: {checkpoint_path}")
    print(f"reward_plot: {reward_plot_path}")
    print(f"cleaned_plot: {cleaned_plot_path}")
    print(f"success_plot: {success_plot_path}")
    print(f"epsilon_plot: {epsilon_plot_path}")
    print(f"learned_q_states: {len(q_table)}")
    print(f"final_epsilon: {get_current_epsilon(agent):.4f}")
    print(f"final_eval: {format_sarsa_eval_result(final_eval_result)}")
    if last_eval_result is not None:
        print(
            f"last_periodic_eval@{last_eval_checkpoint_episodes}: "
            f"{format_sarsa_eval_result(last_eval_result)}"
        )
    if best_eval_result is not None:
        print(
            "best_eval: "
            f"checkpoint_episodes={best_checkpoint_episodes} | "
            f"{format_sarsa_eval_result(best_eval_result)}"
        )
    print(f"last_{final_window}_avg_reward: {mean(rewards[-final_window:]):.2f}")
    print(
        f"last_{final_window}_avg_cleaned_ratio: "
        f"{mean(cleaned_ratios[-final_window:]) * 100:.2f}%"
    )
    print(
        f"last_{final_window}_success_rate: "
        f"{mean(successes[-final_window:]) * 100:.2f}%"
    )


if __name__ == "__main__":
    main()

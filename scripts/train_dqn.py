from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.dqn_agent import DQNAgent, DQNConfig
from configs.map_presets import (
    DISCOUNT_FACTOR,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    PRINT_EVERY,
    TRAIN_EPISODES,
)
from utils.dqn_experiment_utils import (
    ensure_dqn_output_dirs,
    evaluate_dqn_agent,
    format_dqn_eval_result,
    get_dqn_best_checkpoint_path,
    get_dqn_checkpoint_path,
    infer_dqn_checkpoint_episodes,
    is_better_dqn_eval_result,
    read_dqn_checkpoint_metadata,
)
from utils.experiment_utils import build_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DQN agent for SweepAgent.",
    )
    parser.add_argument(
        "--map-name",
        type=str,
        default="default",
        help="Map preset name to train on.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=TRAIN_EPISODES,
        help="Number of training episodes.",
    )
    parser.add_argument(
        "--checkpoint-episodes",
        type=int,
        default=0,
        help=(
            "Optional episode count to embed in the checkpoint filename. "
            "Defaults to cumulative episodes when resuming, otherwise --episodes."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for agent initialization and checkpoint naming.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=PRINT_EVERY,
        help="Print one compact progress line every N episodes.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path to resume DQN training from.",
    )
    parser.add_argument(
        "--starting-checkpoint-episodes",
        type=int,
        default=0,
        help=(
            "Optional explicit episode count already represented by --init-checkpoint. "
            "Used only when it cannot be inferred from the checkpoint filename."
        ),
    )
    parser.add_argument(
        "--checkpoint-tag",
        type=str,
        default="",
        help="Optional suffix appended to the checkpoint filename for experiment separation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--discount-factor",
        type=float,
        default=DISCOUNT_FACTOR,
        help="Discount factor (gamma).",
    )
    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=EPSILON_START,
        help="Initial epsilon for epsilon-greedy exploration.",
    )
    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=EPSILON_DECAY,
        help="Multiplicative epsilon decay applied after each episode.",
    )
    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=EPSILON_MIN,
        help="Minimum epsilon floor.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Replay minibatch size.",
    )
    parser.add_argument(
        "--replay-capacity",
        type=int,
        default=50000,
        help="Maximum number of transitions kept in replay memory.",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=1000,
        help="Number of collected transitions before gradient updates start.",
    )
    parser.add_argument(
        "--train-every",
        type=int,
        default=4,
        help="Run one optimizer step every N environment steps after warmup.",
    )
    parser.add_argument(
        "--target-update-interval",
        type=int,
        default=1000,
        help="Number of optimizer steps between hard target-network syncs.",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
        help="Hidden width for the two-layer MLP.",
    )
    parser.add_argument(
        "--guided-exploration-ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of epsilon exploration actions chosen by shortest-path "
            "dirty/charger guidance instead of uniform random sampling."
        ),
    )
    parser.add_argument(
        "--feature-version",
        type=int,
        choices=(1, 2),
        default=2,
        help="State feature encoder version. Use 2 for route-via-charger context.",
    )
    parser.add_argument(
        "--battery-profile",
        type=str,
        choices=("training", "evaluation"),
        default="training",
        help="Which battery capacity profile to use for environment construction.",
    )
    parser.add_argument(
        "--battery-capacity-override",
        type=int,
        default=0,
        help="Optional explicit battery capacity override for the environment.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Number of greedy evaluation episodes after training.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Run greedy evaluation every N training episodes. Use 0 to disable.",
    )
    parser.add_argument(
        "--save-best-eval-checkpoint",
        action="store_true",
        help="Save an extra checkpoint whenever a new best evaluation result is found.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: auto, cpu, cuda, or a specific torch device string.",
    )
    return parser.parse_args()


def moving_average(values: list[float], window: int = 100) -> list[float]:
    if not values:
        return []

    output: list[float] = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        output.append(mean(values[start : idx + 1]))
    return output


def save_line_plot(
    values: list[float],
    title: str,
    y_label: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def build_fresh_agent(args: argparse.Namespace, battery_capacity: int) -> DQNAgent:
    config = DQNConfig(
        map_name=args.map_name,
        battery_capacity=battery_capacity,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon_start,
        epsilon_decay=args.epsilon_decay,
        epsilon_min=args.epsilon_min,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts,
        train_every=args.train_every,
        target_update_interval=args.target_update_interval,
        hidden_size=args.hidden_size,
        guided_exploration_ratio=args.guided_exploration_ratio,
        feature_version=args.feature_version,
        seed=args.seed,
    )
    return DQNAgent(config=config, device=args.device)


def apply_resume_training_overrides(agent: DQNAgent, args: argparse.Namespace) -> None:
    agent.config.learning_rate = args.learning_rate
    agent.config.discount_factor = args.discount_factor
    agent.config.epsilon_decay = args.epsilon_decay
    agent.config.epsilon_min = args.epsilon_min
    agent.config.batch_size = args.batch_size
    agent.config.learning_starts = args.learning_starts
    agent.config.train_every = args.train_every
    agent.config.target_update_interval = args.target_update_interval
    agent.config.guided_exploration_ratio = args.guided_exploration_ratio

    for param_group in agent.optimizer.param_groups:
        param_group["lr"] = args.learning_rate


def build_agent(args: argparse.Namespace, battery_capacity: int) -> DQNAgent:
    if not args.init_checkpoint:
        return build_fresh_agent(args=args, battery_capacity=battery_capacity)

    checkpoint_path = Path(args.init_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")

    agent = DQNAgent.load(checkpoint_path, device=args.device, training=True)
    if agent.config.map_name != args.map_name:
        raise ValueError(
            f"Loaded checkpoint map_name='{agent.config.map_name}' does not match "
            f"--map-name '{args.map_name}'."
        )
    if agent.config.battery_capacity != battery_capacity:
        raise ValueError(
            f"Loaded checkpoint battery_capacity={agent.config.battery_capacity} does not "
            f"match the requested environment battery_capacity={battery_capacity}."
        )
    if agent.config.feature_version != args.feature_version:
        raise ValueError(
            f"Loaded checkpoint feature_version={agent.config.feature_version} does not "
            f"match --feature-version {args.feature_version}."
        )
    if agent.config.hidden_size != args.hidden_size:
        raise ValueError(
            f"Loaded checkpoint hidden_size={agent.config.hidden_size} does not "
            f"match --hidden-size {args.hidden_size}."
        )
    apply_resume_training_overrides(agent=agent, args=args)
    return agent


def train_one_episode(env, agent: DQNAgent) -> dict[str, float]:
    state = env.reset()
    done = False
    total_reward = 0.0
    losses: list[float] = []
    termination_reason = "ongoing"

    while not done:
        action = agent.select_action(state, training=True)
        next_state, reward, done, termination_reason = env.step_training(action)

        agent.remember(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )
        loss = agent.optimize_if_ready()
        if loss is not None:
            losses.append(loss)

        state = next_state
        total_reward += reward

    final_info = env.get_episode_info(termination_reason=termination_reason)
    agent.decay_epsilon()

    return {
        "total_reward": total_reward,
        "steps_taken": float(final_info["steps_taken"]),
        "cleaned_ratio": float(final_info["cleaned_ratio"]),
        "success": 1.0 if float(final_info["cleaned_ratio"]) == 1.0 else 0.0,
        "loss": mean(losses) if losses else 0.0,
    }


def build_checkpoint_metadata(
    args: argparse.Namespace,
    checkpoint_episodes: int,
    cumulative_episodes: int,
    starting_checkpoint_episodes: int,
    eval_result: dict[str, float],
    best_eval_result: dict[str, float] | None,
    best_checkpoint_episodes: int,
    best_checkpoint_path: Path | None,
    is_best_eval_checkpoint: bool = False,
) -> dict[str, object]:
    return {
        "map_name": args.map_name,
        "episodes": args.episodes,
        "checkpoint_episodes": checkpoint_episodes,
        "cumulative_episodes": cumulative_episodes,
        "seed": args.seed,
        "init_checkpoint": args.init_checkpoint,
        "starting_checkpoint_episodes": starting_checkpoint_episodes,
        "checkpoint_tag": args.checkpoint_tag,
        "battery_profile": args.battery_profile,
        "battery_capacity_override": args.battery_capacity_override,
        "feature_version": args.feature_version,
        "eval_episodes": args.eval_episodes,
        "eval_every": args.eval_every,
        "save_best_eval_checkpoint": args.save_best_eval_checkpoint,
        "eval_result": eval_result,
        "best_eval_result": best_eval_result or {},
        "best_checkpoint_episodes": best_checkpoint_episodes,
        "best_checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else "",
        "is_best_eval_checkpoint": is_best_eval_checkpoint,
    }


def maybe_load_existing_best_eval(
    args: argparse.Namespace,
    best_checkpoint_path: Path | None,
) -> tuple[dict[str, float] | None, int]:
    if not args.init_checkpoint or best_checkpoint_path is None or not best_checkpoint_path.exists():
        return None, 0

    metadata = read_dqn_checkpoint_metadata(best_checkpoint_path)
    eval_result = metadata.get("eval_result")
    if not isinstance(eval_result, dict) or not eval_result:
        return None, 0

    checkpoint_episodes = int(
        metadata.get(
            "best_checkpoint_episodes",
            metadata.get(
                "checkpoint_episodes",
                metadata.get("cumulative_episodes", metadata.get("episodes", 0)),
            ),
        )
    )
    return {
        "avg_reward": float(eval_result.get("avg_reward", 0.0)),
        "avg_steps": float(eval_result.get("avg_steps", 0.0)),
        "avg_cleaned_ratio": float(eval_result.get("avg_cleaned_ratio", 0.0)),
        "success_rate": float(eval_result.get("success_rate", 0.0)),
    }, checkpoint_episodes


def main() -> None:
    args = parse_args()
    env = build_env(
        map_name=args.map_name,
        battery_profile=args.battery_profile,
        battery_capacity_override=(
            args.battery_capacity_override if args.battery_capacity_override > 0 else None
        ),
    )
    if env.battery_capacity is None:
        raise ValueError("DQN training requires a battery-enabled map preset.")

    starting_checkpoint_episodes = infer_dqn_checkpoint_episodes(
        path_str=args.init_checkpoint,
        explicit_value=args.starting_checkpoint_episodes,
    )
    checkpoint_episodes = (
        args.checkpoint_episodes
        if args.checkpoint_episodes > 0
        else starting_checkpoint_episodes + args.episodes
    )
    cumulative_episodes = starting_checkpoint_episodes + args.episodes

    agent = build_agent(args=args, battery_capacity=env.battery_capacity)
    _, plot_dir, _ = ensure_dqn_output_dirs()
    checkpoint_path = get_dqn_checkpoint_path(
        map_name=args.map_name,
        episodes=checkpoint_episodes,
        seed=args.seed,
        checkpoint_tag=args.checkpoint_tag,
    )
    best_checkpoint_path = (
        get_dqn_best_checkpoint_path(
            map_name=args.map_name,
            seed=args.seed,
            checkpoint_tag=args.checkpoint_tag,
        )
        if args.save_best_eval_checkpoint
        else None
    )
    best_eval_result, best_checkpoint_episodes = maybe_load_existing_best_eval(
        args=args,
        best_checkpoint_path=best_checkpoint_path,
    )
    last_eval_result: dict[str, float] | None = None
    last_eval_checkpoint_episodes = 0

    rewards: list[float] = []
    cleaned_ratios: list[float] = []
    successes: list[float] = []
    epsilons: list[float] = []
    losses: list[float] = []

    print(f"Training DQN agent on map='{args.map_name}'...")
    print(f"episodes: {args.episodes}")
    print(f"checkpoint_episodes: {checkpoint_episodes}")
    print(f"cumulative_episodes: {cumulative_episodes}")
    print(f"seed: {args.seed}")
    print(f"init_checkpoint: {args.init_checkpoint if args.init_checkpoint else 'none'}")
    print(f"starting_checkpoint_episodes: {starting_checkpoint_episodes}")
    print(f"checkpoint_tag: {args.checkpoint_tag if args.checkpoint_tag else 'none'}")
    print(f"device: {agent.device}")
    print(f"learning_rate: {agent.config.learning_rate}")
    print(f"discount_factor: {agent.config.discount_factor}")
    print(f"epsilon: {agent.epsilon}")
    print(f"epsilon_decay: {agent.config.epsilon_decay}")
    print(f"epsilon_min: {agent.config.epsilon_min}")
    print(f"batch_size: {agent.config.batch_size}")
    print(f"replay_capacity: {agent.config.replay_capacity}")
    print(f"learning_starts: {agent.config.learning_starts}")
    print(f"train_every: {agent.config.train_every}")
    print(f"target_update_interval: {agent.config.target_update_interval}")
    print(f"hidden_size: {agent.config.hidden_size}")
    print(f"guided_exploration_ratio: {agent.config.guided_exploration_ratio}")
    print(f"feature_version: {agent.config.feature_version}")
    print(f"battery_profile: {args.battery_profile}")
    print(f"battery_capacity: {env.battery_capacity}")
    print(f"eval_episodes: {args.eval_episodes}")
    print(f"eval_every: {args.eval_every}")
    print(f"save_best_eval_checkpoint: {args.save_best_eval_checkpoint}")
    if best_checkpoint_path is not None:
        print(f"best_checkpoint_path: {best_checkpoint_path}")
        if best_eval_result is not None:
            print(
                "resume_best_eval: "
                f"checkpoint_episodes={best_checkpoint_episodes} | "
                f"{format_dqn_eval_result(best_eval_result)}"
            )

    for episode_idx in range(1, args.episodes + 1):
        episode_result = train_one_episode(env=env, agent=agent)
        rewards.append(episode_result["total_reward"])
        cleaned_ratios.append(episode_result["cleaned_ratio"])
        successes.append(episode_result["success"])
        epsilons.append(agent.epsilon)
        losses.append(episode_result["loss"])

        if episode_idx % args.print_every == 0 or episode_idx == args.episodes:
            start_idx = max(0, len(rewards) - args.print_every)
            avg_reward = mean(rewards[start_idx:])
            avg_cleaned_ratio = mean(cleaned_ratios[start_idx:])
            success_rate = mean(successes[start_idx:])
            avg_loss = mean(losses[start_idx:])

            print(
                f"Episode {episode_idx}/{args.episodes} | "
                f"avg_reward={avg_reward:.2f} | "
                f"avg_cleaned_ratio={avg_cleaned_ratio * 100:.2f}% | "
                f"success_rate={success_rate * 100:.2f}% | "
                f"epsilon={agent.epsilon:.4f} | "
                f"loss={avg_loss:.4f} | "
                f"replay={len(agent.replay_buffer)}"
            )

        if (
            args.eval_episodes > 0
            and args.eval_every > 0
            and episode_idx % args.eval_every == 0
        ):
            current_eval_checkpoint_episodes = starting_checkpoint_episodes + episode_idx
            eval_result = evaluate_dqn_agent(
                map_name=args.map_name,
                eval_episodes=args.eval_episodes,
                agent=agent,
                battery_capacity_override=(
                    args.battery_capacity_override if args.battery_capacity_override > 0 else None
                ),
            )
            last_eval_result = eval_result
            last_eval_checkpoint_episodes = current_eval_checkpoint_episodes

            print(
                f"eval@{current_eval_checkpoint_episodes}: "
                f"{format_dqn_eval_result(eval_result)}"
            )

            if is_better_dqn_eval_result(eval_result, best_eval_result):
                best_eval_result = eval_result
                best_checkpoint_episodes = current_eval_checkpoint_episodes
                print(
                    "best_eval_updated: "
                    f"checkpoint_episodes={best_checkpoint_episodes} | "
                    f"{format_dqn_eval_result(best_eval_result)}"
                )

                if best_checkpoint_path is not None:
                    agent.save(
                        best_checkpoint_path,
                        metadata=build_checkpoint_metadata(
                            args=args,
                            checkpoint_episodes=best_checkpoint_episodes,
                            cumulative_episodes=current_eval_checkpoint_episodes,
                            starting_checkpoint_episodes=starting_checkpoint_episodes,
                            eval_result=eval_result,
                            best_eval_result=best_eval_result,
                            best_checkpoint_episodes=best_checkpoint_episodes,
                            best_checkpoint_path=best_checkpoint_path,
                            is_best_eval_checkpoint=True,
                        ),
                    )

    reward_plot_path = save_line_plot(
        values=moving_average(rewards, window=100),
        title=f"SweepAgent DQN Training Reward ({args.map_name})",
        y_label="Reward (100-episode moving average)",
        output_path=plot_dir / f"dqn_training_reward_{args.map_name}.png",
    )
    cleaned_plot_path = save_line_plot(
        values=moving_average(cleaned_ratios, window=100),
        title=f"SweepAgent DQN Cleaned Ratio ({args.map_name})",
        y_label="Cleaned Ratio (100-episode moving average)",
        output_path=plot_dir / f"dqn_training_cleaned_ratio_{args.map_name}.png",
    )
    success_plot_path = save_line_plot(
        values=moving_average(successes, window=100),
        title=f"SweepAgent DQN Success Rate ({args.map_name})",
        y_label="Success Rate (100-episode moving average)",
        output_path=plot_dir / f"dqn_training_success_{args.map_name}.png",
    )
    epsilon_plot_path = save_line_plot(
        values=epsilons,
        title=f"SweepAgent DQN Epsilon Decay ({args.map_name})",
        y_label="Epsilon",
        output_path=plot_dir / f"dqn_training_epsilon_{args.map_name}.png",
    )
    loss_plot_path = save_line_plot(
        values=moving_average(losses, window=100),
        title=f"SweepAgent DQN Loss ({args.map_name})",
        y_label="Loss (100-episode moving average)",
        output_path=plot_dir / f"dqn_training_loss_{args.map_name}.png",
    )

    if args.eval_episodes > 0 and last_eval_checkpoint_episodes == cumulative_episodes:
        eval_result = last_eval_result if last_eval_result is not None else {
            "avg_reward": 0.0,
            "avg_steps": 0.0,
            "avg_cleaned_ratio": 0.0,
            "success_rate": 0.0,
        }
    else:
        eval_result = evaluate_dqn_agent(
            map_name=args.map_name,
            eval_episodes=args.eval_episodes,
            agent=agent,
            battery_capacity_override=(
                args.battery_capacity_override if args.battery_capacity_override > 0 else None
            ),
        )
        last_eval_result = eval_result
        last_eval_checkpoint_episodes = cumulative_episodes

    if is_better_dqn_eval_result(eval_result, best_eval_result):
        best_eval_result = eval_result
        best_checkpoint_episodes = cumulative_episodes
        if best_checkpoint_path is not None:
            agent.save(
                best_checkpoint_path,
                metadata=build_checkpoint_metadata(
                    args=args,
                    checkpoint_episodes=best_checkpoint_episodes,
                    cumulative_episodes=cumulative_episodes,
                    starting_checkpoint_episodes=starting_checkpoint_episodes,
                    eval_result=eval_result,
                    best_eval_result=best_eval_result,
                    best_checkpoint_episodes=best_checkpoint_episodes,
                    best_checkpoint_path=best_checkpoint_path,
                    is_best_eval_checkpoint=True,
                ),
            )

    agent.save(
        checkpoint_path,
        metadata=build_checkpoint_metadata(
            args=args,
            checkpoint_episodes=checkpoint_episodes,
            cumulative_episodes=cumulative_episodes,
            starting_checkpoint_episodes=starting_checkpoint_episodes,
            eval_result=eval_result,
            best_eval_result=best_eval_result,
            best_checkpoint_episodes=best_checkpoint_episodes,
            best_checkpoint_path=best_checkpoint_path,
        ),
    )

    final_window = min(100, len(rewards))
    print("\n=== Training Complete ===")
    print(f"checkpoint: {checkpoint_path}")
    print(f"reward_plot: {reward_plot_path}")
    print(f"cleaned_plot: {cleaned_plot_path}")
    print(f"success_plot: {success_plot_path}")
    print(f"epsilon_plot: {epsilon_plot_path}")
    print(f"loss_plot: {loss_plot_path}")
    print(f"device: {agent.device}")
    print(f"final_epsilon: {agent.epsilon:.4f}")
    print(f"replay_size: {len(agent.replay_buffer)}")
    print(f"optimization_steps: {agent.optimization_steps}")
    print(f"last_{final_window}_avg_reward: {mean(rewards[-final_window:]):.2f}")
    print(f"last_{final_window}_avg_cleaned_ratio: {mean(cleaned_ratios[-final_window:]) * 100:.2f}%")
    print(f"last_{final_window}_success_rate: {mean(successes[-final_window:]) * 100:.2f}%")
    if args.eval_episodes > 0:
        print(f"eval: {format_dqn_eval_result(eval_result)}")
    if best_eval_result is not None:
        print(
            "best_eval: "
            f"checkpoint_episodes={best_checkpoint_episodes} | "
            f"{format_dqn_eval_result(best_eval_result)}"
        )
        if best_checkpoint_path is not None:
            print(f"best_checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()

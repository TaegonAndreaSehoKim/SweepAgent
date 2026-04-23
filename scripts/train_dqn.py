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
        "--device",
        type=str,
        default="auto",
        help="Torch device: auto, cpu, cuda, or a specific torch device string.",
    )
    return parser.parse_args()


def ensure_output_dirs() -> tuple[Path, Path]:
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    plot_dir = PROJECT_ROOT / "outputs" / "plots"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, plot_dir


def get_checkpoint_path(map_name: str, episodes: int, seed: int) -> Path:
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"dqn_agent_{map_name}_ep_{episodes}_seed_{seed}.pt"


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


def build_agent(args: argparse.Namespace, battery_capacity: int) -> DQNAgent:
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
        seed=args.seed,
    )
    return DQNAgent(config=config, device=args.device)


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


def evaluate_agent(args: argparse.Namespace, agent: DQNAgent) -> dict[str, float]:
    if args.eval_episodes <= 0:
        return {
            "avg_reward": 0.0,
            "avg_steps": 0.0,
            "avg_cleaned_ratio": 0.0,
            "success_rate": 0.0,
        }

    env = build_env(
        map_name=args.map_name,
        battery_profile="evaluation",
        battery_capacity_override=(
            args.battery_capacity_override if args.battery_capacity_override > 0 else None
        ),
    )
    rewards: list[float] = []
    steps: list[float] = []
    cleaned_ratios: list[float] = []
    successes: list[float] = []
    was_training = agent.policy_net.training
    agent.policy_net.eval()

    for _ in range(args.eval_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0
        final_info: dict[str, float | str] = {}

        while not done:
            action = agent.select_action(state, training=False)
            state, reward, done, final_info = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        steps.append(float(final_info["steps_taken"]))
        cleaned_ratios.append(float(final_info["cleaned_ratio"]))
        successes.append(1.0 if float(final_info["cleaned_ratio"]) == 1.0 else 0.0)

    if was_training:
        agent.policy_net.train()

    return {
        "avg_reward": mean(rewards),
        "avg_steps": mean(steps),
        "avg_cleaned_ratio": mean(cleaned_ratios),
        "success_rate": mean(successes),
    }


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

    agent = build_agent(args=args, battery_capacity=env.battery_capacity)
    _, plot_dir = ensure_output_dirs()
    checkpoint_path = get_checkpoint_path(
        map_name=args.map_name,
        episodes=args.episodes,
        seed=args.seed,
    )

    rewards: list[float] = []
    cleaned_ratios: list[float] = []
    successes: list[float] = []
    epsilons: list[float] = []
    losses: list[float] = []

    print(f"Training DQN agent on map='{args.map_name}'...")
    print(f"episodes: {args.episodes}")
    print(f"seed: {args.seed}")
    print(f"device: {agent.device}")
    print(f"learning_rate: {args.learning_rate}")
    print(f"discount_factor: {args.discount_factor}")
    print(f"epsilon_start: {args.epsilon_start}")
    print(f"epsilon_decay: {args.epsilon_decay}")
    print(f"epsilon_min: {args.epsilon_min}")
    print(f"batch_size: {args.batch_size}")
    print(f"replay_capacity: {args.replay_capacity}")
    print(f"learning_starts: {args.learning_starts}")
    print(f"train_every: {args.train_every}")
    print(f"target_update_interval: {args.target_update_interval}")
    print(f"hidden_size: {args.hidden_size}")
    print(f"guided_exploration_ratio: {args.guided_exploration_ratio}")
    print(f"battery_profile: {args.battery_profile}")
    print(f"battery_capacity: {env.battery_capacity}")

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

    eval_result = evaluate_agent(args=args, agent=agent)
    agent.save(
        checkpoint_path,
        metadata={
            "map_name": args.map_name,
            "episodes": args.episodes,
            "seed": args.seed,
            "battery_profile": args.battery_profile,
            "battery_capacity_override": args.battery_capacity_override,
            "eval_episodes": args.eval_episodes,
            "eval_result": eval_result,
        },
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
        print(
            "eval: "
            f"avg_reward={eval_result['avg_reward']:.2f} | "
            f"avg_steps={eval_result['avg_steps']:.2f} | "
            f"avg_cleaned_ratio={eval_result['avg_cleaned_ratio'] * 100:.2f}% | "
            f"success_rate={eval_result['success_rate'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()

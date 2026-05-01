from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.sarsa_agent import SarsaAgent
from configs.map_presets import (
    DISCOUNT_FACTOR,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    LEARNING_RATE,
    PRINT_EVERY,
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
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--reset-epsilon", action="store_true")
    parser.add_argument("--override-loaded-hparams", action="store_true")
    return parser.parse_args()


def get_sarsa_checkpoint_path(map_name: str, episodes: int, seed: int) -> Path:
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"sarsa_agent_{map_name}_ep_{episodes}_seed_{seed}.json"


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


def train_one_episode(env, agent: SarsaAgent) -> dict[str, float]:
    state = env.reset()
    action = agent.select_action(state, training=True)
    done = False
    total_reward = 0.0
    termination_reason = "ongoing"

    while not done:
        next_state, reward, done, termination_reason = env.step_training(action)
        next_action = (
            agent.select_action(next_state, training=True)
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
    )
    agent = build_agent(args)

    _, plot_dir = ensure_output_dirs()
    checkpoint_path = get_sarsa_checkpoint_path(
        map_name=args.map_name,
        episodes=checkpoint_episodes,
        seed=args.seed,
    )

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
    print(f"init_checkpoint: {args.init_checkpoint if args.init_checkpoint else 'none'}")
    print(f"reset_epsilon: {args.reset_epsilon}")
    print(f"override_loaded_hparams: {args.override_loaded_hparams}")

    for episode_idx in range(1, args.episodes + 1):
        episode_result = train_one_episode(env=env, agent=agent)

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

    agent.save(checkpoint_path)
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

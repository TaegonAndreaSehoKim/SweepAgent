from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from agents.ppo_agent import PPOAgent, PPOConfig, PPORolloutStep
from configs.map_presets import PRINT_EVERY
from utils.experiment_utils import build_env
from utils.ppo_experiment_utils import (
    ensure_ppo_output_dirs,
    evaluate_ppo_agent,
    format_ppo_eval_result,
    get_ppo_best_checkpoint_path,
    get_ppo_checkpoint_path,
    infer_ppo_checkpoint_episodes,
    is_better_ppo_eval_result,
    read_ppo_checkpoint_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PPO agent for SweepAgent.")
    parser.add_argument("--map-name", type=str, default="default")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--checkpoint-episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=PRINT_EVERY)
    parser.add_argument("--init-checkpoint", type=str, default="")
    parser.add_argument("--starting-checkpoint-episodes", type=int, default=0)
    parser.add_argument("--checkpoint-tag", type=str, default="")
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--feature-version", type=int, choices=(1, 2), default=2)
    parser.add_argument(
        "--battery-profile",
        type=str,
        choices=("training", "evaluation"),
        default="training",
    )
    parser.add_argument("--battery-capacity-override", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--save-best-eval-checkpoint", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
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


def build_fresh_agent(args: argparse.Namespace, battery_capacity: int) -> PPOAgent:
    return PPOAgent(
        config=PPOConfig(
            map_name=args.map_name,
            battery_capacity=battery_capacity,
            learning_rate=args.learning_rate,
            discount_factor=args.discount_factor,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            max_grad_norm=args.max_grad_norm,
            rollout_steps=args.rollout_steps,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            hidden_size=args.hidden_size,
            feature_version=args.feature_version,
            seed=args.seed,
        ),
        device=args.device,
    )


def apply_resume_training_overrides(agent: PPOAgent, args: argparse.Namespace) -> None:
    agent.config.learning_rate = args.learning_rate
    agent.config.discount_factor = args.discount_factor
    agent.config.gae_lambda = args.gae_lambda
    agent.config.clip_ratio = args.clip_ratio
    agent.config.entropy_coef = args.entropy_coef
    agent.config.value_coef = args.value_coef
    agent.config.max_grad_norm = args.max_grad_norm
    agent.config.rollout_steps = args.rollout_steps
    agent.config.update_epochs = args.update_epochs
    agent.config.minibatch_size = args.minibatch_size

    for param_group in agent.optimizer.param_groups:
        param_group["lr"] = args.learning_rate


def build_agent(args: argparse.Namespace, battery_capacity: int) -> PPOAgent:
    if not args.init_checkpoint:
        return build_fresh_agent(args=args, battery_capacity=battery_capacity)

    checkpoint_path = Path(args.init_checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint_path}")

    agent = PPOAgent.load(checkpoint_path, device=args.device, training=True)
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

    metadata = read_ppo_checkpoint_metadata(best_checkpoint_path)
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


def finalize_rollout_update(
    agent: PPOAgent,
    rollout: list[PPORolloutStep],
    state,
) -> dict[str, float]:
    if not rollout:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
        }
    last_value = 0.0
    if not rollout[-1].done:
        with torch.no_grad():
            _, _, last_value = agent.select_action(state, training=False)
    return agent.update(rollout=rollout, last_value=last_value)


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
        raise ValueError("PPO training requires a battery-enabled map preset.")

    starting_checkpoint_episodes = infer_ppo_checkpoint_episodes(
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
    _, plot_dir, _ = ensure_ppo_output_dirs()
    checkpoint_path = get_ppo_checkpoint_path(
        map_name=args.map_name,
        episodes=checkpoint_episodes,
        seed=args.seed,
        checkpoint_tag=args.checkpoint_tag,
    )
    best_checkpoint_path = (
        get_ppo_best_checkpoint_path(
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

    rewards: list[float] = []
    cleaned_ratios: list[float] = []
    successes: list[float] = []
    steps_taken: list[float] = []
    losses: list[float] = []
    entropies: list[float] = []
    rollout: list[PPORolloutStep] = []

    print(f"Training PPO agent on map='{args.map_name}'...")
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
    print(f"gae_lambda: {agent.config.gae_lambda}")
    print(f"clip_ratio: {agent.config.clip_ratio}")
    print(f"entropy_coef: {agent.config.entropy_coef}")
    print(f"value_coef: {agent.config.value_coef}")
    print(f"rollout_steps: {agent.config.rollout_steps}")
    print(f"update_epochs: {agent.config.update_epochs}")
    print(f"minibatch_size: {agent.config.minibatch_size}")
    print(f"hidden_size: {agent.config.hidden_size}")
    print(f"feature_version: {agent.config.feature_version}")
    print(f"battery_profile: {args.battery_profile}")
    print(f"battery_capacity: {env.battery_capacity}")
    print(f"eval_episodes: {args.eval_episodes}")
    print(f"eval_every: {args.eval_every}")
    print(f"save_best_eval_checkpoint: {args.save_best_eval_checkpoint}")
    if best_checkpoint_path is not None:
        print(f"best_checkpoint_path: {best_checkpoint_path}")

    state = env.reset()
    done = False
    episode_reward = 0.0
    episode_steps = 0
    termination_reason = "ongoing"

    for episode_idx in range(1, args.episodes + 1):
        while not done:
            action, log_prob, value = agent.select_action(state, training=True)
            next_state, reward, done, termination_reason = env.step_training(action)
            rollout.append(
                PPORolloutStep(
                    state=state,
                    action=action,
                    reward=reward,
                    done=done,
                    log_prob=log_prob,
                    value=value,
                )
            )
            state = next_state
            episode_reward += reward
            episode_steps += 1

            if len(rollout) >= args.rollout_steps:
                metrics = finalize_rollout_update(
                    agent=agent,
                    rollout=rollout,
                    state=state,
                )
                losses.append(metrics["loss"])
                entropies.append(metrics["entropy"])
                rollout = []

        final_info = env.get_episode_info(termination_reason=termination_reason)
        rewards.append(episode_reward)
        cleaned_ratios.append(float(final_info["cleaned_ratio"]))
        successes.append(1.0 if float(final_info["cleaned_ratio"]) == 1.0 else 0.0)
        steps_taken.append(float(final_info["steps_taken"]))

        if episode_idx % args.print_every == 0 or episode_idx == args.episodes:
            start_idx = max(0, len(rewards) - args.print_every)
            recent_losses = losses[-max(1, min(len(losses), args.print_every)) :]
            recent_entropies = entropies[-max(1, min(len(entropies), args.print_every)) :]
            print(
                f"Episode {episode_idx}/{args.episodes} | "
                f"avg_reward={mean(rewards[start_idx:]):.2f} | "
                f"avg_cleaned_ratio={mean(cleaned_ratios[start_idx:]) * 100:.2f}% | "
                f"success_rate={mean(successes[start_idx:]) * 100:.2f}% | "
                f"avg_steps={mean(steps_taken[start_idx:]):.2f} | "
                f"loss={(mean(recent_losses) if recent_losses else 0.0):.4f} | "
                f"entropy={(mean(recent_entropies) if recent_entropies else 0.0):.4f}"
            )

        if (
            args.eval_episodes > 0
            and args.eval_every > 0
            and episode_idx % args.eval_every == 0
        ):
            current_eval_checkpoint_episodes = starting_checkpoint_episodes + episode_idx
            eval_result = evaluate_ppo_agent(
                map_name=args.map_name,
                eval_episodes=args.eval_episodes,
                agent=agent,
                battery_capacity_override=(
                    args.battery_capacity_override if args.battery_capacity_override > 0 else None
                ),
            )
            print(
                f"eval@{current_eval_checkpoint_episodes}: "
                f"{format_ppo_eval_result(eval_result)}"
            )
            if is_better_ppo_eval_result(eval_result, best_eval_result):
                best_eval_result = eval_result
                best_checkpoint_episodes = current_eval_checkpoint_episodes
                print(
                    "best_eval_updated: "
                    f"checkpoint_episodes={best_checkpoint_episodes} | "
                    f"{format_ppo_eval_result(best_eval_result)}"
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

        state = env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0
        termination_reason = "ongoing"

    if rollout:
        metrics = finalize_rollout_update(
            agent=agent,
            rollout=rollout,
            state=state,
        )
        losses.append(metrics["loss"])
        entropies.append(metrics["entropy"])

    reward_plot_path = save_line_plot(
        values=moving_average(rewards, window=100),
        title=f"SweepAgent PPO Training Reward ({args.map_name})",
        y_label="Reward (100-episode moving average)",
        output_path=plot_dir / f"ppo_training_reward_{args.map_name}.png",
    )
    cleaned_plot_path = save_line_plot(
        values=moving_average(cleaned_ratios, window=100),
        title=f"SweepAgent PPO Cleaned Ratio ({args.map_name})",
        y_label="Cleaned Ratio (100-episode moving average)",
        output_path=plot_dir / f"ppo_training_cleaned_ratio_{args.map_name}.png",
    )
    success_plot_path = save_line_plot(
        values=moving_average(successes, window=100),
        title=f"SweepAgent PPO Success Rate ({args.map_name})",
        y_label="Success Rate (100-episode moving average)",
        output_path=plot_dir / f"ppo_training_success_{args.map_name}.png",
    )
    loss_plot_path = save_line_plot(
        values=moving_average(losses, window=20),
        title=f"SweepAgent PPO Loss ({args.map_name})",
        y_label="Loss (20-update moving average)",
        output_path=plot_dir / f"ppo_training_loss_{args.map_name}.png",
    )

    eval_result = evaluate_ppo_agent(
        map_name=args.map_name,
        eval_episodes=args.eval_episodes,
        agent=agent,
        battery_capacity_override=(
            args.battery_capacity_override if args.battery_capacity_override > 0 else None
        ),
    )
    if is_better_ppo_eval_result(eval_result, best_eval_result):
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
    print(f"loss_plot: {loss_plot_path}")
    print(f"device: {agent.device}")
    print(f"training_steps: {agent.training_steps}")
    print(f"optimization_steps: {agent.optimization_steps}")
    print(f"last_{final_window}_avg_reward: {mean(rewards[-final_window:]):.2f}")
    print(f"last_{final_window}_avg_cleaned_ratio: {mean(cleaned_ratios[-final_window:]) * 100:.2f}%")
    print(f"last_{final_window}_success_rate: {mean(successes[-final_window:]) * 100:.2f}%")
    if args.eval_episodes > 0:
        print(f"eval: {format_ppo_eval_result(eval_result)}")
    if best_eval_result is not None:
        print(
            "best_eval: "
            f"checkpoint_episodes={best_checkpoint_episodes} | "
            f"{format_ppo_eval_result(best_eval_result)}"
        )
        if best_checkpoint_path is not None:
            print(f"best_checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()

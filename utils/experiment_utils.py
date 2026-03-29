from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from agents.q_learning_agent import QLearningAgent
from configs.map_presets import (
    DISCOUNT_FACTOR,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    LEARNING_RATE,
    MAP_PRESETS,
    PRINT_EVERY,
    REWARD_CLEAN,
    REWARD_FINISH,
    REWARD_INVALID,
    REWARD_MOVE,
    REWARD_REVISIT,
    TRAIN_EPISODES,
    TRAIN_SEED,
)
from env.grid_clean_env import GridCleanEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_env(map_name: str = "default") -> GridCleanEnv:
    # Build one of the shared map presets.
    if map_name not in MAP_PRESETS:
        supported_maps = ", ".join(MAP_PRESETS.keys())
        raise ValueError(
            f"Unknown map_name='{map_name}'. Supported maps: {supported_maps}"
        )

    preset = MAP_PRESETS[map_name]

    return GridCleanEnv(
        grid_map=preset["grid_map"],
        max_steps=preset["max_steps"],
        reward_clean=REWARD_CLEAN,
        reward_move=REWARD_MOVE,
        reward_revisit=REWARD_REVISIT,
        reward_invalid=REWARD_INVALID,
        reward_finish=REWARD_FINISH,
    )


def build_q_learning_agent(
    action_space_size: int,
    seed: int = TRAIN_SEED,
    learning_rate: float = LEARNING_RATE,
    discount_factor: float = DISCOUNT_FACTOR,
    epsilon_start: float = EPSILON_START,
    epsilon_decay: float = EPSILON_DECAY,
    epsilon_min: float = EPSILON_MIN,
) -> QLearningAgent:
    # Build a Q-learning agent using the shared default hyperparameters.
    return QLearningAgent(
        action_space_size=action_space_size,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon_start,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        seed=seed,
    )


def get_checkpoint_path(map_name: str, seed: int) -> Path:
    # Use a map-specific checkpoint name so policies do not get mixed.
    return (
        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / f"q_learning_agent_{map_name}_seed_{seed}.json"
    )


def get_legacy_default_checkpoint_path(seed: int) -> Path:
    # Support the older default-map checkpoint naming for backward compatibility.
    return (
        PROJECT_ROOT
        / "outputs"
        / "checkpoints"
        / f"q_learning_agent_seed_{seed}.json"
    )


def train_q_learning_agent(
    env: GridCleanEnv,
    num_episodes: int = TRAIN_EPISODES,
    seed: int = TRAIN_SEED,
    print_every: int = PRINT_EVERY,
) -> QLearningAgent:
    # Train a Q-learning agent on the provided environment.
    agent = build_q_learning_agent(
        action_space_size=len(env.ACTIONS),
        seed=seed,
    )

    rewards: List[float] = []

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0
        final_info: Dict[str, float] = {}

        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            agent.update(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            state = next_state
            total_reward += reward
            final_info = info

        agent.decay_epsilon()
        rewards.append(total_reward)

        if episode % print_every == 0:
            recent_rewards = rewards[-print_every:]
            print(
                f"Episode {episode}/{num_episodes} | "
                f"avg_reward={sum(recent_rewards) / len(recent_rewards):.2f} | "
                f"cleaned_ratio={final_info['cleaned_ratio']:.2%} | "
                f"epsilon={agent.epsilon:.4f}"
            )

    return agent


def load_or_train_q_agent(
    map_name: str = "default",
    num_episodes: int = TRAIN_EPISODES,
    seed: int = TRAIN_SEED,
    print_every: int = PRINT_EVERY,
) -> QLearningAgent:
    # Reuse a saved checkpoint when possible, otherwise train and save a new one.
    checkpoint_path = get_checkpoint_path(map_name=map_name, seed=seed)

    if checkpoint_path.exists():
        print(
            "Loading trained Q-learning agent from "
            f"{checkpoint_path.relative_to(PROJECT_ROOT)}"
        )
        return QLearningAgent.load(checkpoint_path)

    # Migrate the legacy default-map checkpoint to the new naming convention.
    if map_name == "default":
        legacy_checkpoint_path = get_legacy_default_checkpoint_path(seed=seed)

        if legacy_checkpoint_path.exists():
            print(
                "Loading legacy default-map checkpoint from "
                f"{legacy_checkpoint_path.relative_to(PROJECT_ROOT)}"
            )
            agent = QLearningAgent.load(legacy_checkpoint_path)

            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            agent.save(checkpoint_path)

            print(
                "Migrated checkpoint to "
                f"{checkpoint_path.relative_to(PROJECT_ROOT)}"
            )
            return agent

    print(f"No checkpoint found for map='{map_name}'. Training Q-learning agent...")
    train_env = build_env(map_name=map_name)
    agent = train_q_learning_agent(
        env=train_env,
        num_episodes=num_episodes,
        seed=seed,
        print_every=print_every,
    )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(checkpoint_path)

    print(f"Saved checkpoint to: {checkpoint_path.relative_to(PROJECT_ROOT)}")
    return agent
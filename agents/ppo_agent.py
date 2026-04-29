from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from agents.dqn_agent import State, StateFeatureEncoder


class PPORolloutStep(NamedTuple):
    state: State
    action: int
    reward: float
    done: bool
    log_prob: float
    value: float


@dataclass
class PPOConfig:
    map_name: str
    battery_capacity: int
    action_space_size: int = 4
    learning_rate: float = 0.0003
    discount_factor: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 2048
    update_epochs: int = 4
    minibatch_size: int = 256
    hidden_size: int = 128
    feature_version: int = 2
    seed: int | None = None


class PPOActorCritic(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, action_space_size: int) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_size, action_space_size)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(features)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


class PPOAgent:
    def __init__(self, config: PPOConfig, device: str = "auto") -> None:
        self.config = config
        self.device = self._resolve_device(device)
        self.encoder = StateFeatureEncoder(
            map_name=config.map_name,
            battery_capacity=config.battery_capacity,
            feature_version=config.feature_version,
        )
        if config.seed is not None:
            torch.manual_seed(config.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(config.seed)

        self.network = PPOActorCritic(
            input_size=self.encoder.input_size,
            hidden_size=config.hidden_size,
            action_space_size=config.action_space_size,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
        )
        self.training_steps = 0
        self.optimization_steps = 0

    def _resolve_device(self, device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but torch.cuda.is_available() is false.")
        return resolved_device

    def _distribution_and_value(
        self,
        states: list[State],
    ) -> tuple[Categorical, torch.Tensor]:
        features = self.encoder.encode_batch(states, device=self.device)
        logits, values = self.network(features)
        valid_masks = self.encoder.valid_action_masks(states, device=self.device)
        masked_logits = logits.masked_fill(~valid_masks, -torch.inf)
        return Categorical(logits=masked_logits), values

    def select_action(self, state: State, training: bool = True) -> tuple[int, float, float]:
        distribution, values = self._distribution_and_value([state])
        if training:
            action_tensor = distribution.sample()
        else:
            action_tensor = torch.argmax(distribution.logits, dim=-1)
        log_prob = distribution.log_prob(action_tensor)
        return (
            int(action_tensor.item()),
            float(log_prob.item()),
            float(values[0].item()),
        )

    def get_policy_action(self, state: State) -> int:
        action, _, _ = self.select_action(state, training=False)
        return action

    def compute_returns_and_advantages(
        self,
        rollout: list[PPORolloutStep],
        last_value: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages: list[float] = []
        gae = 0.0
        next_value = last_value

        for step in reversed(rollout):
            non_terminal = 0.0 if step.done else 1.0
            delta = (
                step.reward
                + self.config.discount_factor * next_value * non_terminal
                - step.value
            )
            gae = (
                delta
                + self.config.discount_factor
                * self.config.gae_lambda
                * non_terminal
                * gae
            )
            advantages.append(gae)
            next_value = step.value

        advantages.reverse()
        advantage_tensor = torch.tensor(
            advantages,
            dtype=torch.float32,
            device=self.device,
        )
        value_tensor = torch.tensor(
            [step.value for step in rollout],
            dtype=torch.float32,
            device=self.device,
        )
        returns = advantage_tensor + value_tensor
        return returns, advantage_tensor

    def update(self, rollout: list[PPORolloutStep], last_value: float) -> dict[str, float]:
        if not rollout:
            return {
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
            }

        returns, advantages = self.compute_returns_and_advantages(
            rollout=rollout,
            last_value=last_value,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        states = [step.state for step in rollout]
        actions = torch.tensor(
            [step.action for step in rollout],
            dtype=torch.int64,
            device=self.device,
        )
        old_log_probs = torch.tensor(
            [step.log_prob for step in rollout],
            dtype=torch.float32,
            device=self.device,
        )

        batch_size = len(rollout)
        minibatch_size = min(self.config.minibatch_size, batch_size)
        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []

        for _ in range(self.config.update_epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                batch_states = [states[int(idx)] for idx in indices.tolist()]
                distribution, values = self._distribution_and_value(batch_states)
                batch_actions = actions[indices]
                batch_old_log_probs = old_log_probs[indices]
                batch_returns = returns[indices]
                batch_advantages = advantages[indices]

                new_log_probs = distribution.log_prob(batch_actions)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                )
                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages,
                ).mean()
                value_loss = F.mse_loss(values, batch_returns)
                entropy = distribution.entropy().mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                self.optimization_steps += 1
                losses.append(float(loss.item()))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy.item()))

        self.training_steps += len(rollout)
        return {
            "loss": sum(losses) / len(losses),
            "policy_loss": sum(policy_losses) / len(policy_losses),
            "value_loss": sum(value_losses) / len(value_losses),
            "entropy": sum(entropies) / len(entropies),
        }

    def to_checkpoint(self, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "algorithm": "ppo",
            "config": asdict(self.config),
            "training_steps": self.training_steps,
            "optimization_steps": self.optimization_steps,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
            "network_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": metadata or {},
        }

    def save(self, file_path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        output_path = Path(file_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_checkpoint(metadata=metadata), output_path)

    @classmethod
    def load(
        cls,
        file_path: str | Path,
        device: str = "auto",
        training: bool = False,
    ) -> "PPOAgent":
        input_path = Path(file_path)
        resolved_device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        payload = torch.load(input_path, map_location=resolved_device)
        config_payload = dict(payload["config"])
        config_payload.setdefault(
            "feature_version",
            StateFeatureEncoder.FEATURE_VERSION_LEGACY,
        )
        agent = cls(config=PPOConfig(**config_payload), device=str(resolved_device))
        agent.training_steps = int(payload.get("training_steps", 0))
        agent.optimization_steps = int(payload.get("optimization_steps", 0))
        torch_rng_state = payload.get("torch_rng_state")
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state.cpu())
        cuda_rng_state_all = payload.get("cuda_rng_state_all")
        if agent.device.type == "cuda" and cuda_rng_state_all:
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_state_all])
        agent.network.load_state_dict(payload["network_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if training:
            agent.network.train()
        else:
            agent.network.eval()
        return agent

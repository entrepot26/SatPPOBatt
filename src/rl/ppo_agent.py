from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Beta

from src.rl.buffer import RolloutBuffer
from src.rl.networks import ActorNet, CriticNet


class PPOAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: dict, device: torch.device) -> None:
        hidden_sizes = list(cfg["hidden_sizes"])
        self.device = device
        self.clip_ratio = float(cfg["clip_ratio"])
        self.gamma = float(cfg["gamma"])
        self.gae_lambda = float(cfg["gae_lambda"])
        self.ppo_epochs = int(cfg["ppo_epochs"] if "ppo_epochs" in cfg else cfg["update_epochs"])
        self.mini_batch_size = int(cfg["mini_batch_size"])
        self.entropy_coef = float(cfg["entropy_coef"])
        self.value_coef = float(cfg["value_coef"])

        self.actor = ActorNet(obs_dim, action_dim, hidden_sizes).to(device)
        self.critic = CriticNet(obs_dim, hidden_sizes).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=float(cfg["actor_lr"]))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=float(cfg["critic_lr"]))

    def _distribution(self, states: torch.Tensor) -> Beta:
        alpha, beta = self.actor(states)
        return Beta(alpha, beta)

    def act(self, states: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            dist = self._distribution(s)
            if deterministic:
                action = dist.mean
            else:
                action = dist.sample()
            action = torch.clamp(action, 1e-6, 1.0 - 1e-6)
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(s)
        return action.cpu().numpy(), log_prob.cpu().numpy(), value.cpu().numpy()

    def estimate_values(self, states: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            values = self.critic(s)
        return values.cpu().numpy()

    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = self._distribution(states)
        actions = torch.clamp(actions, 1e-6, 1.0 - 1e-6)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(states)
        return log_prob, entropy, values

    def update(self, buffer: RolloutBuffer, last_values: np.ndarray) -> dict[str, float]:
        data = buffer.build_tensors(last_values=last_values, gamma=self.gamma, gae_lambda=self.gae_lambda, device=self.device)
        states = data["states"]
        actions = data["actions"]
        old_log_probs = data["old_log_probs"]
        returns = data["returns"]
        advantages = data["advantages"]

        actor_loss_sum = 0.0
        critic_loss_sum = 0.0
        entropy_sum = 0.0
        steps = 0

        for _ in range(self.ppo_epochs):
            for idx in RolloutBuffer.iterate_minibatches(states.shape[0], self.mini_batch_size):
                b_idx = torch.as_tensor(idx, dtype=torch.long, device=self.device)
                b_states = states[b_idx]
                b_actions = actions[b_idx]
                b_old_log_probs = old_log_probs[b_idx]
                b_returns = returns[b_idx]
                b_adv = advantages[b_idx]

                log_probs, entropy, values = self.evaluate_actions(b_states, b_actions)
                ratio = torch.exp(log_probs - b_old_log_probs)
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
                actor_loss = -torch.min(ratio * b_adv, clipped * b_adv).mean()
                critic_loss = F.mse_loss(values, b_returns)
                entropy_bonus = entropy.mean()

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_bonus

                self.actor_opt.zero_grad(set_to_none=True)
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.actor_opt.step()
                self.critic_opt.step()

                actor_loss_sum += float(actor_loss.item())
                critic_loss_sum += float(critic_loss.item())
                entropy_sum += float(entropy_bonus.item())
                steps += 1

        if steps == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}
        return {
            "actor_loss": actor_loss_sum / steps,
            "critic_loss": critic_loss_sum / steps,
            "entropy": entropy_sum / steps,
        }

    def save(self, path: str | Path) -> None:
        payload = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }
        torch.save(payload, path)

    def load(self, path: str | Path, strict: bool = True) -> None:
        payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"], strict=strict)
        self.critic.load_state_dict(payload["critic"], strict=strict)
        if "actor_opt" in payload and "critic_opt" in payload:
            self.actor_opt.load_state_dict(payload["actor_opt"])
            self.critic_opt.load_state_dict(payload["critic_opt"])

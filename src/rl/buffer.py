from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, rollout_steps: int, n_envs: int, state_dim: int, action_dim: int) -> None:
        if rollout_steps <= 0:
            raise ValueError(f"rollout_steps must be positive, got {rollout_steps}")
        if n_envs <= 0:
            raise ValueError(f"n_envs must be positive, got {n_envs}")
        if state_dim <= 0:
            raise ValueError(f"state_dim must be positive, got {state_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")
        self.rollout_steps = int(rollout_steps)
        self.n_envs = int(n_envs)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.states = np.zeros((self.rollout_steps, self.n_envs, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.rollout_steps, self.n_envs, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros((self.rollout_steps, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.rollout_steps, self.n_envs), dtype=np.float32)
        self.rewards = np.zeros((self.rollout_steps, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.rollout_steps, self.n_envs), dtype=np.float32)
        self._step = 0

    def clear(self) -> None:
        self._step = 0

    @property
    def size(self) -> int:
        return self._step * self.n_envs

    @property
    def is_full(self) -> bool:
        return self._step == self.rollout_steps

    def _assert_shape(self, name: str, arr: np.ndarray, expected_shape: tuple[int, ...]) -> None:
        if tuple(arr.shape) != expected_shape:
            raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {tuple(arr.shape)}")

    def add_step(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        if self._step >= self.rollout_steps:
            raise ValueError(
                f"RolloutBuffer is full: rollout_steps={self.rollout_steps}, n_envs={self.n_envs}, size={self.size}"
            )

        states_np = np.asarray(states, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.float32)
        log_probs_np = np.asarray(log_probs, dtype=np.float32)
        values_np = np.asarray(values, dtype=np.float32)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.float32)

        self._assert_shape("states", states_np, (self.n_envs, self.state_dim))
        self._assert_shape("actions", actions_np, (self.n_envs, self.action_dim))
        self._assert_shape("log_probs", log_probs_np, (self.n_envs,))
        self._assert_shape("values", values_np, (self.n_envs,))
        self._assert_shape("rewards", rewards_np, (self.n_envs,))
        self._assert_shape("dones", dones_np, (self.n_envs,))

        self.states[self._step] = states_np
        self.actions[self._step] = actions_np
        self.log_probs[self._step] = log_probs_np
        self.values[self._step] = values_np
        self.rewards[self._step] = rewards_np
        self.dones[self._step] = dones_np
        self._step += 1

    def build_tensors(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        if not self.is_full:
            raise ValueError(
                f"RolloutBuffer is not full: collected_steps={self._step}, expected_steps={self.rollout_steps}"
            )

        last_values_np = np.asarray(last_values, dtype=np.float32).reshape(-1)
        if tuple(last_values_np.shape) != (self.n_envs,):
            raise ValueError(f"last_values shape mismatch: expected {(self.n_envs,)}, got {tuple(last_values_np.shape)}")

        advantages = np.zeros_like(self.rewards, dtype=np.float32)  # [T, N]
        next_advantage = np.zeros(self.n_envs, dtype=np.float32)  # [N]
        next_values = last_values_np.copy()  # [N]

        for t in reversed(range(self.rollout_steps)):
            mask = 1.0 - self.dones[t]  # [N]
            delta = self.rewards[t] + gamma * next_values * mask - self.values[t]  # [N]
            next_advantage = delta + gamma * gae_lambda * mask * next_advantage  # [N]
            advantages[t] = next_advantage
            next_values = self.values[t]

        returns = advantages + self.values  # [T, N]

        states_flat = self.states.reshape(-1, self.state_dim)
        actions_flat = self.actions.reshape(-1, self.action_dim)
        old_log_probs_flat = self.log_probs.reshape(-1)
        returns_flat = returns.reshape(-1)
        advantages_flat = advantages.reshape(-1)
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)
        return {
            "states": torch.as_tensor(states_flat, dtype=torch.float32, device=device),
            "actions": torch.as_tensor(actions_flat, dtype=torch.float32, device=device),
            "old_log_probs": torch.as_tensor(old_log_probs_flat, dtype=torch.float32, device=device),
            "returns": torch.as_tensor(returns_flat, dtype=torch.float32, device=device),
            "advantages": torch.as_tensor(advantages_flat, dtype=torch.float32, device=device),
        }

    @staticmethod
    def iterate_minibatches(total_size: int, mini_batch_size: int) -> list[np.ndarray]:
        if total_size <= 0:
            raise ValueError(f"total_size must be positive, got {total_size}")
        if mini_batch_size <= 0:
            raise ValueError(f"mini_batch_size must be positive, got {mini_batch_size}")
        indices = np.arange(total_size)
        np.random.shuffle(indices)
        return [indices[i : i + mini_batch_size] for i in range(0, total_size, mini_batch_size)]

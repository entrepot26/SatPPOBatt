from __future__ import annotations

import torch
import torch.nn as nn


def build_mlp(input_dim: int, hidden_sizes: list[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = input_dim
    for hidden in hidden_sizes:
        layers.append(nn.Linear(prev, hidden))
        layers.append(nn.Tanh())
        prev = hidden
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class ActorNet(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        self.backbone = build_mlp(obs_dim, hidden_sizes, 2 * action_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.backbone(obs)
        alpha_raw, beta_raw = torch.chunk(x, 2, dim=-1)
        alpha = torch.nn.functional.softplus(alpha_raw) + 1.0
        beta = torch.nn.functional.softplus(beta_raw) + 1.0
        return alpha, beta


class CriticNet(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        self.value = build_mlp(obs_dim, hidden_sizes, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(obs).squeeze(-1)


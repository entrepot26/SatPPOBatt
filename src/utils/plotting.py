from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .io import ensure_dir


def _save_fig(path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    plt.tight_layout()
    plt.savefig(p, dpi=160)
    plt.close()


def plot_train_rewards(episode_ids: list[int], rewards: list[float], out_path: str | Path) -> None:
    if not episode_ids:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(episode_ids, rewards, marker="o", linewidth=1.5)
    plt.title("PPO Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(alpha=0.3)
    _save_fig(out_path)


def plot_round_time(round_ids: list[int], round_times: list[float], out_path: str | Path) -> None:
    if not round_ids:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(round_ids, round_times, linewidth=1.5)
    plt.title("Round Total Time")
    plt.xlabel("Round")
    plt.ylabel("Seconds")
    plt.grid(alpha=0.3)
    _save_fig(out_path)


def plot_avg_utility(round_ids: list[int], utility: list[float], out_path: str | Path) -> None:
    if not round_ids:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(round_ids, utility, linewidth=1.5)
    plt.title("Average Battery Utility")
    plt.xlabel("Round")
    plt.ylabel("Utility")
    plt.grid(alpha=0.3)
    _save_fig(out_path)


def plot_reward_indicator(
    round_ids: list[int],
    values: list[float],
    out_path: str | Path,
    title: str,
    ylabel: str = "Value",
) -> None:
    if not round_ids:
        return
    plt.figure(figsize=(8, 4))
    plt.plot(round_ids, values, linewidth=1.5)
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    _save_fig(out_path)


def plot_cumulative_aging(round_ids: list[int], avg_aging: list[float], out_path: str | Path) -> None:
    if not round_ids:
        return
    cumulative = []
    total = 0.0
    for value in avg_aging:
        total += value
        cumulative.append(total)
    plt.figure(figsize=(8, 4))
    plt.plot(round_ids, cumulative, linewidth=1.5)
    plt.title("Cumulative Battery Aging")
    plt.xlabel("Round")
    plt.ylabel("Aging")
    plt.grid(alpha=0.3)
    _save_fig(out_path)


def plot_energy_breakdown(
    round_ids: list[int],
    comp_energy: list[float],
    comm_energy: list[float],
    out_path: str | Path,
) -> None:
    if not round_ids:
        return
    plt.figure(figsize=(8, 4))
    plt.stackplot(round_ids, comp_energy, comm_energy, labels=["Computation", "Communication"], alpha=0.8)
    plt.title("Energy Breakdown")
    plt.xlabel("Round")
    plt.ylabel("Energy")
    plt.legend(loc="upper left")
    _save_fig(out_path)

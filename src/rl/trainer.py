from __future__ import annotations

from pathlib import Path

import numpy as np

from src.env.leo_env import LEOEnv
from src.rl.buffer import RolloutBuffer
from src.rl.ppo_agent import PPOAgent
from src.simulator.metrics import MetricsStore


def _apply_eval_sunlit_action_mask(state: np.ndarray, actions: np.ndarray, sats_per_orbit: int) -> np.ndarray:
    masked_actions = np.asarray(actions, dtype=np.float32).copy()
    state_np = np.asarray(state, dtype=np.float32)
    sunlit = state_np[:, 2::4]
    if sunlit.shape != (state_np.shape[0], sats_per_orbit):
        raise ValueError(
            f"sunlit slice shape mismatch: expected {(state_np.shape[0], sats_per_orbit)}, got {tuple(sunlit.shape)}"
        )
    sunlit_mask = sunlit > 0.5
    masked_actions[:, 0::2] = np.where(sunlit_mask, 1.0, masked_actions[:, 0::2])
    masked_actions[:, 1::2] = np.where(sunlit_mask, 1.0, masked_actions[:, 1::2])
    return masked_actions


def train_ppo(
    env: LEOEnv,
    agent: PPOAgent,
    cfg: dict,
    metrics: MetricsStore,
    checkpoint_last: str | Path,
    checkpoint_best: str | Path,
) -> dict[str, float]:
    total_episodes = int(cfg["total_episodes"] if "total_episodes" in cfg else cfg["num_episodes"])
    n_envs = int(env.num_orbits)
    rollout_size = int(cfg["rollout_size"])
    assert rollout_size % n_envs == 0, f"rollout_size must be divisible by n_envs: rollout_size={rollout_size}, n_envs={n_envs}"
    simulator_steps_per_episode = rollout_size // n_envs
    cfg_steps = int(
        cfg["simulator_steps_per_episode"]
        if "simulator_steps_per_episode" in cfg
        else cfg["train_rounds_per_episode"]
    )
    assert cfg_steps == simulator_steps_per_episode, (
        "simulator_steps_per_episode must equal rollout_size // n_envs: "
        f"simulator_steps_per_episode={cfg_steps}, rollout_size={rollout_size}, n_envs={n_envs}"
    )
    buffer = RolloutBuffer(
        rollout_steps=simulator_steps_per_episode,
        n_envs=n_envs,
        state_dim=env.obs_dim,
        action_dim=env.action_dim,
    )

    best_reward = -float("inf")
    last_update = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

    for episode in range(1, total_episodes + 1):
        state = env.reset()
        episode_rewards: list[float] = []
        for _ in range(simulator_steps_per_episode):
            actions, log_probs, values = agent.act(state, deterministic=False)
            next_state, reward, info = env.step(actions)

            if "orbit_rewards" in info:
                rewards = np.asarray(info["orbit_rewards"], dtype=np.float32)
            else:
                rewards = np.full(n_envs, reward, dtype=np.float32)
            if tuple(rewards.shape) != (n_envs,):
                raise ValueError(f"orbit reward shape mismatch: expected {(n_envs,)}, got {tuple(rewards.shape)}")
            dones = np.zeros(n_envs, dtype=np.float32)
            buffer.add_step(
                states=state,
                actions=actions,
                log_probs=log_probs,
                values=values,
                rewards=rewards,
                dones=dones,
            )

            metrics.add_round(info["round_metrics"])
            metrics.add_orbits(info["orbit_metrics"])
            metrics.add_sats(info["sat_metrics"])

            episode_rewards.append(float(np.mean(rewards)))
            state = next_state

        last_values = agent.estimate_values(state)
        last_update = agent.update(buffer=buffer, last_values=last_values)
        buffer.clear()

        avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
        metrics.add_train_reward(episode_id=episode, avg_reward=avg_reward)
        agent.save(checkpoint_last)
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(checkpoint_best)

    return {
        "best_reward": best_reward,
        "actor_loss": last_update["actor_loss"],
        "critic_loss": last_update["critic_loss"],
    }


def evaluate_policy(
    env: LEOEnv,
    agent: PPOAgent,
    num_rounds: int,
    metrics: MetricsStore,
    enable_sunlit_mask: bool = True,
) -> float:
    state = env.reset()
    rewards: list[float] = []
    for _ in range(num_rounds):
        actions, _, _ = agent.act(state, deterministic=True)
        if enable_sunlit_mask:
            actions = _apply_eval_sunlit_action_mask(state=state, actions=actions, sats_per_orbit=env.sats_per_orbit)
        next_state, reward, info = env.step(actions)
        metrics.add_round(info["round_metrics"])
        metrics.add_orbits(info["orbit_metrics"])
        metrics.add_sats(info["sat_metrics"])
        rewards.append(reward)
        state = next_state
    return float(np.mean(rewards)) if rewards else 0.0

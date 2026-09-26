from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.env.leo_env import LEOEnv
from src.rl.ppo_agent import PPOAgent
from src.rl.trainer import train_ppo
from src.simulator.logger import setup_logger
from src.simulator.metrics import MetricsStore
from src.simulator.trace_loader import load_trace_csv
from src.utils.config import load_config, resolve_path
from src.utils.io import build_run_info, create_experiment_dir, save_experiment_summary, save_json, save_yaml
from src.utils.plotting import (
    plot_avg_utility,
    plot_cumulative_aging,
    plot_energy_breakdown,
    plot_reward_indicator,
    plot_round_time,
    plot_train_rewards,
)
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for LEO FL simulator")
    parser.add_argument("--config", default="configs/default.yaml", help="Base config yaml path")
    parser.add_argument("--override", action="append", default=[], help="Extra yaml paths to override base config")
    parser.add_argument("--results-root", default=None, help="Override results root directory")
    return parser.parse_args()


def pick_device(device_cfg: str) -> torch.device:
    if device_cfg != "auto":
        return torch.device(device_cfg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    cfg = load_config(args.config, args.override)
    if args.results_root:
        cfg["results_root"] = args.results_root

    seed_record = set_global_seed(int(cfg["seed"]))
    device = pick_device(str(cfg["device"]))

    exp_dir = create_experiment_dir(resolve_path(root, cfg["results_root"]))
    logger = setup_logger(exp_dir / "train.log", level=logging.INFO)
    logger.info("Experiment directory: %s", exp_dir)

    trace_path = resolve_path(root, cfg["trace_csv_path"])
    trace = load_trace_csv(
        path=trace_path,
        num_orbits=cfg["num_orbits"],
        sats_per_orbit=cfg["sats_per_orbit"],
        slot_len=cfg["slot_len"],
    )

    env = LEOEnv(cfg=cfg, trace=trace)
    agent = PPOAgent(obs_dim=env.obs_dim, action_dim=env.action_dim, cfg=cfg, device=device)
    metrics = MetricsStore()

    checkpoint_last = exp_dir / "ppo_checkpoint_last.pt"
    checkpoint_best = exp_dir / "ppo_checkpoint_best.pt"
    train_stats = train_ppo(
        env=env,
        agent=agent,
        cfg=cfg,
        metrics=metrics,
        checkpoint_last=checkpoint_last,
        checkpoint_best=checkpoint_best,
    )
    metrics.save_train(exp_dir)
    partition_path = env.export_partition_history(
        resolve_path(root, cfg["partitions_dir"]) / f"local_train_images_train_{exp_dir.name}.csv"
    )

    reward_rows = metrics.train_reward_rows
    episode_ids = [int(row["episode_id"]) for row in reward_rows]
    rewards = [float(row["avg_reward"]) for row in reward_rows]
    plot_train_rewards(episode_ids, rewards, exp_dir / "rl_train_rewards.png")

    round_rows = metrics.round_rows
    summary = save_experiment_summary(round_rows, exp_dir / "summary.csv", sat_rows=metrics.sat_rows)
    print(
        f"[summary][train] 总轮次={summary['total_rounds']}, "
        f"总时间={float(summary['total_time']):.6f}, "
        f"总损耗(阴影卫星battery_degradation总和)={float(summary['total_loss']):.6f}"
    )
    logger.info(
        "Summary: rounds=%d, total_time=%.6f, total_loss=%.6f",
        int(summary["total_rounds"]),
        float(summary["total_time"]),
        float(summary["total_loss"]),
    )
    round_ids = list(range(len(round_rows)))
    plot_round_time(round_ids, [float(r["round_time"]) for r in round_rows], exp_dir / "round_total_time.png")
    plot_avg_utility(
        round_ids,
        [float(r["avg_battery_utility"]) for r in round_rows],
        exp_dir / "avg_battery_utility.png",
    )
    plot_reward_indicator(
        round_ids,
        [float(r["reward3_ideal_dod"]) for r in round_rows],
        exp_dir / "reward3_ideal_dod.png",
        "Reward3: Ideal DoD",
    )
    plot_reward_indicator(
        round_ids,
        [float(r["reward4_in_orbit_late"]) for r in round_rows],
        exp_dir / "reward4_in_orbit_late.png",
        "Reward4: In-Orbit Late",
    )
    plot_reward_indicator(
        round_ids,
        [float(r["reward5_ex_orbit_late"]) for r in round_rows],
        exp_dir / "reward5_ex_orbit_late.png",
        "Reward5: Ex-Orbit Late",
    )
    plot_reward_indicator(
        round_ids,
        [float(r["reward6_energy_consumed"]) for r in round_rows],
        exp_dir / "reward6_energy_consumed.png",
        "Reward6: Shadow Energy Consumed",
        ylabel="Energy",
    )
    plot_reward_indicator(
        round_ids,
        [float(r["reward7_battery_degradation"]) for r in round_rows],
        exp_dir / "reward7_battery_degradation.png",
        "Reward7: Shadow Battery Degradation",
        ylabel="Degradation",
    )
    plot_cumulative_aging(round_ids, [float(r["avg_aging"]) for r in round_rows], exp_dir / "cumulative_aging.png")
    plot_energy_breakdown(
        round_ids,
        [float(r["comp_energy"]) for r in round_rows],
        [float(r["comm_energy"]) for r in round_rows],
        exp_dir / "energy_breakdown.png",
    )

    save_yaml(cfg, exp_dir / "config_snapshot.yaml")
    run_info = build_run_info(cfg, mode="train")
    run_info.update(
        {
            "device": str(device),
            "partition_file": str(partition_path),
            "seed_record": seed_record,
            "train_stats": train_stats,
        }
    )
    save_json(run_info, exp_dir / "run_info.json")

    # 本项目不做真实FL，这里保存占位模型文件用于结果归档。
    torch.save({"model_name": cfg["model_name"], "model_bits": cfg["model_bits"]}, exp_dir / "final_global_model.pt")

    logger.info("Training completed. Best episode reward: %.4f", train_stats["best_reward"])


if __name__ == "__main__":
    main()

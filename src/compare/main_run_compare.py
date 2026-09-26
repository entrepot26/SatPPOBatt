from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.compare.env_fixed import FixedPolicyEnv
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
)
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-policy comparison experiments")
    parser.add_argument("--config", default="configs/default.yaml", help="Base config yaml path")
    parser.add_argument("--override", action="append", default=[], help="Extra yaml paths to override base config")
    parser.add_argument("--results-root", default=None, help="Override results root directory")
    parser.add_argument(
        "--mode",
        default="all",
        choices=["compare1", "compare2", "all"],
        help="Comparison mode to run",
    )
    return parser.parse_args()


def _run_single_compare(
    mode: str,
    cfg: dict,
    trace,
    root: Path,
    compare_root: Path,
) -> dict:
    exp_dir = create_experiment_dir(compare_root / mode)
    logger = setup_logger(exp_dir / f"{mode}.log", level=logging.INFO)
    logger.info("Comparison mode: %s", mode)
    logger.info("Experiment directory: %s", exp_dir)

    env = FixedPolicyEnv(cfg=cfg, trace=trace, mode=mode)
    metrics = MetricsStore()

    _ = env.reset()
    rewards: list[float] = []
    num_rounds = int(cfg["eval_rounds"])
    for _ in range(num_rounds):
        _, reward, info = env.step()
        metrics.add_round(info["round_metrics"])
        metrics.add_orbits(info["orbit_metrics"])
        metrics.add_sats(info["sat_metrics"])
        rewards.append(float(reward))

    metrics.save_eval(exp_dir)
    partition_path = env.export_partition_history(
        resolve_path(root, cfg["partitions_dir"]) / f"local_train_images_{mode}_{exp_dir.name}.csv"
    )

    round_rows = metrics.round_rows
    summary = save_experiment_summary(round_rows, exp_dir / "summary.csv", sat_rows=metrics.sat_rows)
    print(
        f"[summary][{mode}] 总轮次={summary['total_rounds']}, "
        f"总时间={float(summary['total_time']):.6f}, "
        f"总损耗(阴影卫星battery_degradation总和)={float(summary['total_loss']):.6f}"
    )
    logger.info(
        "Summary: rounds=%d, total_time=%.6f, total_loss=%.6f",
        int(summary["total_rounds"]),
        float(summary["total_time"]),
        float(summary["total_loss"]),
    )
    round_ids = [int(r["round_id"]) for r in round_rows]
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
    run_info = build_run_info(cfg, mode=mode)
    run_info.update(
        {
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "partition_file": str(partition_path),
            "comm_coef_sun": float(cfg["comm_coef_sun"]),
            "comm_coef_dark": float(cfg["comm_coef_dark"]),
            "comp_coef_sun": float(cfg["comp_coef_sun"]),
            "comp_coef_dark": float(cfg["comp_coef_dark"]),
            "no_inter_satellite_link": mode == "compare2",
        }
    )
    save_json(run_info, exp_dir / "run_info.json")

    torch.save({"model_name": cfg["model_name"], "model_bits": cfg["model_bits"]}, exp_dir / "final_global_model.pt")
    logger.info("Comparison completed. Avg reward: %.4f", run_info["avg_reward"])
    return {"mode": mode, "exp_dir": str(exp_dir), "avg_reward": run_info["avg_reward"]}


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    cfg = load_config(args.config, args.override)
    if args.results_root:
        cfg["results_root"] = args.results_root

    seed_record = set_global_seed(int(cfg["seed"]))
    trace_path = resolve_path(root, cfg["trace_csv_path"])
    trace = load_trace_csv(
        path=trace_path,
        num_orbits=cfg["num_orbits"],
        sats_per_orbit=cfg["sats_per_orbit"],
        slot_len=cfg["slot_len"],
    )

    base_results_root = resolve_path(root, cfg["results_root"])
    compare_root = base_results_root / "compare"
    modes = ["compare1", "compare2"] if args.mode == "all" else [args.mode]

    summary = []
    for mode in modes:
        result = _run_single_compare(
            mode=mode,
            cfg=cfg,
            trace=trace,
            root=root,
            compare_root=compare_root,
        )
        result["seed_record"] = seed_record
        summary.append(result)

    summary_dir = create_experiment_dir(compare_root / "summary")
    save_yaml({"results": summary, "mode": args.mode}, summary_dir / "summary.yaml")


if __name__ == "__main__":
    main()

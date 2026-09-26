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

from src.env.battery import battery_degradation, battery_utility_new, cycle_count_from_dod, depth_of_discharge, mean_nonzero
from src.env.communication import (
    DEFAULT_SAT_GROUND_DISTANCE_KM,
    LIGHT_SPEED_VACUUM_MPS,
    propagation_delay,
    step4_upload_energy,
    uplink_time_per_satellite,
)
from src.env.computation import local_train_energy, local_train_time
from src.simulator.logger import setup_logger
from src.simulator.metrics import MetricsStore
from src.simulator.trace_loader import TraceData, load_trace_csv
from src.utils.config import load_config, resolve_path
from src.utils.io import (
    build_run_info,
    create_experiment_dir,
    save_experiment_summary,
    save_json,
    save_rows_csv,
    save_yaml,
)
from src.utils.new_data_partion_simulation import satellite_client_names, simulate_clients_data_number
from src.utils.plotting import (
    plot_avg_utility,
    plot_cumulative_aging,
    plot_energy_breakdown,
    plot_reward_indicator,
    plot_round_time,
)
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compare3 baseline with fixed 900-second rounds")
    parser.add_argument("--config", default="configs/default.yaml", help="Base config yaml path")
    parser.add_argument("--override", action="append", default=[], help="Extra yaml paths to override base config")
    parser.add_argument("--results-root", default=None, help="Override results root directory")
    return parser.parse_args()


def _sample_interval_status(
    trace: TraceData,
    round_start_time: float,
    round_duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Compare3 uses all status samples in the current 900s window.
    slot_len = max(float(trace.slot_len), 1.0)
    steps = max(int(np.ceil(round_duration / slot_len)), 1)
    sample_times = round_start_time + np.arange(steps, dtype=np.float64) * slot_len

    connected_list = []
    sunlit_list = []
    for t in sample_times:
        connected, sunlit = trace.status_at_time(float(t))
        connected_list.append(connected)
        sunlit_list.append(sunlit)
    return np.stack(connected_list, axis=0), np.stack(sunlit_list, axis=0)


def _compute_num_rounds_from_trace(trace: TraceData, round_duration: float) -> int:
    total_seconds = max(float(trace.times[-1] - trace.times[0]), 0.0)
    if round_duration <= 0.0:
        raise ValueError("compare3 round duration must be positive")
    return max(int(np.floor(total_seconds / round_duration)), 1)


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    cfg = load_config(args.config, args.override)
    if args.results_root:
        cfg["results_root"] = args.results_root

    seed_record = set_global_seed(int(cfg["seed"]))

    # Compare3 definition: fixed 900 seconds per round.
    round_duration = float(cfg["compare3_round_seconds"])

    # Keep the trace source explicit to match the requested method.
    trace_path = resolve_path(root, str(cfg["trace_csv_path"]))
    trace = load_trace_csv(
        path=trace_path,
        num_orbits=cfg["num_orbits"],
        sats_per_orbit=cfg["sats_per_orbit"],
        slot_len=cfg["slot_len"],
    )

    base_results_root = resolve_path(root, cfg["results_root"])
    exp_dir = create_experiment_dir(base_results_root / "compare" / "compare3")
    logger = setup_logger(exp_dir / "compare3.log", level=logging.INFO)
    logger.info("Compare3 directory: %s", exp_dir)
    logger.info("Trace path: %s", trace_path)
    logger.info("Round duration (seconds): %.1f", round_duration)

    num_orbits = int(cfg["num_orbits"])
    sats_per_orbit = int(cfg["sats_per_orbit"])
    n_sats_total = num_orbits * sats_per_orbit
    num_rounds = _compute_num_rounds_from_trace(trace, round_duration)

    # Compare3 uses fixed max CPU/Tx settings, no RL action control.
    cpu_freq_fixed = float(cfg["cpu_freq_max"])
    tx_power_fixed = float(cfg["tx_power_max"])

    propagation_speed_mps = float(cfg["propagation_speed_mps"])
    sat_ground_distance_km = float(cfg["distance_ground_station_sat_km"])
    prop_delay_sat_ground = propagation_delay(sat_ground_distance_km, propagation_speed_mps)

    battery_capacity = float(cfg["battery_capacity"])

    partition_columns = satellite_client_names(num_orbits, sats_per_orbit)
    partition_rows: list[dict[str, int]] = []
    partition_rng = np.random.default_rng(int(cfg["partition_seed"]))

    metrics = MetricsStore()
    total_round_loss = 0.0
    trace_start_time = float(trace.times[0])

    for round_id in range(num_rounds):
        round_start_time = trace_start_time + round_id * round_duration
        round_end_time = round_start_time + round_duration

        connected_start, sunlit_start = trace.status_at_time(round_start_time)
        connected_series, sunlit_series = _sample_interval_status(trace, round_start_time, round_duration)

        # Satellite participates if it has at least one GS connection during current round.
        connected_any = np.any(connected_series > 0, axis=0)
        # True means all connected moments are sunlit; these satellites produce no battery loss this round.
        fully_sunlit_when_connected = np.ones((num_orbits, sats_per_orbit), dtype=bool)
        for orbit_id in range(num_orbits):
            for sat_id in range(sats_per_orbit):
                if not connected_any[orbit_id, sat_id]:
                    continue
                conn_mask = connected_series[:, orbit_id, sat_id] > 0
                fully_sunlit_when_connected[orbit_id, sat_id] = bool(
                    np.all(sunlit_series[:, orbit_id, sat_id][conn_mask] > 0)
                )
        loss_mask = connected_any & (~fully_sunlit_when_connected)
        participant_count = int(np.count_nonzero(connected_any))

        local_train_images_flat = simulate_clients_data_number(
            data_total_sample=int(cfg["data_total_sample"]),
            client_number=n_sats_total,
            alpha=float(cfg["dirichlet_alpha"]),
            min_images=int(cfg["data_partition_min_images"]),
            num_classes=int(cfg["data_partition_num_classes"]),
            rng=partition_rng,
        )
        local_train_images = local_train_images_flat.reshape(num_orbits, sats_per_orbit)

        # Save round partition sample for later export, same style as other experiments.
        partition_row: dict[str, int] = {"round_id": int(round_id)}
        for name, count in zip(partition_columns, local_train_images_flat):
            partition_row[name] = int(count)
        partition_rows.append(partition_row)

        step2_time_arr = np.zeros((num_orbits, sats_per_orbit), dtype=np.float64)
        comp_energy_arr = np.zeros((num_orbits, sats_per_orbit), dtype=np.float64)
        comm_energy_arr = np.zeros((num_orbits, sats_per_orbit), dtype=np.float64)
        consume_energy_arr = np.zeros((num_orbits, sats_per_orbit), dtype=np.float64)
        degradation_arr = np.zeros((num_orbits, sats_per_orbit), dtype=np.float64)
        utility_arr = np.zeros((num_orbits, sats_per_orbit), dtype=np.float64)
        dod_before_arr = np.zeros((num_orbits, sats_per_orbit), dtype=np.float64)
        dod_after_arr = np.zeros((num_orbits, sats_per_orbit), dtype=np.float64)
        upload_time_arr = np.zeros((num_orbits, sats_per_orbit), dtype=np.float64)

        sat_rows: list[dict] = []
        round_loss = 0.0

        for orbit_id in range(num_orbits):
            for sat_id in range(sats_per_orbit):
                # Compare3 simplification:
                # round-level degradation is computed independently for each round.
                # So each satellite starts this round from full battery for loss accounting.
                e_before = float(battery_capacity)
                d_before = depth_of_discharge(e_before, battery_capacity)
                dod_before_arr[orbit_id, sat_id] = d_before

                images = int(local_train_images[orbit_id, sat_id])
                is_participating = bool(connected_any[orbit_id, sat_id])
                has_loss = bool(loss_mask[orbit_id, sat_id])

                step2_time = 0.0
                local_energy = 0.0
                upload_time = 0.0
                upload_energy = 0.0
                consume_energy = 0.0

                if is_participating:
                    # Local compute and uplink are fixed to max settings in compare3.
                    step2_time = float(
                        local_train_time(
                            np.asarray([images], dtype=np.float64),
                            cfg["cpu_cycles_per_image"],
                            np.asarray([cpu_freq_fixed], dtype=np.float64),
                        )[0]
                    )
                    upload_time = float(
                        uplink_time_per_satellite(
                            model_bits=cfg["model_bits"],
                            # Simplification: use all round participants as shared uplink count.
                            num_connected=max(participant_count, 1),
                            tx_power=tx_power_fixed,
                            bandwidth_ul=cfg["bandwidth_ul"],
                            channel_const=cfg["channel_const"],
                            prop_delay_ground_sat=prop_delay_sat_ground,
                        )
                    )

                    if has_loss:
                        local_energy = float(
                            local_train_energy(
                                np.asarray([images], dtype=np.float64),
                                cfg["cpu_cycles_per_image"],
                                np.asarray([cpu_freq_fixed], dtype=np.float64),
                                cfg["kappa"],
                            )[0]
                        )
                        upload_energy = float(step4_upload_energy(tx_power_fixed, max(upload_time - prop_delay_sat_ground, 0.0)))
                        consume_energy = local_energy + upload_energy

                e_after = max(e_before - consume_energy, 0.0)
                d_after = depth_of_discharge(e_after, battery_capacity)
                dod_after_arr[orbit_id, sat_id] = d_after

                # Explicitly compute cycle-count terms per requested compare3 definition.
                _ = cycle_count_from_dod(
                    d_before,
                    cfg["degrade_k1"],
                    cfg["degrade_k2"],
                    cfg["degrade_k3"],
                    cfg["degrade_k4"],
                    cfg["degrade_k5"],
                )
                _ = cycle_count_from_dod(
                    d_after,
                    cfg["degrade_k1"],
                    cfg["degrade_k2"],
                    cfg["degrade_k3"],
                    cfg["degrade_k4"],
                    cfg["degrade_k5"],
                )

                degradation = battery_degradation(
                    d_before,
                    d_after,
                    cfg["degrade_k1"],
                    cfg["degrade_k2"],
                    cfg["degrade_k3"],
                    cfg["degrade_k4"],
                    cfg["degrade_k5"],
                )
                utility = battery_utility_new(d_before, d_after)

                step2_time_arr[orbit_id, sat_id] = step2_time
                upload_time_arr[orbit_id, sat_id] = upload_time
                comp_energy_arr[orbit_id, sat_id] = local_energy
                comm_energy_arr[orbit_id, sat_id] = upload_energy
                consume_energy_arr[orbit_id, sat_id] = consume_energy
                degradation_arr[orbit_id, sat_id] = degradation
                utility_arr[orbit_id, sat_id] = utility

                round_loss += float(degradation)

                sat_rows.append(
                    {
                        "round_id": round_id,
                        "orbit_id": orbit_id + 1,
                        "sat_id": sat_id + 1,
                        "step1_time": 0.0,
                        "step2_time": float(step2_time),
                        "local_train_images": images,
                        "cpu_freq": cpu_freq_fixed if is_participating else 0.0,
                        "tx_power": tx_power_fixed if is_participating else 0.0,
                        "comp_energy": float(local_energy),
                        "comm_energy": float(upload_energy),
                        "battery_energy_before": float(e_before),
                        "battery_energy_after": float(e_after),
                        "dod_before": float(d_before),
                        "dod_after": float(d_after),
                        "battery_degradation": float(degradation),
                        "battery_utility": float(utility),
                        "is_connected_round_start": int(connected_start[orbit_id, sat_id]),
                        "is_sunlit_round_start": int(sunlit_start[orbit_id, sat_id]),
                    }
                )

        orbit_rows: list[dict] = []
        for orbit_id in range(num_orbits):
            orbit_participate = connected_any[orbit_id]
            orbit_local_time = float(np.max(step2_time_arr[orbit_id][orbit_participate])) if np.any(orbit_participate) else 0.0
            orbit_step4_time = float(np.max(upload_time_arr[orbit_id][orbit_participate])) if np.any(orbit_participate) else 0.0
            orbit_rows.append(
                {
                    "round_id": round_id,
                    "orbit_id": orbit_id + 1,
                    "step3_time": 0.0,
                    "step4_time": orbit_step4_time,
                    "orbit_local_time": orbit_local_time,
                    "orbit_wait_time": 0.0,
                    "orbit_energy_consumed": float(np.sum(consume_energy_arr[orbit_id])),
                    "orbit_battery_utility": float(mean_nonzero(utility_arr[orbit_id])),
                    "orbit_upload_energy": float(np.sum(comm_energy_arr[orbit_id])),
                }
            )

        reward3_ideal_dod = float(np.mean(np.abs(dod_after_arr - float(cfg["ideal_dod"]))))
        round_energy = float(np.sum(consume_energy_arr))
        round_loss = float(round_loss)
        total_round_loss += round_loss

        round_row = {
            "round_id": round_id,
            "round_start_time": float(round_start_time),
            "round_end_time": float(round_end_time),
            "round_time": float(round_duration),
            "round_time_penalty": float(round_duration),
            "step5_time": 0.0,
            "avg_dod": float(np.mean(dod_after_arr)),
            "avg_aging": float(np.mean(degradation_arr)),
            "avg_battery_utility": float(mean_nonzero(utility_arr.flatten())),
            "total_energy": round_energy,
            "reward": -round_loss,
            "comp_energy": float(np.sum(comp_energy_arr)),
            "comm_energy": float(np.sum(comm_energy_arr)),
            "reward1_utility": float(mean_nonzero(utility_arr.flatten())),
            "reward2_round_time": float(round_duration),
            "reward3_ideal_dod": reward3_ideal_dod,
            "reward4_in_orbit_late": 0.0,
            "reward5_ex_orbit_late": 0.0,
            "reward6_energy_consumed": round_energy,
            "reward7_battery_degradation": round_loss,
            "low_energy_events": 0,
            "negative_energy_events": 0,
        }

        metrics.add_round(round_row)
        metrics.add_orbits(orbit_rows)
        metrics.add_sats(sat_rows)

        loss_sat_count = int(np.count_nonzero(loss_mask))
        #print(
        #    f"[compare3] round={round_id + 1:03d}/{num_rounds}, "
        #    f"participants={participant_count}, loss_sats={loss_sat_count}, round_loss={round_loss:.6f}"
        #)

    metrics.save_eval(exp_dir)
    partition_path = resolve_path(root, cfg["partitions_dir"]) / f"local_train_images_compare3_{exp_dir.name}.csv"
    save_rows_csv(partition_path, partition_rows, ["round_id", *partition_columns])

    round_rows = metrics.round_rows
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

    # Compare3 total_loss should be sum of round-level reward7_battery_degradation.
    summary = save_experiment_summary(round_rows, exp_dir / "summary.csv")
    print(
        f"[summary][compare3] 总轮次={summary['total_rounds']}, "
        f"总时间={float(summary['total_time']):.6f}, "
        f"总损耗(阴影卫星battery_degradation总和)={float(summary['total_loss']):.6f}"
    )
    logger.info(
        "Summary: rounds=%d, total_time=%.6f, total_loss=%.6f",
        int(summary["total_rounds"]),
        float(summary["total_time"]),
        float(summary["total_loss"]),
    )

    save_yaml(cfg, exp_dir / "config_snapshot.yaml")
    run_info = build_run_info(cfg, mode="compare3")
    run_info.update(
        {
            "round_seconds": round_duration,
            "num_rounds": num_rounds,
            "partition_file": str(partition_path),
            "seed_record": seed_record,
            "trace_path": str(trace_path),
            "cpu_freq_fixed": cpu_freq_fixed,
            "tx_power_fixed": tx_power_fixed,
            "total_round_loss": total_round_loss,
        }
    )
    save_json(run_info, exp_dir / "run_info.json")

    torch.save({"model_name": cfg["model_name"], "model_bits": cfg["model_bits"]}, exp_dir / "final_global_model.pt")
    logger.info("Compare3 completed. Total loss: %.6f", total_round_loss)


if __name__ == "__main__":
    main()

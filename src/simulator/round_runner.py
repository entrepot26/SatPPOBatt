from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.env.battery import (
    battery_degradation,
    battery_utility_new,
    depth_of_discharge,
    mean_nonzero,
    update_battery_energy,
)
from src.env.communication import (
    DEFAULT_INTER_SATELLITE_DISTANCE_KM,
    DEFAULT_SAT_GROUND_DISTANCE_KM,
    LIGHT_SPEED_VACUUM_MPS,
    action_to_tx_power,
    propagation_delay,
    step1_forward_energy,
    step1_satellite_time,
    step3_energy_per_satellite,
    step3_ring_allreduce_time,
    step4_upload_energy as compute_step4_upload_energy,
    uplink_time_per_satellite,
)
from src.env.computation import action_to_cpu_freq, local_train_energy, local_train_time, step5_ground_aggregation_time
from src.env.timing import next_slot_time, orbit_local_time, round_time_penalty_for_reward, round_total_time
from src.simulator.trace_loader import TraceData

EPS = 1e-12


@dataclass
class RoundResult:
    next_time: float
    reward: float
    orbit_rewards: np.ndarray  # [O]
    battery_after: np.ndarray  # [O, J]
    round_row: dict
    orbit_rows: list[dict]
    sat_rows: list[dict]


def _norm_positive(value: np.ndarray | float, reference: float) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    ref = max(float(reference), EPS)
    return arr / (arr + ref)


def _shadow_mean(values: np.ndarray, sunlit_status: np.ndarray) -> float:
    shadow_mask = np.asarray(sunlit_status, dtype=np.int64) <= 0
    if not np.any(shadow_mask):
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)[shadow_mask]))


def _normalized_reward_terms(
    orbit_util: np.ndarray,
    round_time_penalty: float,
    best_dod: np.ndarray,
    in_orbit_late: np.ndarray,
    orbit_late: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    util_ref = float(cfg["reward_norm_utility_ref"])
    time_ref = float(cfg["reward_norm_round_time_ref"])
    late_ref = float(cfg["reward_norm_late_ref"])
    best_dod_ref = max(float(cfg["reward_norm_best_dod_ref"]), EPS)

    util_norm = _norm_positive(orbit_util, util_ref)
    round_time_norm = float(_norm_positive(round_time_penalty, time_ref))
    best_dod_norm = np.clip(np.asarray(best_dod, dtype=np.float64) / best_dod_ref, 0.0, 1.0)
    in_orbit_late_norm = _norm_positive(in_orbit_late, late_ref)
    orbit_late_norm = _norm_positive(orbit_late, late_ref)
    return util_norm, round_time_norm, best_dod_norm, in_orbit_late_norm, orbit_late_norm


def _wait_until_connected(trace: TraceData, orbit_id: int, start_time: float, slot_len: float) -> tuple[float, np.ndarray]:
    current = float(start_time)
    total_wait = 0.0
    max_loops = trace.num_slots + 2
    for _ in range(max_loops):
        connected, _ = trace.status_at_time(current)
        orbit_connected = connected[orbit_id]
        idx = np.where(orbit_connected > 0)[0]
        if idx.size > 0:
            return total_wait, idx
        nxt = next_slot_time(current, slot_len)
        total_wait += nxt - current
        current = nxt
    # 轨迹周期内没有可连接卫星时退化到不等待，以避免死循环
    connected, _ = trace.status_at_time(start_time)
    idx = np.where(connected[orbit_id] > 0)[0]
    return total_wait, idx


def run_round(
    round_id: int,
    start_time: float,
    battery_energy: np.ndarray,
    action: np.ndarray,
    local_train_images: np.ndarray,
    trace: TraceData,
    cfg: dict,
) -> RoundResult:
    num_orbits = int(cfg["num_orbits"])
    sats_per_orbit = int(cfg["sats_per_orbit"])
    model_bits = float(cfg["model_bits"])
    slot_len = float(cfg["slot_len"])
    propagation_speed_mps = float(cfg["propagation_speed_mps"])
    sat_ground_distance_km = float(cfg["distance_ground_station_sat_km"])
    inter_sat_distance_km = float(cfg["distance_satellite_satellite_km"])
    prop_delay_sat_ground = propagation_delay(sat_ground_distance_km, propagation_speed_mps)
    prop_delay_isl = propagation_delay(inter_sat_distance_km, propagation_speed_mps)

    connected_start, sunlit_start = trace.status_at_time(start_time)

    comp_ratio = np.asarray(action[:, 0::2], dtype=np.float64)
    tx_ratio = np.asarray(action[:, 1::2], dtype=np.float64)
    action_min_ratio = float(cfg["action_min_ratio"])
    action_max_ratio = float(cfg["action_max_ratio"])
    cpu_freq = action_to_cpu_freq(comp_ratio, cfg["cpu_freq_max"], action_min_ratio, action_max_ratio)
    tx_power = action_to_tx_power(tx_ratio, cfg["tx_power_max"], action_min_ratio, action_max_ratio)

    step2_time = local_train_time(local_train_images, cfg["cpu_cycles_per_image"], cpu_freq)
    step2_energy = local_train_energy(local_train_images, cfg["cpu_cycles_per_image"], cpu_freq, cfg["kappa"])

    step1_base = step1_satellite_time(
        model_bits,
        cfg["bandwidth_dl"],
        sats_per_orbit,
        prop_delay_ground_sat=prop_delay_sat_ground,
        prop_delay_isl=prop_delay_isl,
    )
    step1_energy = step1_forward_energy(cfg["tx_power_isl_fixed"], model_bits, cfg["bandwidth_isl"])
    step3_tx_time = step3_ring_allreduce_time(model_bits, cfg["bandwidth_isl"])
    step3_time = step3_ring_allreduce_time(model_bits, cfg["bandwidth_isl"], prop_delay_isl=prop_delay_isl)
    step3_energy = step3_energy_per_satellite(cfg["tx_power_isl_fixed"], step3_tx_time)

    step1_time_arr = np.zeros((num_orbits, sats_per_orbit), dtype=np.float64)
    step4_time_arr = np.zeros(num_orbits, dtype=np.float64)
    step4_wait_arr = np.zeros(num_orbits, dtype=np.float64)
    step4_upload_energy_arr = np.zeros((num_orbits, sats_per_orbit), dtype=np.float64)
    in_orbit_late_arr = np.zeros(num_orbits, dtype=np.float64)
    orbit_local_arr = np.zeros(num_orbits, dtype=np.float64)

    for orbit_id in range(num_orbits):
        wait1, _ = _wait_until_connected(trace, orbit_id, start_time, slot_len)
        step1_time_arr[orbit_id, :] = wait1 + step1_base

        sat_local_finish = step1_time_arr[orbit_id] + step2_time[orbit_id]
        orbit_local = orbit_local_time(step1_time_arr[orbit_id], step2_time[orbit_id], step3_time)
        orbit_local_arr[orbit_id] = orbit_local
        in_orbit_late_arr[orbit_id] = float(np.mean(np.max(sat_local_finish) - sat_local_finish))

        step4_start = start_time + orbit_local
        wait4, connected_idx = _wait_until_connected(trace, orbit_id, step4_start, slot_len)
        step4_wait_arr[orbit_id] = wait4
        k = int(connected_idx.size)
        if k > 0:
            upload_time = np.zeros(sats_per_orbit, dtype=np.float64)
            for sat_id in connected_idx:
                t_ul = uplink_time_per_satellite(
                    model_bits=model_bits,
                    num_connected=k,
                    tx_power=tx_power[orbit_id, sat_id],
                    bandwidth_ul=cfg["bandwidth_ul"],
                    channel_const=cfg["channel_const"],
                    prop_delay_ground_sat=prop_delay_sat_ground,
                )
                upload_time[sat_id] = t_ul
                step4_upload_energy_arr[orbit_id, sat_id] = compute_step4_upload_energy(
                    tx_power[orbit_id, sat_id],
                    max(t_ul - prop_delay_sat_ground, 0.0),
                    tx_power_max=cfg["tx_power_max"],
                    #power_sensitivity=float(cfg.get("step4_energy_power_sensitivity", 0.0)),
                    #reference_ratio=float(cfg.get("step4_energy_reference_ratio", 0.5)),
                )
            step4_time_arr[orbit_id] = wait4 + float(np.max(upload_time))
        else:
            # 在极端轨迹中允许该轨道本轮上传失败并仅记录等待时间
            step4_time_arr[orbit_id] = wait4

    step5_time = step5_ground_aggregation_time(
        num_orbits=cfg["num_orbits"],
        model_bits=cfg["model_bits"],
        ground_cpu_cycles_per_bit=cfg["ground_cpu_cycles_per_bit"],
        ground_cpu_freq=cfg["ground_cpu_freq"],
    )
    round_time = round_total_time(orbit_local_arr, step4_time_arr, step5_time)
    round_time_penalty = round_time_penalty_for_reward(
        round_time=round_time,
    )
    end_time = start_time + round_time

    orbit_finish = orbit_local_arr + step4_time_arr
    orbit_late = np.max(orbit_finish) - orbit_finish

    battery_after = np.zeros_like(battery_energy, dtype=np.float64)
    dod_before = np.zeros_like(battery_energy, dtype=np.float64)
    dod_after = np.zeros_like(battery_energy, dtype=np.float64)
    degradation = np.zeros_like(battery_energy, dtype=np.float64)
    utility = np.zeros_like(battery_energy, dtype=np.float64)

    e1 = np.full((num_orbits, sats_per_orbit), step1_energy, dtype=np.float64)
    e3 = np.full((num_orbits, sats_per_orbit), step3_energy, dtype=np.float64)
    e4 = step4_upload_energy_arr
    e_cons = e1 + step2_energy + e3 + e4
    shadow_avg_energy_consumed = _shadow_mean(e_cons, sunlit_start)

    low_energy_events = 0
    negative_energy_events = 0
    for orbit_id in range(num_orbits):
        for sat_id in range(sats_per_orbit):
            e_before = float(battery_energy[orbit_id, sat_id])
            d_before = depth_of_discharge(e_before, cfg["battery_capacity"])
            charge = float(cfg["solar_power"]) * round_time if sunlit_start[orbit_id, sat_id] > 0 else 0.0
            update = update_battery_energy(
                energy_before=e_before,
                battery_capacity=cfg["battery_capacity"],
                charge_energy=charge,
                consume_energy=float(e_cons[orbit_id, sat_id]),
                min_warning=cfg["min_battery_warning"],
            )
            e_after = update.energy_after
            d_after = depth_of_discharge(e_after, cfg["battery_capacity"])
            deg = battery_degradation(
                d_before,
                d_after,
                cfg["degrade_k1"],
                cfg["degrade_k2"],
                cfg["degrade_k3"],
                cfg["degrade_k4"],
                cfg["degrade_k5"],
            )
            util = battery_utility_new(d_before, d_after)

            battery_after[orbit_id, sat_id] = e_after
            dod_before[orbit_id, sat_id] = d_before
            dod_after[orbit_id, sat_id] = d_after
            degradation[orbit_id, sat_id] = deg
            utility[orbit_id, sat_id] = util
            low_energy_events += int(update.warning_low_energy)
            negative_energy_events += int(update.warning_energy_negative)

    shadow_avg_battery_degradation = _shadow_mean(degradation, sunlit_start)
    ideal_dod = float(cfg["ideal_dod"])
    orbit_util = np.array([mean_nonzero(utility[o]) for o in range(num_orbits)], dtype=np.float64)
    best_dod = np.array([float(np.mean(np.abs(dod_after[o] - ideal_dod))) for o in range(num_orbits)], dtype=np.float64)
    (
        orbit_util_norm,
        round_time_penalty_norm,
        best_dod_norm,
        in_orbit_late_norm,
        orbit_late_norm,
    ) = _normalized_reward_terms(
        orbit_util=orbit_util,
        round_time_penalty=round_time_penalty,
        best_dod=best_dod,
        in_orbit_late=in_orbit_late_arr,
        orbit_late=orbit_late,
        cfg=cfg,
    )
    shadow_energy_consumed_norm = float(
        _norm_positive(shadow_avg_energy_consumed, float(cfg["reward_norm_energy_consumed_ref"]))
    )
    shadow_battery_degradation_norm = float(
        _norm_positive(shadow_avg_battery_degradation, float(cfg["reward_norm_battery_degradation_ref"]))
    )

    orbit_reward = (
        float(cfg["reward_w1"]) * orbit_util_norm
        - float(cfg["reward_w2"]) * round_time_penalty_norm
        - float(cfg["reward_w3"]) * best_dod_norm
        - float(cfg["reward_w4"]) * in_orbit_late_norm
        - float(cfg["reward_w5"]) * orbit_late_norm
        - float(cfg["reward_w6"]) * shadow_energy_consumed_norm
        - float(cfg["reward_w7"]) * shadow_battery_degradation_norm
    )
    reward = float(np.mean(orbit_reward))

    reward1_utility = float(np.mean(orbit_util))
    reward2_round_time = float(round_time)
    reward3_ideal_dod = float(np.mean(best_dod))
    reward4_in_orbit_late = float(np.mean(in_orbit_late_arr))
    reward5_ex_orbit_late = float(np.mean(orbit_late))

    orbit_rows: list[dict] = []
    for orbit_id in range(num_orbits):
        orbit_rows.append(
            {
                "round_id": round_id,
                "orbit_id": orbit_id + 1,
                "step3_time": float(step3_time),
                "step4_time": float(step4_time_arr[orbit_id]),
                "orbit_local_time": float(orbit_local_arr[orbit_id]),
                "orbit_wait_time": float(step4_wait_arr[orbit_id]),
                "orbit_energy_consumed": float(np.sum(e_cons[orbit_id])),
                "orbit_battery_utility": float(orbit_util[orbit_id]),
                "orbit_upload_energy": float(np.sum(e4[orbit_id])),
            }
        )

    sat_rows: list[dict] = []
    for orbit_id in range(num_orbits):
        for sat_id in range(sats_per_orbit):
            sat_rows.append(
                {
                    "round_id": round_id,
                    "orbit_id": orbit_id + 1,
                    "sat_id": sat_id + 1,
                    "step1_time": float(step1_time_arr[orbit_id, sat_id]),
                    "step2_time": float(step2_time[orbit_id, sat_id]),
                    "local_train_images": int(local_train_images[orbit_id, sat_id]),
                    "cpu_freq": float(cpu_freq[orbit_id, sat_id]),
                    "tx_power": float(tx_power[orbit_id, sat_id]),
                    "comp_energy": float(step2_energy[orbit_id, sat_id]),
                    "comm_energy": float(e1[orbit_id, sat_id] + e3[orbit_id, sat_id] + e4[orbit_id, sat_id]),
                    "battery_energy_before": float(battery_energy[orbit_id, sat_id]),
                    "battery_energy_after": float(battery_after[orbit_id, sat_id]),
                    "dod_before": float(dod_before[orbit_id, sat_id]),
                    "dod_after": float(dod_after[orbit_id, sat_id]),
                    "battery_degradation": float(degradation[orbit_id, sat_id]),
                    "battery_utility": float(utility[orbit_id, sat_id]),
                    "is_connected_round_start": int(connected_start[orbit_id, sat_id]),
                    "is_sunlit_round_start": int(sunlit_start[orbit_id, sat_id]),
                }
            )

    round_row = {
        "round_id": round_id,
        "round_start_time": float(start_time),
        "round_end_time": float(end_time),
        "round_time": float(round_time),
        "round_time_penalty": float(round_time_penalty),
        "step5_time": float(step5_time),
        "avg_dod": float(np.mean(dod_after)),
        "avg_aging": float(np.mean(degradation)),
        "avg_battery_utility": float(mean_nonzero(utility.flatten())),
        "total_energy": float(np.sum(e_cons)),
        "reward": reward,
        "comp_energy": float(np.sum(step2_energy)),
        "comm_energy": float(np.sum(e1 + e3 + e4)),
        "reward1_utility": reward1_utility,
        "reward2_round_time": reward2_round_time,
        "reward3_ideal_dod": reward3_ideal_dod,
        "reward4_in_orbit_late": reward4_in_orbit_late,
        "reward5_ex_orbit_late": reward5_ex_orbit_late,
        "reward6_energy_consumed": float(shadow_avg_energy_consumed),
        "reward7_battery_degradation": float(shadow_avg_battery_degradation),
        "low_energy_events": int(low_energy_events),
        "negative_energy_events": int(negative_energy_events),
    }

    return RoundResult(
        next_time=float(end_time),
        reward=reward,
        orbit_rewards=np.asarray(orbit_reward, dtype=np.float32),
        battery_after=battery_after,
        round_row=round_row,
        orbit_rows=orbit_rows,
        sat_rows=sat_rows,
    )

from __future__ import annotations

import numpy as np


EPS = 1e-12
LIGHT_SPEED_VACUUM_MPS = 299_792_458.0
DEFAULT_SAT_GROUND_DISTANCE_KM = 700.0
DEFAULT_INTER_SATELLITE_DISTANCE_KM = 5100.0


def propagation_delay(distance_km: float, propagation_speed_mps: float = LIGHT_SPEED_VACUUM_MPS) -> float:
    distance_m = max(float(distance_km), 0.0) * 1000.0
    speed = max(float(propagation_speed_mps), EPS)
    return distance_m / speed


def shannon_rate(bandwidth: float, tx_power: float, channel_const: float) -> float:
    """R = B * log2(1 + pC)."""
    p = max(float(tx_power), 0.0)
    return float(bandwidth) * np.log2(1.0 + p * float(channel_const)/float(bandwidth))


def step1_downlink_time(model_bits: float, bandwidth_dl: float) -> float:
    return float(model_bits) / max(float(bandwidth_dl), EPS)


def step1_satellite_time(
    model_bits: float,
    bandwidth_dl: float,
    sats_per_orbit: int,
    prop_delay_ground_sat: float = 0.0,
    prop_delay_isl: float = 0.0,
) -> float:
    # 按文档 8.1.3: T_step1 = T_dl + (J/2)*T^prop，额外计入传播时延
    relay_hops = float(sats_per_orbit) / 2.0
    return (
        step1_downlink_time(model_bits, bandwidth_dl)
        + float(prop_delay_ground_sat)
        + relay_hops * float(prop_delay_isl)
        #+ relay_hops
    )


def step1_forward_energy(tx_power_isl_fixed: float, model_bits: float, bandwidth_isl: float) -> float:
    t_hop_isl = float(model_bits) / max(float(bandwidth_isl), EPS)
    return float(tx_power_isl_fixed) * t_hop_isl


def step3_ring_allreduce_time(model_bits: float, bandwidth_isl: float, prop_delay_isl: float = 0.0) -> float:
    # 文档 8.3.3，并额外计入双向环传传播时延
    return 2.0 * float(model_bits) / max(float(bandwidth_isl), EPS) + 2.0 * float(prop_delay_isl)


def step3_energy_per_satellite(tx_power_isl_fixed: float, step3_time: float) -> float:
    return float(tx_power_isl_fixed) * float(step3_time)


def action_to_tx_power(
    action_ratio: np.ndarray,
    tx_power_max: float,
    action_min_ratio: float,
    action_max_ratio: float = 1.0,
) -> np.ndarray:
    ratio = np.clip(np.asarray(action_ratio, dtype=np.float64), 0.0, 1.0)
    low = float(np.clip(float(action_min_ratio), 0.0, 1.0))
    high = float(np.clip(float(action_max_ratio), 0.0, 1.0))
    if high < low:
        high = low
    scaled = low + (high - low) * ratio
    return scaled * float(tx_power_max)


def uplink_time_per_satellite(
    model_bits: float,
    num_connected: int,
    tx_power: float,
    bandwidth_ul: float,
    channel_const: float,
    prop_delay_ground_sat: float = 0.0,
) -> float:
    if num_connected <= 0:
        return 0.0
    rate = shannon_rate(bandwidth_ul, tx_power, channel_const)
    bits_per_sat = float(model_bits) / float(num_connected)
    return bits_per_sat / max(rate, EPS) + float(prop_delay_ground_sat)


def step4_upload_energy(
    tx_power: float,
    upload_time: float,
    tx_power_max: float,
) -> float:

    p = max(float(tx_power), 0.0)
    t = max(float(upload_time), 0.0)

    return p * t



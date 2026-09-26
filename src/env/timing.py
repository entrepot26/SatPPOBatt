from __future__ import annotations

import numpy as np


def next_slot_time(current_time: float, slot_len: float) -> float:
    if slot_len <= 0:
        return current_time
    steps = np.floor(float(current_time) / float(slot_len)) + 1.0
    return float(steps * float(slot_len))


def orbit_local_time(step1_time: np.ndarray, step2_time: np.ndarray, step3_time: float) -> float:
    local = np.asarray(step1_time, dtype=np.float64) + np.asarray(step2_time, dtype=np.float64)
    return float(np.max(local) + float(step3_time))


def round_total_time(orbit_local: np.ndarray, orbit_step4: np.ndarray, step5_time: float) -> float:
    total = np.asarray(orbit_local, dtype=np.float64) + np.asarray(orbit_step4, dtype=np.float64)
    return float(np.max(total) + float(step5_time))


def round_time_penalty_for_reward(
    round_time: float,
) -> float:
    return max(float(round_time), 0.0)


def slot_index_from_time(main_time: float, slot_len: float, slot_count: int) -> int:
    if slot_count <= 0:
        raise ValueError("slot_count must be positive")
    idx = int(np.floor(float(main_time) / float(slot_len)))
    return idx % slot_count

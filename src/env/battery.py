from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12


@dataclass
class BatteryUpdate:
    energy_after: float
    warning_low_energy: bool
    warning_energy_negative: bool


def depth_of_discharge(energy: float, capacity: float) -> float:
    if capacity <= 0:
        return 1.0
    d = 1.0 - float(energy) / float(capacity)
    return float(np.clip(d, 0.0, 1.0))


def cycle_count_from_dod(
    d: float,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    k5: float,

) -> float:
    if d <= 0:
        return float("inf")

    value = (
        k1 * d**3
        + k2 * d**2
        + k3 * d
        + k4
        + k5 / d

    )
    return max(float(value), EPS) #这里的max实际上没用,准备删除
#y = 88338x^3 - 144920x^2 + 48311x + 5992.1 + 4879.5/x

def battery_degradation(
    d_before: float,
    d_after: float,
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    k5: float,

) -> float:
    if d_after <= d_before:
        return 0.0
    c1 = cycle_count_from_dod(d_before, k1, k2, k3, k4, k5)
    c2 = cycle_count_from_dod(d_after, k1, k2, k3, k4, k5)
    inv1 = 0.0 if np.isinf(c1) else 1.0 / max(c1, EPS)
    inv2 = 1.0 / max(c2, EPS)
    return max(1000*(inv2 - inv1), 0.0)



def battery_utility_new(d_before: float, d_after: float) -> float:
    if d_after <= d_before:
        return 0.0
    utile_line1_after=(17667.6*d_after**5
                       -36230*d_after**4
                       +16103.6667*d_after**3
                       +2996.05*d_after**2
                       +4879.5*d_after)
    utile_line1_before=(17667.6*d_before**5
                       -36230*d_before**4
                       +16103.6667*d_before**3
                       +2996.05*d_before**2
                       +4879.5*d_before)

    return (utile_line1_after - utile_line1_before) / (d_after-d_before)


#旧的def battery_utility函数已经被废弃了,目前使用def battery_utility_new衡量电池的寿命效用.
def battery_utility(d_before: float, d_after: float, degradation: float) -> float:
    if d_after <= d_before or degradation <= 0:
        return 0.0
    return (d_after - d_before) / max(degradation, EPS)


def update_battery_energy(
    energy_before: float,
    battery_capacity: float,
    charge_energy: float,
    consume_energy: float,
    min_warning: float,
) -> BatteryUpdate:
    raw = min(float(battery_capacity), float(energy_before) + float(charge_energy) - float(consume_energy))
    warning_negative = raw < 0.0
    energy_after = max(raw, 0.0)
    warning_low = energy_after < float(min_warning)
    return BatteryUpdate(
        energy_after=energy_after,
        warning_low_energy=warning_low,
        warning_energy_negative=warning_negative,
    )


def mean_nonzero(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64)
    nz = v[v > 0.0]
    if nz.size == 0:
        return 0.0
    return float(np.mean(nz))


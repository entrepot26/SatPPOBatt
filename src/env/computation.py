from __future__ import annotations

import numpy as np


EPS = 1e-12


def action_to_cpu_freq(
    action_ratio: np.ndarray,
    cpu_freq_max: float,
    action_min_ratio: float,
    action_max_ratio: float = 1.0,
) -> np.ndarray:
    ratio = np.clip(np.asarray(action_ratio, dtype=np.float64), 0.0, 1.0)
    low = float(np.clip(float(action_min_ratio), 0.0, 1.0))
    high = float(np.clip(float(action_max_ratio), 0.0, 1.0))
    if high < low:
        high = low
    scaled = low + (high - low) * ratio
    return scaled * float(cpu_freq_max)


def local_train_time(local_train_images: np.ndarray, cpu_cycles_per_image: float, cpu_freq: np.ndarray) -> np.ndarray:
    images = np.asarray(local_train_images, dtype=np.float64)
    freq = np.asarray(cpu_freq, dtype=np.float64)
    return images * float(cpu_cycles_per_image) / np.maximum(freq, EPS)


def local_train_energy(
    local_train_images: np.ndarray,
    cpu_cycles_per_image: float,
    cpu_freq: np.ndarray,
    kappa: float,
) -> np.ndarray:
    images = np.asarray(local_train_images, dtype=np.float64)
    freq = np.asarray(cpu_freq, dtype=np.float64)
    return float(kappa) * images * float(cpu_cycles_per_image) * np.square(freq) + 3*(images * float(cpu_cycles_per_image)/freq)
# +images * float(cpu_cycles_per_image)/freq 是后来加上的运行时的额定能耗

def step5_ground_aggregation_time(
    num_orbits: int,
    model_bits: float,
    ground_cpu_cycles_per_bit: float,
    ground_cpu_freq: float,
) -> float:
    # 文档 8.5.2
    return (
        float(num_orbits)
        * float(model_bits)
        * float(ground_cpu_cycles_per_bit)
        / max(float(ground_cpu_freq), EPS)
    )

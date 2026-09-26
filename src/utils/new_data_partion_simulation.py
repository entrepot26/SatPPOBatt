from __future__ import annotations

import numpy as np


def _class_sizes(total: int, num_classes: int) -> np.ndarray:
    base = total // num_classes
    rem = total % num_classes
    sizes = np.full(num_classes, base, dtype=np.int64)
    if rem > 0:
        sizes[:rem] += 1
    return sizes


def _rebalance_to_minimum(counts: np.ndarray, min_images: int) -> np.ndarray:
    out = counts.astype(np.int64, copy=True)
    if min_images <= 0:
        return out

    need_ids = np.where(out < min_images)[0]
    for nid in need_ids:
        gap = int(min_images - out[nid])
        while gap > 0:
            donor_ids = np.where(out > min_images)[0]
            if donor_ids.size == 0:
                break
            donor = int(donor_ids[np.argmax(out[donor_ids])])
            movable = int(out[donor] - min_images)
            if movable <= 0:
                break
            moved = min(gap, movable)
            out[donor] -= moved
            out[nid] += moved
            gap -= moved
    return out


def simulate_clients_data_number(
    data_total_sample: int,
    client_number: int,
    alpha: float,
    min_images: int,
    num_classes: int = 10,
    rng: np.random.Generator | None = None,
    max_retries: int = 128,
) -> np.ndarray:
    if data_total_sample <= 0:
        raise ValueError("data_total_sample must be positive")
    if client_number <= 0:
        raise ValueError("client_number must be positive")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if min_images < 0:
        raise ValueError("min_images must be non-negative")
    if data_total_sample < client_number * min_images:
        raise ValueError("data_total_sample is too small for the requested min_images constraint")

    generator = rng if rng is not None else np.random.default_rng()
    per_class = _class_sizes(int(data_total_sample), int(num_classes))

    for _ in range(max(1, int(max_retries))):
        client_counts = np.zeros(int(client_number), dtype=np.int64)
        for class_size in per_class:
            proportions = generator.dirichlet(np.full(int(client_number), float(alpha), dtype=np.float64))
            class_alloc = np.floor(proportions * int(class_size)).astype(np.int64)
            diff = int(class_size - int(class_alloc.sum()))
            if diff > 0:
                class_alloc[np.argsort(-proportions)[:diff]] += 1
            client_counts += class_alloc

        total_diff = int(data_total_sample - int(client_counts.sum()))
        if total_diff != 0:
            client_counts[0] += total_diff
        if min_images <= 0 or int(client_counts.min()) >= min_images:
            return client_counts

    repaired = _rebalance_to_minimum(client_counts, int(min_images))
    repaired_diff = int(data_total_sample - int(repaired.sum()))
    if repaired_diff != 0:
        repaired[0] += repaired_diff
    return repaired


def satellite_client_names(num_orbits: int, sats_per_orbit: int) -> list[str]:
    sat_width = 2 if sats_per_orbit >= 10 else 1
    return [
        f"sat{orbit_id}{sat_id:0{sat_width}d}"
        for orbit_id in range(1, num_orbits + 1)
        for sat_id in range(1, sats_per_orbit + 1)
    ]

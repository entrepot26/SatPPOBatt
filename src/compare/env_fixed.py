from __future__ import annotations

from pathlib import Path

import numpy as np

from src.compare.round_runner_fixed import RoundResult, run_round_compare1, run_round_compare2
from src.env.battery import depth_of_discharge
from src.simulator.clock import SimulationClock
from src.simulator.trace_loader import TraceData
from src.utils.io import save_rows_csv
from src.utils.new_data_partion_simulation import satellite_client_names, simulate_clients_data_number


class FixedPolicyEnv:
    def __init__(self, cfg: dict, trace: TraceData, mode: str) -> None:
        if mode not in {"compare1", "compare2"}:
            raise ValueError(f"Unsupported compare mode: {mode}")
        self.cfg = cfg
        self.trace = trace
        self.mode = mode
        self.num_orbits = int(cfg["num_orbits"])
        self.sats_per_orbit = int(cfg["sats_per_orbit"])
        self.obs_dim = 4 * self.sats_per_orbit

        self.local_train_images = np.zeros((self.num_orbits, self.sats_per_orbit), dtype=np.int64)
        self.partition_columns = satellite_client_names(self.num_orbits, self.sats_per_orbit)
        self.partition_fieldnames = ["round_id", *self.partition_columns]
        self.partition_rows: list[dict[str, int]] = []
        self.partition_rng = np.random.default_rng(int(cfg["partition_seed"]))
        self.global_round_id = 0

        self.clock = SimulationClock()
        self.round_id = 0
        self.battery = np.full((self.num_orbits, self.sats_per_orbit), float(cfg["battery_capacity"]), dtype=np.float64)

    def _sample_local_train_images(self) -> None:
        counts = simulate_clients_data_number(
            data_total_sample=int(self.cfg["data_total_sample"]),
            client_number=self.num_orbits * self.sats_per_orbit,
            alpha=float(self.cfg["dirichlet_alpha"]),
            min_images=int(self.cfg["data_partition_min_images"]),
            num_classes=int(self.cfg["data_partition_num_classes"]),
            rng=self.partition_rng,
        )
        self.local_train_images = counts.reshape(self.num_orbits, self.sats_per_orbit)

    def _record_current_round_partition(self) -> None:
        row: dict[str, int] = {"round_id": int(self.global_round_id)}
        flat_counts = self.local_train_images.reshape(-1)
        for name, value in zip(self.partition_columns, flat_counts):
            row[name] = int(value)
        self.partition_rows.append(row)

    def export_partition_history(self, path: str | Path) -> Path:
        out = Path(path)
        save_rows_csv(out, self.partition_rows, self.partition_fieldnames)
        return out

    def _normalized_local_images(self, orbit_id: int) -> np.ndarray:
        local_images = self.local_train_images[orbit_id].astype(np.float32)
        mode = str(self.cfg["state_local_images_norm"]).strip().lower()
        if mode == "none":
            return local_images

        total_images = max(float(self.cfg["data_total_sample"]), 1.0)
        if mode == "linear":
            return np.clip(local_images / total_images, 0.0, 1.0)

        denom = np.log1p(total_images)
        if denom <= 0.0:
            return np.zeros_like(local_images, dtype=np.float32)
        return np.log1p(local_images) / denom

    def _build_state(self) -> np.ndarray:
        connected, sunlit = self.trace.status_at_time(self.clock.now)
        state = np.zeros((self.num_orbits, self.obs_dim), dtype=np.float32)
        for o in range(self.num_orbits):
            dod = np.array(
                [depth_of_discharge(self.battery[o, j], self.cfg["battery_capacity"]) for j in range(self.sats_per_orbit)],
                dtype=np.float32,
            )
            link = connected[o].astype(np.float32)
            sun = sunlit[o].astype(np.float32)
            local_images_norm = self._normalized_local_images(o)
            sat_state = np.stack([dod, link, sun, local_images_norm], axis=1)
            state[o] = sat_state.reshape(-1)
        return state

    def reset(self) -> np.ndarray:
        self.clock.reset(0.0)
        self.round_id = 0
        self.battery.fill(float(self.cfg["battery_capacity"]))
        self._sample_local_train_images()
        return self._build_state()

    def step(self) -> tuple[np.ndarray, float, dict]:
        self._record_current_round_partition()
        runner = run_round_compare1 if self.mode == "compare1" else run_round_compare2
        result: RoundResult = runner(
            round_id=self.round_id,
            start_time=self.clock.now,
            battery_energy=self.battery,
            local_train_images=self.local_train_images,
            trace=self.trace,
            cfg=self.cfg,
        )
        self.battery = result.battery_after
        self.clock.reset(result.next_time)
        self.round_id += 1
        self.global_round_id += 1
        self._sample_local_train_images()
        next_state = self._build_state()
        info = {
            "round_metrics": result.round_row,
            "orbit_metrics": result.orbit_rows,
            "sat_metrics": result.sat_rows,
            "orbit_rewards": result.orbit_rewards,
        }
        return next_state, result.reward, info

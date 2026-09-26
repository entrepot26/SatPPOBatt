from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.utils.io import save_rows_csv


ROUND_FIELDS = [
    "round_id",
    "round_start_time",
    "round_end_time",
    "round_time",
    "round_time_penalty",
    "step5_time",
    "avg_dod",
    "avg_aging",
    "avg_battery_utility",
    "total_energy",
    "reward",
    "comp_energy",
    "comm_energy",
    "reward1_utility",
    "reward2_round_time",
    "reward3_ideal_dod",
    "reward4_in_orbit_late",
    "reward5_ex_orbit_late",
    "reward6_energy_consumed",
    "reward7_battery_degradation",
]

ORBIT_FIELDS = [
    "round_id",
    "orbit_id",
    "step3_time",
    "step4_time",
    "orbit_local_time",
    "orbit_wait_time",
    "orbit_energy_consumed",
    "orbit_battery_utility",
    "orbit_upload_energy",
]

SAT_FIELDS = [
    "round_id",
    "orbit_id",
    "sat_id",
    "step1_time",
    "step2_time",
    "local_train_images",
    "cpu_freq",
    "tx_power",
    "comp_energy",
    "comm_energy",
    "battery_energy_before",
    "battery_energy_after",
    "dod_before",
    "dod_after",
    "battery_degradation",
    "battery_utility",
    "is_connected_round_start",
    "is_sunlit_round_start",
]

TRAIN_REWARD_FIELDS = ["episode_id", "avg_reward"]


@dataclass
class MetricsStore:
    round_rows: list[dict] = field(default_factory=list)
    orbit_rows: list[dict] = field(default_factory=list)
    sat_rows: list[dict] = field(default_factory=list)
    train_reward_rows: list[dict] = field(default_factory=list)

    def add_round(self, row: dict) -> None:
        self.round_rows.append(row)

    def add_orbits(self, rows: list[dict]) -> None:
        self.orbit_rows.extend(rows)

    def add_sats(self, rows: list[dict]) -> None:
        self.sat_rows.extend(rows)

    def add_train_reward(self, episode_id: int, avg_reward: float) -> None:
        self.train_reward_rows.append({"episode_id": episode_id, "avg_reward": avg_reward})

    def save_train(self, exp_dir: str | Path) -> None:
        base = Path(exp_dir)
        save_rows_csv(base / "rl_train_rewards.csv", self.train_reward_rows, TRAIN_REWARD_FIELDS)
        save_rows_csv(base / "rl_train_round_metrics.csv", self.round_rows, ROUND_FIELDS)
        save_rows_csv(base / "train_orbit_round_metrics.csv", self.orbit_rows, ORBIT_FIELDS)
        save_rows_csv(base / "train_satellite_round_metrics.csv", self.sat_rows, SAT_FIELDS)

    def save_eval(self, exp_dir: str | Path) -> None:
        base = Path(exp_dir)
        save_rows_csv(base / "rl_eval_round_metrics.csv", self.round_rows, ROUND_FIELDS)
        save_rows_csv(base / "orbit_round_metrics.csv", self.orbit_rows, ORBIT_FIELDS)
        save_rows_csv(base / "satellite_round_metrics.csv", self.sat_rows, SAT_FIELDS)

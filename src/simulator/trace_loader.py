from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.env.timing import slot_index_from_time
from src.utils.io import ensure_dir


# Support both legacy `satXY` and new `satXYZ` naming.
# - X: orbit id (1 digit)
# - Y / YZ: satellite id in orbit (1 or 2 digits)
SAT_PATTERN = re.compile(r"^sat(?P<orbit>\d)(?P<sat>\d{1,2})$", re.IGNORECASE)


@dataclass
class TraceData:
    times: np.ndarray
    connected: np.ndarray  # [T, O, J]
    sunlit: np.ndarray  # [T, O, J]
    sat_names: list[str]
    slot_len: float

    @property
    def num_slots(self) -> int:
        return int(self.times.shape[0])

    @property
    def num_orbits(self) -> int:
        return int(self.connected.shape[1])

    @property
    def sats_per_orbit(self) -> int:
        return int(self.connected.shape[2])

    def status_at_time(self, main_time: float) -> tuple[np.ndarray, np.ndarray]:
        idx = slot_index_from_time(main_time, self.slot_len, self.num_slots)
        return self.connected[idx].copy(), self.sunlit[idx].copy()


def _parse_status(code: str) -> tuple[int, int]:
    raw = str(code).strip().replace(" ", "")
    if len(raw) < 2:
        raise ValueError(f"Invalid status code: {code!r}")
    if not raw[:2].isdigit():
        raise ValueError(f"Status code must be 2-digit number: {code!r}")
    return int(raw[0]), int(raw[1])


def _format_sat_name(orbit_id: int, sat_id: int, sats_per_orbit: int) -> str:
    # For >=10 satellites per orbit, use zero-padded 2-digit in-orbit id (e.g. sat101, sat612).
    sat_token = f"{sat_id:02d}" if sats_per_orbit >= 10 else str(sat_id)
    return f"sat{orbit_id}{sat_token}"


def generate_sample_trace_csv(
    path: str | Path,
    num_orbits: int,
    sats_per_orbit: int,
    slot_len: int,
    num_slots: int = 200,
) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    sat_names = [
        _format_sat_name(o, s, sats_per_orbit)
        for o in range(1, num_orbits + 1)
        for s in range(1, sats_per_orbit + 1)
    ]

    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_epsec", *sat_names])
        for t in range(num_slots):
            row = [t * slot_len]
            for o in range(1, num_orbits + 1):
                for s in range(1, sats_per_orbit + 1):
                    connected = 1 if ((t + o + s) % 4 in (0, 1)) else 0
                    sunlit = 1 if ((t + 2 * o + s) % 6 < 3) else 0
                    row.append(f"{connected}{sunlit}")
            writer.writerow(row)
    return p


def load_trace_csv(path: str | Path, num_orbits: int, sats_per_orbit: int, slot_len: int) -> TraceData:
    p = Path(path)
    if not p.exists():
        generate_sample_trace_csv(p, num_orbits=num_orbits, sats_per_orbit=sats_per_orbit, slot_len=slot_len)

    # `utf-8-sig` transparently strips UTF-8 BOM so column checks keep working.
    with p.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        raw_columns = reader.fieldnames or []
        columns = [str(c).strip() if c is not None else "" for c in raw_columns]
        if not columns or columns[0] != "time_epsec":
            raise ValueError("Trace CSV first column must be `time_epsec`")
        raw_time_col = raw_columns[0]
        sat_names = columns[1:]
        sat_raw_names = raw_columns[1:]
        expected = num_orbits * sats_per_orbit
        if len(sat_names) != expected:
            raise ValueError(f"Expected {expected} satellites, found {len(sat_names)}")

        sat_map: list[tuple[int, int, str]] = []
        seen_sat_slots: set[tuple[int, int]] = set()
        for name, raw_name in zip(sat_names, sat_raw_names):
            m = SAT_PATTERN.match(name)
            if not m:
                raise ValueError(f"Invalid satellite column name: {name}")
            orbit_id = int(m.group("orbit")) - 1
            sat_id = int(m.group("sat")) - 1
            if not (0 <= orbit_id < num_orbits and 0 <= sat_id < sats_per_orbit):
                raise ValueError(f"Satellite id out of range: {name}")
            sat_slot = (orbit_id, sat_id)
            if sat_slot in seen_sat_slots:
                raise ValueError(f"Duplicate satellite slot in columns: {name}")
            seen_sat_slots.add(sat_slot)
            sat_map.append((orbit_id, sat_id, raw_name))

        times = []
        connected_rows = []
        sunlit_rows = []
        for row in reader:
            times.append(float(str(row[raw_time_col]).strip()))
            conn = np.zeros((num_orbits, sats_per_orbit), dtype=np.int64)
            sun = np.zeros((num_orbits, sats_per_orbit), dtype=np.int64)
            for orbit_id, sat_id, raw_name in sat_map:
                c, s = _parse_status(row[raw_name])
                conn[orbit_id, sat_id] = c
                sun[orbit_id, sat_id] = s
            connected_rows.append(conn)
            sunlit_rows.append(sun)

    if len(times) < 2:
        raise ValueError("Trace CSV must have at least 2 rows")

    arr_times = np.asarray(times, dtype=np.float64)
    connected = np.stack(connected_rows, axis=0).astype(np.int64)
    sunlit = np.stack(sunlit_rows, axis=0).astype(np.int64)
    return TraceData(times=arr_times, connected=connected, sunlit=sunlit, sat_names=sat_names, slot_len=float(slot_len))

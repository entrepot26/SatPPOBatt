from __future__ import annotations

import csv
import json
from datetime import datetime
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable

import yaml

CSV_FLOAT_DIGITS = 6
CSV_SCI_THRESHOLD = 1e-6
CSV_SCI_DIGITS = 8


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_experiment_dir(results_root: str | Path) -> Path:
    root = ensure_dir(results_root)
    existing = sorted(p for p in root.glob("exp_*") if p.is_dir())
    next_idx = 1
    if existing:
        last = existing[-1].name.split("_")[-1]
        if last.isdigit():
            next_idx = int(last) + 1
    exp_dir = root / f"exp_{next_idx:03d}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def save_json(data: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_rows_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    rows = list(rows)
    if not rows and fieldnames is None:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _format_csv_value(v) for k, v in row.items()})


def _format_csv_value(value: Any) -> Any:
    if isinstance(value, Real) and not isinstance(value, Integral):
        v = float(value)
        if v != 0.0 and abs(v) < CSV_SCI_THRESHOLD:
            return f"{v:.{CSV_SCI_DIGITS}e}"
        return f"{v:.{CSV_FLOAT_DIGITS}f}"
    return value


def build_run_info(config: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "project_name": config["project_name"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": config.get("seed"),
    }


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_experiment_summary(
    round_rows: Iterable[dict[str, Any]],
    sat_rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = list(round_rows)
    total_rounds = int(len(rows))
    total_time = float(sum(float(row.get("round_time", 0.0)) for row in rows))

    # total_loss: sum of shadow satellites' battery degradation across all rounds.
    if sat_rows is not None:
        total_loss = 0.0
        for row in sat_rows:
            is_sunlit = int(_to_float(row.get("is_sunlit_round_start", 1), 1.0))
            if is_sunlit <= 0:
                total_loss += _to_float(row.get("battery_degradation", 0.0), 0.0)
    else:
        # Fallback when satellite-level rows are unavailable.
        total_loss = float(sum(float(row.get("reward7_battery_degradation", 0.0)) for row in rows))
    return {
        "total_rounds": total_rounds,
        "total_time": total_time,
        "total_loss": total_loss,
    }


def save_experiment_summary(
    round_rows: Iterable[dict[str, Any]],
    path: str | Path,
    sat_rows: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = build_experiment_summary(round_rows, sat_rows=sat_rows)
    save_rows_csv(path, [summary], fieldnames=list(summary.keys()))
    return summary

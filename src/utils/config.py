from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REQUIRED_KEYS = {
    "num_orbits",
    "sats_per_orbit",
    "slot_len",
    "trace_csv_path",
    "eval_rounds",
    "model_name",
    "model_bits",
    "bandwidth_isl",
    "bandwidth_ul",
    "bandwidth_dl",
    "channel_const",
    "tx_power_max",
    "tx_power_isl_fixed",
    "cpu_freq_max",
    "cpu_cycles_per_image",
    "kappa",
    "battery_capacity",
    "solar_power",
    "degrade_k1",
    "degrade_k2",
    "degrade_k3",
    "ground_cpu_cycles_per_bit",
    "ground_cpu_freq",
    "total_episodes",
    "simulator_steps_per_episode",
    "rollout_size",
    "gamma",
    "gae_lambda",
    "clip_ratio",
    "actor_lr",
    "critic_lr",
    "ppo_epochs",
    "mini_batch_size",
    "entropy_coef",
    "value_coef",
    "action_min_ratio",
    "reward_w1",
    "reward_w2",
    "reward_w3",
    "reward_w4",
    "reward_w5",
    "reward_w6",
    "reward_w7",
    "reward_norm_utility_ref",
    "reward_norm_round_time_ref",
    "reward_norm_late_ref",
    "reward_norm_best_dod_ref",
    "reward_norm_energy_consumed_ref",
    "reward_norm_battery_degradation_ref",
    "partitions_dir",
    "dirichlet_alpha",
    "partition_seed",
    "data_total_sample",
    "data_partition_min_images",
}

LEGACY_KEY_ALIASES = {
    "num_episodes": "total_episodes",
    "train_rounds_per_episode": "simulator_steps_per_episode",
    "update_epochs": "ppo_epochs",
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {p}")
    return data


def merge_dict(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged.update(extra)
    return merged


def load_config(base_path: str | Path, override_paths: list[str | Path] | None = None) -> dict[str, Any]:
    cfg = load_yaml(base_path)
    for path in override_paths or []:
        cfg = merge_dict(cfg, load_yaml(path))
    cfg = normalize_legacy_keys(cfg)
    cfg = prompt_for_missing_values(cfg)
    validate_config(cfg)
    return cfg


def prompt_for_missing_values(cfg: dict[str, Any]) -> dict[str, Any]:
    """Prompt for redacted settings when the program starts."""
    completed = dict(cfg)
    missing = [key for key, value in completed.items() if value is None]
    if not missing:
        return completed

    print("检测到未设置的参数，请根据论文设置逐项输入。")
    print("输入采用 YAML 格式，例如：3、0.5、true、example、[1, 2]。")
    for key in missing:
        while True:
            try:
                raw_value = input(f"{key}（参考论文设置）: ").strip()
            except EOFError as exc:
                raise RuntimeError(f"无法从标准输入读取参数：{key}") from exc
            if not raw_value:
                print("输入不能为空，请重新输入。")
                continue
            value = yaml.safe_load(raw_value)
            if value is None or isinstance(value, dict):
                print("请输入单个标量或列表值。")
                continue
            completed[key] = value
            break
    return completed


def normalize_legacy_keys(cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(cfg)
    for legacy_key, new_key in LEGACY_KEY_ALIASES.items():
        if new_key not in normalized and legacy_key in normalized:
            normalized[new_key] = normalized[legacy_key]
    return normalized


def validate_config(cfg: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_KEYS - set(cfg))
    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")


def resolve_path(root: str | Path, raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return Path(root) / p

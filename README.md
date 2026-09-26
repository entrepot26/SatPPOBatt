# LEO FL + PPO 仿真实验代码

此部分为本文的DPOM求解算法，请配合STK工具与基本联邦学习代码使用。

该项目按 `sa2.md` 规范实现了一个可运行的 LEO 卫星联邦学习流程仿真框架，核心是：

- 用轨迹 CSV 驱动 `is_connected/is_sunlit`
- 用 PPO 联合控制每颗卫星的 CPU 频率和上传功率
- 自动输出实验目录、CSV 指标、checkpoint 和图表

## 目录结构

```text
configs/
  default.yaml
  rl.yaml
  battery.yaml
  trace.yaml
src/
  main_train_rl.py
  main_eval_rl.py
  simulator/
  env/
  rl/
  utils/
data/
  traces/
  partitions/
results/
```

## 环境要求

- Python 3.10+
- PyTorch 2.x
- `numpy` `pyyaml` `matplotlib`

## 快速开始

在项目根目录执行：

```bash
python -m src.main_train_rl --config configs/default.yaml
```

训练完成后，再执行推理评估：

```bash
python -m src.main_eval_rl --config configs/default.yaml --checkpoint results/exp_076/ppo_checkpoint_last.pt
```

若不指定 `--checkpoint`，评估脚本会自动读取 `results/` 下最新实验目录的 `ppo_checkpoint_best.pt`。

## 轨迹输入

- `trace_csv_path` 指向宽表 CSV
- 第一列必须是 `time_epsec`
- 其余列必须是 `satXYZ`（例如 `sat101`、`sat612`）
- 单元格状态编码为两位数：第一位 `is_connected`，第二位 `is_sunlit`

若默认轨迹文件不存在，程序会在 `data/traces/sample_trace.csv` 自动生成一个可运行样例。

## 配置说明

程序启动时会提示逐项输入参数，请参考论文设置填写。输入采用 YAML 格式，例如数字 `3`、布尔值 `true`、字符串 `example` 或列表 `[1, 2]`。

所有关键参数名均保留在 `configs/default.yaml`，包括：

- 星座规模、时隙、轨迹路径
- 通信/计算/电池/地面聚合参数
- PPO 超参数和奖励权重
- 结果目录、随机种子、数据划分参数

运行实验前请按照论文设置参数。

## 输出结果

每次运行自动创建 `results/exp_xxx/`，包含至少以下文件：

- `config_snapshot.yaml`
- `run_info.json`
- `rl_train_rewards.csv`（训练模式）
- `rl_train_rewards.png`（训练模式）
- `ppo_checkpoint_last.pt`
- `ppo_checkpoint_best.pt`
- `rl_eval_round_metrics.csv`（评估模式）
- `orbit_round_metrics.csv`（评估模式）
- `satellite_round_metrics.csv`（评估模式）
- `final_global_model.pt`
- `round_total_time.png`
- `avg_battery_utility.png`
- `cumulative_aging.png`
- `energy_breakdown.png`


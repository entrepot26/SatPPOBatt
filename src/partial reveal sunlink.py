import re
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import yaml

# 设置全局字体为 SimHei（黑体），Windows 和大多数环境都支持
plt.rcParams['font.sans-serif'] = ['SimHei'] 
# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False  

# ============================================================
# 1. 基本配置：启动时根据提示输入
# ============================================================

CSV_PATH = None  # 参考论文设置
OUTPUT_DIR = None  # 参考论文设置
SAVE_DPI = None  # 参考论文设置


# ----------------------------
# 1.1 选择要画哪些轨道
# ----------------------------
SELECTED_ORBITS = None  # 参考论文设置

# ----------------------------
# 1.2 选择绘图时隙范围
# ----------------------------
# 采用 [START_SLOT, END_SLOT) 规则；输入 null 表示不限制。
START_SLOT = None  # 参考论文设置
END_SLOT = None  # 参考论文设置


# ----------------------------
# 1.3 是否绘制两类状态图
# ----------------------------
DRAW_SUNLIGHT_FIGURE = None  # 参考论文设置
DRAW_CONNECTION_FIGURE = None  # 参考论文设置


# ----------------------------
# 1.4 图表尺寸
# ----------------------------
FIG_WIDTH = None  # 参考论文设置
FIG_HEIGHT = None  # 参考论文设置


# ----------------------------
# 1.5 字体大小设置
# ----------------------------
# 坐标轴上数字的大小
TICK_FONT_SIZE = None  # 参考论文设置

# 坐标轴文字的大小，例如 Time / seconds, Satellite Index
AXIS_LABEL_FONT_SIZE = None  # 参考论文设置

# 上方颜色标签文字大小
LEGEND_FONT_SIZE = None  # 参考论文设置


# ----------------------------
# 1.6 上方颜色标签位置设置
# ----------------------------
# 绘图区顶部位置，数值越小，绘图区越往下，上方留白越大
AXES_TOP = None  # 参考论文设置

# 上方颜色标签距离绘图区的距离
# 数值越大，颜色标签越远离绘图区
LEGEND_DISTANCE_FROM_AXES = None  # 参考论文设置

# 上方颜色标签的列数
LEGEND_NCOL = None  # 参考论文设置


# ----------------------------
# 1.7 图表边距
# ----------------------------
FIG_LEFT = None  # 参考论文设置
FIG_RIGHT = None  # 参考论文设置
FIG_BOTTOM = None  # 参考论文设置


# ----------------------------
# 1.8 柱状时间带高度
# ----------------------------
BAR_HEIGHT = None  # 参考论文设置


def prompt_setting(name: str, allow_none: bool = False):
    """启动时读取已清除的公开参数。"""
    while True:
        raw_value = input(f"{name}（参考论文设置）: ").strip()
        if not raw_value:
            print("输入不能为空，请重新输入。")
            continue
        value = yaml.safe_load(raw_value)
        if value is None and not allow_none:
            print("该参数不能为 null，请重新输入。")
            continue
        if isinstance(value, dict):
            print("请输入单个标量或列表值。")
            continue
        return value


# ============================================================
# 2. 数据读取与状态解析
# ============================================================

def load_state_csv(csv_path: str) -> pd.DataFrame:
    """
    读取卫星状态 CSV。

    第一列为 time_epsec，其余列为 satXY 或 satXXY。

    单元格状态为两位数：
        第一位：连接状态，1=连接地面站，0=未连接
        第二位：光照状态，1=光照，0=无光照
    """
    df = pd.read_csv(csv_path)

    if "time_epsec" not in df.columns:
        raise ValueError("CSV 文件中必须包含 time_epsec 列。")

    df = df.sort_values("time_epsec").reset_index(drop=True)

    return df


def normalize_state(value) -> str:
    """
    将状态值统一转为两位字符串。

    例如：
        11   -> "11"
        10   -> "10"
        1    -> "01"
        0    -> "00"
        1.0  -> "01"
        10.0 -> "10"
    """
    if pd.isna(value):
        raise ValueError("发现空状态值 NaN，请检查 CSV 数据。")

    state = str(value).strip()

    if state.endswith(".0"):
        state = state[:-2]

    state = state.zfill(2)

    if state not in {"00", "01", "10", "11"}:
        raise ValueError(f"发现非法状态值：{value}")

    return state


def parse_satellite_name(col_name: str):
    """
    解析卫星列名。

    支持：
        sat11  -> 轨道 1，轨道内编号 1
        sat18  -> 轨道 1，轨道内编号 8
        sat121 -> 轨道 12，轨道内编号 1

    默认最后一位数字表示轨道内卫星编号，
    前面的数字表示轨道编号。
    """
    name = col_name.lower()

    match = re.fullmatch(r"sat(\d+)(\d)", name)
    if not match:
        return None

    orbit_id = int(match.group(1))
    sat_id = int(match.group(2))

    return orbit_id, sat_id


def detect_orbit_ids(df: pd.DataFrame) -> list[int]:
    """
    自动检测 CSV 中包含哪些轨道编号。
    """
    orbit_ids = set()

    for col in df.columns:
        parsed = parse_satellite_name(col)

        if parsed is not None:
            orbit_id, _ = parsed
            orbit_ids.add(orbit_id)

    return sorted(orbit_ids)


def get_satellites_in_orbit(df: pd.DataFrame, orbit_id: int) -> list[tuple[int, str]]:
    """
    获取某个轨道中的所有卫星。

    返回结果格式：
        [(轨道内编号, 列名), ...]

    例如：
        [(1, "sat31"), (2, "sat32"), ..., (8, "sat38")]
    """
    sats = []

    for col in df.columns:
        parsed = parse_satellite_name(col)

        if parsed is None:
            continue

        col_orbit_id, sat_id = parsed

        if col_orbit_id == orbit_id:
            sats.append((sat_id, col))

    sats = sorted(sats, key=lambda x: x[0])

    if not sats:
        raise ValueError(f"CSV 中没有找到轨道 {orbit_id} 对应的卫星列。")

    return sats


# ============================================================
# 3. 时隙范围处理
# ============================================================

def get_full_time_intervals(df: pd.DataFrame) -> list[tuple[float, float]]:
    """
    根据 time_epsec 生成完整时间区间。

    例如：
        time_epsec = [0, 20, 40, 60]

    则：
        第 0 个时隙：0  到 20
        第 1 个时隙：20 到 40
        第 2 个时隙：40 到 60
        第 3 个时隙：60 到 80

    最后一行没有下一个时间点，因此用前面时隙长度的中位数补齐。
    """
    times = df["time_epsec"].astype(float).tolist()

    if len(times) == 0:
        raise ValueError("CSV 中没有任何时间数据。")

    if len(times) == 1:
        return [(times[0], 20.0)]

    intervals = []

    for i in range(len(times) - 1):
        start_time = times[i]
        duration = times[i + 1] - times[i]

        if duration <= 0:
            raise ValueError("time_epsec 必须严格递增，请检查 CSV 数据。")

        intervals.append((start_time, duration))

    durations = [duration for _, duration in intervals]
    last_duration = pd.Series(durations).median()

    intervals.append((times[-1], last_duration))

    return intervals


def slice_by_slot_range(
    df: pd.DataFrame,
    intervals: list[tuple[float, float]],
    start_slot: Optional[int],
    end_slot: Optional[int]
):
    """
    按时隙编号截取数据。

    采用 Python 切片规则：[start_slot, end_slot)

    例如：
        start_slot = 20
        end_slot   = 120

    表示绘制第 20 个时隙到第 119 个时隙，
    一共 100 个时隙。

    如果每个时隙是 20 秒，则对应 2000 秒。
    """
    total_slots = len(df)

    if start_slot is None:
        start_slot = 0

    if end_slot is None:
        end_slot = total_slots

    if start_slot < 0:
        raise ValueError("START_SLOT 不能小于 0。")

    if end_slot > total_slots:
        raise ValueError(f"END_SLOT 不能超过总时隙数 {total_slots}。")

    if start_slot >= end_slot:
        raise ValueError("START_SLOT 必须小于 END_SLOT。")

    df_subset = df.iloc[start_slot:end_slot].reset_index(drop=True)
    intervals_subset = intervals[start_slot:end_slot]

    return df_subset, intervals_subset, start_slot, end_slot


# ============================================================
# 4. 绘图函数
# ============================================================

def plot_one_state_timeline(
    df_subset: pd.DataFrame,
    intervals_subset: list[tuple[float, float]],
    orbit_id: int,
    satellite_items: list[tuple[int, str]],
    output_dir: Path,
    state_type: str,
    start_slot: int,
    end_slot: int
):
    """
    绘制单个轨道的某一种状态图。

    state_type:
        "sunlight"   -> 光照状态图
        "connection" -> 连接状态图
    """

    if state_type == "sunlight":
        bit_index = 1

        colors = {
            "1": "#FFD700",  # 金色：光照中
            "0": "#808080",  # 灰色：无光照
        }

        labels = {
            "1": "光照",
            "0": "阴影",
        }

        output_name = f"orbit_{orbit_id}_sunlight_slot_{start_slot}_{end_slot}.png"

    elif state_type == "connection":
        bit_index = 0

        colors = {
            "1": "#1f77b4",  # 蓝色：连接
            "0": "#ff7f0e",  # 橙色：未连接
        }

        labels = {
            "1": "连接",
            "0": "未连接",
        }

        output_name = f"orbit_{orbit_id}_connection_slot_{start_slot}_{end_slot}.png"

    else:
        raise ValueError("state_type 只能是 'sunlight' 或 'connection'。")

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

    # 只取轨道内编号作为纵坐标显示内容
    sat_ids = [sat_id for sat_id, _ in satellite_items]
    sat_cols = [sat_col for _, sat_col in satellite_items]

    for sat_idx, sat_col in enumerate(sat_cols):
        y_pos = sat_idx

        for row_idx, (start_time, duration) in enumerate(intervals_subset):
            state = normalize_state(df_subset.loc[row_idx, sat_col])
            state_value = state[bit_index]

            ax.broken_barh(
                [(start_time, duration)],
                (y_pos - BAR_HEIGHT / 2, BAR_HEIGHT),
                facecolors=colors[state_value],
                edgecolors="none"
            )

    # 纵坐标只显示轨道内编号，不显示 sat，也不显示轨道编号
    ax.set_yticks(range(len(sat_ids)))
    ax.set_yticklabels(
        [str(sat_id) for sat_id in sat_ids],
        fontsize=TICK_FONT_SIZE
    )

    ax.tick_params(
        axis="x",
        labelsize=TICK_FONT_SIZE
    )

    ax.tick_params(
        axis="y",
        labelsize=TICK_FONT_SIZE
    )

    ax.set_xlabel(
        "时间（秒）",
        fontsize=AXIS_LABEL_FONT_SIZE
    )

    ax.set_ylabel(
        "卫星编号",
        fontsize=AXIS_LABEL_FONT_SIZE
    )

    # 不设置标题
    # 不使用：
    # title = f"Connection State of Satellites in Orbit {orbit_id}"
    # title = f"Sunlight State of Satellites in Orbit {orbit_id}"

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.4
    )

    legend_handles = [
        Patch(facecolor=colors["1"], label=labels["1"]),
        Patch(facecolor=colors["0"], label=labels["0"]),
    ]

    legend_y = AXES_TOP + LEGEND_DISTANCE_FROM_AXES

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=LEGEND_NCOL,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE
    )

    fig.subplots_adjust(
        top=AXES_TOP,
        bottom=FIG_BOTTOM,
        left=FIG_LEFT,
        right=FIG_RIGHT
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    plt.savefig(
        output_path,
        dpi=SAVE_DPI,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"轨道 {orbit_id} 的 {state_type} 图已保存到：{output_path}")


def plot_orbit_state_timeline(
    df: pd.DataFrame,
    orbit_id: int,
    output_dir: Path,
    start_slot: Optional[int],
    end_slot: Optional[int]
):
    """
    绘制某一轨道的光照状态图和连接状态图。
    """
    satellite_items = get_satellites_in_orbit(df, orbit_id)

    full_intervals = get_full_time_intervals(df)

    df_subset, intervals_subset, real_start_slot, real_end_slot = slice_by_slot_range(
        df=df,
        intervals=full_intervals,
        start_slot=start_slot,
        end_slot=end_slot
    )

    if DRAW_SUNLIGHT_FIGURE:
        plot_one_state_timeline(
            df_subset=df_subset,
            intervals_subset=intervals_subset,
            orbit_id=orbit_id,
            satellite_items=satellite_items,
            output_dir=output_dir,
            state_type="sunlight",
            start_slot=real_start_slot,
            end_slot=real_end_slot
        )

    if DRAW_CONNECTION_FIGURE:
        plot_one_state_timeline(
            df_subset=df_subset,
            intervals_subset=intervals_subset,
            orbit_id=orbit_id,
            satellite_items=satellite_items,
            output_dir=output_dir,
            state_type="connection",
            start_slot=real_start_slot,
            end_slot=real_end_slot
        )


# ============================================================
# 5. 轨道选择处理
# ============================================================

def get_selected_orbits(
    selected_orbits_config: Union[str, list[int]],
    available_orbits: list[int]
) -> list[int]:
    """
    根据代码顶部的 SELECTED_ORBITS 参数确定实际绘图轨道。
    """
    if isinstance(selected_orbits_config, str):
        if selected_orbits_config.lower() == "all":
            return available_orbits
        else:
            raise ValueError(
                "SELECTED_ORBITS 如果是字符串，只能设置为 'all'。"
            )

    selected_orbits = []

    for orbit_id in selected_orbits_config:
        if orbit_id not in available_orbits:
            raise ValueError(
                f"轨道 {orbit_id} 不存在。当前 CSV 中检测到的轨道编号为：{available_orbits}"
            )

        selected_orbits.append(orbit_id)

    if not selected_orbits:
        raise ValueError("SELECTED_ORBITS 中没有任何有效轨道编号。")

    return selected_orbits


# ============================================================
# 6. 主程序
# ============================================================

def main():
    global CSV_PATH, OUTPUT_DIR, SAVE_DPI, SELECTED_ORBITS
    global START_SLOT, END_SLOT, DRAW_SUNLIGHT_FIGURE, DRAW_CONNECTION_FIGURE
    global FIG_WIDTH, FIG_HEIGHT, TICK_FONT_SIZE, AXIS_LABEL_FONT_SIZE
    global LEGEND_FONT_SIZE, AXES_TOP, LEGEND_DISTANCE_FROM_AXES, LEGEND_NCOL
    global FIG_LEFT, FIG_RIGHT, FIG_BOTTOM, BAR_HEIGHT

    print("请根据论文设置输入绘图参数（采用 YAML 格式）。")
    CSV_PATH = str(prompt_setting("CSV_PATH"))
    OUTPUT_DIR = Path(str(prompt_setting("OUTPUT_DIR")))
    SAVE_DPI = int(prompt_setting("SAVE_DPI"))
    SELECTED_ORBITS = prompt_setting("SELECTED_ORBITS")
    START_SLOT = prompt_setting("START_SLOT", allow_none=True)
    END_SLOT = prompt_setting("END_SLOT", allow_none=True)
    DRAW_SUNLIGHT_FIGURE = bool(prompt_setting("DRAW_SUNLIGHT_FIGURE"))
    DRAW_CONNECTION_FIGURE = bool(prompt_setting("DRAW_CONNECTION_FIGURE"))
    FIG_WIDTH = float(prompt_setting("FIG_WIDTH"))
    FIG_HEIGHT = float(prompt_setting("FIG_HEIGHT"))
    TICK_FONT_SIZE = float(prompt_setting("TICK_FONT_SIZE"))
    AXIS_LABEL_FONT_SIZE = float(prompt_setting("AXIS_LABEL_FONT_SIZE"))
    LEGEND_FONT_SIZE = float(prompt_setting("LEGEND_FONT_SIZE"))
    AXES_TOP = float(prompt_setting("AXES_TOP"))
    LEGEND_DISTANCE_FROM_AXES = float(prompt_setting("LEGEND_DISTANCE_FROM_AXES"))
    LEGEND_NCOL = int(prompt_setting("LEGEND_NCOL"))
    FIG_LEFT = float(prompt_setting("FIG_LEFT"))
    FIG_RIGHT = float(prompt_setting("FIG_RIGHT"))
    FIG_BOTTOM = float(prompt_setting("FIG_BOTTOM"))
    BAR_HEIGHT = float(prompt_setting("BAR_HEIGHT"))

    df = load_state_csv(CSV_PATH)

    available_orbits = detect_orbit_ids(df)

    if not available_orbits:
        raise ValueError(
            "CSV 中没有检测到任何卫星列，请检查列名是否类似 sat11、sat12、sat121。"
        )

    selected_orbits = get_selected_orbits(
        selected_orbits_config=SELECTED_ORBITS,
        available_orbits=available_orbits
    )

    print(f"检测到的轨道编号：{available_orbits}")
    print(f"本次绘图轨道编号：{selected_orbits}")
    print(f"绘图时隙范围：[START_SLOT={START_SLOT}, END_SLOT={END_SLOT})")
    print(f"输出文件夹：{OUTPUT_DIR}")

    for orbit_id in selected_orbits:
        plot_orbit_state_timeline(
            df=df,
            orbit_id=orbit_id,
            output_dir=OUTPUT_DIR,
            start_slot=START_SLOT,
            end_slot=END_SLOT
        )

    print()
    print("全部图片生成完成。")


if __name__ == "__main__":
    main()

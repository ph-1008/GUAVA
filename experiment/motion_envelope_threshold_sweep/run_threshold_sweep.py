#!/usr/bin/env python3
"""Compare motion-envelope extraction ranges across MAD threshold settings."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from scipy.ndimage import gaussian_filter1d


GUAVA_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_ROOT = (
    GUAVA_ROOT
    / "EHM-Tracker"
    / "Sign_dataset"
    / "tw_sign_dataset"
    / "sign_net_tracked"
)

HIGH_COEFFICIENTS = (0.8, 1.0, 1.2, 1.5, 2.0)
LOW_COEFFICIENTS = (-0.5, -0.35, 0.0, 0.2, 0.35, 0.5, 0.8)
CURRENT_HIGH = 1.2
CURRENT_LOW = 0.35

WINDOW_SIZE = 5
MIN_PEAK_DISTANCE = 10
OFFSET_FRAMES = 5

LEFT_WRIST_INDEX = 7
RIGHT_WRIST_INDEX = 4


@dataclass(frozen=True)
class WordInput:
    sentence: int
    position: str
    word: str
    video_id: str

    @property
    def tracking_path(self) -> Path:
        return DATA_ROOT / self.video_id / "optim_tracking_ehm.pkl"


WORD_INPUTS = (
    WordInput(1, "middle", "弟弟", "HtxacayoApo"),
    WordInput(1, "middle", "吃", "qJjlVop8ZYc"),
    WordInput(1, "middle", "番茄", "I0eejS0NIt4"),
    WordInput(1, "middle", "不喜歡", "3emK5fPvLH8"),
    WordInput(2, "middle", "今天", "yVJ8AtJMz4w"),
    WordInput(2, "middle", "哪裡", "sOJ7Ugz2Bug"),
    WordInput(2, "middle", "上課", "rieiV5Ba98c"),
)


def configure_plot_font() -> None:
    candidates = (
        Path.home() / ".local/share/fonts/MSJH.TTC",
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    )
    for path in candidates:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_DIR / "results",
        help="Output directory (default: experiment/results).",
    )
    return parser.parse_args()


def load_combined_velocity(item: WordInput) -> tuple[np.ndarray, int, int]:
    with item.tracking_path.open("rb") as handle:
        tracking_data = pickle.load(handle)
    if not isinstance(tracking_data, dict):
        raise TypeError(f"Expected dict in {item.tracking_path}")

    frame_keys = sorted(
        key
        for key in tracking_data
        if isinstance(key, str) and key.startswith("frame_")
    )
    if not frame_keys:
        raise ValueError(f"No frame_* data in {item.tracking_path}")

    left_positions: list[np.ndarray] = []
    right_positions: list[np.ndarray] = []
    missing_frames = 0
    for key in frame_keys:
        frame = tracking_data[key]
        dwpose = frame.get("dwpose_rlt") if isinstance(frame, dict) else None
        keypoints = dwpose.get("keypoints") if isinstance(dwpose, dict) else None
        if isinstance(keypoints, np.ndarray) and len(keypoints) >= 60:
            left_positions.append(
                np.asarray(keypoints[LEFT_WRIST_INDEX], dtype=np.float64)
            )
            right_positions.append(
                np.asarray(keypoints[RIGHT_WRIST_INDEX], dtype=np.float64)
            )
        else:
            left_positions.append(np.zeros(2, dtype=np.float64))
            right_positions.append(np.zeros(2, dtype=np.float64))
            missing_frames += 1

    left = np.asarray(left_positions)
    right = np.asarray(right_positions)
    raw_velocity = np.linalg.norm(np.diff(left, axis=0), axis=1)
    raw_velocity += np.linalg.norm(np.diff(right, axis=0), axis=1)
    sigma = max(1.0, WINDOW_SIZE / 2.5)
    smoothed = gaussian_filter1d(raw_velocity.astype(np.float32), sigma=sigma)
    return np.clip(smoothed, 0, None), len(frame_keys), missing_frames


def velocity_regions(active_indices: np.ndarray) -> list[list[int]]:
    if len(active_indices) == 0:
        return []
    regions: list[list[int]] = []
    start = previous = int(active_indices[0])
    for raw_index in active_indices[1:]:
        index = int(raw_index)
        if index == previous + 1:
            previous = index
        else:
            regions.append([start, previous])
            start = previous = index
    regions.append([start, previous])
    return regions


def expand_to_low(
    regions: list[list[int]], velocity: np.ndarray, low_threshold: float
) -> list[list[int]]:
    expanded: list[list[int]] = []
    last_index = len(velocity) - 1
    for original_start, original_end in regions:
        start, end = original_start, original_end
        while start > 0 and velocity[start - 1] > low_threshold:
            start -= 1
        while end < last_index and velocity[end + 1] > low_threshold:
            end += 1
        expanded.append([start, end])
    return expanded


def filter_regions(
    regions: list[list[int]], velocity: np.ndarray
) -> list[list[int]]:
    if not regions:
        return []
    maximum_peak = max(
        float(np.max(velocity[start : end + 1])) for start, end in regions
    )
    maximum_energy = max(
        float(np.sum(velocity[start : end + 1])) for start, end in regions
    )
    filtered: list[list[int]] = []
    for start, end in regions:
        values = velocity[start : end + 1]
        if end - start + 1 >= 2 and (
            float(np.max(values)) >= maximum_peak * 0.25
            or float(np.sum(values)) >= maximum_energy * 0.15
        ):
            filtered.append([start, end])
    return filtered or regions


def merge_close_regions(
    regions: list[list[int]], maximum_gap: int
) -> list[list[int]]:
    if not regions:
        return []
    ordered = sorted(regions, key=lambda region: region[0])
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        previous = merged[-1]
        if start - previous[1] - 1 <= maximum_gap:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])
    return merged


def enforce_minimum_length(
    start: int, end: int, number_of_frames: int
) -> tuple[int, int]:
    minimum = min(number_of_frames, max(18, int(number_of_frames * 0.45)))
    if end - start >= minimum:
        return start, end
    center = (start + end) // 2
    start = max(0, center - minimum // 2)
    end = min(number_of_frames, start + minimum)
    start = max(0, end - minimum)
    return start, end


def analyze_setting(
    item: WordInput,
    velocity: np.ndarray,
    number_of_frames: int,
    high_coefficient: float,
    low_coefficient: float,
) -> dict[str, Any]:
    median = float(np.median(velocity))
    mad = float(np.median(np.abs(velocity - median)))
    high_threshold = median + high_coefficient * mad
    low_threshold = median + low_coefficient * mad
    base = {
        "sentence": item.sentence,
        "position": item.position,
        "word": item.word,
        "video_id": item.video_id,
        "source_tracking_file": str(item.tracking_path),
        "number_of_frames": number_of_frames,
        "velocity_median": median,
        "velocity_mad": mad,
        "velocity_min": float(np.min(velocity)),
        "velocity_max": float(np.max(velocity)),
        "high_coefficient": high_coefficient,
        "low_coefficient": low_coefficient,
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "low_threshold_below_zero": low_threshold < 0,
        "is_current_setting": (
            high_coefficient == CURRENT_HIGH and low_coefficient == CURRENT_LOW
        ),
    }

    if low_threshold >= high_threshold:
        return {
            **base,
            "valid": False,
            "status": "invalid_low_not_below_high",
            "region_count": 0,
            "regions": "[]",
            "raw_start": "",
            "raw_end_exclusive": "",
            "raw_frame_count": "",
            "raw_retained_ratio": "",
            "pipeline_start": "",
            "pipeline_end_exclusive": "",
            "pipeline_frame_count": "",
            "pipeline_retained_ratio": "",
            "final_start": "",
            "final_end_exclusive": "",
            "final_frame_count": "",
            "final_retained_ratio": "",
            "raw_touches_start": "",
            "raw_touches_end": "",
        }

    regions = velocity_regions(np.where(velocity > high_threshold)[0])
    regions = expand_to_low(regions, velocity, low_threshold)
    regions = filter_regions(regions, velocity)
    regions = merge_close_regions(
        regions, maximum_gap=max(MIN_PEAK_DISTANCE, 10)
    )

    if regions:
        raw_start = regions[0][0]
        # Velocity index e spans frame e -> e+1, hence the exclusive frame end e+2.
        raw_end = min(number_of_frames, regions[-1][1] + 2)
        pipeline_start = regions[0][0] + OFFSET_FRAMES
        # This intentionally mirrors the current production implementation.
        pipeline_end = regions[-1][1] - OFFSET_FRAMES
        status = "motion_regions_detected"
    else:
        raw_start = raw_end = 0
        pipeline_start = OFFSET_FRAMES
        pipeline_end = number_of_frames - OFFSET_FRAMES
        status = "no_stable_region_keep_middle"

    pipeline_start = max(0, pipeline_start)
    pipeline_end = min(number_of_frames, pipeline_end)
    pipeline_start, pipeline_end = enforce_minimum_length(
        pipeline_start, pipeline_end, number_of_frames
    )

    raw_count = max(0, raw_end - raw_start)
    pipeline_count = max(0, pipeline_end - pipeline_start)
    return {
        **base,
        "valid": True,
        "status": status,
        "region_count": len(regions),
        "regions": json.dumps(regions),
        "raw_start": raw_start,
        "raw_end_exclusive": raw_end,
        "raw_frame_count": raw_count,
        "raw_retained_ratio": raw_count / number_of_frames,
        "pipeline_start": pipeline_start,
        "pipeline_end_exclusive": pipeline_end,
        "pipeline_frame_count": pipeline_count,
        "pipeline_retained_ratio": pipeline_count / number_of_frames,
        "final_start": pipeline_start,
        "final_end_exclusive": pipeline_end,
        "final_frame_count": pipeline_count,
        "final_retained_ratio": pipeline_count / number_of_frames,
        "raw_touches_start": raw_start == 0,
        "raw_touches_end": raw_end == number_of_frames,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matrix_for(
    rows: list[dict[str, Any]], field: str, normalize: bool = False
) -> np.ndarray:
    matrix = np.full((len(LOW_COEFFICIENTS), len(HIGH_COEFFICIENTS)), np.nan)
    for row in rows:
        if not row["valid"]:
            continue
        low_index = LOW_COEFFICIENTS.index(row["low_coefficient"])
        high_index = HIGH_COEFFICIENTS.index(row["high_coefficient"])
        value = float(row[field])
        if normalize:
            value /= float(row["number_of_frames"])
        matrix[low_index, high_index] = value
    return matrix


def annotate_heatmap(
    axis: plt.Axes, matrix: np.ndarray, fmt: str, image: Any
) -> None:
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            text = "—" if np.isnan(value) else format(value, fmt)
            text_color = "black"
            if not np.isnan(value):
                red, green, blue, _ = image.cmap(image.norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                text_color = "white" if luminance < 0.48 else "black"
            axis.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color=text_color,
            )


def plot_word_heatmaps(
    item: WordInput, rows: list[dict[str, Any]], output_path: Path
) -> None:
    fields = (
        ("raw_start", False, "Raw start frame", ".0f"),
        ("raw_end_exclusive", False, "Raw end frame (exclusive)", ".0f"),
        ("raw_retained_ratio", False, "Raw retained ratio", ".2f"),
        ("final_retained_ratio", False, "Final retained ratio", ".2f"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    for axis, (field, normalize, title, fmt) in zip(axes.flat, fields):
        matrix = matrix_for(rows, field, normalize=normalize)
        image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        annotate_heatmap(axis, matrix, fmt, image)
        axis.set_xticks(range(len(HIGH_COEFFICIENTS)), HIGH_COEFFICIENTS)
        axis.set_yticks(range(len(LOW_COEFFICIENTS)), LOW_COEFFICIENTS)
        axis.set_xlabel("High coefficient")
        axis.set_ylabel("Low coefficient")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.82)
        current_x = HIGH_COEFFICIENTS.index(CURRENT_HIGH)
        current_y = LOW_COEFFICIENTS.index(CURRENT_LOW)
        axis.add_patch(
            plt.Rectangle(
                (current_x - 0.5, current_y - 0.5),
                1,
                1,
                fill=False,
                edgecolor="red",
                linewidth=2.5,
            )
        )
    figure.suptitle(
        f"{item.word} ({item.video_id}) threshold sweep — red=current 1.2/0.35",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def selected_settings() -> list[tuple[float, float]]:
    settings = [(CURRENT_HIGH, low) for low in LOW_COEFFICIENTS]
    settings.extend(
        (high, CURRENT_LOW)
        for high in HIGH_COEFFICIENTS
        if high != CURRENT_HIGH
    )
    return settings


def plot_selected_ranges(
    item: WordInput,
    velocity: np.ndarray,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    by_setting = {
        (row["high_coefficient"], row["low_coefficient"]): row for row in rows
    }
    settings = selected_settings()
    figure, (envelope_axis, range_axis) = plt.subplots(
        2,
        1,
        figsize=(15, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 2.8]},
        constrained_layout=True,
    )
    envelope_axis.plot(velocity, color="#2878B5", linewidth=2.2)
    current = by_setting[(CURRENT_HIGH, CURRENT_LOW)]
    envelope_axis.axhline(
        current["high_threshold"], color="#8E44AD", linestyle="--", label="目前高閾值"
    )
    envelope_axis.axhline(
        current["low_threshold"], color="#F39C12", linestyle=":", label="目前低閾值"
    )
    minus = by_setting[(CURRENT_HIGH, -0.35)]
    envelope_axis.axhline(
        minus["low_threshold"],
        color="#009E73",
        linestyle="-.",
        label=r"低閾值使用 $-0.35\,\mathrm{MAD}$",
    )
    envelope_axis.set_title(f"Motion envelope 與參考閾值—{item.word}")
    envelope_axis.set_xlabel("幀間速度索引")
    envelope_axis.set_ylabel("手腕位移（pixels/frame）")
    envelope_axis.set_xlim(0, int(current["number_of_frames"]))
    envelope_axis.tick_params(axis="x", labelbottom=True)
    envelope_axis.grid(alpha=0.25)
    envelope_axis.legend()

    colors = plt.cm.tab20(np.linspace(0, 1, len(settings)))
    labels: list[str] = []
    for y, ((high, low), color) in enumerate(zip(settings, colors)):
        row = by_setting[(high, low)]
        if not row["valid"]:
            labels.append(f"H{high:g} / L{low:g}（無效）")
            continue
        start = int(row["final_start"])
        end = int(row["final_end_exclusive"])
        range_axis.barh(y, end - start, left=start, height=0.62, color=color)
        label_is_near_right_edge = end >= int(current["number_of_frames"]) * 0.94
        label_x = end - 0.4 if label_is_near_right_edge else end + 0.4
        range_axis.text(
            label_x,
            y,
            f"[{start}, {end})",
            ha="right" if label_is_near_right_edge else "left",
            va="center",
            fontsize=8,
        )
        marker = "  ← 目前設定" if row["is_current_setting"] else ""
        labels.append(f"H{high:g} / L{low:g}{marker}")
    range_axis.set_yticks(range(len(settings)), labels)
    range_axis.invert_yaxis()
    range_axis.set_xlim(0, int(current["number_of_frames"]))
    range_axis.set_xlabel("幀編號")
    position_label = {"first": "句首", "middle": "句中", "last": "句尾"}[item.position]
    range_axis.set_title(
        f"最終擷取範圍（所有詞均視為{position_label}詞）"
    )
    range_axis.grid(axis="x", alpha=0.25)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current_by_word = {
        row["word"]: row for row in rows if row["is_current_setting"]
    }
    aggregates: list[dict[str, Any]] = []
    for high in HIGH_COEFFICIENTS:
        for low in LOW_COEFFICIENTS:
            selected = [
                row
                for row in rows
                if row["high_coefficient"] == high
                and row["low_coefficient"] == low
            ]
            valid = [row for row in selected if row["valid"]]
            base = {
                "high_coefficient": high,
                "low_coefficient": low,
                "is_current_setting": high == CURRENT_HIGH and low == CURRENT_LOW,
                "valid_word_count": len(valid),
            }
            if not valid:
                aggregates.append(
                    {
                        **base,
                        "mean_raw_retained_ratio": "",
                        "mean_pipeline_retained_ratio": "",
                        "mean_final_retained_ratio": "",
                        "mean_abs_raw_start_delta_from_current": "",
                        "mean_abs_raw_end_delta_from_current": "",
                        "mean_raw_start_shift_from_current": "",
                        "mean_raw_end_shift_from_current": "",
                        "same_raw_range_as_current_count": "",
                        "raw_touches_both_edges_count": "",
                        "no_region_count": "",
                    }
                )
                continue
            start_deltas = []
            end_deltas = []
            start_shifts = []
            end_shifts = []
            same_ranges = 0
            for row in valid:
                current = current_by_word[row["word"]]
                start_deltas.append(abs(int(row["raw_start"]) - int(current["raw_start"])))
                end_deltas.append(
                    abs(
                        int(row["raw_end_exclusive"])
                        - int(current["raw_end_exclusive"])
                    )
                )
                start_shifts.append(
                    int(row["raw_start"]) - int(current["raw_start"])
                )
                end_shifts.append(
                    int(row["raw_end_exclusive"])
                    - int(current["raw_end_exclusive"])
                )
                same_ranges += (
                    row["raw_start"] == current["raw_start"]
                    and row["raw_end_exclusive"] == current["raw_end_exclusive"]
                )
            aggregates.append(
                {
                    **base,
                    "mean_raw_retained_ratio": float(
                        np.mean([row["raw_retained_ratio"] for row in valid])
                    ),
                    "mean_pipeline_retained_ratio": float(
                        np.mean([row["pipeline_retained_ratio"] for row in valid])
                    ),
                    "mean_final_retained_ratio": float(
                        np.mean([row["final_retained_ratio"] for row in valid])
                    ),
                    "mean_abs_raw_start_delta_from_current": float(np.mean(start_deltas)),
                    "mean_abs_raw_end_delta_from_current": float(np.mean(end_deltas)),
                    "mean_raw_start_shift_from_current": float(np.mean(start_shifts)),
                    "mean_raw_end_shift_from_current": float(np.mean(end_shifts)),
                    "same_raw_range_as_current_count": int(same_ranges),
                    "raw_touches_both_edges_count": sum(
                        bool(row["raw_touches_start"] and row["raw_touches_end"])
                        for row in valid
                    ),
                    "no_region_count": sum(
                        row["status"] == "no_stable_region_keep_middle"
                        for row in valid
                    ),
                }
            )
    return aggregates


def plot_aggregate_heatmaps(
    aggregate: list[dict[str, Any]], output_path: Path
) -> None:
    fields = (
        ("mean_raw_retained_ratio", "(a) 平均閾值範圍保留比例", ".2f"),
        ("mean_final_retained_ratio", "(b) 平均最終擷取保留比例", ".2f"),
        (
            "mean_raw_start_shift_from_current",
            r"(c) 平均起點位移 $\Delta s=s-s_{\mathrm{current}}$（幀）",
            ".1f",
        ),
        (
            "mean_raw_end_shift_from_current",
            r"(d) 平均終點位移 $\Delta e=e-e_{\mathrm{current}}$（幀）",
            ".1f",
        ),
    )
    figure, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    for axis, (field, title, fmt) in zip(axes.flat, fields):
        matrix = np.full((len(LOW_COEFFICIENTS), len(HIGH_COEFFICIENTS)), np.nan)
        for row in aggregate:
            if row[field] == "":
                continue
            y = LOW_COEFFICIENTS.index(row["low_coefficient"])
            x = HIGH_COEFFICIENTS.index(row["high_coefficient"])
            matrix[y, x] = float(row[field])
        if "shift" in field:
            finite = matrix[np.isfinite(matrix)]
            absolute_maximum = max(abs(float(np.min(finite))), abs(float(np.max(finite))), 1e-6)
            norm = TwoSlopeNorm(
                vmin=-absolute_maximum, vcenter=0.0, vmax=absolute_maximum
            )
            image = axis.imshow(
                matrix, aspect="auto", cmap="coolwarm", norm=norm
            )
        else:
            image = axis.imshow(matrix, aspect="auto", cmap="viridis")
        annotate_heatmap(axis, matrix, fmt, image)
        axis.set_xticks(range(len(HIGH_COEFFICIENTS)), HIGH_COEFFICIENTS)
        axis.set_yticks(range(len(LOW_COEFFICIENTS)), LOW_COEFFICIENTS)
        axis.set_xlabel("高閾值係數")
        axis.set_ylabel("低閾值係數")
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.82)
        current_x = HIGH_COEFFICIENTS.index(CURRENT_HIGH)
        current_y = LOW_COEFFICIENTS.index(CURRENT_LOW)
        axis.add_patch(
            plt.Rectangle(
                (current_x - 0.5, current_y - 0.5),
                1,
                1,
                fill=False,
                edgecolor="red",
                linewidth=2.5,
            )
        )
    figure.suptitle(
        "Motion envelope 雙閾值敏感度分析（紅框為目前設定 1.2/0.35）",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_report(rows: list[dict[str, Any]], aggregate: list[dict[str, Any]]) -> str:
    by_setting = {
        (row["high_coefficient"], row["low_coefficient"]): row
        for row in aggregate
    }
    current = by_setting[(CURRENT_HIGH, CURRENT_LOW)]
    minus = by_setting[(CURRENT_HIGH, -0.35)]
    lower_high = by_setting[(0.8, CURRENT_LOW)]
    higher_high = by_setting[(2.0, CURRENT_LOW)]

    def percent(value: Any) -> str:
        return f"{float(value) * 100:.1f}%"

    lines = [
        "# Motion envelope threshold sweep report",
        "",
        "## 實驗範圍",
        "",
        "- 詞彙：弟弟、吃、番茄、不喜歡、今天、哪裡、上課。",
        f"- 有效閾值組合：{sum(row['valid_word_count'] > 0 for row in aggregate)} 組。",
        "- 沒有人工標註；以下數據描述擷取範圍，不代表語意正確率。",
        "",
        "## 代表性設定的跨詞平均",
        "",
        "| high / low | raw 保留比例 | 最終擷取保留比例 | 與目前 start 差 | 與目前 end 差 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, row in (
        ("1.2 / -0.35（low 減 MAD）", minus),
        ("0.8 / 0.35（較低 high）", lower_high),
        ("1.2 / 0.35（目前）", current),
        ("2.0 / 0.35（較高 high）", higher_high),
    ):
        lines.append(
            f"| {label} | {percent(row['mean_raw_retained_ratio'])} | "
            f"{percent(row['mean_final_retained_ratio'])} | "
            f"{float(row['mean_abs_raw_start_delta_from_current']):.1f} 幀 | "
            f"{float(row['mean_abs_raw_end_delta_from_current']):.1f} 幀 |"
        )

    lines.extend(
        [
            "",
            "## 解讀限制",
            "",
            "- `low` 越低，已啟動的區域通常會延伸越遠；若低於靜止追蹤雜訊，可能一路延伸到影片邊界。",
            "- `high` 越高，能啟動的區域通常越少；可能排除雜訊，也可能漏掉輕微動作。",
            "- 本實驗忽略詞彙原本的句首或句尾位置，七個詞皆以句中詞規則計算完整起訖點。",
            "- 最短片段限制會把過短結果重新擴張，所以判讀閾值本身時應優先看 `raw_*` 欄位。",
            "- 沒有人工標註時，只能挑選範圍長短與穩定性的折衷，不能證明哪組是最佳語意邊界。",
            "",
        ]
    )

    current_word_rows = [row for row in rows if row["is_current_setting"]]
    lines.extend(
        [
            "## 目前 1.2 / 0.35 的逐詞範圍",
            "",
            "| 詞彙 | 總幀數 | raw 範圍 | 最終擷取範圍 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in current_word_rows:
        lines.append(
            f"| {row['word']} | {row['number_of_frames']} | "
            f"[{row['raw_start']}, {row['raw_end_exclusive']}) | "
            f"[{row['final_start']}, {row['final_end_exclusive']}) |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    per_word_dir = output_dir / "per_word"
    per_word_dir.mkdir(parents=True, exist_ok=True)
    configure_plot_font()

    missing = [str(item.tracking_path) for item in WORD_INPUTS if not item.tracking_path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))

    all_rows: list[dict[str, Any]] = []
    word_summaries: list[dict[str, Any]] = []
    for item in WORD_INPUTS:
        velocity, number_of_frames, missing_frames = load_combined_velocity(item)
        rows = [
            analyze_setting(item, velocity, number_of_frames, high, low)
            for high in HIGH_COEFFICIENTS
            for low in LOW_COEFFICIENTS
        ]
        all_rows.extend(rows)
        target_dir = per_word_dir / item.word
        target_dir.mkdir(parents=True, exist_ok=True)
        plot_word_heatmaps(item, rows, target_dir / "threshold_heatmaps.png")
        plot_selected_ranges(
            item, velocity, rows, target_dir / "selected_range_comparison.png"
        )
        current = next(row for row in rows if row["is_current_setting"])
        summary = {
            **asdict(item),
            "tracking_path": str(item.tracking_path),
            "number_of_frames": number_of_frames,
            "missing_dwpose_frames": missing_frames,
            "gaussian_sigma": max(1.0, WINDOW_SIZE / 2.5),
            "current_setting_result": current,
        }
        (target_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        word_summaries.append(summary)
        print(f"Analyzed {item.word}: {number_of_frames} frames")

    write_csv(output_dir / "threshold_sweep_all_words.csv", all_rows)
    aggregate = aggregate_rows(all_rows)
    write_csv(output_dir / "aggregate_by_threshold.csv", aggregate)
    plot_aggregate_heatmaps(aggregate, output_dir / "aggregate_heatmaps.png")
    (output_dir / "report.md").write_text(
        build_report(all_rows, aggregate), encoding="utf-8"
    )
    experiment_summary = {
        "experiment": "motion_envelope_mad_threshold_sweep",
        "sentences": [
            ["弟弟", "吃", "番茄", "不喜歡"],
            ["今天", "哪裡", "上課"],
        ],
        "high_coefficients": HIGH_COEFFICIENTS,
        "low_coefficients": LOW_COEFFICIENTS,
        "current_setting": {"high": CURRENT_HIGH, "low": CURRENT_LOW},
        "fixed_parameters": {
            "window_size": WINDOW_SIZE,
            "gaussian_sigma": max(1.0, WINDOW_SIZE / 2.5),
            "minimum_peak_distance": MIN_PEAK_DISTANCE,
            "merge_gap": max(MIN_PEAK_DISTANCE, 10),
            "offset_frames": OFFSET_FRAMES,
            "minimum_segment_frames": "min(N, max(18, int(N * 0.45)))",
        },
        "word_inputs": word_summaries,
        "limitations": [
            "No manually annotated start/end frames.",
            "Results compare extraction ranges, not semantic correctness.",
        ],
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(experiment_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()

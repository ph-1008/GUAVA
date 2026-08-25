#!/usr/bin/env python3
"""Analyze left- and right-hand motion envelopes independently.

The segment detection logic intentionally mirrors
``tracking_concatenation_final_5.detect_sign_segment_by_peaks``. The only
difference is that the two wrist velocities are not added together: each hand
is smoothed, thresholded, and segmented independently.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


CJK_FONT_PATH = Path.home() / ".local/share/fonts/MSJH.TTC"
if CJK_FONT_PATH.is_file():
    font_manager.fontManager.addfont(str(CJK_FONT_PATH))
    cjk_font_name = font_manager.FontProperties(fname=CJK_FONT_PATH).get_name()
    plt.rcParams["font.family"] = [cjk_font_name, "DejaVu Sans"]

plt.rcParams["axes.unicode_minus"] = False

EXPERIMENT_DIR = Path(__file__).resolve().parent
LEFT_WRIST_INDEX = 7
RIGHT_WRIST_INDEX = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run independent left/right-hand motion envelope analysis."
    )
    parser.add_argument(
        "--tracking-pkl",
        type=Path,
        help=(
            "Path to optim_tracking_ehm.pkl. If omitted, the script uses the "
            "single parameter file found under this experiment directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for plots and JSON. If omitted, outputs are placed in "
            "results/<word>_<source-directory>/ so existing results are not overwritten."
        ),
    )
    parser.add_argument(
        "--word",
        help="Word shown in plot titles and recorded in the JSON summary.",
    )
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--min-peak-distance", type=int, default=10)
    parser.add_argument("--offset-frames", type=int, default=5)
    return parser.parse_args()


def resolve_tracking_path(requested_path: Path | None) -> Path:
    if requested_path is not None:
        tracking_path = requested_path.expanduser().resolve()
        if not tracking_path.is_file():
            raise FileNotFoundError(f"Tracking parameter file not found: {tracking_path}")
        return tracking_path

    candidates = sorted(EXPERIMENT_DIR.glob("*/optim_tracking_ehm.pkl"))
    if len(candidates) != 1:
        candidate_text = "\n".join(f"  - {path}" for path in candidates) or "  (none)"
        raise RuntimeError(
            "Expected exactly one optim_tracking_ehm.pkl under the experiment "
            f"directory, but found {len(candidates)}:\n{candidate_text}\n"
            "Specify one explicitly with --tracking-pkl."
        )
    return candidates[0].resolve()


def load_tracking_data(tracking_path: Path) -> tuple[dict[str, Any], list[str]]:
    with tracking_path.open("rb") as handle:
        tracking_data = pickle.load(handle)

    if not isinstance(tracking_data, dict):
        raise TypeError("Tracking parameter file must contain a dictionary.")

    frame_keys = sorted(
        key for key in tracking_data if isinstance(key, str) and key.startswith("frame_")
    )
    if not frame_keys:
        raise ValueError("No frame_* entries found in tracking parameter file.")
    return tracking_data, frame_keys


def extract_wrist_positions(
    tracking_data: dict[str, Any], frame_keys: list[str]
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract wrists exactly as the current concatenation implementation does."""
    left_positions: list[np.ndarray] = []
    right_positions: list[np.ndarray] = []
    missing_frames = 0

    for frame_key in frame_keys:
        frame_data = tracking_data[frame_key]
        dwpose = frame_data.get("dwpose_rlt") if isinstance(frame_data, dict) else None
        keypoints = dwpose.get("keypoints") if isinstance(dwpose, dict) else None

        if isinstance(keypoints, np.ndarray) and len(keypoints) >= 60:
            left_positions.append(np.asarray(keypoints[LEFT_WRIST_INDEX], dtype=np.float64))
            right_positions.append(np.asarray(keypoints[RIGHT_WRIST_INDEX], dtype=np.float64))
        else:
            # This matches tracking_concatenation_final_5.py: missing DWPose data
            # is represented by a zero coordinate for both wrists.
            left_positions.append(np.zeros(2, dtype=np.float64))
            right_positions.append(np.zeros(2, dtype=np.float64))
            missing_frames += 1

    return np.asarray(left_positions), np.asarray(right_positions), missing_frames


def calculate_velocity(positions: np.ndarray) -> np.ndarray:
    """Return per-frame-interval 2D wrist displacement."""
    return np.linalg.norm(np.diff(positions, axis=0), axis=1).astype(np.float32)


def velocity_regions(active_indices: np.ndarray) -> list[list[int]]:
    """Convert sorted active velocity indices into inclusive contiguous regions."""
    if len(active_indices) == 0:
        return []

    regions: list[list[int]] = []
    start = int(active_indices[0])
    previous = start
    for raw_index in active_indices[1:]:
        index = int(raw_index)
        if index == previous + 1:
            previous = index
        else:
            regions.append([start, previous])
            start = index
            previous = index
    regions.append([start, previous])
    return regions


def expand_regions_to_low_threshold(
    regions: list[list[int]], velocity: np.ndarray, low_threshold: float
) -> list[list[int]]:
    """Expand high-threshold regions until motion falls below the low threshold."""
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


def merge_close_regions(
    regions: list[list[int]], maximum_gap: int
) -> list[list[int]]:
    """Merge neighboring motion regions separated by a short pause."""
    if not regions:
        return []

    ordered_regions = sorted(regions, key=lambda item: item[0])
    merged = [list(ordered_regions[0])]
    for start, end in ordered_regions[1:]:
        previous = merged[-1]
        gap = start - previous[1] - 1
        if gap <= maximum_gap:
            previous[1] = max(previous[1], end)
        else:
            merged.append([start, end])
    return merged


def enforce_min_segment_length(
    start_frame: int,
    end_frame: int,
    number_of_frames: int,
    minimum_segment_frames: int,
) -> tuple[int, int]:
    """Expand a short segment around its center, matching the current method."""
    if end_frame - start_frame >= minimum_segment_frames:
        return start_frame, end_frame

    center = (start_frame + end_frame) // 2
    half = minimum_segment_frames // 2
    start_frame = max(0, center - half)
    end_frame = min(number_of_frames, start_frame + minimum_segment_frames)
    start_frame = max(0, end_frame - minimum_segment_frames)
    return start_frame, end_frame


def analyze_hand_velocity(
    raw_velocity: np.ndarray,
    number_of_frames: int,
    window_size: int,
    minimum_peak_distance: int,
    offset_frames: int,
) -> dict[str, Any]:
    """Apply the production motion-envelope rules to one hand."""
    if number_of_frames <= 1 or np.all(raw_velocity == 0):
        return {
            "status": "all_zero_or_too_short",
            "smoothed_velocity": raw_velocity,
            "start_frame": 0,
            "end_frame": number_of_frames,
            "regions": [],
            "peaks": np.array([], dtype=int),
            "low_threshold": None,
            "high_threshold": None,
            "statistics": {},
        }

    smooth_sigma = max(1.0, window_size / 2.5)
    smoothed_velocity = gaussian_filter1d(raw_velocity, sigma=smooth_sigma)
    smoothed_velocity = np.clip(smoothed_velocity, 0, None)

    velocity_mean = float(np.mean(smoothed_velocity))
    velocity_std = float(np.std(smoothed_velocity))
    velocity_median = float(np.median(smoothed_velocity))
    velocity_mad = float(
        np.median(np.abs(smoothed_velocity - velocity_median))
    )
    velocity_p35 = float(np.percentile(smoothed_velocity, 35))
    velocity_p65 = float(np.percentile(smoothed_velocity, 65))
    velocity_p90 = float(np.percentile(smoothed_velocity, 90))

    statistics = {
        "mean": velocity_mean,
        "std": velocity_std,
        "median": velocity_median,
        "mad": velocity_mad,
        "p35": velocity_p35,
        "p65": velocity_p65,
        "p90": velocity_p90,
        "total_energy": float(np.sum(smoothed_velocity)),
        "maximum": float(np.max(smoothed_velocity)),
    }

    if number_of_frames <= 25 or velocity_p90 - velocity_p35 < 1e-3:
        peaks, _ = find_peaks(
            smoothed_velocity, distance=max(1, minimum_peak_distance // 2)
        )
        return {
            "status": "insufficient_variation_keep_all",
            "smoothed_velocity": smoothed_velocity,
            "start_frame": 0,
            "end_frame": number_of_frames,
            "regions": [],
            "peaks": peaks,
            "low_threshold": None,
            "high_threshold": None,
            "statistics": statistics,
        }

    if velocity_mad > 1e-3:
        high_threshold = velocity_median + velocity_mad * 1.2
        low_threshold = velocity_median + velocity_mad * 0.35
    else:
        high_threshold = velocity_p65
        low_threshold = velocity_p35

    if high_threshold <= low_threshold:
        high_threshold = velocity_p65
        low_threshold = min(velocity_p35, high_threshold * 0.75)

    peak_prominence = max(velocity_mad, velocity_std * 0.15, 1e-3)
    peaks, _ = find_peaks(
        smoothed_velocity,
        height=high_threshold,
        distance=minimum_peak_distance,
        prominence=peak_prominence,
    )

    active_indices = np.where(smoothed_velocity > high_threshold)[0]
    regions = velocity_regions(active_indices)
    regions = expand_regions_to_low_threshold(
        regions, smoothed_velocity, low_threshold
    )

    if regions:
        maximum_peak = max(
            float(np.max(smoothed_velocity[start : end + 1]))
            for start, end in regions
        )
        maximum_energy = max(
            float(np.sum(smoothed_velocity[start : end + 1]))
            for start, end in regions
        )
        filtered_regions = []
        for start, end in regions:
            region_velocity = smoothed_velocity[start : end + 1]
            region_peak = float(np.max(region_velocity))
            region_energy = float(np.sum(region_velocity))
            region_length = end - start + 1
            if region_length >= 2 and (
                region_peak >= maximum_peak * 0.25
                or region_energy >= maximum_energy * 0.15
            ):
                filtered_regions.append([start, end])
        regions = filtered_regions or regions

    regions = merge_close_regions(
        regions, maximum_gap=max(minimum_peak_distance, 10)
    )

    if regions:
        start_frame = regions[0][0] + offset_frames
        end_frame = regions[-1][1] - offset_frames
        status = "motion_regions_detected"
    else:
        start_frame = offset_frames
        end_frame = number_of_frames - offset_frames
        status = "no_stable_region_keep_middle"

    start_frame = max(0, start_frame)
    end_frame = min(number_of_frames, end_frame)
    minimum_segment_frames = min(
        number_of_frames, max(18, int(number_of_frames * 0.45))
    )
    start_frame, end_frame = enforce_min_segment_length(
        start_frame,
        end_frame,
        number_of_frames,
        minimum_segment_frames,
    )

    return {
        "status": status,
        "smoothed_velocity": smoothed_velocity,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "regions": regions,
        "peaks": peaks,
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
        "statistics": statistics,
    }


def plot_hand_result(
    hand_label: str,
    word_label: str,
    result: dict[str, Any],
    number_of_frames: int,
    y_axis_max: float,
    output_path: Path,
) -> None:
    velocity = result["smoothed_velocity"]
    velocity_indices = np.arange(len(velocity))
    start_frame = result["start_frame"]
    end_frame = result["end_frame"]

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(
        velocity_indices,
        velocity,
        color="#2878B5",
        linewidth=2.2,
        label=f"{hand_label} wrist motion envelope",
    )

    low_threshold = result["low_threshold"]
    high_threshold = result["high_threshold"]
    if low_threshold is not None:
        axis.axhline(
            low_threshold,
            color="#F39C12",
            linestyle=":",
            linewidth=1.8,
            label=f"Low threshold: {low_threshold:.2f}",
        )
    if high_threshold is not None:
        axis.axhline(
            high_threshold,
            color="#8E44AD",
            linestyle="--",
            linewidth=1.8,
            label=f"High threshold: {high_threshold:.2f}",
        )

    axis.axvspan(
        start_frame,
        end_frame,
        color="#69B96B",
        alpha=0.22,
        label=f"Extracted: [{start_frame}, {end_frame})",
    )
    axis.axvline(
        start_frame,
        color="#009E73",
        linestyle="--",
        linewidth=2.2,
        label=f"Start: {start_frame}",
    )
    axis.axvline(
        end_frame,
        color="#D55E00",
        linestyle="--",
        linewidth=2.2,
        label=f"End: {end_frame}",
    )

    axis.set_xlim(0, max(number_of_frames, 1))
    axis.set_ylim(0, y_axis_max)
    axis.set_xlabel("Frame number")
    axis.set_ylabel("Wrist displacement (pixels/frame)")
    axis.set_title(f"{hand_label} Hand Motion Envelope - {word_label}")
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_envelope_comparison(
    results: dict[str, dict[str, Any]],
    word_label: str,
    number_of_frames: int,
    y_axis_max: float,
    output_path: Path,
) -> None:
    """Plot left, right, and the production-style summed envelope together."""
    figure, axis = plt.subplots(figsize=(14, 7))
    colors = {
        "left": "#222222",
        "right": "#E67E22",
        "combined": "#2878B5",
    }
    labels = {
        "left": r"Left hand $v_t^L$",
        "right": r"Right hand $v_t^R$",
        "combined": r"Original $v_t = v_t^L + v_t^R$",
    }

    for name in ("left", "right", "combined"):
        velocity = results[name]["smoothed_velocity"]
        axis.plot(
            np.arange(len(velocity)),
            velocity,
            color=colors[name],
            linewidth=2.0 if name != "combined" else 2.8,
            alpha=0.9,
            label=labels[name],
        )

    combined_result = results["combined"]
    low_threshold = combined_result["low_threshold"]
    high_threshold = combined_result["high_threshold"]
    if low_threshold is not None:
        axis.axhline(
            low_threshold,
            color="#F39C12",
            linestyle=":",
            linewidth=1.8,
            label=f"Original low threshold: {low_threshold:.2f}",
        )
    if high_threshold is not None:
        axis.axhline(
            high_threshold,
            color="#8E44AD",
            linestyle="--",
            linewidth=1.8,
            label=f"Original high threshold: {high_threshold:.2f}",
        )

    start_frame = combined_result["start_frame"]
    end_frame = combined_result["end_frame"]
    axis.axvspan(
        start_frame,
        end_frame,
        color="#69B96B",
        alpha=0.16,
        label=f"Original extracted range: [{start_frame}, {end_frame})",
    )
    axis.axvline(
        start_frame,
        color="#009E73",
        linestyle="--",
        linewidth=1.8,
        label=f"Original start: {start_frame}",
    )
    axis.axvline(
        end_frame,
        color="#D55E00",
        linestyle="--",
        linewidth=1.8,
        label=f"Original end: {end_frame}",
    )

    axis.set_xlim(0, max(number_of_frames, 1))
    axis.set_ylim(0, y_axis_max)
    axis.set_xlabel("Frame number")
    axis.set_ylabel("Wrist displacement (pixels/frame)")
    axis.set_title(
        f"Left, Right, and Original Motion Envelopes - {word_label}"
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="upper right", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def json_ready_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": result["status"],
        "start_frame": int(result["start_frame"]),
        "end_frame_exclusive": int(result["end_frame"]),
        "extracted_frame_count": int(
            result["end_frame"] - result["start_frame"]
        ),
        "motion_regions_velocity_indices_inclusive": result["regions"],
        "auxiliary_peaks": [int(value) for value in result["peaks"]],
        "low_threshold": result["low_threshold"],
        "high_threshold": result["high_threshold"],
        "statistics": result["statistics"],
    }


def main() -> None:
    args = parse_args()
    tracking_path = resolve_tracking_path(args.tracking_pkl)
    word_label = args.word or tracking_path.parent.name
    source_directory_name = tracking_path.parent.name
    safe_output_name = f"{word_label}_{source_directory_name}".replace("/", "_")
    safe_output_name = safe_output_name.replace("\\", "_")
    if args.output_dir is None:
        output_dir = (EXPERIMENT_DIR / "results" / safe_output_name).resolve()
    else:
        output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tracking_data, frame_keys = load_tracking_data(tracking_path)
    number_of_frames = len(frame_keys)
    left_positions, right_positions, missing_frames = extract_wrist_positions(
        tracking_data, frame_keys
    )

    raw_velocities = {
        "left": calculate_velocity(left_positions),
        "right": calculate_velocity(right_positions),
    }
    # This is the original production definition:
    # v_t = ||left[t+1] - left[t]|| + ||right[t+1] - right[t]||.
    raw_velocities["combined"] = raw_velocities["left"] + raw_velocities["right"]
    results = {
        hand: analyze_hand_velocity(
            raw_velocity,
            number_of_frames=number_of_frames,
            window_size=args.window_size,
            minimum_peak_distance=args.min_peak_distance,
            offset_frames=args.offset_frames,
        )
        for hand, raw_velocity in raw_velocities.items()
    }
    combined_maximum = float(
        np.max(results["combined"]["smoothed_velocity"])
    )
    common_y_axis_max = max(1.0, combined_maximum * 1.08)

    plot_paths = {
        "left": output_dir / "left_hand_motion_envelope.png",
        "right": output_dir / "right_hand_motion_envelope.png",
        "comparison": output_dir / "left_right_original_motion_envelopes.png",
    }
    plot_hand_result(
        "Left",
        word_label,
        results["left"],
        number_of_frames,
        common_y_axis_max,
        plot_paths["left"],
    )
    plot_hand_result(
        "Right",
        word_label,
        results["right"],
        number_of_frames,
        common_y_axis_max,
        plot_paths["right"],
    )
    plot_envelope_comparison(
        results,
        word_label,
        number_of_frames,
        common_y_axis_max,
        plot_paths["comparison"],
    )

    summary = {
        "experiment": "independent_left_right_motion_envelope",
        "source_tracking_file": str(tracking_path),
        "source_directory": source_directory_name,
        "word": word_label,
        "number_of_frames": number_of_frames,
        "missing_dwpose_frames": missing_frames,
        "wrist_keypoint_indices": {
            "left": LEFT_WRIST_INDEX,
            "right": RIGHT_WRIST_INDEX,
        },
        "parameters": {
            "window_size": args.window_size,
            "gaussian_sigma": max(1.0, args.window_size / 2.5),
            "min_peak_distance": args.min_peak_distance,
            "offset_frames": args.offset_frames,
            "minimum_segment_frames": min(
                number_of_frames, max(18, int(number_of_frames * 0.45))
            ),
            "plot_y_axis_min": 0.0,
            "plot_y_axis_max": common_y_axis_max,
        },
        "left_hand": json_ready_result(results["left"]),
        "right_hand": json_ready_result(results["right"]),
        "original_combined": json_ready_result(results["combined"]),
        "plots": {hand: str(path) for hand, path in plot_paths.items()},
    }
    summary_path = output_dir / "single_hand_motion_envelope_results.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Source: {tracking_path}")
    print(f"Frames: {number_of_frames}; missing DWPose frames: {missing_frames}")
    for hand in ("left", "right"):
        result = results[hand]
        print(
            f"{hand.capitalize()} hand: "
            f"[{result['start_frame']}, {result['end_frame']}) "
            f"({result['end_frame'] - result['start_frame']} frames), "
            f"status={result['status']}"
        )
        print(f"  Plot: {plot_paths[hand]}")
    combined_result = results["combined"]
    print(
        "Original combined: "
        f"[{combined_result['start_frame']}, {combined_result['end_frame']}) "
        f"({combined_result['end_frame'] - combined_result['start_frame']} frames)"
    )
    print(f"  Comparison plot: {plot_paths['comparison']}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

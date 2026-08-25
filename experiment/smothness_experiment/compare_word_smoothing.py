#!/usr/bin/env python3
"""
Compare EHM tracking outputs before/after word-level smoothing.

The folders produced by the UI pipeline contain one optim_tracking_ehm.pkl and
one gloss_pipeline_manifest.json. This script groups folders by gloss token and
compares the run with smooth_word_segments=false against the run with
smooth_word_segments=true. If the manifest flag is missing, earlier timestamped
folders are treated as "before" and later folders as "after".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parent
UI_DIR_RE = re.compile(r"^ui_(?P<date>\d{8})_(?P<time>\d{6})_(?P<label>.+)$")
SYMBOL_TOKENS = {"^^", "。", ".", "!", "?", "！", "？", ",", "，"}


@dataclass(frozen=True)
class RunCase:
    gloss: str
    timestamp: str
    smooth_word_segments: bool | None
    path: Path

    @property
    def phase(self) -> str:
        if self.smooth_word_segments is True:
            return "after"
        if self.smooth_word_segments is False:
            return "before"
        return "unknown"


@dataclass(frozen=True)
class RunPair:
    gloss: str
    before: RunCase
    after: RunCase


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute numerical before/after differences for word smoothing outputs."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_ROOT / "word_smoothing_metrics",
    )
    args = parser.parse_args()

    cases = discover_cases(args.root)
    pairs = pair_cases(cases)
    if not pairs:
        raise SystemExit(f"No before/after word-smoothing pairs found under {args.root}")

    case_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    pair_metric_rows: list[dict[str, Any]] = []

    for pair in pairs:
        before_payload = load_payload(pair.before)
        after_payload = load_payload(pair.after)

        before_metrics = evaluate_case(pair.before, before_payload)
        after_metrics = evaluate_case(pair.after, after_payload)
        case_rows.extend([before_metrics, after_metrics])

        pair_summary = evaluate_pair(pair, before_payload, after_payload)
        pair_rows.append(pair_summary)
        pair_metric_rows.extend(compare_metrics(pair, before_metrics, after_metrics, pair_summary))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    case_csv = args.out_dir / "word_smoothing_case_metrics.csv"
    comparison_csv = args.out_dir / "word_smoothing_pair_comparison.csv"
    summary_json = args.out_dir / "word_smoothing_pair_summary.json"
    report_md = args.out_dir / "word_smoothing_report.md"

    write_csv(case_csv, case_rows)
    write_csv(comparison_csv, pair_metric_rows)
    write_json(summary_json, pair_rows)
    write_report(report_md, pairs, pair_metric_rows, pair_rows)

    print(f"Wrote case metrics: {case_csv}")
    print(f"Wrote pair comparison: {comparison_csv}")
    print(f"Wrote pair summary: {summary_json}")
    print(f"Wrote report: {report_md}")


def discover_cases(root: Path) -> list[RunCase]:
    cases: list[RunCase] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        match = UI_DIR_RE.match(child.name)
        if not match:
            continue
        if not (child / "optim_tracking_ehm.pkl").is_file():
            continue

        manifest = load_manifest(child / "gloss_pipeline_manifest.json")
        gloss = manifest_gloss(manifest) or match.group("label")
        timestamp = f"{match.group('date')}_{match.group('time')}"
        smooth = manifest.get("smooth_word_segments")
        smooth = smooth if isinstance(smooth, bool) else None
        cases.append(RunCase(gloss=gloss, timestamp=timestamp, smooth_word_segments=smooth, path=child))

    return sorted(cases, key=lambda item: (item.gloss, item.timestamp))


def pair_cases(cases: list[RunCase]) -> list[RunPair]:
    by_gloss: dict[str, list[RunCase]] = {}
    for case in cases:
        by_gloss.setdefault(case.gloss, []).append(case)

    pairs: list[RunPair] = []
    for gloss, group in sorted(by_gloss.items()):
        group = sorted(group, key=lambda item: item.timestamp)
        befores = [case for case in group if case.smooth_word_segments is False]
        afters = [case for case in group if case.smooth_word_segments is True]

        if not befores or not afters:
            midpoint = len(group) // 2
            befores = group[:midpoint]
            afters = group[midpoint:]

        for before, after in zip(befores, afters):
            if before.timestamp <= after.timestamp:
                pairs.append(RunPair(gloss=gloss, before=before, after=after))
            else:
                pairs.append(RunPair(gloss=gloss, before=after, after=before))

    return pairs


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def manifest_gloss(manifest: dict[str, Any]) -> str | None:
    tokens = manifest.get("gloss_tokens")
    if not isinstance(tokens, list):
        return None
    content_tokens = [str(token) for token in tokens if str(token) not in SYMBOL_TOKENS]
    return "_".join(content_tokens) if content_tokens else None


def load_payload(case: RunCase) -> dict[str, Any]:
    with (case.path / "optim_tracking_ehm.pkl").open("rb") as f:
        tracking = pickle.load(f)

    frame_keys = load_frame_keys(case.path / "videos_info.json", tracking)
    return {
        "tracking": tracking,
        "frame_keys": frame_keys,
        "sequences": collect_sequences(tracking, frame_keys),
    }


def load_frame_keys(videos_info_path: Path, tracking: dict[str, Any]) -> list[str]:
    if videos_info_path.is_file():
        with videos_info_path.open("r", encoding="utf-8") as f:
            videos_info = json.load(f)
        if isinstance(videos_info, dict) and videos_info:
            first_video = next(iter(videos_info.values()))
            frame_keys = first_video.get("frames_keys")
            if frame_keys:
                return [key for key in frame_keys if key in tracking]

    return sorted(tracking.keys())


def collect_sequences(tracking: dict[str, Any], frame_keys: list[str]) -> dict[str, np.ndarray]:
    sequences: dict[str, list[np.ndarray]] = {
        "dwpose_hands_2d": [],
        "dwpose_hand_scores": [],
        "smplx_left_hand_pose": [],
        "smplx_right_hand_pose": [],
        "smplx_both_hand_pose": [],
        "smplx_body_pose": [],
        "smplx_body_global_pose": [],
        "smplx_global_pose": [],
        "smplx_all_pose": [],
        "smplx_body_cam": [],
        "smplx_camera_translation": [],
    }

    for key in frame_keys:
        frame = tracking[key]
        dwpose = frame.get("dwpose_rlt", {})
        hands = np.asarray(dwpose.get("hands"), dtype=np.float64).reshape(-1)
        scores = np.asarray(dwpose.get("scores"), dtype=np.float64)[-42:]

        smplx = frame["smplx_coeffs"]
        global_pose = flatten(smplx.get("global_pose"))
        body_pose = flatten(smplx.get("body_pose"))
        left_hand_pose = flatten(smplx.get("left_hand_pose"))
        right_hand_pose = flatten(smplx.get("right_hand_pose"))
        both_hand_pose = np.concatenate([left_hand_pose, right_hand_pose])
        body_cam = flatten(smplx.get("body_cam"))
        camera_rt = np.asarray(smplx.get("camera_RT_params"), dtype=np.float64)
        camera_t = camera_rt[:3, 3] if camera_rt.shape == (3, 4) else np.full(3, np.nan)

        sequences["dwpose_hands_2d"].append(hands)
        sequences["dwpose_hand_scores"].append(scores)
        sequences["smplx_left_hand_pose"].append(left_hand_pose)
        sequences["smplx_right_hand_pose"].append(right_hand_pose)
        sequences["smplx_both_hand_pose"].append(both_hand_pose)
        sequences["smplx_body_pose"].append(body_pose)
        sequences["smplx_body_global_pose"].append(np.concatenate([global_pose, body_pose]))
        sequences["smplx_global_pose"].append(global_pose)
        sequences["smplx_all_pose"].append(
            np.concatenate([global_pose, body_pose, left_hand_pose, right_hand_pose])
        )
        sequences["smplx_body_cam"].append(body_cam)
        sequences["smplx_camera_translation"].append(camera_t)

    return {name: np.stack(values) for name, values in sequences.items()}


def evaluate_case(case: RunCase, payload: dict[str, Any]) -> dict[str, Any]:
    seqs = payload["sequences"]
    row: dict[str, Any] = {
        "gloss": case.gloss,
        "phase": case.phase,
        "smooth_word_segments": case.smooth_word_segments,
        "timestamp": case.timestamp,
        "frames": len(payload["frame_keys"]),
        "path": str(case.path),
    }

    hand_scores = seqs["dwpose_hand_scores"]
    row["dwpose_hand_score_mean"] = safe_mean(hand_scores)
    row["dwpose_hand_visible_ratio_0p7"] = float(np.mean(hand_scores >= 0.7))

    for name, seq in seqs.items():
        if name == "dwpose_hand_scores":
            continue
        for metric_name, value in sequence_metrics(seq).items():
            row[f"{name}_{metric_name}"] = value

    row["smplx_hand_speed_corr_with_dwpose"] = safe_corr(
        sequence_speed(seqs["smplx_both_hand_pose"]),
        sequence_speed(seqs["dwpose_hands_2d"]),
    )
    return row


def evaluate_pair(
    pair: RunPair,
    before_payload: dict[str, Any],
    after_payload: dict[str, Any],
) -> dict[str, Any]:
    before_keys = before_payload["frame_keys"]
    after_keys = after_payload["frame_keys"]
    common_keys = [key for key in before_keys if key in set(after_keys)]
    before_index = {key: idx for idx, key in enumerate(before_keys)}
    after_index = {key: idx for idx, key in enumerate(after_keys)}

    row: dict[str, Any] = {
        "gloss": pair.gloss,
        "before_timestamp": pair.before.timestamp,
        "after_timestamp": pair.after.timestamp,
        "before_path": str(pair.before.path),
        "after_path": str(pair.after.path),
        "before_frames": len(before_keys),
        "after_frames": len(after_keys),
        "common_frames": len(common_keys),
    }

    for name in [
        "smplx_both_hand_pose",
        "smplx_left_hand_pose",
        "smplx_right_hand_pose",
        "smplx_all_pose",
        "smplx_body_global_pose",
        "smplx_body_cam",
        "smplx_camera_translation",
    ]:
        before_seq = before_payload["sequences"][name][[before_index[key] for key in common_keys]]
        after_seq = after_payload["sequences"][name][[after_index[key] for key in common_keys]]
        diff = after_seq - before_seq
        row[f"paired_{name}_rmse"] = float(np.sqrt(np.nanmean(diff * diff)))
        row[f"paired_{name}_mean_l2"] = safe_mean(np.linalg.norm(diff, axis=1))

    return row


def compare_metrics(
    pair: RunPair,
    before: dict[str, Any],
    after: dict[str, Any],
    pair_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ignored = {"gloss", "phase", "smooth_word_segments", "timestamp", "frames", "path"}

    for metric in sorted(key for key in before.keys() if key not in ignored):
        before_value = before.get(metric, math.nan)
        after_value = after.get(metric, math.nan)
        rows.append(make_metric_row(pair, metric, before_value, after_value))

    for metric, value in sorted(pair_summary.items()):
        if metric.startswith("paired_"):
            rows.append(
                {
                    "gloss": pair.gloss,
                    "metric": metric,
                    "before": math.nan,
                    "after": value,
                    "delta_after_minus_before": math.nan,
                    "percent_change": math.nan,
                    "preferred_direction": "context",
                    "improved": "",
                    "before_timestamp": pair.before.timestamp,
                    "after_timestamp": pair.after.timestamp,
                }
            )

    return rows


def make_metric_row(
    pair: RunPair,
    metric: str,
    before_value: Any,
    after_value: Any,
) -> dict[str, Any]:
    before_num = as_float(before_value)
    after_num = as_float(after_value)
    delta = after_num - before_num if is_finite(before_num) and is_finite(after_num) else math.nan
    pct = delta / abs(before_num) * 100.0 if is_finite(delta) and abs(before_num) > 1e-12 else math.nan
    direction = preferred_direction(metric)

    improved: bool | str
    if not is_finite(delta) or direction == "context":
        improved = ""
    elif abs(delta) <= 1e-12:
        improved = "same"
    elif direction == "lower":
        improved = delta < 0
    else:
        improved = delta > 0

    return {
        "gloss": pair.gloss,
        "metric": metric,
        "before": before_num,
        "after": after_num,
        "delta_after_minus_before": delta,
        "percent_change": pct,
        "preferred_direction": direction,
        "improved": improved,
        "before_timestamp": pair.before.timestamp,
        "after_timestamp": pair.after.timestamp,
    }


def preferred_direction(metric: str) -> str:
    if metric.endswith("spectral_arc_length") or metric.endswith("corr_with_dwpose"):
        return "higher"
    if any(token in metric for token in ["acceleration", "jerk", "normalized_jerk_cost"]):
        return "lower"
    return "context"


def sequence_metrics(seq: np.ndarray) -> dict[str, float]:
    speed = sequence_speed(seq)
    accel = sequence_acceleration(seq)
    jerk = sequence_jerk(seq)
    return {
        "velocity_mean": safe_mean(speed),
        "velocity_rms": safe_rms(speed),
        "acceleration_mean": safe_mean(accel),
        "acceleration_rms": safe_rms(accel),
        "jerk_mean": safe_mean(jerk),
        "jerk_rms": safe_rms(jerk),
        "normalized_jerk_cost": normalized_jerk_cost(seq),
        "spectral_arc_length": spectral_arc_length(speed),
    }


def sequence_speed(seq: np.ndarray) -> np.ndarray:
    if len(seq) < 2:
        return np.array([], dtype=np.float64)
    return np.linalg.norm(np.diff(seq, axis=0), axis=1)


def sequence_acceleration(seq: np.ndarray) -> np.ndarray:
    if len(seq) < 3:
        return np.array([], dtype=np.float64)
    return np.linalg.norm(np.diff(seq, n=2, axis=0), axis=1)


def sequence_jerk(seq: np.ndarray) -> np.ndarray:
    if len(seq) < 4:
        return np.array([], dtype=np.float64)
    return np.linalg.norm(np.diff(seq, n=3, axis=0), axis=1)


def normalized_jerk_cost(seq: np.ndarray) -> float:
    if len(seq) < 4:
        return math.nan
    speed = sequence_speed(seq)
    path_length = float(np.nansum(speed))
    if path_length <= 1e-12:
        return math.nan
    jerk = np.diff(seq, n=3, axis=0)
    jerk_energy = float(np.nansum(np.linalg.norm(jerk, axis=1) ** 2))
    duration = float(len(seq) - 1)
    return (duration**5 / path_length**2) * jerk_energy


def spectral_arc_length(speed: np.ndarray) -> float:
    speed = np.asarray(speed, dtype=np.float64)
    speed = speed[np.isfinite(speed)]
    if len(speed) < 4 or np.nanmax(np.abs(speed)) <= 1e-12:
        return math.nan

    n_fft = int(2 ** math.ceil(math.log2(len(speed)) + 4))
    spectrum = np.abs(np.fft.rfft(speed, n=n_fft))
    if np.max(spectrum) <= 1e-12:
        return math.nan
    spectrum = spectrum / np.max(spectrum)
    freq = np.fft.rfftfreq(n_fft, d=1.0)

    keep = spectrum >= 0.05
    if np.any(keep):
        last = int(np.max(np.flatnonzero(keep)))
        spectrum = spectrum[: last + 1]
        freq = freq[: last + 1]

    if len(freq) < 2:
        return math.nan
    freq = freq / max(freq[-1], 1e-12)
    return -float(np.sum(np.sqrt(np.diff(freq) ** 2 + np.diff(spectrum) ** 2)))


def flatten(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def safe_mean(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return math.nan
    return float(np.nanmean(arr))


def safe_rms(values: Any) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return math.nan
    return float(np.sqrt(np.nanmean(arr * arr)))


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n < 3:
        return math.nan

    a = a[:n]
    b = b[:n]
    valid = ~(np.isnan(a) | np.isnan(b))
    if np.sum(valid) < 3:
        return math.nan

    a = a[valid]
    b = b[valid]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def as_float(value: Any) -> float:
    if isinstance(value, (bool, str)) or value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def is_finite(value: float) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "gloss",
        "phase",
        "metric",
        "before",
        "after",
        "delta_after_minus_before",
        "percent_change",
        "preferred_direction",
        "improved",
        "timestamp",
        "before_timestamp",
        "after_timestamp",
        "frames",
        "path",
    ]
    columns = preferred + [col for col in columns if col not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_report(
    path: Path,
    pairs: list[RunPair],
    metric_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> None:
    selected_metrics = [
        "smplx_body_pose_acceleration_mean",
        "smplx_body_pose_jerk_mean",
        "smplx_body_pose_normalized_jerk_cost",
        "smplx_body_pose_spectral_arc_length",
        "smplx_body_global_pose_jerk_mean",
        "smplx_body_global_pose_normalized_jerk_cost",
        "smplx_all_pose_jerk_mean",
        "smplx_both_hand_pose_jerk_mean",
        "smplx_hand_speed_corr_with_dwpose",
    ]

    by_gloss_metric = {(row["gloss"], row["metric"]): row for row in metric_rows}
    by_gloss_pair = {row["gloss"]: row for row in pair_rows}

    lines = [
        "# Word-Level Smoothing Experiment",
        "",
        "Pairing rule: `smooth_word_segments=false` is the before run and `smooth_word_segments=true` is the after run. If the manifest flag is unavailable, timestamp order is used.",
        "",
        "## Recommended thesis metrics",
        "",
        "- `jerk_mean` / `jerk_rms`: third-order temporal difference. Lower values indicate less abrupt motion changes.",
        "- `normalized_jerk_cost`: integrated squared jerk normalized by duration and path length. Lower values are better and are easier to compare across different sequence lengths or motion amplitudes.",
        "- `spectral_arc_length`: smoothness from the Fourier spectrum of the speed profile. Values closer to zero are smoother.",
        "- `smplx_hand_speed_corr_with_dwpose`: correlation between optimized SMPL-X hand-pose speed and observed DWPose hand motion. Use it as a fidelity check so smoothing is not interpreted as motion collapse.",
        "",
        "## Summary",
        "",
    ]

    for pair in pairs:
        lines.append(f"### {pair.gloss}")
        lines.append("")
        pair_summary = by_gloss_pair[pair.gloss]
        lines.append(
            f"Before `{pair.before.timestamp}` -> after `{pair.after.timestamp}`; "
            f"common frames: {pair_summary['common_frames']}."
        )
        lines.append("")
        lines.append("| metric | before | after | delta | percent change | preferred | improved |")
        lines.append("|---|---:|---:|---:|---:|---|---|")
        for metric in selected_metrics:
            row = by_gloss_metric.get((pair.gloss, metric))
            if row is None:
                continue
            lines.append(
                "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                    metric,
                    fmt(row["before"]),
                    fmt(row["after"]),
                    fmt(row["delta_after_minus_before"]),
                    fmt_percent(row["percent_change"]),
                    row["preferred_direction"],
                    row["improved"],
                )
            )
        lines.append("")
        for metric in [
            "paired_smplx_both_hand_pose_rmse",
            "paired_smplx_all_pose_rmse",
            "paired_smplx_camera_translation_rmse",
        ]:
            if metric in pair_summary:
                lines.append(f"- `{metric}`: {fmt(pair_summary[metric])}")
        lines.append("")

    lines.extend(
        [
            "## Notes for writing",
            "",
            "For the thesis table, report `smplx_body_pose_normalized_jerk_cost`, `smplx_body_pose_spectral_arc_length`, and `smplx_all_pose_jerk_mean` as the main result. In the current pkl files, the paired hand-pose RMSE is zero, so word-level smoothing changed body pose but did not change the explicit SMPL-X hand-pose parameters. Keep `smplx_hand_speed_corr_with_dwpose` as context/fidelity rather than claiming hand-pose smoothing from these files alone.",
            "",
            "Suggested citations:",
            "",
            "- Flash and Hogan, 1985, minimum-jerk model for human arm movement: https://www.jneurosci.org/content/5/7/1688",
            "- Balasubramanian et al., 2012, spectral arc length movement-smoothness metric: https://doi.org/10.1109/TBME.2011.2179545",
            "- Balasubramanian et al., 2015, movement-smoothness analysis and dimensionless jerk discussion: https://doi.org/10.1186/s12984-015-0090-9",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def fmt(value: Any) -> str:
    value = as_float(value)
    if not is_finite(value):
        return "nan"
    return f"{value:.6g}"


def fmt_percent(value: Any) -> str:
    value = as_float(value)
    if not is_finite(value):
        return "nan"
    return f"{value:.2f}%"


if __name__ == "__main__":
    main()

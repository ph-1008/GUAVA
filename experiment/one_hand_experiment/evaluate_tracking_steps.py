#!/usr/bin/env python3
"""
Evaluate EHM-Tracker optimization-step variants for one-hand cases.

This script is designed for paired folders such as:

  one_hand_experiment/can                 -> assumed step1000
  one_hand_experiment/can_step100/can     -> step100
  one_hand_experiment/find                -> assumed step1000
  one_hand_experiment/find_step100/find   -> step100

It computes lightweight metrics that do not require loading the EHM/GUAVA
model or running GPU projection:

  1. DWPose hand visibility statistics
  2. DWPose hand 2D motion smoothness
  3. SMPL-X body/hand/camera parameter smoothness
  4. Correlation between observed hand motion envelope and hand-pose motion

Important limitation:
  True 2D reprojection error and PCK require projecting optimized 3D joints
  back to image coordinates with the EHM/SMPL-X model. This script does not
  compute those model-projection metrics; it gives a quick pilot analysis for
  deciding whether a larger step100 vs. step1000 experiment is worth running.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path(__file__).resolve().parent
HAND_SCORE_THRESHOLD = 0.7


@dataclass(frozen=True)
class TrackCase:
    gloss: str
    step_label: str
    path: Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute pilot metrics for EHM-Tracker step100 vs step1000 outputs."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_ROOT / "tracking_step_metrics.csv",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_ROOT / "tracking_step_report.md",
    )
    parser.add_argument("--score-threshold", type=float, default=HAND_SCORE_THRESHOLD)
    args = parser.parse_args()

    cases = discover_cases(args.root)
    if not cases:
        raise SystemExit(f"No tracking cases found under {args.root}")

    rows = []
    for case in cases:
        rows.append(evaluate_case(case, args.score_threshold))

    write_csv(args.out_csv, rows)
    write_markdown(args.out_md, rows)

    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote report: {args.out_md}")


def discover_cases(root: Path) -> list[TrackCase]:
    cases: list[TrackCase] = []

    # Direct folders without "step100" are treated as the long optimization baseline.
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.endswith("_step100"):
            continue
        if has_tracking_payload(child):
            cases.append(TrackCase(gloss=child.name, step_label="step1000", path=child))

    # Folders named *_step100 usually contain the actual tracked subfolder.
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.endswith("_step100"):
            continue
        gloss = child.name[: -len("_step100")]
        nested = child / gloss
        if has_tracking_payload(nested):
            cases.append(TrackCase(gloss=gloss, step_label="step100", path=nested))
        elif has_tracking_payload(child):
            cases.append(TrackCase(gloss=gloss, step_label="step100", path=child))

    return sorted(cases, key=lambda item: (item.gloss, item.step_label))


def has_tracking_payload(path: Path) -> bool:
    return (path / "optim_tracking_ehm.pkl").is_file() and (path / "videos_info.json").is_file()


def evaluate_case(case: TrackCase, score_threshold: float) -> dict[str, Any]:
    tracking = load_pickle(case.path / "optim_tracking_ehm.pkl")
    frame_keys = load_frame_keys(case.path / "videos_info.json", tracking)

    dwpose_keypoints = []
    dwpose_scores = []
    smplx_vectors: dict[str, list[np.ndarray]] = {
        "body_pose": [],
        "left_hand_pose": [],
        "right_hand_pose": [],
        "both_hand_pose": [],
        "camera_translation": [],
        "all_pose": [],
    }

    for key in frame_keys:
        frame = tracking[key]
        dwpose = frame.get("dwpose_rlt", {})
        dwpose_keypoints.append(np.asarray(dwpose.get("keypoints"), dtype=np.float64))
        dwpose_scores.append(np.asarray(dwpose.get("scores"), dtype=np.float64))

        smplx = frame["smplx_coeffs"]
        body_pose = flatten(smplx.get("body_pose"))
        left_hand_pose = flatten(smplx.get("left_hand_pose"))
        right_hand_pose = flatten(smplx.get("right_hand_pose"))
        global_pose = flatten(smplx.get("global_pose"))
        camera_rt = np.asarray(smplx.get("camera_RT_params"), dtype=np.float64)
        camera_t = camera_rt[:3, 3] if camera_rt.shape == (3, 4) else np.full(3, np.nan)

        smplx_vectors["body_pose"].append(body_pose)
        smplx_vectors["left_hand_pose"].append(left_hand_pose)
        smplx_vectors["right_hand_pose"].append(right_hand_pose)
        smplx_vectors["both_hand_pose"].append(np.concatenate([left_hand_pose, right_hand_pose]))
        smplx_vectors["camera_translation"].append(camera_t)
        smplx_vectors["all_pose"].append(
            np.concatenate([global_pose, body_pose, left_hand_pose, right_hand_pose])
        )

    keypoints = np.stack(dwpose_keypoints)
    scores = np.stack(dwpose_scores)

    hand_kpts = keypoints[:, -42:, :2]
    hand_scores = scores[:, -42:]
    left_scores = scores[:, -42:-21]
    right_scores = scores[:, -21:]

    observed_hand_speed = weighted_keypoint_speed(hand_kpts, hand_scores, score_threshold)
    model_hand_speed = sequence_speed(np.stack(smplx_vectors["both_hand_pose"]))

    row: dict[str, Any] = {
        "gloss": case.gloss,
        "step": case.step_label,
        "path": str(case.path),
        "frames": len(frame_keys),
        "hand_score_mean": safe_mean(hand_scores),
        "left_hand_score_mean": safe_mean(left_scores),
        "right_hand_score_mean": safe_mean(right_scores),
        "hand_visible_ratio": visible_ratio(hand_scores, score_threshold),
        "left_hand_visible_ratio": visible_ratio(left_scores, score_threshold),
        "right_hand_visible_ratio": visible_ratio(right_scores, score_threshold),
        "one_hand_low_visibility_frame_ratio": one_hand_low_visibility_frame_ratio(
            left_scores, right_scores, score_threshold
        ),
        "dwpose_hand_speed_mean": safe_mean(observed_hand_speed),
        "dwpose_hand_accel_mean": safe_mean(second_difference_norm(hand_kpts, hand_scores, score_threshold)),
        "model_hand_motion_corr_with_dwpose": safe_corr(model_hand_speed, observed_hand_speed),
    }

    for name, vectors in smplx_vectors.items():
        seq = np.stack(vectors)
        row[f"{name}_velocity_mean"] = safe_mean(sequence_speed(seq))
        row[f"{name}_acceleration_mean"] = safe_mean(sequence_acceleration(seq))
        row[f"{name}_jerk_mean"] = safe_mean(sequence_jerk(seq))

    return row


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_frame_keys(videos_info_path: Path, tracking: dict[str, Any]) -> list[str]:
    with videos_info_path.open("r", encoding="utf-8") as f:
        videos_info = json.load(f)

    if isinstance(videos_info, dict) and videos_info:
        first_video = next(iter(videos_info.values()))
        frame_keys = first_video.get("frames_keys")
        if frame_keys:
            return list(frame_keys)

    return sorted(tracking.keys())


def flatten(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(0, dtype=np.float64)
    return np.asarray(value, dtype=np.float64).reshape(-1)


def visible_ratio(scores: np.ndarray, threshold: float) -> float:
    return float(np.mean(scores >= threshold))


def one_hand_low_visibility_frame_ratio(
    left_scores: np.ndarray,
    right_scores: np.ndarray,
    threshold: float,
) -> float:
    left_frame_ratio = np.mean(left_scores >= threshold, axis=1)
    right_frame_ratio = np.mean(right_scores >= threshold, axis=1)
    one_low = ((left_frame_ratio < 0.25) & (right_frame_ratio >= 0.5)) | (
        (right_frame_ratio < 0.25) & (left_frame_ratio >= 0.5)
    )
    return float(np.mean(one_low))


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


def weighted_keypoint_speed(
    keypoints: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if len(keypoints) < 2:
        return np.array([], dtype=np.float64)

    delta = np.linalg.norm(np.diff(keypoints, axis=0), axis=-1)
    valid = (scores[1:] >= threshold) & (scores[:-1] >= threshold)
    return weighted_frame_average(delta, valid)


def second_difference_norm(
    keypoints: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    if len(keypoints) < 3:
        return np.array([], dtype=np.float64)

    accel = np.linalg.norm(np.diff(keypoints, n=2, axis=0), axis=-1)
    valid = (scores[2:] >= threshold) & (scores[1:-1] >= threshold) & (scores[:-2] >= threshold)
    return weighted_frame_average(accel, valid)


def weighted_frame_average(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for idx in range(values.shape[0]):
        if np.any(valid[idx]):
            out[idx] = float(np.mean(values[idx][valid[idx]]))
    return out


def safe_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return math.nan
    return float(np.nanmean(arr))


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
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row.keys()})
    preferred = ["gloss", "step", "frames", "path"]
    columns = preferred + [col for col in columns if col not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_gloss: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_gloss.setdefault(row["gloss"], []).append(row)

    metric_cols = [
        "hand_visible_ratio",
        "left_hand_visible_ratio",
        "right_hand_visible_ratio",
        "one_hand_low_visibility_frame_ratio",
        "model_hand_motion_corr_with_dwpose",
        "both_hand_pose_velocity_mean",
        "both_hand_pose_acceleration_mean",
        "both_hand_pose_jerk_mean",
        "all_pose_acceleration_mean",
        "camera_translation_acceleration_mean",
    ]

    lines = [
        "# EHM-Tracker Step Pilot Metrics",
        "",
        "Lower acceleration/jerk usually indicates smoother parameters, but excessive smoothness can also mean motion collapse.",
        "Higher model/DWPose motion correlation is a useful pilot signal that optimized hand-pose dynamics still follow observed hand motion.",
        "For a formal thesis experiment, add model projection metrics such as 2D reprojection error or PCK.",
        "",
    ]

    for gloss, group in sorted(by_gloss.items()):
        lines.append(f"## {gloss}")
        lines.append("")
        lines.append("| metric | step100 | step1000 | step100 - step1000 |")
        lines.append("|---|---:|---:|---:|")
        by_step = {row["step"]: row for row in group}
        for metric in metric_cols:
            v100 = by_step.get("step100", {}).get(metric, math.nan)
            v1000 = by_step.get("step1000", {}).get(metric, math.nan)
            delta = v100 - v1000 if is_number(v100) and is_number(v1000) else math.nan
            lines.append(
                f"| `{metric}` | {fmt(v100)} | {fmt(v1000)} | {fmt(delta)} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.floating)) and not math.isnan(float(value))


def fmt(value: Any) -> str:
    if not is_number(value):
        return "nan"
    return f"{float(value):.6g}"


if __name__ == "__main__":
    main()

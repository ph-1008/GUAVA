#!/usr/bin/env python3
"""
Evaluate before/after word smoothing on paired one-hand and two-hand samples.

This script computes:
  1. motion-parameter smoothness metrics from tracking pkl files
  2. 2D reprojection error and PCK against DWPose pseudo ground truth

Input layout:
  before_smooth/{1_hand,2_hands}/<sample_id>/optim_tracking_ehm.pkl
  after_smooth/{1_hand,2_hands}/<sample_id>/optim_tracking_ehm.pkl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


GUAVA_ROOT = Path("/home/paohan/GUAVA")
EHM_ROOT = GUAVA_ROOT / "EHM-Tracker"
DEFAULT_ROOT = GUAVA_ROOT / "experiment" / "smothness_experiment"
DEFAULT_BEFORE_ROOT = DEFAULT_ROOT / "before_smooth"
DEFAULT_AFTER_ROOT = DEFAULT_ROOT / "after_smooth"
DEFAULT_OUT_DIR = DEFAULT_ROOT / "smoothing_eval_metrics"

if str(EHM_ROOT) not in sys.path:
    sys.path.insert(0, str(EHM_ROOT))

from src.modules.ehm import EHM  # noqa: E402
from src.modules.refiner.smplx_utils import smplx_joints_to_dwpose  # noqa: E402
from src.utils.graphics import GS_Camera  # noqa: E402


LEFT_HAND = np.arange(92, 113)
RIGHT_HAND = np.arange(113, 134)
HANDS = np.arange(92, 134)
BODY_NO_FACE = np.r_[0:24, 92:134]


@dataclass(frozen=True)
class PairedSample:
    hand_group: str
    sample_id: str
    before_dir: Path
    after_dir: Path


@dataclass(frozen=True)
class EvalCase:
    hand_group: str
    sample_id: str
    phase: str
    path: Path


def main() -> None:
    args = parse_args()
    pairs = discover_pairs(args.before_root, args.after_root)
    if args.only:
        wanted = set(args.only)
        pairs = [pair for pair in pairs if pair.sample_id in wanted]
    if args.exclude:
        excluded = set(args.exclude)
        pairs = [pair for pair in pairs if pair.sample_id not in excluded]
    for spec in args.extra_pair:
        pairs.append(parse_extra_pair(spec))
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit("No before/after pairs found.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        case
        for pair in pairs
        for case in [
            EvalCase(pair.hand_group, pair.sample_id, "before", pair.before_dir),
            EvalCase(pair.hand_group, pair.sample_id, "after", pair.after_dir),
        ]
    ]

    smooth_rows: list[dict[str, Any]] = []
    smooth_compare_rows: list[dict[str, Any]] = []
    if not args.skip_smoothness:
        print(f"Computing smoothness metrics for {len(cases)} cases...")
        smooth_by_key = {}
        for case in cases:
            row = evaluate_smoothness_case(case)
            smooth_rows.append(row)
            smooth_by_key[(case.hand_group, case.sample_id, case.phase)] = row
        smooth_compare_rows = compare_case_rows(pairs, smooth_by_key, preferred_direction)
        write_csv(args.out_dir / "smoothness_case_metrics.csv", smooth_rows)
        write_csv(args.out_dir / "smoothness_pair_comparison.csv", smooth_compare_rows)

    reproj_rows: list[dict[str, Any]] = []
    reproj_compare_rows: list[dict[str, Any]] = []
    if not args.skip_reprojection:
        print(f"Computing reprojection/PCK metrics for {len(cases)} cases...")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA is not available; falling back to CPU. This may be slow.")
            args.device = "cpu"
        model = load_ehm_model(args.device)
        pck_thresholds = parse_float_list(args.pck_thresholds)
        reproj_by_key = {}
        for idx, case in enumerate(cases, start=1):
            print(f"[{idx}/{len(cases)}] {case.hand_group}/{case.sample_id}/{case.phase}")
            row = evaluate_reprojection_case(
                case,
                model,
                args.device,
                args.batch_size,
                args.image_size,
                args.tanfov,
                args.score_threshold,
                pck_thresholds,
            )
            reproj_rows.append(row)
            reproj_by_key[(case.hand_group, case.sample_id, case.phase)] = row
        reproj_compare_rows = compare_case_rows(pairs, reproj_by_key, reproj_preferred_direction)
        write_csv(args.out_dir / "reprojection_pck_case_metrics.csv", reproj_rows)
        write_csv(args.out_dir / "reprojection_pck_pair_comparison.csv", reproj_compare_rows)

    write_report(
        args.out_dir / "smoothing_experiment_report.md",
        pairs,
        smooth_compare_rows,
        reproj_compare_rows,
    )

    print(f"Wrote outputs to: {args.out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate before/after word smoothing.")
    parser.add_argument("--before-root", type=Path, default=DEFAULT_BEFORE_ROOT)
    parser.add_argument("--after-root", type=Path, default=DEFAULT_AFTER_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--exclude", nargs="*", default=None)
    parser.add_argument(
        "--extra-pair",
        action="append",
        default=[],
        metavar="HAND_GROUP:SAMPLE_ID:BEFORE_DIR:AFTER_DIR",
        help="Add a before/after pair outside the before_smooth/after_smooth layout.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-smoothness", action="store_true")
    parser.add_argument("--skip-reprojection", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--tanfov", type=float, default=1.0 / 24.0)
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--pck-thresholds", default="0.05,0.10")
    return parser.parse_args()


def discover_pairs(before_root: Path, after_root: Path) -> list[PairedSample]:
    pairs: list[PairedSample] = []
    for hand_group in ["1_hand", "2_hands"]:
        before_group = before_root / hand_group
        after_group = after_root / hand_group
        if not before_group.is_dir() or not after_group.is_dir():
            continue
        for before_dir in sorted(before_group.iterdir()):
            after_dir = after_group / before_dir.name
            if before_dir.is_dir() and has_tracking_payload(before_dir) and has_tracking_payload(after_dir):
                pairs.append(PairedSample(hand_group, before_dir.name, before_dir, after_dir))
    return pairs


def parse_extra_pair(spec: str) -> PairedSample:
    parts = spec.split(":", 3)
    if len(parts) != 4:
        raise ValueError("--extra-pair must use HAND_GROUP:SAMPLE_ID:BEFORE_DIR:AFTER_DIR")
    hand_group, sample_id, before_dir, after_dir = parts
    pair = PairedSample(hand_group, sample_id, Path(before_dir), Path(after_dir))
    if not has_tracking_payload(pair.before_dir):
        raise FileNotFoundError(f"Missing before payload for extra pair: {pair.before_dir}")
    if not has_tracking_payload(pair.after_dir):
        raise FileNotFoundError(f"Missing after payload for extra pair: {pair.after_dir}")
    return pair


def has_tracking_payload(path: Path) -> bool:
    return (
        (path / "optim_tracking_ehm.pkl").is_file()
        and (path / "id_share_params.pkl").is_file()
        and (path / "videos_info.json").is_file()
    )


def evaluate_smoothness_case(case: EvalCase) -> dict[str, Any]:
    tracking = load_pickle(case.path / "optim_tracking_ehm.pkl")
    frame_keys = load_frame_keys(case.path / "videos_info.json", tracking)
    seqs = collect_parameter_sequences(tracking, frame_keys)

    row: dict[str, Any] = {
        "hand_group": case.hand_group,
        "sample_id": case.sample_id,
        "phase": case.phase,
        "frames": len(frame_keys),
        "path": str(case.path),
    }
    for name, seq in seqs.items():
        for metric_name, value in sequence_metrics(seq).items():
            row[f"{name}_{metric_name}"] = value
    return row


def collect_parameter_sequences(tracking: dict[str, Any], frame_keys: list[str]) -> dict[str, np.ndarray]:
    names = [
        "smplx_body_pose",
        "smplx_global_pose",
        "smplx_both_hand_pose",
        "smplx_all_pose",
        "smplx_body_cam",
        "smplx_camera_translation",
        "mano_left_hand_pose",
        "mano_right_hand_pose",
        "mano_both_hand_pose",
        "mano_left_global_orient",
        "mano_right_global_orient",
        "mano_both_global_orient",
        "flame_expression",
        "flame_jaw",
        "flame_neck",
    ]
    seqs: dict[str, list[np.ndarray]] = {name: [] for name in names}

    for key in frame_keys:
        frame = tracking[key]
        smplx = frame.get("smplx_coeffs", {})
        left_mano = frame.get("left_mano_coeffs", {})
        right_mano = frame.get("right_mano_coeffs", {})
        flame = frame.get("flame_coeffs", {})

        smplx_global = flatten(smplx.get("global_pose"))
        smplx_body = flatten(smplx.get("body_pose"))
        smplx_left_hand = flatten(smplx.get("left_hand_pose"))
        smplx_right_hand = flatten(smplx.get("right_hand_pose"))
        smplx_both_hand = np.concatenate([smplx_left_hand, smplx_right_hand])

        mano_left_hand = flatten(left_mano.get("hand_pose"))
        mano_right_hand = flatten(right_mano.get("hand_pose"))
        mano_left_orient = flatten(left_mano.get("global_orient"))
        mano_right_orient = flatten(right_mano.get("global_orient"))
        camera_rt = np.asarray(smplx.get("camera_RT_params"), dtype=np.float64)
        camera_t = camera_rt[:3, 3] if camera_rt.shape == (3, 4) else np.full(3, np.nan)

        seqs["smplx_body_pose"].append(smplx_body)
        seqs["smplx_global_pose"].append(smplx_global)
        seqs["smplx_both_hand_pose"].append(smplx_both_hand)
        seqs["smplx_all_pose"].append(np.concatenate([smplx_global, smplx_body, smplx_both_hand]))
        seqs["smplx_body_cam"].append(flatten(smplx.get("body_cam")))
        seqs["smplx_camera_translation"].append(camera_t)
        seqs["mano_left_hand_pose"].append(mano_left_hand)
        seqs["mano_right_hand_pose"].append(mano_right_hand)
        seqs["mano_both_hand_pose"].append(np.concatenate([mano_left_hand, mano_right_hand]))
        seqs["mano_left_global_orient"].append(mano_left_orient)
        seqs["mano_right_global_orient"].append(mano_right_orient)
        seqs["mano_both_global_orient"].append(np.concatenate([mano_left_orient, mano_right_orient]))
        seqs["flame_expression"].append(flatten(flame.get("expression_params")))
        seqs["flame_jaw"].append(flatten(flame.get("jaw_params")))
        seqs["flame_neck"].append(flatten(flame.get("neck_pose", flame.get("neck_pose_params"))))

    return {name: np.stack(values) for name, values in seqs.items() if values and values[0].size > 0}


def sequence_metrics(seq: np.ndarray) -> dict[str, float]:
    speed = sequence_speed(seq)
    accel = sequence_acceleration(seq)
    jerk = sequence_jerk(seq)
    return {
        "velocity_mean": safe_mean(speed),
        "acceleration_mean": safe_mean(accel),
        "jerk_mean": safe_mean(jerk),
        "jerk_rms": safe_rms(jerk),
        "normalized_jerk_cost": normalized_jerk_cost(seq),
        "spectral_arc_length": spectral_arc_length(speed),
    }


def evaluate_reprojection_case(
    case: EvalCase,
    model: EHM,
    device: str,
    batch_size: int,
    image_size: int,
    tanfov: float,
    score_threshold: float,
    pck_thresholds: list[float],
) -> dict[str, Any]:
    tracking = load_pickle(case.path / "optim_tracking_ehm.pkl")
    id_share = load_pickle(case.path / "id_share_params.pkl")
    frame_keys = load_frame_keys(case.path / "videos_info.json", tracking)

    pred_chunks: list[np.ndarray] = []
    gt_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(frame_keys), batch_size):
            keys = frame_keys[start : start + batch_size]
            flame_coeffs, smplx_coeffs = build_batch(tracking, id_share, keys, device)
            ret_body = model(smplx_coeffs, flame_coeffs, pose_type="aa")

            camera_rt = smplx_coeffs["camera_RT_params"]
            r_matrix, translation = camera_rt.split([3, 1], dim=-1)
            translation = translation.squeeze(-1)

            cameras = build_camera(len(keys), image_size, tanfov, device)
            proj_joints = cameras.transform_points_screen(
                ret_body["joints"], R=r_matrix, T=translation
            )
            pred = smplx_joints_to_dwpose(proj_joints)[0][..., :2]

            pred_chunks.append(pred.detach().cpu().numpy())
            gt_chunks.append(
                np.stack(
                    [
                        np.asarray(tracking[key]["dwpose_rlt"]["keypoints"], dtype=np.float64)[:, :2]
                        for key in keys
                    ]
                )
            )
            score_chunks.append(
                np.stack(
                    [
                        np.asarray(tracking[key]["dwpose_rlt"]["scores"], dtype=np.float64)
                        for key in keys
                    ]
                )
            )

    pred_2d = np.concatenate(pred_chunks, axis=0)
    gt_2d = np.concatenate(gt_chunks, axis=0)
    scores = np.concatenate(score_chunks, axis=0)

    row: dict[str, Any] = {
        "hand_group": case.hand_group,
        "sample_id": case.sample_id,
        "phase": case.phase,
        "frames": len(frame_keys),
        "path": str(case.path),
        "all_valid_keypoints": int(np.sum(scores >= score_threshold)),
        "hand_valid_keypoints": int(np.sum(scores[:, HANDS] >= score_threshold)),
        "left_hand_valid_keypoints": int(np.sum(scores[:, LEFT_HAND] >= score_threshold)),
        "right_hand_valid_keypoints": int(np.sum(scores[:, RIGHT_HAND] >= score_threshold)),
        "all_reproj_error_px": weighted_reprojection_error(
            pred_2d, gt_2d, scores, np.arange(scores.shape[1]), score_threshold
        ),
        "body_hand_reproj_error_px": weighted_reprojection_error(
            pred_2d, gt_2d, scores, BODY_NO_FACE, score_threshold
        ),
        "hand_reproj_error_px": weighted_reprojection_error(
            pred_2d, gt_2d, scores, HANDS, score_threshold
        ),
        "left_hand_reproj_error_px": weighted_reprojection_error(
            pred_2d, gt_2d, scores, LEFT_HAND, score_threshold
        ),
        "right_hand_reproj_error_px": weighted_reprojection_error(
            pred_2d, gt_2d, scores, RIGHT_HAND, score_threshold
        ),
    }

    for alpha in pck_thresholds:
        label = pck_label(alpha)
        row[f"all_pck_{label}"] = pck(
            pred_2d, gt_2d, scores, np.arange(scores.shape[1]), np.arange(scores.shape[1]), score_threshold, alpha, image_size
        )
        row[f"body_hand_pck_{label}"] = pck(
            pred_2d, gt_2d, scores, BODY_NO_FACE, BODY_NO_FACE, score_threshold, alpha, image_size
        )
        row[f"hand_pck_{label}"] = pck(
            pred_2d, gt_2d, scores, HANDS, HANDS, score_threshold, alpha, image_size
        )
        row[f"left_hand_pck_{label}"] = pck(
            pred_2d, gt_2d, scores, LEFT_HAND, LEFT_HAND, score_threshold, alpha, image_size
        )
        row[f"right_hand_pck_{label}"] = pck(
            pred_2d, gt_2d, scores, RIGHT_HAND, RIGHT_HAND, score_threshold, alpha, image_size
        )
    return row


def compare_case_rows(
    pairs: list[PairedSample],
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]],
    direction_func,
) -> list[dict[str, Any]]:
    ignored = {"hand_group", "sample_id", "phase", "frames", "path"}
    out: list[dict[str, Any]] = []
    for pair in pairs:
        before = rows_by_key.get((pair.hand_group, pair.sample_id, "before"))
        after = rows_by_key.get((pair.hand_group, pair.sample_id, "after"))
        if before is None or after is None:
            continue
        for metric in sorted(key for key in before if key not in ignored):
            before_value = as_float(before.get(metric))
            after_value = as_float(after.get(metric))
            delta = after_value - before_value if is_finite(before_value) and is_finite(after_value) else math.nan
            pct = delta / abs(before_value) * 100.0 if is_finite(delta) and abs(before_value) > 1e-12 else math.nan
            direction = direction_func(metric)
            improved: bool | str
            if not is_finite(delta) or direction == "context":
                improved = ""
            elif abs(delta) <= 1e-12:
                improved = "same"
            elif direction == "lower":
                improved = delta < 0
            else:
                improved = delta > 0

            out.append(
                {
                    "hand_group": pair.hand_group,
                    "sample_id": pair.sample_id,
                    "metric": metric,
                    "before": before_value,
                    "after": after_value,
                    "delta_after_minus_before": delta,
                    "percent_change": pct,
                    "preferred_direction": direction,
                    "improved": improved,
                }
            )
    return out


def preferred_direction(metric: str) -> str:
    if metric.endswith("spectral_arc_length"):
        return "higher"
    if any(token in metric for token in ["acceleration", "jerk", "normalized_jerk_cost"]):
        return "lower"
    return "context"


def reproj_preferred_direction(metric: str) -> str:
    if "pck" in metric:
        return "higher"
    if "reproj_error" in metric:
        return "lower"
    return "context"


def load_ehm_model(device: str) -> EHM:
    model = EHM(
        str(EHM_ROOT / "assets" / "FLAME"),
        str(EHM_ROOT / "assets" / "SMPLX"),
        str(EHM_ROOT / "assets" / "MANO"),
    ).to(device)
    model.eval()
    return model


def build_batch(
    tracking: dict[str, Any],
    id_share: dict[str, Any],
    keys: list[str],
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    flame_keys = tracking[keys[0]]["flame_coeffs"].keys()
    smplx_keys = tracking[keys[0]]["smplx_coeffs"].keys()
    flame = {
        key: stack_frame_value([tracking[frame_key]["flame_coeffs"][key] for frame_key in keys], device)
        for key in flame_keys
    }
    smplx = {
        key: stack_frame_value([tracking[frame_key]["smplx_coeffs"][key] for frame_key in keys], device)
        for key in smplx_keys
    }
    n = len(keys)
    smplx["shape"] = expand_shared(id_share["smplx_shape"], n, device)
    flame["shape_params"] = expand_shared(id_share["flame_shape"], n, device)
    smplx["joints_offset"] = expand_shared(id_share["joints_offset"], n, device)
    smplx["head_scale"] = expand_shared(id_share["head_scale"], n, device)
    smplx["hand_scale"] = expand_shared(id_share["hand_scale"], n, device)
    return flame, smplx


def stack_frame_value(values: list[Any], device: str) -> torch.Tensor:
    arr = np.stack([np.asarray(value) for value in values], axis=0)
    if arr.ndim >= 2 and arr.shape[1] == 1:
        arr = np.squeeze(arr, axis=1)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def expand_shared(value: Any, n: int, device: str) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    return tensor.expand(n, *tensor.shape[1:])


def build_camera(batch_size: int, image_size: int, tanfov: float, device: str) -> GS_Camera:
    screen_size = torch.full((batch_size, 2), float(image_size), device=device)
    return GS_Camera(
        principal_point=torch.zeros(batch_size, 2, device=device),
        focal_length=1.0 / tanfov,
        image_size=screen_size,
        device=device,
    ).to(device)


def weighted_reprojection_error(
    pred: np.ndarray,
    gt: np.ndarray,
    scores: np.ndarray,
    indices: np.ndarray,
    score_threshold: float,
) -> float:
    pred_i = pred[:, indices]
    gt_i = gt[:, indices]
    score_i = scores[:, indices]
    valid = np.isfinite(pred_i).all(axis=-1) & np.isfinite(gt_i).all(axis=-1) & (score_i >= score_threshold)
    if not np.any(valid):
        return math.nan
    error = np.linalg.norm(pred_i - gt_i, axis=-1)
    weights = np.where(valid, score_i, 0.0)
    denom = np.sum(weights)
    return float(np.sum(error * weights) / denom) if denom > 1e-8 else math.nan


def pck(
    pred: np.ndarray,
    gt: np.ndarray,
    scores: np.ndarray,
    eval_indices: np.ndarray,
    scale_indices: np.ndarray,
    score_threshold: float,
    alpha: float,
    fallback_scale: float,
) -> float:
    pred_i = pred[:, eval_indices]
    gt_i = gt[:, eval_indices]
    score_i = scores[:, eval_indices]
    valid = np.isfinite(pred_i).all(axis=-1) & np.isfinite(gt_i).all(axis=-1) & (score_i >= score_threshold)
    if not np.any(valid):
        return math.nan
    scales = bbox_scales(gt, scores, scale_indices, score_threshold, fallback_scale)
    error = np.linalg.norm(pred_i - gt_i, axis=-1)
    correct = error <= alpha * scales[:, None]
    return float(np.sum(correct & valid) / np.sum(valid))


def bbox_scales(
    gt: np.ndarray,
    scores: np.ndarray,
    indices: np.ndarray,
    score_threshold: float,
    fallback_scale: float,
) -> np.ndarray:
    scales = np.full(gt.shape[0], float(fallback_scale), dtype=np.float64)
    for frame_idx in range(gt.shape[0]):
        valid = scores[frame_idx, indices] >= score_threshold
        pts = gt[frame_idx, indices][valid]
        if len(pts) >= 2:
            wh = np.nanmax(pts, axis=0) - np.nanmin(pts, axis=0)
            scale = float(np.nanmax(wh))
            if np.isfinite(scale) and scale > 1.0:
                scales[frame_idx] = scale
    return scales


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


def write_report(
    path: Path,
    pairs: list[PairedSample],
    smooth_compare_rows: list[dict[str, Any]],
    reproj_compare_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Word Smoothing Evaluation",
        "",
        f"Samples: {len(pairs)} paired gloss videos ({sum(1 for p in pairs if p.hand_group == '1_hand')} one-hand, {sum(1 for p in pairs if p.hand_group == '2_hands')} two-hand).",
        "",
        "## Smoothness Summary",
        "",
    ]
    smooth_metrics = [
        "smplx_body_pose_normalized_jerk_cost",
        "smplx_body_pose_jerk_mean",
        "mano_both_hand_pose_normalized_jerk_cost",
        "mano_both_hand_pose_jerk_mean",
        "smplx_all_pose_jerk_mean",
    ]
    lines.extend(summary_table(smooth_compare_rows, smooth_metrics))
    lines.extend(["", "## Reprojection / PCK Summary", ""])
    reproj_metrics = [
        "body_hand_reproj_error_px",
        "hand_reproj_error_px",
        "body_hand_pck_0p05",
        "body_hand_pck_0p10",
        "hand_pck_0p05",
        "hand_pck_0p10",
    ]
    lines.extend(summary_table(reproj_compare_rows, reproj_metrics))
    lines.extend(
        [
            "",
            "Interpretation: smoothness metrics verify that the smoothing module reduces temporal variation in targeted motion parameters. Reprojection/PCK metrics are used as a fidelity check; the desired outcome is large smoothness improvement with no substantial degradation in 2D pose consistency.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def summary_table(rows: list[dict[str, Any]], metrics: list[str]) -> list[str]:
    if not rows:
        return ["_Not computed._"]
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_metric.setdefault(row["metric"], []).append(row)
    lines = [
        "| metric | before mean | after mean | mean change | mean % change | improved / total |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        group = by_metric.get(metric, [])
        if not group:
            continue
        before = finite_values(row["before"] for row in group)
        after = finite_values(row["after"] for row in group)
        delta = finite_values(row["delta_after_minus_before"] for row in group)
        pct = finite_values(row["percent_change"] for row in group)
        improved = sum(1 for row in group if row.get("improved") is True)
        total = sum(1 for row in group if row.get("improved") in {True, False, "same"})
        lines.append(
            f"| `{metric}` | {fmt(np.mean(before))} | {fmt(np.mean(after))} | {fmt(np.mean(delta))} | {fmt_percent(np.mean(pct))} | {improved}/{total} |"
        )
    return lines


def finite_values(values: Any) -> list[float]:
    out = []
    for value in values:
        value = as_float(value)
        if is_finite(value):
            out.append(value)
    return out


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
            return [key for key in frame_keys if key in tracking]
    return sorted(tracking.keys())


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


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def pck_label(alpha: float) -> str:
    return f"{alpha:.2f}".replace("0.", "0p")


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
        "hand_group",
        "sample_id",
        "phase",
        "metric",
        "before",
        "after",
        "delta_after_minus_before",
        "percent_change",
        "preferred_direction",
        "improved",
        "frames",
        "path",
    ]
    columns = preferred + [col for col in columns if col not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


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

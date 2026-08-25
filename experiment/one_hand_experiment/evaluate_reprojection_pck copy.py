#!/usr/bin/env python3
"""
Evaluate EHM-Tracker step variants with 2D reprojection error and PCK.

This script loads optimized EHM/SMPL-X tracking results, reconstructs 3D
joints, projects them back to the 1024 x 1024 body crop, converts the projected
SMPL-X joints to the DWPose keypoint order, and compares them against DWPose
2D keypoints stored in optim_tracking_ehm.pkl.

Important interpretation note:
  DWPose keypoints are used here as pseudo ground truth. These metrics measure
  agreement with the 2D detector, not agreement with manually annotated human
  labels. Low-confidence keypoints are excluded from PCK and weighted
  reprojection error.
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


DEFAULT_ROOT = Path(__file__).resolve().parent
GUAVA_ROOT = Path(__file__).resolve().parents[2]
EHM_ROOT = GUAVA_ROOT / "EHM-Tracker"

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
class TrackCase:
    gloss: str
    step_label: str
    path: Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute 2D reprojection error and PCK for step100/step1000 tracking results."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--tanfov", type=float, default=1.0 / 24.0)
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--pck-thresholds", default="0.05,0.10")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_ROOT / "tracking_step_reprojection_pck.csv",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_ROOT / "tracking_step_reprojection_pck_report.md",
    )
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU. This may be slow.")
        args.device = "cpu"

    pck_thresholds = parse_float_list(args.pck_thresholds)
    cases = discover_cases(args.root)
    if not cases:
        raise SystemExit(f"No tracking cases found under {args.root}")

    model = load_ehm_model(args.device)
    rows = [
        evaluate_case(case, model, args.device, args.batch_size, args.image_size, args.tanfov, args.score_threshold, pck_thresholds)
        for case in cases
    ]

    write_csv(args.out_csv, rows)
    write_markdown(args.out_md, rows, pck_thresholds)

    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote report: {args.out_md}")


def discover_cases(root: Path) -> list[TrackCase]:
    cases: list[TrackCase] = []

    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.endswith("_step100"):
            continue
        if has_tracking_payload(child):
            cases.append(TrackCase(gloss=child.name, step_label="step1000", path=child))

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
    return (
        (path / "optim_tracking_ehm.pkl").is_file()
        and (path / "id_share_params.pkl").is_file()
        and (path / "videos_info.json").is_file()
    )


def load_ehm_model(device: str) -> EHM:
    flame_assets = EHM_ROOT / "assets" / "FLAME"
    smplx_assets = EHM_ROOT / "assets" / "SMPLX"
    mano_assets = EHM_ROOT / "assets" / "MANO"

    model = EHM(str(flame_assets), str(smplx_assets), str(mano_assets)).to(device)
    model.eval()
    return model


def evaluate_case(
    case: TrackCase,
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
            R, T = camera_rt.split([3, 1], dim=-1)
            T = T.squeeze(-1)

            cameras = build_camera(len(keys), image_size, tanfov, device)
            proj_joints = cameras.transform_points_screen(ret_body["joints"], R=R, T=T)
            pred = smplx_joints_to_dwpose(proj_joints)[0][..., :2]

            pred_chunks.append(pred.detach().cpu().numpy())
            gt_chunks.append(np.stack([np.asarray(tracking[key]["dwpose_rlt"]["keypoints"], dtype=np.float64)[:, :2] for key in keys]))
            score_chunks.append(np.stack([np.asarray(tracking[key]["dwpose_rlt"]["scores"], dtype=np.float64) for key in keys]))

    pred_2d = np.concatenate(pred_chunks, axis=0)
    gt_2d = np.concatenate(gt_chunks, axis=0)
    scores = np.concatenate(score_chunks, axis=0)

    row: dict[str, Any] = {
        "gloss": case.gloss,
        "step": case.step_label,
        "frames": len(frame_keys),
        "path": str(case.path),
        "all_valid_keypoints": int(np.sum(scores >= score_threshold)),
        "hand_valid_keypoints": int(np.sum(scores[:, HANDS] >= score_threshold)),
        "left_hand_valid_keypoints": int(np.sum(scores[:, LEFT_HAND] >= score_threshold)),
        "right_hand_valid_keypoints": int(np.sum(scores[:, RIGHT_HAND] >= score_threshold)),
        "all_reproj_error_px": weighted_reprojection_error(pred_2d, gt_2d, scores, np.arange(scores.shape[1]), score_threshold),
        "body_hand_reproj_error_px": weighted_reprojection_error(pred_2d, gt_2d, scores, BODY_NO_FACE, score_threshold),
        "hand_reproj_error_px": weighted_reprojection_error(pred_2d, gt_2d, scores, HANDS, score_threshold),
        "left_hand_reproj_error_px": weighted_reprojection_error(pred_2d, gt_2d, scores, LEFT_HAND, score_threshold),
        "right_hand_reproj_error_px": weighted_reprojection_error(pred_2d, gt_2d, scores, RIGHT_HAND, score_threshold),
        "lowconf_left_hand_pred_in_crop_ratio": lowconf_pred_in_crop_ratio(pred_2d, scores, LEFT_HAND, score_threshold, image_size),
        "lowconf_right_hand_pred_in_crop_ratio": lowconf_pred_in_crop_ratio(pred_2d, scores, RIGHT_HAND, score_threshold, image_size),
    }

    for alpha in pck_thresholds:
        label = pck_label(alpha)
        row[f"all_pck_{label}"] = pck(pred_2d, gt_2d, scores, np.arange(scores.shape[1]), np.arange(scores.shape[1]), score_threshold, alpha, image_size)
        row[f"body_hand_pck_{label}"] = pck(pred_2d, gt_2d, scores, BODY_NO_FACE, BODY_NO_FACE, score_threshold, alpha, image_size)
        row[f"hand_pck_{label}"] = pck(pred_2d, gt_2d, scores, HANDS, HANDS, score_threshold, alpha, image_size)
        row[f"left_hand_pck_{label}"] = pck(pred_2d, gt_2d, scores, LEFT_HAND, LEFT_HAND, score_threshold, alpha, image_size)
        row[f"right_hand_pck_{label}"] = pck(pred_2d, gt_2d, scores, RIGHT_HAND, RIGHT_HAND, score_threshold, alpha, image_size)

    return row


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
    if denom <= 1e-8:
        return math.nan
    return float(np.sum(error * weights) / denom)


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


def lowconf_pred_in_crop_ratio(
    pred: np.ndarray,
    scores: np.ndarray,
    indices: np.ndarray,
    score_threshold: float,
    image_size: int,
) -> float:
    score_i = scores[:, indices]
    pred_i = pred[:, indices]
    low_conf = score_i < score_threshold
    if not np.any(low_conf):
        return math.nan

    in_crop = (
        (pred_i[..., 0] >= 0)
        & (pred_i[..., 0] < image_size)
        & (pred_i[..., 1] >= 0)
        & (pred_i[..., 1] < image_size)
    )
    return float(np.sum(in_crop & low_conf) / np.sum(low_conf))


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


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def pck_label(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row.keys()})
    preferred = ["gloss", "step", "frames", "path"]
    columns = preferred + [col for col in columns if col not in preferred]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], pck_thresholds: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_gloss: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_gloss.setdefault(row["gloss"], []).append(row)

    metric_cols = [
        "all_reproj_error_px",
        "body_hand_reproj_error_px",
        "hand_reproj_error_px",
        "left_hand_reproj_error_px",
        "right_hand_reproj_error_px",
    ]
    for alpha in pck_thresholds:
        label = pck_label(alpha)
        metric_cols.extend(
            [
                f"body_hand_pck_{label}",
                f"hand_pck_{label}",
                f"left_hand_pck_{label}",
                f"right_hand_pck_{label}",
            ]
        )
    metric_cols.extend(
        [
            "lowconf_left_hand_pred_in_crop_ratio",
            "lowconf_right_hand_pred_in_crop_ratio",
            "hand_valid_keypoints",
        ]
    )

    lines = [
        "# EHM-Tracker Reprojection and PCK Metrics",
        "",
        "DWPose 2D keypoints are used as pseudo ground truth.",
        "Reprojection error is measured in pixels on the 1024 x 1024 body crop; lower is better.",
        "PCK reports the ratio of valid keypoints whose projected distance is within alpha times the visible-keypoint bounding-box scale; higher is better.",
        "Low-confidence hand predicted-in-crop ratio is diagnostic only: high values may indicate the model keeps a low-confidence/offscreen hand inside the crop, but low DWPose confidence can also occur for in-frame occlusion.",
        "",
    ]

    for gloss, group in sorted(by_gloss.items()):
        lines.append(f"## {gloss}")
        lines.append("")
        lines.append("| metric | step100 | step1000 | step100 - step1000 | interpretation |")
        lines.append("|---|---:|---:|---:|---|")
        by_step = {row["step"]: row for row in group}
        for metric in metric_cols:
            v100 = by_step.get("step100", {}).get(metric, math.nan)
            v1000 = by_step.get("step1000", {}).get(metric, math.nan)
            delta = v100 - v1000 if is_number(v100) and is_number(v1000) else math.nan
            lines.append(
                f"| `{metric}` | {fmt(v100)} | {fmt(v1000)} | {fmt(delta)} | {interpretation(metric)} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def interpretation(metric: str) -> str:
    if "pck" in metric:
        return "higher is better"
    if "reproj_error" in metric:
        return "lower is better"
    if "valid_keypoints" in metric:
        return "more observed evidence"
    return "diagnostic"


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.floating)) and not math.isnan(float(value))


def fmt(value: Any) -> str:
    if not is_number(value):
        return "nan"
    return f"{float(value):.6g}"


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Apply the same word-segment smoothing used by app.py and render each sample.

Input layout:
  before_smooth/
    1_hand/<sample_id>/{optim_tracking_ehm.pkl,id_share_params.pkl,videos_info.json,img_lmdb}
    2_hands/<sample_id>/...

Output layout:
  after_smooth/
    1_hand/<sample_id>/...
    2_hands/<sample_id>/...
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


GUAVA_ROOT = Path("/home/paohan/GUAVA")
DEFAULT_BEFORE_ROOT = GUAVA_ROOT / "experiment" / "smothness_experiment" / "before_smooth"
DEFAULT_AFTER_ROOT = GUAVA_ROOT / "experiment" / "smothness_experiment" / "after_smooth"
DEFAULT_MODEL_PATH = GUAVA_ROOT / "assets" / "GUAVA"
DEFAULT_SOURCE_DATA_PATH = (
    GUAVA_ROOT
    / "outputs"
    / "app"
    / "tracked_source_image"
    / "Gemini_Generated_Image_kzne4skzne4skzne"
)


@dataclass(frozen=True)
class Sample:
    hand_group: str
    sample_id: str
    input_dir: Path
    output_dir: Path


def main() -> None:
    args = parse_args()
    samples = discover_samples(args.before_root, args.after_root)
    if args.only:
        wanted = set(args.only)
        samples = [sample for sample in samples if sample.sample_id in wanted]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit(f"No samples found under {args.before_root}")

    print(f"Found {len(samples)} samples.")
    failures: list[tuple[Sample, str]] = []

    for idx, sample in enumerate(samples, start=1):
        print(f"\n{'=' * 80}")
        print(f"[{idx}/{len(samples)}] {sample.hand_group}/{sample.sample_id}")
        print(f"Input : {sample.input_dir}")
        print(f"Output: {sample.output_dir}")

        if args.skip_existing and find_rendered_video(sample.output_dir):
            print("Rendered video already exists; skipping.")
            continue

        try:
            if not args.render_only:
                smooth_sample(sample, visualize=not args.no_visualize)
                write_manifest(sample, args)
            if not args.no_render:
                render_sample(sample, args)
        except Exception as exc:  # noqa: BLE001 - batch script should continue.
            failures.append((sample, str(exc)))
            print(f"FAILED {sample.hand_group}/{sample.sample_id}: {exc}")
            if not args.keep_going:
                break

    print(f"\n{'=' * 80}")
    print("Batch finished.")
    print(f"Succeeded/attempted: {len(samples) - len(failures)}/{len(samples)}")
    if failures:
        print("\nFailures:")
        for sample, reason in failures:
            print(f"  - {sample.hand_group}/{sample.sample_id}: {reason}")
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smooth before_smooth samples with app.py word smoothing and render them."
    )
    parser.add_argument("--before-root", type=Path, default=DEFAULT_BEFORE_ROOT)
    parser.add_argument("--after-root", type=Path, default=DEFAULT_AFTER_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--source-data-path", type=Path, default=DEFAULT_SOURCE_DATA_PATH)
    parser.add_argument("--devices", "-d", default="0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="Only process these sample ids.")
    parser.add_argument("--no-render", action="store_true", help="Only generate smoothed tracking data.")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render existing after_smooth data without regenerating smoothed tracking.",
    )
    parser.add_argument("--no-visualize", action="store_true", help="Skip peak analysis PNG output.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples with an existing rendered video.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a sample fails.")
    return parser.parse_args()


def discover_samples(before_root: Path, after_root: Path) -> list[Sample]:
    samples: list[Sample] = []
    for hand_group in ["1_hand", "2_hands"]:
        group_dir = before_root / hand_group
        if not group_dir.is_dir():
            continue
        for child in sorted(group_dir.iterdir()):
            if not child.is_dir() or not has_tracking_payload(child):
                continue
            samples.append(
                Sample(
                    hand_group=hand_group,
                    sample_id=child.name,
                    input_dir=child,
                    output_dir=after_root / hand_group / child.name,
                )
            )
    return samples


def has_tracking_payload(path: Path) -> bool:
    return (
        (path / "optim_tracking_ehm.pkl").is_file()
        and (path / "id_share_params.pkl").is_file()
        and (path / "videos_info.json").is_file()
        and (path / "img_lmdb").is_dir()
    )


def smooth_sample(sample: Sample, visualize: bool) -> None:
    sys.path.insert(0, str(GUAVA_ROOT))
    from tracking_concatenation_final_5 import concatenate_tracking_data_with_peaks

    sample.output_dir.mkdir(parents=True, exist_ok=True)
    concatenate_tracking_data_with_peaks(
        [str(sample.input_dir)],
        str(sample.output_dir),
        window_size=5,
        min_peak_distance=10,
        offset_frames=5,
        num_transition_frames=15,
        smooth_params=True,
        body_sigma=3,
        hand_sigma=3,
        face_sigma=3,
        global_smooth_sigma=0,
        use_median_for_hands=True,
        fix_hand_jitter=False,
        visualize=visualize,
    )


def write_manifest(sample: Sample, args: argparse.Namespace) -> None:
    render_cmd = build_render_command(sample, args)
    manifest = {
        "sample_id": sample.sample_id,
        "hand_group": sample.hand_group,
        "before_dir": str(sample.input_dir),
        "after_dir": str(sample.output_dir),
        "smooth_word_segments": True,
        "fix_hand_jitter": False,
        "smoothing_matches_app_checkbox": True,
        "smoothing_parameters": {
            "window_size": 5,
            "min_peak_distance": 10,
            "offset_frames": 5,
            "num_transition_frames": 15,
            "body_sigma": 3,
            "hand_sigma": 3,
            "face_sigma": 3,
            "global_smooth_sigma": 0,
            "use_median_for_hands": True,
        },
        "render_command": render_cmd,
    }
    with (sample.output_dir / "smooth_render_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def build_render_command(sample: Sample, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        str(GUAVA_ROOT / "main" / "test.py"),
        "-d",
        args.devices,
        "-m",
        str(args.model_path),
        "-s",
        str(sample.output_dir),
        "--data_path",
        str(sample.output_dir),
        "--source_data_path",
        str(args.source_data_path),
        "--skip_self_act",
        "--render_cross_act",
    ]


def render_sample(sample: Sample, args: argparse.Namespace) -> None:
    command = build_render_command(sample, args)
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(GUAVA_ROOT)
        if not existing_pythonpath
        else f"{GUAVA_ROOT}{os.pathsep}{existing_pythonpath}"
    )

    log_path = sample.output_dir / "render.log"
    print("\nRender command:")
    print(" ".join(str(part) for part in command))

    output_lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(GUAVA_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
                output_lines.append(line)
            process.stdout.close()

        return_code = process.wait()

    if return_code != 0:
        tail = "".join(output_lines[-80:])
        raise subprocess.CalledProcessError(return_code, command, output=tail)

    video = find_rendered_video(sample.output_dir)
    if not video:
        raise FileNotFoundError(f"Render completed but no mp4 was found under {sample.output_dir}")
    print(f"Rendered video: {video}")


def find_rendered_video(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    candidates = sorted(output_dir.rglob("*_video.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        candidates = sorted(output_dir.rglob("*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


if __name__ == "__main__":
    main()

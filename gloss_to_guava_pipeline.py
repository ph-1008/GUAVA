#!/usr/bin/env python3
"""
Run the GUAVA gloss-to-render pipeline from run_translate_to_guava.sh output.

The script reads Gloss joined: ... from gloss_output.txt, resolves each gloss
token through merged_labels_tracked.json, concatenates the tracked motions in
that order, and optionally launches main/test.py for cross rendering.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import unicodedata
from datetime import datetime
from pathlib import Path


GUAVA_ROOT = Path("/home/paohan/GUAVA")
DEFAULT_GLOSS_OUTPUT = GUAVA_ROOT / "sign_translate_code" / "gloss_output.txt"
DEFAULT_LABELS_JSON = (
    GUAVA_ROOT
    / "EHM-Tracker"
    / "Sign_dataset"
    / "tw_sign_dataset"
    / "merged_labels_tracked.json"
)
DEFAULT_OUTPUT_ROOT = GUAVA_ROOT / "outputs_2"
DEFAULT_MODEL_PATH = GUAVA_ROOT / "assets" / "GUAVA"
DEFAULT_SOURCE_DATA_PATH = (
    GUAVA_ROOT
    / "outputs"
    / "app"
    / "tracked_source_image"
    / "Gemini_Generated_Image_kzne4skzne4skzne"
)
FIX_HAND_JITTER = False

PUNCTUATION_TOKENS = {
    "",
    ".",
    ",",
    "!",
    "?",
    ";",
    ":",
    "。",
    "，",
    "、",
    "！",
    "？",
    "；",
    "：",
}

# Gloss normalizations that appear in the current translation output.
TOKEN_ALIASES = {
    "臺北": ["台北"],
    "臺灣": ["台灣"],
    "一個小時": ["一小時"],
}

COMPOUND_TOKEN_EXPANSIONS = {
    "二個小時": ["二", "小時"],
    "二小時": ["二", "小時"],
    "兩個小時": ["兩個", "小時"],
    "兩小時": ["兩個", "小時"],
}

SEARCH_ROOTS = [
    GUAVA_ROOT / "EHM-Tracker" / "Sign_dataset" / "tw_sign_dataset",
    GUAVA_ROOT / "EHM-Tracker" / "results" / "dad_drive_1hr_step700",
    GUAVA_ROOT / "EHM-Tracker" / "results" / "TWTSL_tracked",
    GUAVA_ROOT / "EHM-Tracker" / "results",
    GUAVA_ROOT / "outputs" / "TWTSL_tracked",
    GUAVA_ROOT / "outputs" / "app" / "tracked_driven_video",
]


def parse_gloss_tokens(gloss_output: Path) -> list[str]:
    text = gloss_output.read_text(encoding="utf-8")

    joined_matches = re.findall(r"^Gloss joined:\s*(.+?)\s*$", text, re.MULTILINE)
    if joined_matches:
        return split_gloss_joined(joined_matches[-1])

    generated_matches = re.findall(r"^Generated gloss:\s*(.+?)\s*$", text, re.MULTILINE)
    if generated_matches:
        try:
            parsed = ast.literal_eval(generated_matches[-1])
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"Cannot parse Generated gloss line: {generated_matches[-1]}") from exc
        if not isinstance(parsed, list):
            raise ValueError("Generated gloss is not a list")
        return [str(item).strip() for item in parsed]

    raise ValueError(f"No 'Gloss joined:' line found in {gloss_output}")


def split_gloss_joined(gloss_joined: str) -> list[str]:
    return [part.strip() for part in gloss_joined.split("/") if part.strip()]


def load_labels(labels_json: Path) -> dict[str, str]:
    with labels_json.open("r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, dict):
        raise ValueError(f"{labels_json} must contain a JSON object")
    return {str(k).strip(): str(v).strip() for k, v in labels.items()}


def normalize_label_path(raw_path: str) -> Path:
    rel_path = raw_path.replace("\\", "/")
    path = Path(rel_path)
    if path.is_absolute():
        return path
    return GUAVA_ROOT / path


def has_tracking_payload(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "optim_tracking_ehm.pkl").is_file()
        and (path / "id_share_params.pkl").is_file()
        and (path / "videos_info.json").is_file()
        and (path / "img_lmdb").is_dir()
    )


def lookup_token_candidates(token: str, labels: dict[str, str]) -> list[tuple[str, Path, str]]:
    candidates: list[tuple[str, Path, str]] = []
    seen: set[Path] = set()

    lookup_keys = [token]
    lookup_keys.extend(TOKEN_ALIASES.get(token, []))
    nfkc = unicodedata.normalize("NFKC", token)
    if nfkc not in lookup_keys:
        lookup_keys.append(nfkc)
    simplified_variant = nfkc.replace("臺", "台")
    if simplified_variant not in lookup_keys:
        lookup_keys.append(simplified_variant)

    for key in lookup_keys:
        raw_path = labels.get(key)
        if not raw_path:
            continue
        direct_path = normalize_label_path(raw_path)
        for path in [direct_path, *fallback_paths_for(direct_path.name)]:
            if path in seen:
                continue
            seen.add(path)
            candidates.append((key, path, raw_path))

    return candidates


def fallback_paths_for(dirname: str) -> list[Path]:
    paths = []
    for root in SEARCH_ROOTS:
        paths.append(root / dirname)
    return paths


def resolve_tokens(tokens: list[str], labels: dict[str, str]) -> tuple[list[str], list[dict[str, str]]]:
    tracked_dirs: list[str] = []
    records: list[dict[str, str]] = []
    missing: list[str] = []

    for token in tokens:
        resolved_dirs, resolved_records = resolve_one_token(token, labels)
        if not resolved_dirs and not any(record["status"].startswith("skipped") for record in resolved_records):
            missing.append(token)
            continue
        tracked_dirs.extend(resolved_dirs)
        records.extend(resolved_records)

    if missing:
        detail = "\n".join(f"  - {token}" for token in missing)
        raise FileNotFoundError(
            "Some gloss tokens could not be resolved to tracked GUAVA data:\n"
            f"{detail}\n"
            "Check merged_labels_tracked.json or add a tracked directory under one of SEARCH_ROOTS."
        )

    if not tracked_dirs:
        raise ValueError("No gloss token resolved to a tracked directory")

    return tracked_dirs, records


def resolve_one_token(token: str, labels: dict[str, str]) -> tuple[list[str], list[dict[str, str]]]:
    skip_reason = get_skip_reason(token)
    if skip_reason:
        return [], [{"token": token, "status": skip_reason}]

    candidates = lookup_token_candidates(token, labels)
    for matched_key, path, raw_path in candidates:
        if has_tracking_payload(path):
            return [
                str(path)
            ], [
                {
                    "token": token,
                    "status": "resolved",
                    "matched_key": matched_key,
                    "json_path": raw_path,
                    "tracked_dir": str(path),
                }
            ]

    expansion = COMPOUND_TOKEN_EXPANSIONS.get(token)
    if expansion:
        tracked_dirs: list[str] = []
        records: list[dict[str, str]] = [
            {"token": token, "status": f"expanded to {'/'.join(expansion)}"}
        ]
        for child_token in expansion:
            child_dirs, child_records = resolve_one_token(child_token, labels)
            if not child_dirs:
                candidate_paths = [str(path) for _, path, _ in candidates[:8]]
                return [], [
                    {
                        "token": token,
                        "status": "missing",
                        "candidate_paths": "\n".join(candidate_paths),
                    }
                ]
            tracked_dirs.extend(child_dirs)
            records.extend(child_records)
        return tracked_dirs, records

    candidate_paths = [str(path) for _, path, _ in candidates[:8]]
    return [], [
        {
            "token": token,
            "status": "missing",
            "candidate_paths": "\n".join(candidate_paths),
        }
    ]


def get_skip_reason(token: str) -> str | None:
    stripped = token.strip()
    if stripped in PUNCTUATION_TOKENS:
        return "skipped punctuation"

    if stripped and all(is_punctuation_or_symbol(char) for char in stripped):
        return "skipped symbol/emoticon"

    return None


def is_punctuation_or_symbol(char: str) -> bool:
    return unicodedata.category(char)[0] in {"P", "S"}


def make_output_dir(output_root: Path, tokens: list[str], name: str | None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    if name:
        base = safe_folder_name(name)
    else:
        content_tokens = [token for token in tokens if not get_skip_reason(token)]
        base = safe_folder_name("_".join(content_tokens[:10]))
        if not base:
            base = "gloss"
        base = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{base}"

    output_dir = output_root / base
    if not output_dir.exists():
        return output_dir

    for idx in range(2, 1000):
        candidate = output_root / f"{base}_{idx}"
        if not candidate.exists():
            return candidate

    raise FileExistsError(f"Cannot create a unique output directory for {output_dir}")


def safe_folder_name(value: str) -> str:
    value = value.strip().replace("/", "_").replace("\\", "_")
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value, flags=re.UNICODE)
    value = re.sub(r"_+", "_", value).strip("._")
    return value[:120]


def write_manifest(
    output_dir: Path,
    tokens: list[str],
    records: list[dict[str, str]],
    render_cmd: list[str],
    smooth_word_segments: bool,
    fix_hand_jitter: bool,
) -> None:
    manifest = {
        "gloss_tokens": tokens,
        "resolved_records": records,
        "render_command": render_cmd,
        "smooth_word_segments": smooth_word_segments,
        "fix_hand_jitter": fix_hand_jitter,
    }
    with (output_dir / "gloss_pipeline_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def run_concatenation(
    tracked_dirs: list[str],
    output_dir: Path,
    visualize: bool,
    smooth_word_segments: bool,
) -> None:
    sys.path.insert(0, str(GUAVA_ROOT))
    from tracking_concatenation_final_5 import concatenate_tracking_data_with_peaks

    if smooth_word_segments:
        smooth_params = True
        body_sigma = 3
        hand_sigma = 3
        face_sigma = 3
        global_smooth_sigma = 0
        use_median_for_hands = True
    else:
        smooth_params = False
        body_sigma = 0
        hand_sigma = 0
        face_sigma = 0
        global_smooth_sigma = 0
        use_median_for_hands = False

    concatenate_tracking_data_with_peaks(
        tracked_dirs,
        str(output_dir),
        window_size=5,
        min_peak_distance=10,
        offset_frames=5,
        num_transition_frames=15,
        smooth_params=smooth_params,
        body_sigma=body_sigma,
        hand_sigma=hand_sigma,
        face_sigma=face_sigma,
        global_smooth_sigma=global_smooth_sigma,
        use_median_for_hands=use_median_for_hands,
        fix_hand_jitter=FIX_HAND_JITTER,
        visualize=visualize,
    )


def build_render_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(GUAVA_ROOT / "main" / "test.py"),
        "-d",
        args.devices,
        "-m",
        str(args.model_path),
        "-s",
        str(output_dir),
        "--data_path",
        str(output_dir),
        "--source_data_path",
        str(args.source_data_path),
        "--skip_self_act",
        "--render_cross_act",
    ]


def run_render(command: list[str]) -> None:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(GUAVA_ROOT)
        if not existing_pythonpath
        else f"{GUAVA_ROOT}{os.pathsep}{existing_pythonpath}"
    )

    save_path = None
    if "-s" in command:
        save_index = command.index("-s") + 1
        if save_index < len(command):
            save_path = Path(command[save_index])
    log_path = (save_path or GUAVA_ROOT) / "render.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

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
        print(f"\nRender failed. Full log: {log_path}")
        print("\nLast render log lines:")
        print(tail)
        raise subprocess.CalledProcessError(return_code, command, output=tail)


def print_resolution(records: list[dict[str, str]]) -> None:
    print("\nGloss resolution:")
    for record in records:
        token = record["token"]
        status = record["status"]
        if status == "resolved":
            print(f"  {token} -> {record['tracked_dir']} (matched: {record['matched_key']})")
        else:
            print(f"  {token} -> {status}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve gloss_output.txt tokens, concatenate GUAVA tracking data, and render."
    )
    parser.add_argument("--gloss-output", type=Path, default=DEFAULT_GLOSS_OUTPUT)
    parser.add_argument("--labels-json", type=Path, default=DEFAULT_LABELS_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--source-data-path", type=Path, default=DEFAULT_SOURCE_DATA_PATH)
    parser.add_argument("--devices", "-d", type=str, default="0")
    parser.add_argument("--dry-run", action="store_true", help="Resolve gloss tokens and print paths only.")
    parser.add_argument("--no-render", action="store_true", help="Only create the concatenated outputs_2 data.")
    parser.add_argument("--no-visualize", action="store_true", help="Skip peak analysis PNG files.")
    parser.add_argument(
        "--smooth-word-segments",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply light smoothing to each resolved word segment before rendering. "
            "Default is false, which keeps word motion parameters closest to the original EHM-Tracker output."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokens = parse_gloss_tokens(args.gloss_output)
    labels = load_labels(args.labels_json)
    tracked_dirs, records = resolve_tokens(tokens, labels)
    output_dir = make_output_dir(args.output_root, tokens, args.output_name)
    render_cmd = build_render_command(args, output_dir)

    print(f"Gloss tokens: {'/'.join(tokens)}")
    print_resolution(records)
    print(f"\nConcatenation output: {output_dir}")

    if args.dry_run:
        print("\n--dry-run enabled; no files were generated.")
        return

    print(f"Smooth word segments: {args.smooth_word_segments}")
    print(f"Fix hand jitter: {FIX_HAND_JITTER}")

    run_concatenation(
        tracked_dirs,
        output_dir,
        visualize=not args.no_visualize,
        smooth_word_segments=args.smooth_word_segments,
    )
    write_manifest(
        output_dir,
        tokens,
        records,
        render_cmd,
        args.smooth_word_segments,
        FIX_HAND_JITTER,
    )

    print("\nRender command:")
    print(" ".join(str(part) for part in render_cmd))

    if args.no_render:
        print("\n--no-render enabled; finished after concatenation.")
        return

    print("\nStarting final render...")
    run_render(render_cmd)


if __name__ == "__main__":
    main()

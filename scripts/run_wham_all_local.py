#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def discover_videos(input_root: Path) -> list[Path]:
    """Find all supported video files recursively under the input root."""
    return sorted(
        path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def run_command(cmd: list[str], cwd: Path) -> None:
    """Run one WHAM demo command with writable temp config dirs."""
    print("$", " ".join(shlex.quote(part) for part in cmd))
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-wham")
    env.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics-wham")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def output_dir_for_video(output_root: Path, input_root: Path, video_path: Path) -> Path:
    """Mirror the B2S_Data folder structure in the output directory."""
    relative_parent = video_path.parent.relative_to(input_root)
    return output_root / relative_parent


def already_done(output_dir: Path, video_path: Path) -> bool:
    """Skip re-running WHAM if the expected pickle output is already there."""
    sequence = video_path.stem
    return (output_dir / sequence / "wham_output.pkl").exists()


def run_wham_local(wham_root: Path, video_path: Path, output_dir: Path, run_smplify: bool) -> None:
    """Run WHAM in local-only mode on one full video."""
    cmd = [
        sys.executable,
        "demo.py",
        "--video",
        str(video_path),
        "--output_pth",
        str(output_dir),
        "--estimate_local_only",
        "--save_pkl",
    ]
    if run_smplify:
        cmd.append("--run_smplify")
    run_command(cmd, cwd=wham_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WHAM local-only pose estimation on all local videos.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/home/lala/Documents/GitHub/B2S/B2S_Data"),
        help="Root directory containing videos.",
    )
    parser.add_argument(
        "--wham-root",
        type=Path,
        default=Path("/home/lala/Documents/GitHub/B2S/WHAM"),
        help="Path to the local WHAM repository.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/lala/Documents/GitHub/B2S/B2S_Data/wham_all_local_outputs"),
        help="Root directory where WHAM results will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on how many videos to process.",
    )
    parser.add_argument(
        "--run-smplify",
        action="store_true",
        help="Run Temporal SMPLify refinement after WHAM inference.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run videos even if a wham_output.pkl already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    videos = discover_videos(args.input_root)
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"No supported videos found under {args.input_root}")

    print(f"Found {len(videos)} videos under {args.input_root}")
    for video_path in videos:
        per_video_output_root = output_dir_for_video(args.output_root, args.input_root, video_path)
        per_video_output_root.mkdir(parents=True, exist_ok=True)
        if not args.force and already_done(per_video_output_root, video_path):
            print(f"Skipping {video_path} because output already exists.")
            continue
        print(f"Running WHAM local-only on {video_path}")
        run_wham_local(args.wham_root, video_path, per_video_output_root, args.run_smplify)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

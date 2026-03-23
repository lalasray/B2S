#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import os
import shutil
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}


def discover_videos(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def run_command(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(shlex.quote(part) for part in cmd))
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-wham")
    env.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics-wham")
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def resolve_ffmpeg(ffmpeg_bin: str | None) -> str:
    if ffmpeg_bin:
        return ffmpeg_bin

    # Prefer the system ffmpeg over a possibly broken conda-forge binary.
    if Path("/usr/bin/ffmpeg").exists():
        return "/usr/bin/ffmpeg"

    resolved = shutil.which("ffmpeg")
    if resolved:
        return resolved

    raise FileNotFoundError("Could not find a usable ffmpeg binary")


def split_video(video_path: Path, chunk_dir: Path, window_seconds: int, ffmpeg_bin: str) -> list[Path]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_pattern = chunk_dir / f"{video_path.stem}_chunk_%03d{video_path.suffix.lower()}"
    run_command(
        [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-f",
            "segment",
            "-segment_time",
            str(window_seconds),
            "-reset_timestamps",
            "1",
            str(chunk_pattern),
        ],
        cwd=video_path.parent,
    )
    return sorted(chunk_dir.glob(f"{video_path.stem}_chunk_*{video_path.suffix.lower()}"))


def run_wham_on_chunk(wham_root: Path, chunk_path: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "demo.py",
            "--video",
            str(chunk_path),
            "--output_pth",
            str(output_root),
            "--estimate_local_only",
            "--save_pkl",
        ],
        cwd=wham_root,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split videos into fixed windows and run WHAM local-only inference on each chunk."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/lala/Documents/GitHub/B2S/B2S_Data/Stairs"),
        help="Directory containing source videos.",
    )
    parser.add_argument(
        "--wham-root",
        type=Path,
        default=Path("/home/lala/Documents/GitHub/B2S/WHAM"),
        help="Path to the WHAM repository.",
    )
    parser.add_argument(
        "--chunk-root",
        type=Path,
        default=Path("/home/lala/Documents/GitHub/B2S/B2S_Data/Stairs/wham_chunks_5s"),
        help="Directory where 5-second chunks will be written.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/lala/Documents/GitHub/B2S/B2S_Data/Stairs/wham_local_outputs"),
        help="Directory where WHAM outputs will be written.",
    )
    parser.add_argument(
        "--window-seconds",
        type=int,
        default=5,
        help="Chunk duration in seconds.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default=None,
        help="Optional path to a specific ffmpeg binary.",
    )
    parser.add_argument(
        "--split-only",
        action="store_true",
        help="Only split videos into chunks without running WHAM.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of source videos to process.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ffmpeg_bin = resolve_ffmpeg(args.ffmpeg_bin)
    videos = discover_videos(args.input_dir)
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        raise SystemExit(f"No supported videos found in {args.input_dir}")

    print(f"Found {len(videos)} videos in {args.input_dir}")
    for video_path in videos:
        print(f"Splitting {video_path.name} into {args.window_seconds}s chunks...")
        per_video_chunk_dir = args.chunk_root / video_path.stem
        chunks = split_video(video_path, per_video_chunk_dir, args.window_seconds, ffmpeg_bin)
        print(f"Created {len(chunks)} chunks for {video_path.name}")

        if args.split_only:
            continue

        for chunk_path in chunks:
            print(f"Running WHAM local-only on {chunk_path.name}...")
            run_wham_on_chunk(args.wham_root, chunk_path, args.output_root / video_path.stem)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

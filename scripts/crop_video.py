#!/usr/bin/env python3
"""Interactively crop a local video and save the result beside the source file."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    """Allow both interactive use and fully scripted CLI use."""
    parser = argparse.ArgumentParser(
        description="Crop a video using ffmpeg and save it beside the original file."
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Path to the input video. If omitted, the script will prompt for it.",
    )
    parser.add_argument(
        "--start",
        help="Crop start time, for example 12.5, 00:28, or 00:00:12.500.",
    )
    parser.add_argument(
        "--end",
        help="Crop end time, for example 30:40, 20.0, or 00:30:40.000.",
    )
    parser.add_argument(
        "--suffix",
        default="_cropped",
        help="Suffix to append to the output filename before the extension.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def prompt(message: str, default: str | None = None) -> str:
    """Read input from the terminal with an optional default value."""
    if default:
        response = input(f"{message} [{default}]: ").strip()
        return response or default
    return input(f"{message}: ").strip()


def normalize_time(value: str) -> str:
    """Accept seconds, mm:ss, or hh:mm:ss input and return ffmpeg-friendly text."""
    value = value.strip()
    if not value:
        raise ValueError("Time value cannot be empty.")

    if ":" in value:
        parts = value.split(":")
        if len(parts) > 3 or any(not part for part in parts):
            raise ValueError(f"Invalid time format: {value}")
        try:
            numeric_parts = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"Invalid time format: {value}") from exc

        total_seconds = 0.0
        for part in numeric_parts:
            total_seconds = total_seconds * 60 + part
    else:
        try:
            total_seconds = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric time value: {value}") from exc

    if total_seconds < 0:
        raise ValueError("Time value must be non-negative.")
    return f"{total_seconds:.3f}"


def time_to_seconds(value: str) -> float:
    """Convert the normalized time string back into seconds for validation."""
    return float(value)


def choose_video(provided_path: Path | None) -> Path:
    """Resolve and validate the video path."""
    raw_path = provided_path or Path(prompt("Enter the video path"))
    video_path = raw_path.expanduser().resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {video_path}")
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(
            f"Unsupported video extension '{video_path.suffix}'. "
            f"Expected one of: {', '.join(sorted(VIDEO_EXTENSIONS))}"
        )
    return video_path


def build_output_path(video_path: Path, suffix: str) -> Path:
    """Keep the cropped result beside the source video."""
    return video_path.with_name(f"{video_path.stem}{suffix}{video_path.suffix}")


def ensure_ffmpeg() -> None:
    """Fail fast with a clear message if ffmpeg is missing."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. Install ffmpeg before using this script."
        )


def run_crop(
    video_path: Path,
    start_time: str,
    end_time: str,
    output_path: Path,
    overwrite: bool,
) -> None:
    """Run ffmpeg with stream copy so the crop stays fast and local."""
    command = [
        "ffmpeg",
        "-hide_banner",
        "-y" if overwrite else "-n",
        "-ss",
        start_time,
        "-to",
        end_time,
        "-i",
        str(video_path),
        "-c",
        "copy",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def main() -> int:
    """Gather inputs, validate them, and crop the selected video."""
    args = parse_args()

    try:
        ensure_ffmpeg()
        video_path = choose_video(args.video)

        start_raw = args.start or prompt("Enter crop start time", "0")
        end_raw = args.end or prompt("Enter crop end time")
        start_time = normalize_time(start_raw)
        end_time = normalize_time(end_raw)

        if time_to_seconds(end_time) <= time_to_seconds(start_time):
            raise ValueError("End time must be greater than start time.")

        output_path = build_output_path(video_path, args.suffix)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}\n"
                "Use --overwrite or change the suffix."
            )

        print(f"Input:  {video_path}")
        print(f"Start:  {start_time}s")
        print(f"End:    {end_time}s")
        print(f"Output: {output_path}")

        run_crop(
            video_path=video_path,
            start_time=start_time,
            end_time=end_time,
            output_path=output_path,
            overwrite=args.overwrite,
        )
        print("Crop complete.")
        return 0
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        return 130
    except Exception as exc:  # pragma: no cover - simple CLI error handling
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

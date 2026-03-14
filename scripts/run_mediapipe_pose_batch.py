#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe import tasks


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
POSE_LANDMARKS = list(tasks.vision.PoseLandmark)


@dataclass
class VideoResult:
    video_path: Path
    csv_path: Path
    summary_path: Path
    total_frames: int
    detected_frames: int
    fps: float


def build_header() -> list[str]:
    header = ["frame_idx", "timestamp_sec"]
    for landmark in POSE_LANDMARKS:
        name = landmark.name.lower()
        header.extend(
            [
                f"image_{name}_x",
                f"image_{name}_y",
                f"image_{name}_z",
                f"image_{name}_visibility",
                f"image_{name}_presence",
                f"world_{name}_x",
                f"world_{name}_y",
                f"world_{name}_z",
                f"world_{name}_visibility",
                f"world_{name}_presence",
            ]
        )
    return header


def discover_videos(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def write_summary(result: VideoResult) -> None:
    payload = {
        "video_path": str(result.video_path),
        "csv_path": str(result.csv_path),
        "total_frames": result.total_frames,
        "detected_frames": result.detected_frames,
        "detection_rate": (
            result.detected_frames / result.total_frames if result.total_frames else 0.0
        ),
        "fps": result.fps,
    }
    result.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_pose_landmarker(model_path: Path) -> tasks.vision.PoseLandmarker:
    options = tasks.vision.PoseLandmarkerOptions(
        base_options=tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return tasks.vision.PoseLandmarker.create_from_options(options)


def process_video(
    video_path: Path,
    output_dir: Path,
    pose_landmarker: tasks.vision.PoseLandmarker,
) -> VideoResult:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    csv_path = output_dir / f"{video_path.stem}_mediapipe_pose3d.csv"
    summary_path = output_dir / f"{video_path.stem}_mediapipe_pose3d_summary.json"
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    total_frames = 0
    detected_frames = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(build_header())

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp_ms = int(round((total_frames / fps) * 1000)) if fps else total_frames
            timestamp_sec = timestamp_ms / 1000.0
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            image_landmarks = result.pose_landmarks[0] if result.pose_landmarks else None
            world_landmarks = (
                result.pose_world_landmarks[0] if result.pose_world_landmarks else None
            )

            row: list[float | int | str] = [total_frames, f"{timestamp_sec:.6f}"]

            if image_landmarks:
                detected_frames += 1

            for index, _ in enumerate(POSE_LANDMARKS):
                if image_landmarks:
                    image_point = image_landmarks[index]
                    row.extend(
                        [
                            image_point.x,
                            image_point.y,
                            image_point.z,
                            getattr(image_point, "visibility", ""),
                            getattr(image_point, "presence", ""),
                        ]
                    )
                else:
                    row.extend(["", "", "", "", ""])

                if world_landmarks:
                    world_point = world_landmarks[index]
                    row.extend(
                        [
                            world_point.x,
                            world_point.y,
                            world_point.z,
                            getattr(world_point, "visibility", ""),
                            getattr(world_point, "presence", ""),
                        ]
                    )
                else:
                    row.extend(["", "", "", "", ""])

            writer.writerow(row)
            total_frames += 1

    capture.release()

    result = VideoResult(
        video_path=video_path,
        csv_path=csv_path,
        summary_path=summary_path,
        total_frames=total_frames,
        detected_frames=detected_frames,
        fps=fps,
    )
    write_summary(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MediaPipe 3D pose estimation on every video in a directory."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/lala/Documents/GitHub/B2S/B2S_Data/Stairs"),
        help="Directory containing stair videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/lala/Documents/GitHub/B2S/B2S_Data/Stairs/mediapipe_pose_3d"),
        help="Directory to write CSV and summary outputs.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/lala/Documents/GitHub/B2S/models/pose_landmarker_heavy.task"),
        help="Path to the MediaPipe pose landmarker task file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(args.input_dir)
    if not videos:
        raise SystemExit(f"No supported videos found in {args.input_dir}")
    if not args.model_path.exists():
        raise SystemExit(f"Model file not found: {args.model_path}")

    pose_landmarker = build_pose_landmarker(args.model_path)
    results: list[VideoResult] = []

    try:
        for video_path in videos:
            print(f"Processing {video_path.name}...")
            results.append(process_video(video_path, args.output_dir, pose_landmarker))
    finally:
        pose_landmarker.close()

    for result in results:
        print(
            f"{result.video_path.name}: {result.detected_frames}/{result.total_frames} "
            f"frames with pose -> {result.csv_path.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

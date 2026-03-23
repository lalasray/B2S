#!/usr/bin/env python3
"""Visual video crop tool with folder browsing, preview, and start/end selection."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
DEFAULT_PREVIEW_WIDTH = 720
DEFAULT_PREVIEW_HEIGHT = 420


def parse_args() -> argparse.Namespace:
    """Allow a starting folder to be passed from the command line."""
    parser = argparse.ArgumentParser(
        description="Browse local videos, visually mark start/end, and crop beside the source file."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Initial directory to scan for videos.",
    )
    parser.add_argument(
        "--suffix",
        default="_cropped",
        help="Suffix to add to the cropped filename.",
    )
    return parser.parse_args()


def ensure_dependencies() -> None:
    """Surface missing local tools immediately instead of failing mid-crop."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not on PATH.")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe is not installed or not on PATH.")


def discover_videos(root: Path) -> list[Path]:
    """Recursively find supported videos under the selected root folder."""
    if not root.exists():
        return []
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def ffprobe_duration(video_path: Path) -> float:
    """Read the video duration in seconds for validation and timeline scaling."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def ffprobe_rotation(video_path: Path) -> int:
    """Read stream rotation metadata so previewed phone videos appear upright."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream_tags=rotate:stream_side_data=rotation",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams", [])
    if not streams:
        return 0

    stream = streams[0]
    rotation = 0

    tags = stream.get("tags", {})
    if "rotate" in tags:
        try:
            rotation = int(float(tags["rotate"]))
        except (TypeError, ValueError):
            rotation = 0

    for item in stream.get("side_data_list", []):
        if "rotation" in item:
            try:
                rotation = int(float(item["rotation"]))
            except (TypeError, ValueError):
                pass

    # Many phone videos store a display transform that should be applied to the
    # raw decoded frame. OpenCV usually ignores it, so we apply the inverse of
    # the stored transform for preview.
    return (-rotation) % 360


def format_seconds(value: float) -> str:
    """Show times in an easy-to-read hh:mm:ss.mmm format."""
    value = max(0.0, float(value))
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    seconds = value % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def build_output_path(video_path: Path, suffix: str) -> Path:
    """Save cropped clips beside the source video."""
    return video_path.with_name(f"{video_path.stem}{suffix}{video_path.suffix}")


class VideoCropApp:
    """Tiny desktop app for browsing videos and selecting crop bounds visually."""

    def __init__(self, root_window: tk.Tk, initial_root: Path, suffix: str) -> None:
        self.root_window = root_window
        self.root_window.title("B2S Video Crop Tool")
        self.root_window.geometry("1180x760")

        self.current_root = initial_root.expanduser().resolve()
        self.suffix = suffix

        self.video_paths: list[Path] = []
        self.current_video_path: Path | None = None
        self.capture: cv2.VideoCapture | None = None
        self.duration_seconds = 0.0
        self.frame_count = 0
        self.fps = 0.0
        self.current_frame_index = 0
        self.current_preview_image: ImageTk.PhotoImage | None = None
        self.rotation_degrees = 0
        self.manual_rotation_degrees = 0
        self.vertical_flip_enabled = True
        self.last_frame = None
        self.is_playing = False
        self.playback_job: str | None = None
        self.is_dragging_timeline = False
        self.ignore_scale_callback = False

        self.start_seconds: float | None = None
        self.end_seconds: float | None = None

        self._build_ui()
        self._bind_shortcuts()
        self._load_video_list()

    def _build_ui(self) -> None:
        """Create a left-side browser and right-side preview/cropping controls."""
        self.root_window.columnconfigure(1, weight=1)
        self.root_window.rowconfigure(0, weight=1)

        left_panel = ttk.Frame(self.root_window, padding=12)
        left_panel.grid(row=0, column=0, sticky="nsw")
        left_panel.rowconfigure(2, weight=1)

        ttk.Label(left_panel, text="Video Root").grid(row=0, column=0, sticky="w")
        root_row = ttk.Frame(left_panel)
        root_row.grid(row=1, column=0, sticky="ew", pady=(4, 10))
        root_row.columnconfigure(0, weight=1)

        self.root_var = tk.StringVar(value=str(self.current_root))
        ttk.Entry(root_row, textvariable=self.root_var, width=40).grid(row=0, column=0, sticky="ew")
        ttk.Button(root_row, text="Browse", command=self._choose_root).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(left_panel, text="Refresh Videos", command=self._load_video_list).grid(
            row=2, column=0, sticky="ew", pady=(0, 8)
        )

        self.video_listbox = tk.Listbox(left_panel, width=42, height=32)
        self.video_listbox.grid(row=3, column=0, sticky="nsew")
        self.video_listbox.bind("<<ListboxSelect>>", self._on_video_selected)

        right_panel = ttk.Frame(self.root_window, padding=12)
        right_panel.grid(row=0, column=1, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(1, weight=1)

        ttk.Label(right_panel, text="Preview").grid(row=0, column=0, sticky="w")
        self.preview_label = ttk.Label(right_panel, anchor="center", relief="solid")
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(6, 10))
        self.preview_label.bind("<Configure>", self._on_preview_resized)

        self.position_scale = ttk.Scale(
            right_panel,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            command=self._on_scrub,
        )
        self.position_scale.grid(row=2, column=0, sticky="ew")
        self.position_scale.bind("<ButtonPress-1>", self._on_timeline_press)
        self.position_scale.bind("<ButtonRelease-1>", self._on_timeline_release)

        info_frame = ttk.Frame(right_panel)
        info_frame.grid(row=3, column=0, sticky="ew", pady=(10, 8))
        info_frame.columnconfigure(1, weight=1)

        self.video_var = tk.StringVar(value="No video selected")
        self.time_var = tk.StringVar(value="Current: 00:00:00.000")
        self.range_var = tk.StringVar(value="Start: not set | End: not set")
        self.output_var = tk.StringVar(value="Output: -")

        ttk.Label(info_frame, textvariable=self.video_var).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(info_frame, textvariable=self.time_var).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Label(info_frame, textvariable=self.range_var).grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Label(info_frame, textvariable=self.output_var).grid(row=3, column=0, columnspan=2, sticky="w")

        button_row = ttk.Frame(right_panel)
        button_row.grid(row=4, column=0, sticky="ew", pady=(4, 8))

        self.play_button = ttk.Button(button_row, text="Play", command=self._toggle_playback)
        self.play_button.grid(row=0, column=0, padx=(0, 8))
        ttk.Button(button_row, text="Mark Start", command=self._mark_start).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(button_row, text="Mark End", command=self._mark_end).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(button_row, text="Clear Marks", command=self._clear_marks).grid(row=0, column=3, padx=(0, 8))
        ttk.Button(button_row, text="Crop Video", command=self._crop_video).grid(row=0, column=4)

        hint_text = (
            "Tips: select a video on the left, drag the timeline to preview frames, "
            "then mark start and end before cropping. Shortcuts: S=start, E=end, "
            "C=clear, Left/Right=step, Shift+Left/Right=jump, R=rotate, P=play/pause."
        )
        ttk.Label(right_panel, text=hint_text, wraplength=760).grid(row=5, column=0, sticky="w")

    def _bind_shortcuts(self) -> None:
        """Add keyboard shortcuts for the most common trimming actions."""
        self.root_window.bind("<KeyPress-s>", lambda _event: self._mark_start())
        self.root_window.bind("<KeyPress-S>", lambda _event: self._mark_start())
        self.root_window.bind("<KeyPress-e>", lambda _event: self._mark_end())
        self.root_window.bind("<KeyPress-E>", lambda _event: self._mark_end())
        self.root_window.bind("<KeyPress-c>", lambda _event: self._clear_marks())
        self.root_window.bind("<KeyPress-C>", lambda _event: self._clear_marks())
        self.root_window.bind("<Left>", lambda _event: self._step_frames(-1))
        self.root_window.bind("<Right>", lambda _event: self._step_frames(1))
        self.root_window.bind("<Shift-Left>", lambda _event: self._step_frames(-10))
        self.root_window.bind("<Shift-Right>", lambda _event: self._step_frames(10))
        self.root_window.bind("<KeyPress-r>", lambda _event: self._cycle_rotation())
        self.root_window.bind("<KeyPress-R>", lambda _event: self._cycle_rotation())
        self.root_window.bind("<KeyPress-p>", lambda _event: self._toggle_playback())
        self.root_window.bind("<KeyPress-P>", lambda _event: self._toggle_playback())
        self.root_window.bind("<space>", lambda _event: self._toggle_playback())

    def _choose_root(self) -> None:
        """Let the user visually choose the folder that contains videos."""
        selected = filedialog.askdirectory(initialdir=str(self.current_root))
        if not selected:
            return
        self.current_root = Path(selected).expanduser().resolve()
        self.root_var.set(str(self.current_root))
        self._load_video_list()

    def _load_video_list(self) -> None:
        """Refresh the listbox contents from the currently selected root folder."""
        self.current_root = Path(self.root_var.get()).expanduser().resolve()
        self.video_paths = discover_videos(self.current_root)

        self.video_listbox.delete(0, tk.END)
        for path in self.video_paths:
            self.video_listbox.insert(tk.END, str(path.relative_to(self.current_root)))

        if not self.video_paths:
            self.video_var.set("No videos found in selected folder")
            self.preview_label.configure(image="", text="No preview available")
            self.output_var.set("Output: -")

    def _on_video_selected(self, _event: object) -> None:
        """Open the chosen video and display its first frame."""
        selection = self.video_listbox.curselection()
        if not selection:
            return

        self._release_capture()
        self.current_video_path = self.video_paths[selection[0]]
        self.capture = cv2.VideoCapture(str(self.current_video_path))
        if not self.capture.isOpened():
            messagebox.showerror("Video Error", f"Could not open {self.current_video_path}")
            self.capture = None
            return

        self.fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration_seconds = ffprobe_duration(self.current_video_path)
        self.rotation_degrees = ffprobe_rotation(self.current_video_path)
        self.manual_rotation_degrees = 0
        self._stop_playback()
        self.current_frame_index = 0
        self.start_seconds = None
        self.end_seconds = None

        self.position_scale.configure(from_=0, to=max(self.frame_count - 1, 1))
        self._set_timeline_position(0)
        self.video_var.set(f"Selected: {self.current_video_path}")
        self.output_var.set(f"Output: {build_output_path(self.current_video_path, self.suffix)}")
        self._update_range_text()
        self._show_frame(0)

    def _on_preview_resized(self, _event: object) -> None:
        """Redraw the last frame when the preview panel size changes."""
        if self.last_frame is not None:
            self._render_preview(self.last_frame)

    def _on_scrub(self, value: str) -> None:
        """Render the frame nearest to the slider position."""
        if self.capture is None:
            return
        if self.ignore_scale_callback:
            return
        self._show_frame(int(float(value)))

    def _on_timeline_press(self, _event: object) -> None:
        """Remember when the user starts dragging so playback does not fight the slider."""
        self.is_dragging_timeline = True

    def _on_timeline_release(self, _event: object) -> None:
        """Jump playback to the released position and continue smoothly if playing."""
        self.is_dragging_timeline = False
        if self.capture is None:
            return
        self._show_frame(int(float(self.position_scale.get())))

    def _step_frames(self, delta: int) -> None:
        """Move the preview cursor by a small frame offset."""
        if self.capture is None:
            return
        self._stop_playback()
        target_frame = self.current_frame_index + delta
        self._show_frame(target_frame)

    def _show_frame(self, frame_index: int) -> None:
        """Seek to one frame, convert it for Tk, and update the preview panel."""
        if self.capture is None:
            return

        frame_index = max(0, min(frame_index, max(self.frame_count - 1, 0)))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok:
            return

        self.current_frame_index = frame_index
        self._set_timeline_position(frame_index)
        frame = self._apply_preview_rotation(frame)
        self.last_frame = frame
        self._render_preview(frame)

        current_seconds = self._current_seconds()
        self.time_var.set(
            f"Current: {format_seconds(current_seconds)} / {format_seconds(self.duration_seconds)}"
        )

    def _current_seconds(self) -> float:
        """Map the currently previewed frame back to seconds."""
        if self.fps > 0:
            return self.current_frame_index / self.fps
        if self.frame_count > 1 and self.duration_seconds > 0:
            return self.duration_seconds * (self.current_frame_index / (self.frame_count - 1))
        return 0.0

    def _apply_preview_rotation(self, frame):
        """Rotate preview frames based on container metadata that OpenCV ignores."""
        rotation = (self.rotation_degrees + self.manual_rotation_degrees) % 360
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if self.vertical_flip_enabled:
            frame = cv2.flip(frame, 0)
        return frame

    def _render_preview(self, frame) -> None:
        """Scale the current frame to match the live preview area."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)

        target_width = max(self.preview_label.winfo_width(), DEFAULT_PREVIEW_WIDTH)
        target_height = max(self.preview_label.winfo_height(), DEFAULT_PREVIEW_HEIGHT)
        image.thumbnail((target_width, target_height))

        self.current_preview_image = ImageTk.PhotoImage(image=image)
        self.preview_label.configure(image=self.current_preview_image, text="")

    def _cycle_rotation(self) -> None:
        """Let the user manually rotate the preview when file metadata is inconsistent."""
        if self.last_frame is None:
            return
        self._stop_playback()
        self.manual_rotation_degrees = (self.manual_rotation_degrees + 90) % 360
        self._show_frame(self.current_frame_index)

    def _toggle_playback(self) -> None:
        """Start or pause preview playback."""
        if self.capture is None:
            return
        if self.is_playing:
            self._stop_playback()
            return
        self.is_playing = True
        self.play_button.configure(text="Pause")
        self._play_next_frame()

    def _play_next_frame(self) -> None:
        """Advance playback using the video FPS as the timer."""
        if not self.is_playing or self.capture is None:
            return

        if self.is_dragging_timeline:
            delay_ms = max(int(1000 / max(self.fps, 1.0)), 15)
            self.playback_job = self.root_window.after(delay_ms, self._play_next_frame)
            return

        next_frame = self.current_frame_index + 1
        if next_frame >= max(self.frame_count, 1):
            self._stop_playback()
            return

        self._show_frame(next_frame)

        delay_ms = max(int(1000 / max(self.fps, 1.0)), 15)
        self.playback_job = self.root_window.after(delay_ms, self._play_next_frame)

    def _stop_playback(self) -> None:
        """Cancel any scheduled playback callbacks and reset the button text."""
        self.is_playing = False
        self.play_button.configure(text="Play")
        if self.playback_job is not None:
            self.root_window.after_cancel(self.playback_job)
            self.playback_job = None

    def _set_timeline_position(self, frame_index: int) -> None:
        """Update the slider without triggering a second seek through the callback."""
        self.ignore_scale_callback = True
        try:
            self.position_scale.set(frame_index)
        finally:
            self.ignore_scale_callback = False

    def _mark_start(self) -> None:
        """Store the current preview time as the crop start."""
        if self.current_video_path is None:
            return
        self.start_seconds = self._current_seconds()
        self._update_range_text()

    def _mark_end(self) -> None:
        """Store the current preview time as the crop end."""
        if self.current_video_path is None:
            return
        self.end_seconds = self._current_seconds()
        self._update_range_text()

    def _clear_marks(self) -> None:
        """Reset the selected crop bounds."""
        self.start_seconds = None
        self.end_seconds = None
        self._update_range_text()

    def _update_range_text(self) -> None:
        """Keep the current crop range visible in the UI."""
        start_text = format_seconds(self.start_seconds) if self.start_seconds is not None else "not set"
        end_text = format_seconds(self.end_seconds) if self.end_seconds is not None else "not set"
        self.range_var.set(f"Start: {start_text} | End: {end_text}")

    def _crop_video(self) -> None:
        """Validate the chosen interval and write the cropped clip beside the source."""
        if self.current_video_path is None:
            messagebox.showwarning("No Video", "Select a video first.")
            return
        if self.start_seconds is None or self.end_seconds is None:
            messagebox.showwarning("Missing Marks", "Mark both a start and an end time first.")
            return
        if self.end_seconds <= self.start_seconds:
            messagebox.showerror("Invalid Range", "End time must be greater than start time.")
            return

        output_path = build_output_path(self.current_video_path, self.suffix)
        if output_path.exists():
            overwrite = messagebox.askyesno(
                "Overwrite Existing File",
                f"{output_path.name} already exists beside the source video.\nOverwrite it?",
            )
            if not overwrite:
                return
        else:
            overwrite = False

        self._stop_playback()

        filters = []
        rotation = (self.rotation_degrees + self.manual_rotation_degrees) % 360
        if rotation == 90:
            filters.append("transpose=1")
        elif rotation == 180:
            filters.extend(["transpose=1", "transpose=1"])
        elif rotation == 270:
            filters.append("transpose=2")
        if self.vertical_flip_enabled:
            filters.append("vflip")

        command = [
            "ffmpeg",
            "-hide_banner",
            "-y" if overwrite else "-n",
            "-ss",
            f"{self.start_seconds:.3f}",
            "-to",
            f"{self.end_seconds:.3f}",
            "-i",
            str(self.current_video_path),
        ]

        if filters:
            command.extend(
                [
                    "-vf",
                    ",".join(filters),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                ]
            )
        else:
            command.extend(["-c", "copy"])

        command.append(str(output_path))

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as exc:
            messagebox.showerror("Crop Failed", f"ffmpeg could not crop the video.\n\n{exc}")
            return

        messagebox.showinfo("Crop Complete", f"Saved cropped video to:\n{output_path}")

    def _release_capture(self) -> None:
        """Release the previously open OpenCV handle before loading another video."""
        self._stop_playback()
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def close(self) -> None:
        """Clean up resources when the window is closed."""
        self._release_capture()
        self.root_window.destroy()


def main() -> int:
    """Start the Tk app after checking the local prerequisites."""
    args = parse_args()
    try:
        ensure_dependencies()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    root_window = tk.Tk()
    app = VideoCropApp(root_window=root_window, initial_root=args.root, suffix=args.suffix)
    root_window.protocol("WM_DELETE_WINDOW", app.close)
    root_window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

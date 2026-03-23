from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F


@dataclass
class SensorStream:
    """One time-stamped sensor stream such as watch IMU or phone IMU."""

    timestamps: torch.Tensor
    values: torch.Tensor
    name: str


@dataclass
class PreprocessedWindow:
    """Normalized model-ready sensor window plus metadata."""

    timestamps: torch.Tensor
    values: torch.Tensor
    gravity: torch.Tensor
    calibration: Dict[str, torch.Tensor]


def synchronize_streams(streams: List[SensorStream], target_rate_hz: float = 50.0) -> Dict[str, SensorStream]:
    """Resample all streams onto one shared timeline using linear interpolation."""
    t_min = max(float(stream.timestamps[0].item()) for stream in streams)
    t_max = min(float(stream.timestamps[-1].item()) for stream in streams)
    num_steps = max(int((t_max - t_min) * target_rate_hz), 2)
    shared_t = torch.linspace(t_min, t_max, steps=num_steps)
    synced = {}
    for stream in streams:
        values = []
        for dim in range(stream.values.shape[-1]):
            values.append(torch.from_numpy(__import__("numpy").interp(shared_t.numpy(), stream.timestamps.numpy(), stream.values[:, dim].numpy())))
        synced[stream.name] = SensorStream(shared_t, torch.stack(values, dim=-1).float(), stream.name)
    return synced


def smooth_signal(values: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """Apply a simple moving-average denoiser."""
    if kernel_size <= 1:
        return values
    pad = kernel_size // 2
    x = values.T.unsqueeze(0)
    x = F.pad(x, (pad, pad), mode="replicate")
    kernel = torch.ones(values.shape[-1], 1, kernel_size, dtype=values.dtype, device=values.device) / kernel_size
    smoothed = F.conv1d(x, kernel, groups=values.shape[-1])
    return smoothed.squeeze(0).T


def calibrate_imu(values: torch.Tensor, stationary_frames: int = 20) -> torch.Tensor:
    """Remove static bias using the first few frames as a calibration segment."""
    bias = values[:stationary_frames].mean(dim=0, keepdim=True)
    return values - bias


def estimate_gravity_vector(accel: torch.Tensor, stationary_frames: int = 20) -> torch.Tensor:
    """Estimate gravity from early frames where the device is assumed mostly still."""
    gravity = accel[:stationary_frames].mean(dim=0)
    return gravity / gravity.norm().clamp(min=1e-6)


def align_to_gravity(values: torch.Tensor, gravity: torch.Tensor) -> torch.Tensor:
    """Rotate accelerometer/gyro data so gravity points along +Z."""
    target = torch.tensor([0.0, 0.0, 1.0], dtype=values.dtype, device=values.device)
    v = torch.cross(gravity, target)
    c = torch.dot(gravity, target).clamp(-1.0, 1.0)
    s = v.norm()
    if s < 1e-6:
        rotation = torch.eye(3, dtype=values.dtype, device=values.device)
    else:
        vx = torch.tensor(
            [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
            dtype=values.dtype,
            device=values.device,
        )
        rotation = torch.eye(3, dtype=values.dtype, device=values.device) + vx + vx @ vx * ((1 - c) / (s**2))
    accel = values[:, :3] @ rotation.T
    gyro = values[:, 3:6] @ rotation.T if values.shape[-1] >= 6 else values[:, 3:]
    return torch.cat([accel, gyro], dim=-1)


def normalize_device_frame(values: torch.Tensor) -> torch.Tensor:
    """Standardize each feature dimension to zero mean and unit variance."""
    mean = values.mean(dim=0, keepdim=True)
    std = values.std(dim=0, keepdim=True).clamp(min=1e-6)
    return (values - mean) / std


def preprocess_stream(stream: SensorStream) -> PreprocessedWindow:
    """Full per-stream preprocessing: calibration, denoising, gravity alignment, normalization."""
    calibrated = calibrate_imu(stream.values)
    smoothed = smooth_signal(calibrated)
    gravity = estimate_gravity_vector(smoothed[:, :3])
    aligned = align_to_gravity(smoothed, gravity)
    normalized = normalize_device_frame(aligned)
    return PreprocessedWindow(
        timestamps=stream.timestamps,
        values=normalized,
        gravity=gravity,
        calibration={"bias": stream.values[:20].mean(dim=0)},
    )


def window_sensor_data(values: torch.Tensor, seq_len: int, stride: int | None = None) -> torch.Tensor:
    """Segment a long time series into fixed-length windows."""
    stride = seq_len if stride is None else stride
    windows = []
    for start in range(0, max(values.shape[0] - seq_len + 1, 1), stride):
        end = start + seq_len
        if end <= values.shape[0]:
            windows.append(values[start:end])
    return torch.stack(windows, dim=0) if windows else values[:seq_len].unsqueeze(0)

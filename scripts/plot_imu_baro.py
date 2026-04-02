#!/usr/bin/env python3
from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_CSV = (
    Path(__file__).resolve().parents[1]
    / "B2S_Data"
    / "Indoor"
    / "Daniel_2026-02-24_10_54_02_iPhoneRed.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the first minute of IMU and barometer data."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default=str(DEFAULT_CSV),
        help=f"Path to the input CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--minutes",
        type=float,
        default=1.0,
        help="Duration from the start of the recording to plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()

    cols = [
        "loggingTime(txt)",
        "accelerometerAccelerationX(G)",
        "accelerometerAccelerationY(G)",
        "accelerometerAccelerationZ(G)",
        "gyroRotationX(rad/s)",
        "gyroRotationY(rad/s)",
        "gyroRotationZ(rad/s)",
        "magnetometerX(µT)",
        "magnetometerY(µT)",
        "magnetometerZ(µT)",
        "altimeterRelativeAltitude(m)",
        "altimeterPressure(kPa)",
    ]

    df = pd.read_csv(csv_path, usecols=cols)
    df["loggingTime(txt)"] = pd.to_datetime(df["loggingTime(txt)"])

    start = df["loggingTime(txt)"].iloc[0]
    end = start + pd.Timedelta(minutes=args.minutes)
    df = df[df["loggingTime(txt)"] < end].copy()
    df["elapsed_s"] = (df["loggingTime(txt)"] - start).dt.total_seconds()

    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )

    fig, axes = plt.subplots(5, 1, figsize=(18, 8.6), sharex=True)
    fig.subplots_adjust(left=0.055, right=0.94, top=0.97, bottom=0.085, hspace=0.24)

    imu_groups = [
        (
            "Accelerometer (G)",
            [
                "accelerometerAccelerationX(G)",
                "accelerometerAccelerationY(G)",
                "accelerometerAccelerationZ(G)",
            ],
            ["Acc X", "Acc Y", "Acc Z"],
        ),
        (
            "Gyroscope (rad/s)",
            ["gyroRotationX(rad/s)", "gyroRotationY(rad/s)", "gyroRotationZ(rad/s)"],
            ["Gyro X", "Gyro Y", "Gyro Z"],
        ),
        (
            "Magnetometer (µT)",
            ["magnetometerX(µT)", "magnetometerY(µT)", "magnetometerZ(µT)"],
            ["Mag X", "Mag Y", "Mag Z"],
        ),
    ]
    colors = ["tab:red", "tab:green", "tab:blue"]

    for ax, (title, series, labels) in zip(axes[:3], imu_groups):
        for col, label, color in zip(series, labels, colors):
            ax.plot(df["elapsed_s"], df[col], label=label, linewidth=0.75, color=color)
        ax.set_title(title, pad=4)
        ax.grid(True, alpha=0.25)
        ax.legend(
            loc="upper right",
            ncol=3,
            fontsize=12,
            frameon=True,
            borderpad=0.3,
            handlelength=1.8,
        )
        ax.margins(x=0.01)

    altitude_ax = axes[3]
    altitude_ax.plot(
        df["elapsed_s"],
        df["altimeterRelativeAltitude(m)"],
        label="Relative Altitude (m)",
        color="mediumpurple",
        linewidth=0.95,
    )
    altitude_ax.set_title("Relative Altitude (m)", pad=4)
    altitude_ax.set_ylabel("Altitude", color="mediumpurple")
    altitude_ax.tick_params(axis="y", labelcolor="mediumpurple")
    altitude_ax.grid(True, alpha=0.25)
    altitude_ax.legend(loc="upper right", fontsize=12, frameon=True)
    altitude_ax.margins(x=0.01)

    pressure_ax = axes[4]
    pressure_ax.plot(
        df["elapsed_s"],
        df["altimeterPressure(kPa)"],
        label="Pressure (kPa)",
        color="darkorange",
        linewidth=0.95,
    )
    pressure_ax.set_title("Pressure (kPa)", pad=4)
    pressure_ax.set_ylabel("Pressure", color="darkorange")
    pressure_ax.tick_params(axis="y", labelcolor="darkorange")
    pressure_ax.grid(True, alpha=0.25)
    pressure_ax.legend(loc="upper right", fontsize=12, frameon=True)
    pressure_ax.margins(x=0.01)
    pressure_ax.set_xlabel("Elapsed Time (s)")
    plt.show()


if __name__ == "__main__":
    main()

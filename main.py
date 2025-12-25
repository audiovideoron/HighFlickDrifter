#!/usr/bin/env python3
"""
Flicker Detection Tool

Detects brightness anomalies (flicker) in video recordings by:
1. Extracting frames at a low sample rate (2 fps)
2. Computing mean brightness for each frame
3. Flagging frames with abnormal brightness changes using rolling z-scores
4. Grouping nearby events into reviewable segments

Usage:
    python main.py <video_file>
    python main.py test_footage/test_flicker.mp4
"""

import argparse
import re
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np


# Configuration
FPS_SAMPLE = 2          # Sample rate for initial scan
Z_THRESHOLD = 3.0       # Z-score threshold for anomaly detection (lower = more sensitive)
WINDOW_SIZE = 30        # Rolling window size in samples (~15 seconds at 2 fps)
GROUP_GAP_SEC = 15      # Group events within this many seconds


def extract_frame_timestamps(video_path: Path, fps: int) -> list[float]:
    """Extract PTS timestamps from video using ffmpeg showinfo filter."""

    cmd = [
        "ffmpeg", "-hide_banner",
        "-i", str(video_path),
        "-vf", f"fps={fps},showinfo",
        "-f", "null", "-"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    lines = result.stderr.splitlines()

    pts_times = []
    for line in lines:
        match = re.search(r"pts_time:([0-9]+\.?[0-9]*)", line)
        if match:
            pts_times.append(float(match.group(1)))

    return pts_times


def extract_frames(video_path: Path, output_dir: Path, fps: int) -> int:
    """Extract frames from video at specified fps."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean any existing frames
    for f in output_dir.glob("frame_*.jpg"):
        f.unlink()

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-vsync", "vfr",
        str(output_dir / "frame_%06d.jpg")
    ]

    subprocess.run(cmd, check=True)

    return len(list(output_dir.glob("frame_*.jpg")))


def analyze_frames(frame_dir: Path, pts_times: list[float]) -> tuple[list[float], list[float], list[float]]:
    """Analyze extracted frames for brightness and banding metrics."""

    frames = sorted(frame_dir.glob("frame_*.jpg"))
    n = min(len(frames), len(pts_times))

    # Warn if frame/timestamp counts don't match (tolerance: 1 frame)
    if abs(len(frames) - len(pts_times)) > 1:
        print(f"Warning: Frame/timestamp count mismatch - {len(frames)} frames vs {len(pts_times)} timestamps")

    brightness = []
    banding = []
    timestamps = []

    for i in range(n):
        img = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Mean brightness (0-255)
        mean_b = float(np.mean(img))

        # Banding metric: variance of row means (detects horizontal bands)
        row_means = img.mean(axis=1)
        band = float(np.var(row_means))

        brightness.append(mean_b)
        banding.append(band)
        timestamps.append(pts_times[i])

    return brightness, banding, timestamps


def detect_anomalies(
    brightness: list[float],
    banding: list[float],
    timestamps: list[float],
    window_size: int = WINDOW_SIZE,
    z_threshold: float = Z_THRESHOLD
) -> list[dict]:
    """Detect anomalies using rolling z-score analysis."""

    n = len(brightness)
    b_window = deque(maxlen=window_size)
    g_window = deque(maxlen=window_size)

    events = []

    for i in range(n):
        b = brightness[i]
        g = banding[i]

        # Need at least half the window filled before detecting
        if len(b_window) >= window_size // 2:
            b_mean = np.mean(b_window)
            b_std = np.std(b_window) + 1e-6  # Avoid division by zero
            g_mean = np.mean(g_window)
            g_std = np.std(g_window) + 1e-6

            z_brightness = (b - b_mean) / b_std
            z_banding = (g - g_mean) / g_std

            # Flag if brightness or banding exceeds threshold
            if abs(z_brightness) > z_threshold or z_banding > z_threshold * 2:
                events.append({
                    "t_sec": timestamps[i],
                    "z_brightness": float(z_brightness),
                    "z_banding": float(z_banding),
                    "brightness": b,
                    "banding": g
                })

        b_window.append(b)
        g_window.append(g)

    return events


def group_events(events: list[dict], gap_sec: float = GROUP_GAP_SEC) -> list[list[dict]]:
    """Group nearby events into segments."""

    if not events:
        return []

    events = sorted(events, key=lambda x: x["t_sec"])
    groups = []
    current_group = []

    for event in events:
        if not current_group or (event["t_sec"] - current_group[-1]["t_sec"]) <= gap_sec:
            current_group.append(event)
        else:
            groups.append(current_group)
            current_group = [event]

    if current_group:
        groups.append(current_group)

    return groups


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.s"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes}:{secs:05.2f}"


def print_report(
    groups: list[list[dict]],
    total_events: int,
    num_frames: int,
    duration_sec: float,
    fps: int
):
    """Print detection report to console."""

    print()
    print("=" * 60)
    print("FLICKER DETECTION REPORT")
    print("=" * 60)
    print()
    print(f"Scanned {num_frames} frames at {fps} fps")
    print(f"Video duration: {format_timestamp(duration_sec)} ({duration_sec:.1f} seconds)")
    print(f"Found {total_events} anomaly points in {len(groups)} grouped segments")
    print()

    if not groups:
        print("No flicker events detected.")
        return

    print("-" * 60)
    for i, group in enumerate(groups, 1):
        t_start = group[0]["t_sec"]
        t_end = group[-1]["t_sec"]
        duration = t_end - t_start

        worst_brightness = max(group, key=lambda x: abs(x["z_brightness"]))

        print(f"Segment {i}: {format_timestamp(t_start)} to {format_timestamp(t_end)}")
        print(f"  Duration: ~{duration:.1f}s  |  Events: {len(group)}")
        print(f"  Worst brightness z-score: {worst_brightness['z_brightness']:+.1f} at {format_timestamp(worst_brightness['t_sec'])}")
        print()

    print("-" * 60)
    print()
    print("Review these timestamps in VLC or your preferred video player.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Detect flicker anomalies in video recordings"
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to video file"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=FPS_SAMPLE,
        help=f"Sample rate in fps (default: {FPS_SAMPLE})"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=Z_THRESHOLD,
        help=f"Z-score threshold for anomaly detection (default: {Z_THRESHOLD})"
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep extracted frames after analysis"
    )

    args = parser.parse_args()

    video_path = args.video.resolve()

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Temporary directory for extracted frames
    frame_dir = Path(__file__).parent / "frames_tmp"

    try:
        print(f"Analyzing: {video_path.name}")
        print(f"Sample rate: {args.fps} fps")
        print(f"Z-score threshold: {args.threshold}")
        print()

        # Step 1: Get frame timestamps
        print("Extracting frame timestamps...")
        pts_times = extract_frame_timestamps(video_path, args.fps)

        if not pts_times:
            print("Error: Could not extract timestamps from video")
            sys.exit(1)

        # Step 2: Extract frames
        print("Extracting frames...")
        num_frames = extract_frames(video_path, frame_dir, args.fps)
        print(f"Extracted {num_frames} frames")

        # Step 3: Analyze frames
        print("Analyzing brightness...")
        brightness, banding, timestamps = analyze_frames(frame_dir, pts_times)

        if not brightness:
            print("Error: No frames could be analyzed")
            sys.exit(1)

        # Step 4: Detect anomalies
        print("Detecting anomalies...")
        events = detect_anomalies(
            brightness, banding, timestamps,
            window_size=WINDOW_SIZE,
            z_threshold=args.threshold
        )

        # Step 5: Group events
        groups = group_events(events)

        # Step 6: Report
        duration_sec = timestamps[-1] if timestamps else 0
        print_report(groups, len(events), len(brightness), duration_sec, args.fps)

    finally:
        # Clean up temporary frames
        if not args.keep_frames and frame_dir.exists():
            shutil.rmtree(frame_dir)


if __name__ == "__main__":
    main()

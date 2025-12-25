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
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import requests


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

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True)
    except subprocess.TimeoutExpired:
        print("Error: ffmpeg timestamp extraction timed out after 300 seconds")
        print("This may indicate a very large video file or ffmpeg hang")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("Error: ffmpeg timestamp extraction failed")
        print(f"ffmpeg returned non-zero exit code: {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        sys.exit(1)

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

    try:
        subprocess.run(cmd, check=True, timeout=600)
    except subprocess.TimeoutExpired:
        print("Error: ffmpeg frame extraction timed out after 600 seconds")
        print("This may indicate a very large video file or ffmpeg hang")
        sys.exit(1)

    return len(list(output_dir.glob("frame_*.jpg")))


def analyze_frames(frame_dir: Path, pts_times: list[float]) -> tuple[list[float], list[float], list[float], int]:
    """Analyze extracted frames for brightness and banding metrics.

    Returns:
        Tuple of (brightness, banding, timestamps, corrupt_count)
    """

    frames = sorted(frame_dir.glob("frame_*.jpg"))
    n = min(len(frames), len(pts_times))

    # Warn if frame/timestamp counts don't match (tolerance: 1 frame)
    if abs(len(frames) - len(pts_times)) > 1:
        print(f"Warning: Frame/timestamp count mismatch - {len(frames)} frames vs {len(pts_times)} timestamps")

    brightness = []
    banding = []
    timestamps = []
    corrupt_count = 0

    for i in range(n):
        img = cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            corrupt_count += 1
            continue

        # Mean brightness (0-255)
        mean_b = float(np.mean(img))

        # Banding metric: variance of row means (detects horizontal bands)
        row_means = img.mean(axis=1)
        band = float(np.var(row_means))

        brightness.append(mean_b)
        banding.append(band)
        timestamps.append(pts_times[i])

    return brightness, banding, timestamps, corrupt_count


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


VLC_HTTP_PORT = 8080
VLC_HTTP_PASSWORD = "flicker"


def find_vlc() -> str:
    """Find VLC executable path."""
    # Check PATH first
    vlc_path = shutil.which("vlc")
    if vlc_path:
        return vlc_path
    # macOS app bundle
    macos_vlc = "/Applications/VLC.app/Contents/MacOS/VLC"
    if Path(macos_vlc).exists():
        return macos_vlc
    raise FileNotFoundError("VLC not found. Install VLC and ensure it's in PATH or /Applications/")


def launch_vlc(video_path: Path) -> subprocess.Popen:
    """Launch VLC with HTTP interface enabled, starting paused."""
    vlc_bin = find_vlc()
    cmd = [
        vlc_bin,
        str(video_path),
        "--extraintf", "http",
        "--http-password", VLC_HTTP_PASSWORD,
        "--http-port", str(VLC_HTTP_PORT),
        "--start-paused",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def vlc_get_state() -> str | None:
    """Get VLC playback state (playing, paused, stopped)."""
    try:
        url = f"http://localhost:{VLC_HTTP_PORT}/requests/status.xml"
        auth = ("", VLC_HTTP_PASSWORD)
        response = requests.get(url, auth=auth, timeout=5)
        if response.status_code == 200:
            match = re.search(r"<state>(\w+)</state>", response.text)
            if match:
                return match.group(1)
        return None
    except requests.RequestException:
        return None


def vlc_command(command: str, val: str | None = None) -> bool:
    """Send a command to VLC via HTTP API."""
    try:
        url = f"http://localhost:{VLC_HTTP_PORT}/requests/status.xml"
        auth = ("", VLC_HTTP_PASSWORD)
        params = {"command": command}
        if val is not None:
            params["val"] = val
        response = requests.get(url, params=params, auth=auth, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def vlc_seek(seconds: float) -> bool:
    """Seek VLC to specified position in seconds."""
    return vlc_command("seek", str(int(seconds)))


def vlc_ensure_paused() -> bool:
    """Ensure VLC is paused (only toggle if currently playing)."""
    state = vlc_get_state()
    if state == "playing":
        return vlc_command("pl_pause")
    return state == "paused"


def vlc_ensure_playing() -> bool:
    """Ensure VLC is playing (only toggle if currently paused)."""
    state = vlc_get_state()
    if state == "paused":
        return vlc_command("pl_pause")
    return state == "playing"


def vlc_is_ready() -> bool:
    """Check if VLC HTTP interface is responding."""
    return vlc_get_state() is not None


def review_anomalies(video_path: Path, groups: list[list[dict]]):
    """Launch VLC and step through detected anomalies."""
    if not groups:
        print("No anomalies to review.")
        return

    print("\nLaunching VLC...")
    vlc_process = launch_vlc(video_path)

    # Wait for VLC HTTP interface to become ready with improved error handling
    max_retries = 30
    retry_interval = 0.5
    for attempt in range(max_retries):
        # Check if VLC process is still alive
        if vlc_process.poll() is not None:
            # Process has died
            print("Error: VLC process terminated unexpectedly.")
            print("This usually means:")
            print("  - VLC failed to start (check if VLC is properly installed)")
            print("  - The video file format is not supported")
            print("  - VLC encountered a startup error")
            return

        # Check if HTTP interface is ready
        if vlc_is_ready():
            break
        time.sleep(retry_interval)
    else:
        # Timeout reached
        elapsed_time = max_retries * retry_interval
        print(f"Error: VLC HTTP interface not responding after {elapsed_time:.1f} seconds.")
        print("VLC process is still running but the HTTP interface is not available.")
        print("This may indicate:")
        print("  - VLC is starting very slowly")
        print("  - HTTP interface is disabled in VLC preferences")
        print(f"  - Port {VLC_HTTP_PORT} is already in use")
        vlc_process.terminate()
        return

    print(f"Reviewing {len(groups)} anomaly segments. Press Enter to play through each.\n")

    try:
        for i, group in enumerate(groups, 1):
            t_start = group[0]["t_sec"]
            seek_to = max(0, t_start - 2)

            # Seek to 2 seconds before anomaly, ensure paused
            print(f"Segment {i}/{len(groups)}: {format_timestamp(t_start)}")
            vlc_seek(seek_to)
            vlc_ensure_paused()

            input("  Press Enter to play...")

            # Play for ~7 seconds then ensure paused
            vlc_ensure_playing()
            time.sleep(7)
            vlc_ensure_paused()

            if i < len(groups):
                input("  Press Enter for next segment...")
            else:
                input("  Press Enter to finish...")

    finally:
        # Gracefully terminate VLC with timeout and kill() fallback
        vlc_process.terminate()
        try:
            vlc_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            # Process didn't terminate gracefully, force kill
            vlc_process.kill()
            vlc_process.wait()  # Clean up zombie process
        except Exception:
            # Handle any other exceptions (e.g., process already dead)
            pass

    print("\nReview complete")


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
    parser.add_argument(
        "--review",
        action="store_true",
        help="Launch VLC and step through detected anomalies"
    )

    args = parser.parse_args()

    video_path = args.video.resolve()

    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)

    # Check for ffmpeg availability
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found in PATH")
        print("Please install ffmpeg: https://ffmpeg.org/download.html")
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

        if num_frames == 0:
            print("Error: No frames extracted. Check video file and ffmpeg installation.")
            sys.exit(1)

        # Step 3: Analyze frames
        print("Analyzing brightness...")
        brightness, banding, timestamps, corrupt_count = analyze_frames(frame_dir, pts_times)

        # Warn if frames failed to load
        if corrupt_count > 0:
            corrupt_pct = (corrupt_count / num_frames) * 100
            print(f"WARNING: {corrupt_count} frames ({corrupt_pct:.1f}%) failed to load (corrupt or unreadable)")
            if corrupt_pct > 5.0:
                print(f"ERROR: More than 5% of frames are corrupt - results may be unreliable")
                sys.exit(1)

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

        # Step 7: Interactive review (if requested)
        if args.review:
            review_anomalies(video_path, groups)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Cleaning up temporary files...")
        raise
    finally:
        # Clean up temporary frames
        if not args.keep_frames and frame_dir.exists():
            shutil.rmtree(frame_dir)


if __name__ == "__main__":
    main()

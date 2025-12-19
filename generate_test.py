#!/usr/bin/env python3
"""
Generate synthetic test footage with known flicker anomalies.

Creates a 60-second video with:
- Base: mid-gray background (RGB 128,128,128)
- Anomaly 1: Black frames at 15-17s
- Anomaly 2: Darker gray at 35-37s
- Anomaly 3: Black frames at 50-52s

Use this to validate the detection script catches all known events.
"""

import subprocess
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "test_footage"
OUTPUT_FILE = OUTPUT_DIR / "test_flicker.mp4"

# Video parameters
DURATION = 60  # seconds
FPS = 30
WIDTH = 1280
HEIGHT = 720

# Colors (RGB)
BASE_GRAY = "808080"      # mid-gray (128,128,128)
DARK_GRAY = "404040"      # darker gray (64,64,64)
BLACK = "000000"

# Anomalies: (start_sec, end_sec, color_hex)
ANOMALIES = [
    (15, 17, BLACK),      # 2 seconds of black
    (35, 37, DARK_GRAY),  # 2 seconds of darker gray
    (50, 52, BLACK),      # 2 seconds of black
]


def generate_test_video():
    """Generate the test video using ffmpeg."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build a complex filtergraph that overlays anomalies on the base
    # Start with base gray video
    filter_parts = []
    inputs = []

    # Input 0: base gray for full duration
    inputs.extend([
        "-f", "lavfi",
        "-i", f"color=c=0x{BASE_GRAY}:s={WIDTH}x{HEIGHT}:d={DURATION}:r={FPS}"
    ])

    # Add inputs for each anomaly
    for i, (start, end, color) in enumerate(ANOMALIES):
        duration = end - start
        inputs.extend([
            "-f", "lavfi",
            "-i", f"color=c=0x{color}:s={WIDTH}x{HEIGHT}:d={duration}:r={FPS}"
        ])

    # Build overlay chain
    # [0] is base, [1], [2], [3] are anomalies
    current = "[0]"
    for i, (start, end, _) in enumerate(ANOMALIES):
        next_label = f"[v{i}]" if i < len(ANOMALIES) - 1 else "[out]"
        filter_parts.append(
            f"{current}[{i+1}]overlay=enable='between(t,{start},{end})'{next_label}"
        )
        current = next_label if i < len(ANOMALIES) - 1 else None

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg", "-y", "-hide_banner",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[out]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        str(OUTPUT_FILE)
    ]

    print(f"Generating test video: {OUTPUT_FILE}")
    print(f"Duration: {DURATION}s, Resolution: {WIDTH}x{HEIGHT}, FPS: {FPS}")
    print()
    print("Anomalies:")
    for start, end, color in ANOMALIES:
        color_name = "BLACK" if color == BLACK else "DARK_GRAY" if color == DARK_GRAY else color
        print(f"  {start}s - {end}s: {color_name}")
    print()

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Error generating video:")
        print(result.stderr)
        return False

    print(f"Success! Test video created at: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024:.1f} KB")
    return True


if __name__ == "__main__":
    generate_test_video()

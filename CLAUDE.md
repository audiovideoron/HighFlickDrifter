# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run flicker detection on a video
python main.py <video_file>
python main.py video.mp4 --fps 4 --threshold 2.5 --keep-frames

# Run detection with interactive VLC review
python main.py video.mp4 --review

# Generate test footage and validate detection
python generate_test.py
python main.py test_footage/test_flicker.mp4
```

## Dependencies

- **ffmpeg** must be available in PATH (used for frame extraction)
- **VLC** must be available in PATH (used for --review mode)
- Python packages managed via uv: numpy, opencv-python, requests

## Architecture

Single-file CLI tool (`main.py`) with a linear processing pipeline:

1. **Frame Extraction** - ffmpeg extracts frames at low sample rate (default 2 fps)
2. **Analysis** - OpenCV computes per-frame brightness mean and horizontal banding variance
3. **Anomaly Detection** - Rolling z-score flags frames deviating from baseline
4. **Event Grouping** - Nearby anomalies (within 15s) collapse into reviewable segments
5. **Reporting** - Outputs timestamped segments for manual review
6. **Interactive Review** (optional) - VLC steps through anomalies via HTTP API

Key configuration constants at top of `main.py`:
- `FPS_SAMPLE` (2) - sample rate
- `Z_THRESHOLD` (3.0) - detection sensitivity
- `WINDOW_SIZE` (30) - rolling window samples
- `GROUP_GAP_SEC` (15) - segment grouping threshold

## Test Footage

`generate_test.py` creates synthetic video with known anomalies at specific timestamps (15-17s, 35-37s, 50-52s) to validate detection accuracy.

## Issue Tracking

BEFORE ANYTHING ELSE: run `bd onboard` and follow the instructions.

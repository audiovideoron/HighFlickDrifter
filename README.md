# HighFlickDrifter

Detects the flicker that makes the room uneasy.

A command-line tool for finding brightness instability and power-induced visual drift in long, static video recordings—without watching hours of footage.

## Overview

HighFlickDrifter analyzes video files to identify intermittent and disruptive brightness changes that occur during live events, recordings, and screen captures. These issues are often felt immediately by an audience but are difficult to locate after the fact.

The tool scans video at a low sampling rate, applies rolling statistical analysis, and reports only the time ranges where the image *should have been still—but wasn’t*.

## Features

* Fast scanning at configurable sample rates (default: 2 fps)
* Rolling z-score detection of global brightness instability
* Horizontal banding detection (power / lighting artifacts)
* Automatic grouping of nearby anomalies into reviewable segments
* Timestamped reports optimized for quick human verification

## Requirements

* Python 3.10+
* ffmpeg (available in `PATH`)

## Installation

```bash
git clone git@github.com:audiovideoron/HighFlickDrifter.git
cd highflickdrifter
uv sync
```

## Usage

```bash
# Basic scan
python main.py <video_file>

# Example
python main.py test_footage/test_flicker.mp4

# Custom sensitivity
python main.py video.mp4 --fps 4 --threshold 2.5 --keep-frames
```

### Options

| Option          | Default | Description                                |
| --------------- | ------- | ------------------------------------------ |
| `--fps`         | 2       | Frame sampling rate                        |
| `--threshold`   | 3.0     | Z-score threshold (lower = more sensitive) |
| `--keep-frames` | false   | Preserve extracted frames for inspection   |

## How It Works

1. **Frame Sampling**
   Extracts frames using ffmpeg at a low, fixed rate.

2. **Brightness Metrics**
   Computes mean luminance and contrast per frame.

3. **Banding Detection**
   Measures variance in horizontal row means to detect rolling or power-related artifacts.

4. **Anomaly Detection**
   Flags frames whose brightness or banding deviates significantly from a rolling baseline.

5. **Event Grouping**
   Collapses nearby anomalies into short, reviewable segments.

6. **Reporting**
   Outputs timestamps and severity—nothing more.

## Example Output

```
FLICKER DETECTION REPORT

Scanned 120 frames at 2 fps
Video duration: 1:00.00
Detected 3 unstable segments

Segment 1: 0:15.0 – 0:17.0
  Worst brightness drift: z = -8.5

Segment 2: 0:35.0 – 0:37.0
  Worst brightness drift: z = -4.2

Segment 3: 0:50.0 – 0:52.0
  Worst brightness drift: z = -8.3

Review these timestamps in your video player.
```

## Testing

Generate synthetic footage with known flicker:

```bash
python generate_test.py
python main.py test_footage/test_flicker.mp4
```

The test video injects controlled brightness drops at known timestamps to validate detection accuracy.

## Configuration

Key defaults in `main.py`:

| Parameter       | Purpose                 |
| --------------- | ----------------------- |
| `FPS_SAMPLE`    | Sampling rate           |
| `Z_THRESHOLD`   | Sensitivity             |
| `WINDOW_SIZE`   | Rolling baseline window |
| `GROUP_GAP_SEC` | Segment grouping gap    |

## Limitations

* Camera auto-exposure can mask or exaggerate flicker.
* LED wall refresh artifacts may require higher sampling rates.
* Designed for *static* content; motion-heavy video is out of scope.

## License

MIT

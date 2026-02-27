# Growing Up — Face Morph Video Generator

Takes a collection of photos of someone spanning years or decades and produces a smooth video showing them aging over time. Faces are automatically detected, aligned by eye position, sorted chronologically, and morphed together so features flow naturally from one photo to the next.

No cloud services required — everything runs on your own machine.

## How It Works

1. **Face Detection** — AI models (insightface + mediapipe) scan each photo to find the subject's face, even in group shots, and extract precise facial landmarks including iris positions.
2. **Face Alignment** — Each face is rotated, scaled, and cropped to a 1024×1024 square with the eyes placed at exactly the same position in every frame. This keeps the video stable.
3. **Chronological Sorting** — Photos are ordered by date, extracted from filenames (e.g. "1985 Birthday.jpg") or EXIF metadata.
4. **Review & Edit** — After processing, review aligned faces in a scrubber/flipbook. Delete any bad detections before generating the video.
5. **Morph Rendering** — Smooth transitions are generated between consecutive faces using Delaunay triangulation, which warps facial features (eyes to eyes, mouth to mouth) rather than doing a simple dissolve.
6. **Video Encoding** — All frames are assembled into an H.264 MP4 using FFmpeg, with optional background music (auto-timed with fade-in/fade-out).

The app includes both a **command-line pipeline** and a **web dashboard** for managing multiple subjects and monitoring progress.

## Features

- **Two-phase pipeline**: Process images first (detect, align, sort), review results, then generate video
- **Image scrubber**: Flip through aligned faces with slider, prev/next buttons, or arrow keys
- **Age overlay**: Displays computed age ("Newborn", "3 months", "Age 5", etc.) on both the scrubber preview and the rendered video
- **Vignette**: Optional oval vignette mask around the face (toggle on/off per subject)
- **Title cards**: Opening title (subject name) with fade, ending title card with fade-in
- **Music timing**: Background MP3 starts after the opening title fades out; music fades to silence over the last 3 seconds
- **Multi-subject support**: Manage multiple people from the web dashboard, each with independent settings and data
- **Delete & review**: Remove bad images from the scrubber; originals are moved to a `deleted/` folder (not permanently removed)

## Getting Started

### Prerequisites

- Python 3.11+ (tested with 3.14)
- FFmpeg (`brew install ffmpeg` on macOS)
- macOS recommended (uses Apple Silicon-optimized ML models)

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Prepare Your Images

Create a subfolder inside `Images/` named after your subject and copy their photos into it:

```
Images/
  YourName/
    1985 baby photo.jpg
    1990 school picture.png
    2005 wedding.jpg
    2024 selfie.png
    ...
```

- Photos can be JPG, JPEG, PNG, TIFF, HEIC, or BMP
- Include the year in the filename for best chronological sorting (e.g. "1992 graduation.jpg")
- Group photos are fine — the app identifies the correct person using face recognition
- The more photos you include, the smoother the final video

### Run via Web Dashboard (Recommended)

```bash
source venv/bin/activate
python webapp/app.py
```

Open **http://localhost:5001** in your browser. From there you can:

1. Create a subject and point it at your image folder
2. Click **"Process Images"** to run face detection, alignment, and sorting
3. **Review aligned faces** in the scrubber — delete any bad ones
4. Click **"Generate Video"** to render morphs and encode the final MP4
5. **Save the video** to your Downloads folder

Or double-click `Start Server.command` to launch directly.

### Run via Command Line

```bash
source venv/bin/activate

# Full pipeline
python Code/run_all.py --crossfade

# Or step by step
python Code/00_convert_to_png.py
python Code/01_detect_faces.py
python Code/02_align_faces.py
python Code/03_sort_images.py
python Code/04_render_morph.py --crossfade --vignette
python Code/05_encode_video.py --music mp3/song.mp3
```

The finished video will be saved to `Output/`.

### Useful Options

| Flag | Script | Effect |
|------|--------|--------|
| `--crossfade` | 04 / run_all | Simple dissolve instead of Delaunay morph |
| `--vignette` | 04 | Apply oval vignette mask around the face |
| `--reference "photo.jpg"` | 01 | Specify reference photos for face recognition |
| `--skip-detect` | run_all | Reuse existing face detection data |
| `--debug-labels` | 04 | Show filename/date on morph frames |
| `--music path.mp3` | 05 | Add background music (auto-timed with delays and fade-out) |
| `--face-scale 0.70` | 04 | Scale of face within vignette (smaller = more border) |
| `--darken-bg` | 04 | Darken background behind face using selfie segmentation |

## Project Structure

```
Code/              Pipeline scripts (00-07) + utilities
webapp/            Flask web dashboard
  app.py           Routes, subject management, SSE progress
  pipeline_runner.py  Background subprocess execution
  static/          CSS + JavaScript
  templates/       Jinja2 HTML templates
Images/<Name>/     Source photos (user-provided)
mp3/               Optional music tracks
subjects/          Per-subject working data (auto-created by webapp)
config.json        Pipeline settings (fps, morph duration, etc.)
manifest.json      Auto-generated face detection data
requirements.txt   Python dependencies
Start Server.command  macOS double-click launcher
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Face detection & recognition | insightface (buffalo_l) |
| Facial landmarks | mediapipe FaceLandmarker (478 points + iris) |
| Image processing | OpenCV, NumPy, Pillow |
| Morphing | Delaunay triangulation warp + blend |
| Video encoding | FFmpeg (H.264, CRF 18) |
| Web dashboard | Flask + Server-Sent Events (SSE) |
| Selfie segmentation | mediapipe (optional, for background darkening) |

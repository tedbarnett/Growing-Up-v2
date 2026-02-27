# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Face-morph video generator: takes hundreds of photos of a person spanning decades, detects/aligns faces, and produces a smooth morphing video showing them aging over time. Includes a Flask web dashboard for managing multiple subjects.

## Commands

```bash
# Activate venv (required for all commands)
source venv/bin/activate

# Run full pipeline
python Code/run_all.py [--crossfade] [--skip-detect] [--skip-to-morph] [--add-year-labels]

# Run individual pipeline steps
python Code/00_convert_to_png.py
python Code/01_detect_faces.py [--reference "photo1.jpg" "photo2.jpg"]
python Code/02_align_faces.py
python Code/03_sort_images.py
python Code/04_render_morph.py [--crossfade] [--debug-labels] [--vignette]
python Code/05_encode_video.py [--music path.mp3]
python Code/06_review_and_fix.py [--dedupe] [--filter]
python Code/07_interactive_review.py [--excluded-only] [--threshold 0.5] [--review-all]

# Launch web dashboard
python webapp/app.py  # http://localhost:5001 (or GROWUP_PORT env var)
```

## Architecture

### Two Interfaces

1. **CLI Pipeline** (`Code/`): Numbered scripts (00-07) run sequentially. `run_all.py` orchestrates them.
2. **Web Dashboard** (`webapp/`): Flask app that manages multiple subjects and runs the pipeline via subprocess with SSE progress streaming.

### Two-Phase Pipeline (Web Dashboard)

The web dashboard splits the pipeline into two phases with a review step:

- **Phase 1 — Process Images** (scripts 00-03): Convert → Detect faces → Align → Sort chronologically
- **Review**: Scrubber/flipbook to review aligned faces, delete bad detections
- **Phase 2 — Generate Video** (scripts 04-05): Render morph frames → Encode MP4

Each phase is launched separately. Aligned faces persist between sessions.

### Pipeline Flow

```
Images/*.jpg → 00_convert → 01_detect → 02_align → 03_sort → [review] → 04_morph → 05_encode → Output/*.mp4
               (manifest.json updated at each step)
```

- **00_convert_to_png.py**: Converts images to PNG, preserves dates (mtime + macOS creation date), moves originals to `deleted/` subfolder.
- **01_detect_faces.py**: Uses insightface (buffalo_l) for face detection + embeddings, mediapipe FaceLandmarker for 478-point landmarks. Hybrid approach: insightface finds/identifies the subject, mediapipe provides precise landmarks + iris positions.
- **02_align_faces.py**: Single-pass affine transform (rotation + scale + translation) placing eyes at fixed coordinates. Outputs 1024×1024 PNGs to `aligned/`.
- **03_sort_images.py**: Orders images chronologically by extracted date (filename or EXIF). Writes `__sequence__` to manifest.
- **04_render_morph.py**: Two modes — `--crossfade` (alpha blend) or default Delaunay triangulation morphing. Renders opening/ending title cards, age labels, optional vignette. Writes `render_info.json` with timing data for the encoder.
- **05_encode_video.py**: FFmpeg H.264 encoding (CRF 18, slow preset, yuv420p). Handles music delay (starts after title card), audio fade-out over last 3 seconds.

### Web App Architecture

- `webapp/app.py`: Flask routes for subject CRUD, two-phase pipeline launch, aligned image serving/deletion, video serving, SSE progress
- `webapp/pipeline_runner.py`: Runs pipeline scripts as subprocesses in background threads, writes `job_status.json` for progress. Defines `PROCESS_STEPS` (00-03) and `GENERATE_STEPS` (04-05).
- SSE endpoint (`/subjects/<name>/status`) streams progress to the browser
- `webapp/static/app.js`: SSE consumption, progress bar, scrubber/flipbook (slider + arrow keys + delete), phase-aware UI updates, age label computation, Finder-style file browser, path basename display, settings change tracking
- `webapp/static/style.css`: Professional pink theme, mobile-responsive layout, scrubber styling, vignette CSS overlay, Finder-style browse modal with sidebar + breadcrumbs, custom path tooltips

### Key Routes

- `POST /subjects/<name>/process-images` — Launch Phase 1
- `POST /subjects/<name>/generate-video` — Launch Phase 2
- `GET /subjects/<name>/aligned-sequence` — Ordered list of aligned images with metadata
- `GET /subjects/<name>/aligned/<filename>` — Serve individual aligned image
- `DELETE /subjects/<name>/aligned/<filename>` — Remove image from set
- `DELETE /subjects/<name>/delete` — Delete subject entirely
- `GET /subjects/<name>/status` — SSE progress stream

### Multi-Subject System

Each subject gets `subjects/<name>/` with its own config.json, manifest.json, and working directories (aligned, frames, output). The webapp manages subjects via `subjects.json` registry. Environment variables (`GROWUP_*`) isolate paths per subject when pipeline runs via webapp.

### Key Data Files

- **manifest.json**: Central data store — maps each image filename to face bbox, 512-dim embedding, 68 landmarks, iris positions, similarity score, date, sequence order, exclusion flags. The `__sequence__` key holds the chronologically sorted filename list.
- **config.json**: Pipeline parameters — output_size, fps, hold_frames, morph_frames, eye target positions, subject_name, subject_birthdate, vignette toggle
- **render_info.json**: Written by 04_render_morph.py to frames dir. Contains `music_delay_s` for the encoder to delay MP3 start.
- **job_status.json**: Written per-subject during pipeline runs. Contains state, phase, step progress, log tail, error info.

## Important Patterns

### insightface Cython Workaround

insightface requires Cython module stubs on Apple Silicon. See `01_detect_faces.py` — must stub `insightface.thirdparty.face3d.*` modules before importing insightface.

### mediapipe API

Uses the tasks API (`mp.tasks.vision.FaceLandmarker`), NOT the deprecated `mp.solutions` API. Model file: `Code/face_landmarker_v2_with_blendshapes.task`.

### Path Resolution (utils.py)

`PROJECT_ROOT` = parent of `Code/`. All paths derived from there. Environment variables (`GROWUP_PROJECT_ROOT`, `GROWUP_IMAGES_DIR`, `GROWUP_ALIGNED_DIR`, `GROWUP_FRAMES_DIR`, `GROWUP_OUTPUT_DIR`) override defaults — this is how the webapp isolates per-subject runs.

### Age Label Computation

Both Python (`04_render_morph.py:compute_age_label()`) and JavaScript (`app.js:computeAgeLabel()`) implementations exist for computing age from birthdate + photo date. The Python version renders into video frames; the JS version shows in the scrubber preview.

### Video Duration Estimation

Both Python (`app.py:get_projected_video_duration()`) and JavaScript (`app.js:updateScrubberDuration()`) compute estimated video length from aligned image count + config (fps, hold_frames, morph_frames). The JS version updates live in the scrubber title when images are deleted.

### Settings Form Change Tracking

The Save Settings button starts disabled/grey and turns blue only when a setting value differs from its initial state. Implemented via `app.js` settings form listener that captures initial values on page load.

## Tech Stack

- Python 3.11+ (tested with 3.14), Flask, insightface, mediapipe, OpenCV, numpy, Pillow, FFmpeg
- Designed for macOS (Apple Silicon), venv at project root

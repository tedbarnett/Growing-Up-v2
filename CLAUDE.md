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
- **04_render_morph.py**: Two modes — `--crossfade` (alpha blend) or default Delaunay triangulation morphing. Renders opening/ending title cards, age labels, optional vignette, and Barnett Labs credit card (©year, fade-in + 3s hold). Writes `render_info.json` with timing data for the encoder.
- **05_encode_video.py**: FFmpeg H.264 encoding (CRF 18, slow preset, yuv420p). Output filename derived from subject_name (e.g. "Growing Up - Ryan.mp4"). Accepts multiple `--music` files: single file loops via `-stream_loop -1`; multiple files crossfade via FFmpeg `acrossfade` filter (3s overlap). Handles music delay (starts after title card), audio fade-out over last 3 seconds.

### Web App Architecture

- `webapp/app.py`: Flask routes for subject CRUD, two-phase pipeline launch, aligned image serving/deletion, video serving, SSE progress
- `webapp/pipeline_runner.py`: Runs pipeline scripts as subprocesses in background threads, writes `job_status.json` for progress. Defines `PROCESS_STEPS` (00-03) and `GENERATE_STEPS` (04-05).
- SSE endpoint (`/subjects/<name>/status`) streams progress to the browser
- `webapp/static/app.js`: SSE consumption, progress bar, scrubber/flipbook (slider + arrow keys + delete), phase-aware UI updates, age label computation, Finder-style file browser, path basename display, settings change tracking, dynamic multi-music row management
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

Each subject gets `subjects/<name>/` with its own config.json, manifest.json, and working directories (aligned, frames, output). The webapp manages subjects via `subjects.json` registry. Environment variables (`GROWUP_*`) isolate paths per subject when pipeline runs via webapp. The `"music"` field in subjects.json is a list of paths (backward-compatible: legacy string values are treated as single-element lists via `get_music_list()`).

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

`PROJECT_ROOT` = parent of `Code/`. All paths derived from there. Environment variables (`GROWUP_PROJECT_ROOT`, `GROWUP_CODE_DIR`, `GROWUP_IMAGES_DIR`, `GROWUP_ALIGNED_DIR`, `GROWUP_FRAMES_DIR`, `GROWUP_OUTPUT_DIR`) override defaults — this is how the webapp isolates per-subject runs. `GROWUP_CODE_DIR` separates bundled Code/ scripts from user data when running as a PyInstaller app.

### Age Label Computation

Both Python (`04_render_morph.py:compute_age_label()`) and JavaScript (`app.js:computeAgeLabel()`) implementations exist for computing age from birthdate + photo date. The Python version renders into video frames; the JS version shows in the scrubber preview.

### Video Duration Estimation

Both Python (`app.py:get_projected_video_duration()`) and JavaScript (`app.js:updateScrubberDuration()`) compute estimated video length from aligned image count + config (fps, hold_frames, morph_frames). The JS version updates live in the scrubber title when images are deleted.

### Settings Form Change Tracking

The Save Settings button starts disabled/grey and turns blue only when a setting value differs from its initial state. Implemented via `app.js` settings form listener that captures initial values on page load. Uses event delegation on the form to handle dynamically added music input rows.

### Multi-Music Support

Music is stored as a JSON list in `subjects.json`. Key helpers in `app.py`:
- `get_music_list(info)`: Returns music as a list (handles legacy string format)
- `resolve_music_paths(info)`: Resolves to list of absolute file paths
The encoder (`05_encode_video.py`) accepts `--music` with `nargs="*"`: single file loops via `-stream_loop -1`; multiple files use FFmpeg `acrossfade` filter (3s crossfade overlap) before applying delay and fade-out.

### PyInstaller macOS App Packaging

`packaging/macos/` contains everything needed to build a standalone `.app`:
- **launcher.py**: Entry point — creates `~/Documents/Growing Up/` user data dir, sets env vars (`GROWUP_PROJECT_ROOT` → user data, `GROWUP_CODE_DIR` → bundle), adds bundled ffmpeg to PATH, imports Flask app, opens browser.
- **GrowingUp.spec**: PyInstaller one-dir spec. Bundles Code/ scripts, ML models, webapp/, ffmpeg binaries. Excludes insightface's x86_64-only `mesh_core_cython.so` (stubbed at runtime). Hidden imports for insightface, onnxruntime, mediapipe, scipy, etc.
- **build.sh**: Generates icon.icns from apple-touch-icon.png, downloads static ffmpeg arm64 binaries (~80 MB from evermeet.cx), runs PyInstaller, optionally code signs (`--sign`) and notarizes (`--notarize`). Output: `dist/Growing Up.app` + `dist/Growing Up.dmg` (~185 MB).
- **entitlements.plist**: Allows JIT memory (numpy/scipy), disables library validation for bundled .so files.
- `pipeline_runner.py` detects `sys.frozen` to use `sys.executable` instead of venv python when running as a bundled app.

## Tech Stack

- Python 3.11+ (tested with 3.14), Flask, insightface, mediapipe, OpenCV, numpy, Pillow, FFmpeg
- PyInstaller for macOS .app packaging
- Designed for macOS (Apple Silicon), venv at project root

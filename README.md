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
- **Title cards**: Opening title (subject name) with fade, ending title card with fade-in, Barnett Labs credit card
- **Multiple music tracks**: Add one or more MP3s per subject — a single track loops; multiple tracks crossfade (3s overlap) between each other.
- **Music timing**: Background music starts after the opening title fades out; music fades to silence over the last 3 seconds
- **Multi-subject support**: Manage multiple people from the web dashboard, each with independent settings and data
- **Delete & review**: Remove bad images from the scrubber; originals are moved to a `deleted/` folder (not permanently removed)
- **Browse anywhere**: Images and music can live anywhere on your disk — just browse to them in the UI. Previously used locations are remembered across subjects.

## Mac App (Recommended)

Download the latest `Growing Up.dmg` from the Releases page. Open the DMG, drag `Growing Up.app` to Applications, and double-click to launch. No Python, FFmpeg, or terminal required — everything is bundled inside the app.

On first launch:
- Your browser opens to the dashboard at `http://localhost:5001`
- User data is stored in `~/Documents/Growing Up/`
- The insightface face recognition model (~300 MB) downloads automatically on first "Process Images"

### Building the App from Source

```bash
# Install PyInstaller
source venv/bin/activate
pip install pyinstaller

# Build unsigned .app + .dmg
./packaging/macos/build.sh

# Build with code signing (for distribution)
DEVELOPER_ID="Developer ID Application: Your Name (TEAMID)" ./packaging/macos/build.sh --sign

# Build with code signing + notarization
DEVELOPER_ID="Developer ID Application: Your Name (TEAMID)" \
TEAM_ID="YOUR_TEAM_ID" \
./packaging/macos/build.sh --notarize
```

Output: `dist/Growing Up.app` and `dist/Growing Up.dmg` (~185 MB).

## Developer Setup

### Prerequisites

- Python 3.11+ (tested with 3.14)
- FFmpeg (`brew install ffmpeg` on macOS)
- macOS recommended (uses Apple Silicon-optimized ML models)

### Setup

```bash
git clone https://github.com/tedbarnett/Growing-Up-v2.git
cd Growing-Up-v2
./setup.sh
```

The setup script will:
- Verify Python 3.11+ and FFmpeg are installed
- Create a virtual environment and install dependencies
- Download the required ML model files (~15 MB)
- Create the `subjects/` working directory

### Run

```bash
source venv/bin/activate
python webapp/app.py
```

Open **http://localhost:5001** in your browser. From there you can:

1. Create a subject — use **Browse** to point at any folder of photos on your disk
2. Click **"Process Images"** to run face detection, alignment, and sorting
3. **Review aligned faces** in the scrubber — delete any bad ones
4. Click **"Generate Video"** to render morphs and encode the final MP4
5. **Save the video** to your Downloads folder

Or double-click `Start Server.command` to launch directly.

### Prepare Your Images

Your photos can live anywhere on your filesystem — just browse to the folder in the UI. For best results:

- Include the year in the filename for chronological sorting (e.g. "1992 graduation.jpg")
- Photos can be JPG, JPEG, PNG, TIFF, HEIC, or BMP
- Group photos are fine — the app identifies the correct person using face recognition
- The more photos you include, the smoother the final video

### Manual Setup (Alternative)

If you prefer to set up manually instead of using `setup.sh`:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download ML models
curl -L -o Code/face_landmarker_v2_with_blendshapes.task \
    "https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2_with_blendshapes.task"
curl -L -o Code/selfie_segmenter.tflite \
    "https://storage.googleapis.com/mediapipe-models/selfie_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"

mkdir -p subjects
```

## Command-Line Pipeline

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
python Code/05_encode_video.py --music song1.mp3 song2.mp3
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
| `--music path.mp3 [path2.mp3 ...]` | 05 | Add background music (single loops; multiple concatenated, auto-timed) |
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
packaging/macos/   PyInstaller app packaging
  launcher.py      App entry point (sets up user data dir, env vars, launches Flask)
  GrowingUp.spec   PyInstaller spec (dependencies, hidden imports, bundle config)
  build.sh         Build + codesign + DMG + notarize script
  entitlements.plist  macOS code signing entitlements
subjects/          Per-subject working data (auto-created)
config.json        Pipeline settings (fps, morph duration, etc.)
requirements.txt   Python dependencies
setup.sh           One-command post-clone setup
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

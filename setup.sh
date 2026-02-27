#!/bin/bash
# Growing Up — Post-clone setup script
# Run this once after cloning the repository.

set -e

echo "========================================"
echo "  Growing Up — Setup"
echo "========================================"
echo

# --- Check Python 3.11+ ---
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null || true)
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null || true)
        if [ "$major" = "3" ] && [ "$minor" -ge 11 ]; then
            PYTHON="$cmd"
            echo "[OK] Found $cmd $version"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[ERROR] Python 3.11+ is required but not found."
    echo "  Install it with: brew install python@3.13"
    exit 1
fi

# --- Check FFmpeg ---
if command -v ffmpeg &>/dev/null; then
    echo "[OK] FFmpeg found"
else
    echo "[WARNING] FFmpeg not found — required for video encoding."
    if command -v brew &>/dev/null; then
        read -p "  Install via Homebrew? (y/N) " yn
        case $yn in
            [Yy]* ) brew install ffmpeg; echo "[OK] FFmpeg installed";;
            * ) echo "  Skipping — install manually before generating videos.";;
        esac
    else
        echo "  Install it with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
    fi
fi

echo

# --- Create virtual environment ---
if [ -d "venv" ]; then
    echo "[OK] Virtual environment already exists"
else
    echo "Creating virtual environment..."
    "$PYTHON" -m venv venv
    echo "[OK] Virtual environment created"
fi

# --- Install dependencies ---
echo "Installing Python dependencies..."
venv/bin/pip install --upgrade pip -q
venv/bin/pip install -r requirements.txt -q
echo "[OK] Dependencies installed"

echo

# --- Download ML model files ---
MODEL_DIR="Code"

if [ -f "$MODEL_DIR/face_landmarker_v2_with_blendshapes.task" ]; then
    echo "[OK] Face landmarker model already exists"
else
    echo "Downloading face landmarker model..."
    curl -L -o "$MODEL_DIR/face_landmarker_v2_with_blendshapes.task" \
        "https://storage.googleapis.com/mediapipe-assets/face_landmarker_v2_with_blendshapes.task"
    echo "[OK] Face landmarker model downloaded"
fi

if [ -f "$MODEL_DIR/selfie_segmenter.tflite" ]; then
    echo "[OK] Selfie segmenter model already exists"
else
    echo "Downloading selfie segmenter model..."
    curl -L -o "$MODEL_DIR/selfie_segmenter.tflite" \
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
    echo "[OK] Selfie segmenter model downloaded"
fi

echo

# --- Create subjects directory ---
mkdir -p subjects
echo "[OK] subjects/ directory ready"

echo
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo
echo "To start the web dashboard:"
echo "  source venv/bin/activate"
echo "  python webapp/app.py"
echo
echo "Then open http://localhost:5001 in your browser."
echo "Images and music can live anywhere on your disk — just browse to them in the UI."
echo

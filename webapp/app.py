#!/usr/bin/env python3
"""
Growing Up - Web Dashboard

A Flask web app for managing subjects and generating face-morph videos.
Run with: venv/bin/python webapp/app.py
"""

import json
import os
import sys
from pathlib import Path

from flask import (
    Flask, render_template, request, redirect, url_for,
    jsonify, Response, send_file, abort,
)

from pipeline_runner import (
    start_pipeline, start_process, start_generate,
    is_job_running, get_job_status, is_processed as check_is_processed,
    PROJECT_ROOT,
)

app = Flask(__name__)

SUBJECTS_JSON = PROJECT_ROOT / "subjects.json"
SUBJECTS_DIR = PROJECT_ROOT / "subjects"
IMAGES_ROOT = PROJECT_ROOT / "Images"
MP3_DIR = PROJECT_ROOT / "mp3"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_subjects():
    """Load subjects.json registry."""
    if SUBJECTS_JSON.exists():
        with open(SUBJECTS_JSON, "r") as f:
            return json.load(f)
    return {}


def save_subjects(data):
    with open(SUBJECTS_JSON, "w") as f:
        json.dump(data, f, indent=2)


def get_subject_dir(name):
    return SUBJECTS_DIR / name


def load_subject_config(name):
    config_path = get_subject_dir(name) / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def save_subject_config(name, config):
    config_path = get_subject_dir(name) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)


def count_images(images_folder):
    """Count image files in an images directory (absolute path or relative to Images/)."""
    if not images_folder:
        return 0
    img_path = Path(images_folder)
    if img_path.is_absolute():
        img_dir = img_path
    else:
        img_dir = IMAGES_ROOT / images_folder
    if not img_dir.exists():
        return 0
    extensions = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".heic", ".bmp"}
    return sum(1 for f in img_dir.iterdir()
               if f.is_file() and f.suffix.lower() in extensions)


def count_aligned(name):
    """Count aligned face images for a subject."""
    aligned_dir = get_subject_dir(name) / "aligned"
    if not aligned_dir.exists():
        return 0
    return sum(1 for f in aligned_dir.iterdir()
               if f.is_file() and f.suffix.lower() == ".png")


def find_video(name):
    """Find the most recent MP4 in the subject's output dir."""
    output_dir = get_subject_dir(name) / "output"
    if not output_dir.exists():
        return None
    mp4s = sorted(output_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return mp4s[0] if mp4s else None


def list_mp3s():
    """List available MP3 files."""
    if not MP3_DIR.exists():
        return []
    return sorted(f.name for f in MP3_DIR.iterdir()
                  if f.is_file() and f.suffix.lower() == ".mp3")


def list_image_folders():
    """List subdirectories under Images/."""
    if not IMAGES_ROOT.exists():
        return []
    return sorted(d.name for d in IMAGES_ROOT.iterdir() if d.is_dir())


def get_mp3_duration(music_ref):
    """Get MP3 duration in seconds. Returns None if unavailable."""
    if not music_ref:
        return None
    mp3_path = Path(music_ref)
    if not mp3_path.is_absolute():
        mp3_path = MP3_DIR / music_ref
    if not mp3_path.exists():
        return None
    try:
        import subprocess
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(mp3_path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def get_projected_video_duration(name):
    """Calculate projected video duration from manifest + config."""
    config = load_subject_config(name)
    subject_dir = get_subject_dir(name)
    manifest_path = subject_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
    seq = manifest.get("__sequence__", [])
    n = len(seq)
    if n < 2:
        return None
    fps = config.get("fps", 30)
    hold = config.get("hold_frames", 15)
    morph = config.get("morph_frames", 30)
    # Title card ~3s, each image gets hold frames, last gets 2x hold,
    # morph transitions between pairs, 3s fade-to-black, 3s end title card
    title_frames = 3 * fps
    image_frames = n * hold + hold + (n - 1) * morph
    ending_frames = 3 * fps + 3 * fps  # fade-to-black + end title
    total_frames = title_frames + image_frames + ending_frames
    return total_frames / fps


# ---------------------------------------------------------------------------
# Auto-migration: detect existing Ted data and create initial subject entry
# ---------------------------------------------------------------------------

def auto_migrate():
    """If subjects.json doesn't exist but config.json does, create Ted entry."""
    if SUBJECTS_JSON.exists():
        return

    existing_config = PROJECT_ROOT / "config.json"
    if not existing_config.exists():
        # Fresh install, just create empty registry
        save_subjects({})
        return

    with open(existing_config, "r") as f:
        config = json.load(f)

    subject_name = config.get("subject_name", "Ted")

    # Create subject directory
    subject_dir = SUBJECTS_DIR / subject_name
    subject_dir.mkdir(parents=True, exist_ok=True)

    # Copy config into subject dir with images_folder field
    subject_config = dict(config)
    subject_config["images_folder"] = subject_name
    with open(subject_dir / "config.json", "w") as f:
        json.dump(subject_config, f, indent=2)

    # Copy manifest if it exists
    existing_manifest = PROJECT_ROOT / "manifest.json"
    if existing_manifest.exists():
        import shutil
        shutil.copy2(str(existing_manifest), str(subject_dir / "manifest.json"))

    # Create symlinks to existing working directories
    _symlink(PROJECT_ROOT / "aligned", subject_dir / "aligned")
    _symlink(PROJECT_ROOT / "frames", subject_dir / "frames")
    _symlink(PROJECT_ROOT / "Output", subject_dir / "output")

    # Save subjects registry
    save_subjects({
        subject_name: {
            "birthdate": config.get("subject_birthdate", ""),
            "images_folder": subject_name,
            "music": "",
            "created_at": "",
        }
    })

    print(f"Auto-migrated existing subject '{subject_name}'")


def _symlink(target, link_path):
    """Create a symlink, skipping if already exists."""
    link_path = Path(link_path)
    target = Path(target)
    if link_path.exists() or link_path.is_symlink():
        return
    if target.exists():
        link_path.symlink_to(target)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Dashboard: list all subjects."""
    subjects = load_subjects()
    subject_list = []
    for name, info in subjects.items():
        video = find_video(name)
        running = is_job_running(name)
        status = get_job_status(get_subject_dir(name))
        subject_list.append({
            "name": name,
            "birthdate": info.get("birthdate", ""),
            "images_folder": info.get("images_folder", ""),
            "image_count": count_images(info.get("images_folder", "")),
            "aligned_count": count_aligned(name),
            "has_video": video is not None,
            "is_running": running,
            "job_state": status.get("state") if status else None,
        })
    return render_template("index.html",
                           subjects=subject_list,
                           image_folders=list_image_folders(),
                           mp3s=list_mp3s())


@app.route("/subjects/new", methods=["POST"])
def create_subject():
    """Create a new subject."""
    name = request.form.get("name", "").strip()
    if not name:
        return redirect(url_for("index"))

    subjects = load_subjects()
    if name in subjects:
        return redirect(url_for("subject_detail", name=name))

    birthdate = request.form.get("birthdate", "")
    images_folder = request.form.get("images_folder", name)
    music = request.form.get("music", "")

    # Create subject directory and subdirs
    subject_dir = get_subject_dir(name)
    subject_dir.mkdir(parents=True, exist_ok=True)
    (subject_dir / "aligned").mkdir(exist_ok=True)
    (subject_dir / "frames").mkdir(exist_ok=True)
    (subject_dir / "output").mkdir(exist_ok=True)

    # Create subject config (based on default project config)
    default_config_path = PROJECT_ROOT / "config.json"
    if default_config_path.exists():
        with open(default_config_path, "r") as f:
            config = json.load(f)
    else:
        config = {
            "output_size": 1024,
            "fps": 30,
            "hold_frames": 15,
            "morph_frames": 30,
            "eye_left_target": [0.401, 0.42],
            "eye_right_target": [0.599, 0.42],
            "face_recognition_tolerance": 0.6,
            "output_format": "mp4",
            "codec": "libx264",
            "reference_photos": [],
        }

    config["subject_name"] = name
    config["subject_birthdate"] = birthdate
    config["images_folder"] = images_folder
    save_subject_config(name, config)

    # Update registry
    subjects[name] = {
        "birthdate": birthdate,
        "images_folder": images_folder,
        "music": music,
        "created_at": "",
    }
    save_subjects(subjects)

    return redirect(url_for("subject_detail", name=name))


@app.route("/subjects/<name>")
def subject_detail(name):
    """Subject detail page."""
    subjects = load_subjects()
    if name not in subjects:
        abort(404)

    info = subjects[name]
    config = load_subject_config(name)
    video = find_video(name)
    status = get_job_status(get_subject_dir(name))

    subject_dir = get_subject_dir(name)
    processed = check_is_processed(subject_dir)

    mp3_duration = get_mp3_duration(info.get("music", ""))
    video_duration = get_projected_video_duration(name)

    return render_template("subject.html",
                           name=name,
                           info=info,
                           config=config,
                           image_count=count_images(info.get("images_folder", "")),
                           aligned_count=count_aligned(name),
                           has_video=video is not None,
                           video_name=video.name if video else None,
                           is_running=is_job_running(name),
                           is_processed=processed,
                           job_status=status,
                           image_folders=list_image_folders(),
                           mp3s=list_mp3s(),
                           mp3_duration=mp3_duration,
                           video_duration=video_duration)


@app.route("/subjects/<name>/edit", methods=["POST"])
def edit_subject(name):
    """Update subject settings."""
    subjects = load_subjects()
    if name not in subjects:
        abort(404)

    birthdate = request.form.get("birthdate", "")
    images_folder = request.form.get("images_folder", "")
    music = request.form.get("music", "")
    vignette = request.form.get("vignette") == "on"

    subjects[name]["birthdate"] = birthdate
    subjects[name]["images_folder"] = images_folder
    subjects[name]["music"] = music
    subjects[name]["vignette"] = vignette
    save_subjects(subjects)

    # Update subject config too
    config = load_subject_config(name)
    config["subject_birthdate"] = birthdate
    config["images_folder"] = images_folder
    config["vignette"] = vignette
    save_subject_config(name, config)

    return redirect(url_for("subject_detail", name=name))


@app.route("/subjects/<name>/delete", methods=["DELETE"])
def delete_subject(name):
    """Remove a subject from the registry and delete its working directory (not original images)."""
    subjects = load_subjects()
    if name not in subjects:
        return jsonify({"error": "Subject not found"}), 404

    if is_job_running(name):
        return jsonify({"error": "Cannot delete while pipeline is running"}), 409

    # Remove from registry
    del subjects[name]
    save_subjects(subjects)

    # Remove subject working directory (aligned, frames, output, config, manifest)
    # but NOT the original images folder
    import shutil
    subject_dir = get_subject_dir(name)
    if subject_dir.exists():
        shutil.rmtree(str(subject_dir), ignore_errors=True)

    return jsonify({"status": "deleted"})


@app.route("/subjects/<name>/make-video", methods=["POST"])
def make_video(name):
    """Launch pipeline in background."""
    subjects = load_subjects()
    if name not in subjects:
        return jsonify({"error": "Subject not found"}), 404

    if is_job_running(name):
        return jsonify({"error": "Pipeline already running"}), 409

    config = load_subject_config(name)
    subject_dir = get_subject_dir(name)

    # Ensure working dirs exist
    for subdir in ("aligned", "frames", "output"):
        p = subject_dir / subdir
        if not p.exists() and not p.is_symlink():
            p.mkdir(exist_ok=True)

    # Music — support absolute paths or filenames in mp3/
    music = subjects[name].get("music", "")
    music_path = None
    if music:
        mp3_path = Path(music)
        if mp3_path.is_absolute() and mp3_path.exists():
            music_path = str(mp3_path)
        else:
            mp3_path = MP3_DIR / music
            if mp3_path.exists():
                music_path = str(mp3_path)

    started = start_pipeline(name, str(subject_dir), config, music_path)
    if started:
        return jsonify({"status": "started"})
    else:
        return jsonify({"error": "Could not start pipeline"}), 500


@app.route("/subjects/<name>/process-images", methods=["POST"])
def process_images(name):
    """Launch Phase 1: convert, detect, align, sort."""
    subjects = load_subjects()
    if name not in subjects:
        return jsonify({"error": "Subject not found"}), 404

    if is_job_running(name):
        return jsonify({"error": "Pipeline already running"}), 409

    config = load_subject_config(name)
    subject_dir = get_subject_dir(name)

    for subdir in ("aligned", "frames", "output"):
        p = subject_dir / subdir
        if not p.exists() and not p.is_symlink():
            p.mkdir(exist_ok=True)

    started = start_process(name, str(subject_dir), config)
    if started:
        return jsonify({"status": "started", "phase": "process"})
    else:
        return jsonify({"error": "Could not start pipeline"}), 500


@app.route("/subjects/<name>/generate-video", methods=["POST"])
def generate_video(name):
    """Launch Phase 2: morph + encode."""
    subjects = load_subjects()
    if name not in subjects:
        return jsonify({"error": "Subject not found"}), 404

    if is_job_running(name):
        return jsonify({"error": "Pipeline already running"}), 409

    config = load_subject_config(name)
    subject_dir = get_subject_dir(name)

    if not check_is_processed(subject_dir):
        return jsonify({"error": "Images must be processed first"}), 400

    for subdir in ("frames", "output"):
        p = subject_dir / subdir
        if not p.exists() and not p.is_symlink():
            p.mkdir(exist_ok=True)

    # Music
    music = subjects[name].get("music", "")
    music_path = None
    if music:
        mp3_path = Path(music)
        if mp3_path.is_absolute() and mp3_path.exists():
            music_path = str(mp3_path)
        else:
            mp3_path = MP3_DIR / music
            if mp3_path.exists():
                music_path = str(mp3_path)

    started = start_generate(name, str(subject_dir), config, music_path)
    if started:
        return jsonify({"status": "started", "phase": "generate"})
    else:
        return jsonify({"error": "Could not start pipeline"}), 500


@app.route("/subjects/<name>/aligned-sequence")
def aligned_sequence(name):
    """Return ordered list of aligned images with metadata for the scrubber."""
    subjects = load_subjects()
    if name not in subjects:
        return jsonify({"error": "Subject not found"}), 404

    subject_dir = get_subject_dir(name)
    manifest_path = subject_dir / "manifest.json"
    aligned_dir = subject_dir / "aligned"

    if not manifest_path.exists():
        return jsonify({"images": []})

    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, IOError):
        return jsonify({"images": []})

    sequence = manifest.get("__sequence__", [])
    images = []
    for fname in sequence:
        aligned_path = aligned_dir / fname
        if not aligned_path.exists():
            continue
        entry = manifest.get(fname, {})
        images.append({
            "filename": fname,
            "date": entry.get("date", ""),
            "sort_date": entry.get("sort_date", ""),
            "year": entry.get("date", "")[:4] if entry.get("date", "") else "",
            "similarity": entry.get("similarity", None),
        })

    return jsonify({"images": images})


@app.route("/subjects/<name>/aligned/<filename>")
def serve_aligned(name, filename):
    """Serve an individual aligned image."""
    subjects = load_subjects()
    if name not in subjects:
        abort(404)

    aligned_path = get_subject_dir(name) / "aligned" / filename
    if not aligned_path.exists():
        abort(404)

    return send_file(str(aligned_path), mimetype="image/png")


@app.route("/subjects/<name>/aligned/<filename>", methods=["DELETE"])
def delete_aligned(name, filename):
    """Remove an aligned image: delete aligned file, move original to deleted/, update manifest."""
    subjects = load_subjects()
    if name not in subjects:
        return jsonify({"error": "Subject not found"}), 404

    subject_dir = get_subject_dir(name)
    aligned_path = subject_dir / "aligned" / filename

    # Delete aligned file
    if aligned_path.exists():
        aligned_path.unlink()

    # Move original to deleted/
    config = load_subject_config(name)
    images_folder = config.get("images_folder", "")
    if images_folder:
        images_path = Path(images_folder)
        if not images_path.is_absolute():
            images_path = IMAGES_ROOT / images_folder
        original = images_path / filename
        if original.exists():
            deleted_dir = images_path / "deleted"
            deleted_dir.mkdir(exist_ok=True)
            import shutil
            shutil.move(str(original), str(deleted_dir / filename))

    # Update manifest
    manifest_path = subject_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            # Remove from sequence
            seq = manifest.get("__sequence__", [])
            if filename in seq:
                seq.remove(filename)
                manifest["__sequence__"] = seq
            # Remove entry
            manifest.pop(filename, None)
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
        except (json.JSONDecodeError, IOError):
            pass

    return jsonify({"status": "deleted", "filename": filename})


@app.route("/subjects/<name>/status")
def subject_status(name):
    """SSE endpoint: stream job progress."""
    subjects = load_subjects()
    if name not in subjects:
        abort(404)

    def event_stream():
        subject_dir = get_subject_dir(name)
        last_data = None
        while True:
            status = get_job_status(subject_dir)
            running = is_job_running(name)

            data = json.dumps({
                "status": status,
                "is_running": running,
                "has_video": find_video(name) is not None,
                "is_processed": check_is_processed(subject_dir),
            })

            if data != last_data:
                yield f"data: {data}\n\n"
                last_data = data

            # Stop streaming when job is done
            if not running and status and status.get("state") in ("complete", "error"):
                yield f"data: {data}\n\n"
                break

            if not running and not status:
                yield f"data: {json.dumps({'status': None, 'is_running': False, 'has_video': find_video(name) is not None})}\n\n"
                break

            import time
            time.sleep(1)

    return Response(event_stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/subjects/<name>/video")
def serve_video(name):
    """Serve the subject's output video."""
    video = find_video(name)
    if not video:
        abort(404)
    return send_file(str(video), mimetype="video/mp4")


@app.route("/api/mp3s")
def api_mp3s():
    return jsonify(list_mp3s())


@app.route("/api/folders")
def api_folders():
    return jsonify(list_image_folders())


@app.route("/api/browse")
def api_browse():
    """Browse the filesystem. Returns dirs or files at a given path.

    Query params:
        path  - directory to list (default: user home)
        type  - "dirs" or "files" (default: "dirs")
        ext   - file extension filter, e.g. ".mp3" (only when type=files)
    """
    raw_path = request.args.get("path", "")
    browse_type = request.args.get("type", "dirs")
    ext_filter = request.args.get("ext", "").lower()

    # Default to home directory
    if not raw_path:
        raw_path = str(Path.home())

    target = Path(raw_path).expanduser().resolve()

    if not target.is_dir():
        return jsonify({"current": str(target), "parent": str(target.parent), "items": [], "error": "Not a directory"})

    parent = str(target.parent) if target != target.parent else str(target)

    items = []
    try:
        for entry in sorted(target.iterdir(), key=lambda e: e.name.lower()):
            # Skip hidden files/dirs
            if entry.name.startswith("."):
                continue
            if browse_type == "dirs" and entry.is_dir():
                items.append({"name": entry.name, "path": str(entry)})
            elif browse_type == "files" and entry.is_file():
                if ext_filter and entry.suffix.lower() != ext_filter:
                    continue
                items.append({"name": entry.name, "path": str(entry)})
            elif browse_type == "all":
                if entry.is_dir():
                    items.append({"name": entry.name, "path": str(entry), "is_dir": True})
                elif entry.is_file():
                    if ext_filter and entry.suffix.lower() != ext_filter:
                        continue
                    items.append({"name": entry.name, "path": str(entry), "is_dir": False})
    except PermissionError:
        return jsonify({"current": str(target), "parent": parent, "items": [], "error": "Permission denied"})

    return jsonify({"current": str(target), "parent": parent, "items": items})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    auto_migrate()
    print(f"Project root: {PROJECT_ROOT}")
    port = int(os.environ.get("GROWUP_PORT", 5001))
    print(f"Starting Growing Up dashboard at http://localhost:{port}")
    app.run(debug=True, host="0.0.0.0", port=port)

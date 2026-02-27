#!/usr/bin/env python3
"""
Growing Up — macOS app launcher.

This is the main entry point that PyInstaller bundles into Growing Up.app.
It sets up user data directories, configures environment variables so the
Flask webapp and pipeline scripts can find everything, then starts the server
and opens the browser.
"""

import json
import os
import shutil
import sys
import threading
import time
import webbrowser
from pathlib import Path


def get_bundle_dir():
    """Return the directory containing bundled resources.

    When frozen by PyInstaller (one-dir mode), sys._MEIPASS points to the
    temporary extraction folder.  For a .app bundle the resources live
    alongside the executable inside Contents/MacOS/ or Contents/Resources/.
    """
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    # Running as plain script (dev mode)
    return Path(__file__).resolve().parent


def get_user_data_dir():
    """Return ~/Documents/Growing Up/ — where subjects and configs live."""
    return Path.home() / "Documents" / "Growing Up"


def ensure_user_data(bundle_dir, user_data_dir):
    """Create user data directory structure on first launch."""
    user_data_dir.mkdir(parents=True, exist_ok=True)
    (user_data_dir / "subjects").mkdir(exist_ok=True)

    # Copy default config.json if missing
    user_config = user_data_dir / "config.json"
    if not user_config.exists():
        bundled_config = bundle_dir / "default_config.json"
        if bundled_config.exists():
            shutil.copy2(str(bundled_config), str(user_config))
        else:
            # Write sensible defaults
            defaults = {
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
            with open(user_config, "w") as f:
                json.dump(defaults, f, indent=2)

    # Create subjects.json if missing
    subjects_json = user_data_dir / "subjects.json"
    if not subjects_json.exists():
        with open(subjects_json, "w") as f:
            json.dump({}, f)


def setup_environment(bundle_dir, user_data_dir):
    """Configure environment variables for the webapp and pipeline scripts."""

    # User data — where subjects, configs, media live
    os.environ["GROWUP_PROJECT_ROOT"] = str(user_data_dir)

    # Code scripts live inside the app bundle
    code_dir = bundle_dir / "Code"
    os.environ["GROWUP_CODE_DIR"] = str(code_dir)

    # Add bundled ffmpeg/ffprobe to PATH
    bin_dir = bundle_dir / "bin"
    if bin_dir.exists():
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


def open_browser_delayed(port, delay=1.5):
    """Open the default browser after a short delay to let Flask start."""
    def _open():
        time.sleep(delay)
        webbrowser.open(f"http://localhost:{port}")
    t = threading.Thread(target=_open, daemon=True)
    t.start()


def main():
    bundle_dir = get_bundle_dir()
    user_data_dir = get_user_data_dir()

    print(f"Growing Up — bundle: {bundle_dir}")
    print(f"Growing Up — user data: {user_data_dir}")

    ensure_user_data(bundle_dir, user_data_dir)
    setup_environment(bundle_dir, user_data_dir)

    # Add the bundled webapp directory to sys.path so Flask can find
    # templates, static files, and pipeline_runner
    webapp_dir = bundle_dir / "webapp"
    sys.path.insert(0, str(webapp_dir))

    # Import the Flask app — must happen AFTER env vars are set so that
    # pipeline_runner picks up GROWUP_PROJECT_ROOT and GROWUP_CODE_DIR
    from app import app, SUBJECTS_DIR, auto_migrate

    # Ensure subjects dir exists and run auto-migration
    SUBJECTS_DIR.mkdir(parents=True, exist_ok=True)
    auto_migrate()

    port = int(os.environ.get("GROWUP_PORT", 5001))
    print(f"Starting Growing Up at http://localhost:{port}")

    open_browser_delayed(port)

    # Run Flask — no debug/reloader in production
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

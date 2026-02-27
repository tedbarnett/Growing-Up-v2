"""
Pipeline runner: executes CLI scripts as subprocesses with per-subject env overrides.
Writes job_status.json for SSE polling.
"""

import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

# Project root (parent of webapp/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = PROJECT_ROOT / "Code"
VENV_PYTHON = str(PROJECT_ROOT / "venv" / "bin" / "python")

# Phase 1: image processing (steps 00-03)
PROCESS_STEPS = [
    {"script": "00_convert_to_png.py", "label": "Converting images to PNG"},
    {"script": "01_detect_faces.py", "label": "Detecting faces"},
    {"script": "02_align_faces.py", "label": "Aligning faces"},
    {"script": "03_sort_images.py", "label": "Sorting chronologically"},
]

# Phase 2: video generation (steps 04-05)
GENERATE_STEPS = [
    {"script": "04_render_morph.py", "label": "Rendering morph frames", "args": ["--crossfade"]},
    {"script": "05_encode_video.py", "label": "Encoding video"},
]

# All steps combined (for backward compat)
PIPELINE_STEPS = PROCESS_STEPS + GENERATE_STEPS

# Track running jobs: subject_name -> threading.Thread
_running_jobs = {}
_job_lock = threading.Lock()


def is_job_running(subject_name):
    with _job_lock:
        thread = _running_jobs.get(subject_name)
        return thread is not None and thread.is_alive()


def get_job_status(subject_dir):
    """Read job_status.json for a subject."""
    status_path = Path(subject_dir) / "job_status.json"
    if status_path.exists():
        try:
            with open(status_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return None


def _write_status(subject_dir, status_dict):
    status_path = Path(subject_dir) / "job_status.json"
    with open(status_path, "w") as f:
        json.dump(status_dict, f, indent=2)


def _build_env(subject_dir, config):
    """Build environment variables for subprocess."""
    import os
    env = os.environ.copy()
    subject_dir = Path(subject_dir)

    env["GROWUP_PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["GROWUP_CONFIG_PATH"] = str(subject_dir / "config.json")
    env["GROWUP_MANIFEST_PATH"] = str(subject_dir / "manifest.json")
    env["GROWUP_ALIGNED_DIR"] = str(subject_dir / "aligned")
    env["GROWUP_FRAMES_DIR"] = str(subject_dir / "frames")
    env["GROWUP_OUTPUT_DIR"] = str(subject_dir / "output")

    # Images dir — absolute path used directly, otherwise relative to Images/
    images_folder = config.get("images_folder", "")
    if images_folder:
        images_path = Path(images_folder)
        if images_path.is_absolute():
            env["GROWUP_IMAGES_DIR"] = str(images_path)
        else:
            env["GROWUP_IMAGES_DIR"] = str(PROJECT_ROOT / "Images" / images_folder)

    return env


def is_processed(subject_dir):
    """Check if Phase 1 (process) has been completed for a subject.

    Returns True if the manifest has a __sequence__ entry with aligned files.
    """
    subject_dir = Path(subject_dir)
    manifest_path = subject_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        seq = manifest.get("__sequence__", [])
        if not seq:
            return False
        # Verify at least one aligned file exists
        aligned_dir = subject_dir / "aligned"
        if not aligned_dir.exists():
            return False
        return any((aligned_dir / fname).exists() for fname in seq[:3])
    except (json.JSONDecodeError, IOError):
        return False


def _run_pipeline(subject_name, subject_dir, config, steps, phase_name, music_path=None):
    """Run pipeline steps sequentially. Called in a background thread."""
    subject_dir = Path(subject_dir)
    steps = list(steps)

    # Add vignette arg to render step if enabled in config
    use_vignette = config.get("vignette", False)

    # Customize step args based on config
    new_steps = []
    for step in steps:
        if step["script"] == "04_render_morph.py" and use_vignette:
            new_steps.append({
                **step,
                "args": step.get("args", []) + ["--vignette"],
            })
        elif step["script"] == "05_encode_video.py" and music_path:
            new_steps.append({
                **step,
                "args": ["--music", str(music_path)],
            })
        else:
            new_steps.append(step)
    steps = new_steps

    total_steps = len(steps)
    env = _build_env(subject_dir, config)
    log_lines = []

    _write_status(subject_dir, {
        "state": "running",
        "phase": phase_name,
        "step_index": 0,
        "total_steps": total_steps,
        "step_label": steps[0]["label"],
        "log_tail": [],
        "error": None,
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
    })

    for i, step in enumerate(steps):
        script_path = str(CODE_DIR / step["script"])
        args = step.get("args", [])
        cmd = [VENV_PYTHON, script_path] + args

        _write_status(subject_dir, {
            "state": "running",
            "phase": phase_name,
            "step_index": i,
            "total_steps": total_steps,
            "step_label": step["label"],
            "log_tail": log_lines[-50:],
            "error": None,
            "started_at": None,
            "finished_at": None,
        })

        log_lines.append(f"--- Step {i+1}/{total_steps}: {step['label']} ---")

        try:
            result = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 min max per step
            )

            stdout_lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
            stderr_lines = result.stderr.strip().split("\n") if result.stderr.strip() else []
            log_lines.extend(stdout_lines[-20:])

            if result.returncode != 0:
                error_msg = result.stderr.strip()[-500:] if result.stderr else f"Exit code {result.returncode}"
                log_lines.append(f"ERROR: {error_msg}")
                _write_status(subject_dir, {
                    "state": "error",
                    "phase": phase_name,
                    "step_index": i,
                    "total_steps": total_steps,
                    "step_label": step["label"],
                    "log_tail": log_lines[-50:],
                    "error": error_msg,
                    "started_at": None,
                    "finished_at": datetime.now().isoformat(),
                })
                return

        except subprocess.TimeoutExpired:
            log_lines.append("ERROR: Step timed out after 30 minutes")
            _write_status(subject_dir, {
                "state": "error",
                "phase": phase_name,
                "step_index": i,
                "total_steps": total_steps,
                "step_label": step["label"],
                "log_tail": log_lines[-50:],
                "error": "Step timed out after 30 minutes",
                "started_at": None,
                "finished_at": datetime.now().isoformat(),
            })
            return
        except Exception as e:
            log_lines.append(f"ERROR: {str(e)}")
            _write_status(subject_dir, {
                "state": "error",
                "phase": phase_name,
                "step_index": i,
                "total_steps": total_steps,
                "step_label": step["label"],
                "log_tail": log_lines[-50:],
                "error": str(e),
                "started_at": None,
                "finished_at": datetime.now().isoformat(),
            })
            return

    log_lines.append(f"{phase_name.title()} phase complete!")
    _write_status(subject_dir, {
        "state": "complete",
        "phase": phase_name,
        "step_index": total_steps - 1,
        "total_steps": total_steps,
        "step_label": "Done",
        "log_tail": log_lines[-50:],
        "error": None,
        "started_at": None,
        "finished_at": datetime.now().isoformat(),
    })

    with _job_lock:
        _running_jobs.pop(subject_name, None)


def _start_job(subject_name, subject_dir, config, steps, phase_name, music_path=None):
    """Launch a pipeline phase in a background thread. Returns True if started."""
    if is_job_running(subject_name):
        return False

    thread = threading.Thread(
        target=_run_pipeline,
        args=(subject_name, subject_dir, config, steps, phase_name, music_path),
        daemon=True,
    )

    with _job_lock:
        _running_jobs[subject_name] = thread

    thread.start()
    return True


def start_process(subject_name, subject_dir, config):
    """Launch Phase 1 (process images: convert, detect, align, sort)."""
    return _start_job(subject_name, subject_dir, config, PROCESS_STEPS, "process")


def start_generate(subject_name, subject_dir, config, music_path=None):
    """Launch Phase 2 (generate video: morph, encode)."""
    return _start_job(subject_name, subject_dir, config, GENERATE_STEPS, "generate", music_path)


def start_pipeline(subject_name, subject_dir, config, music_path=None):
    """Launch full pipeline (all steps). Kept for backward compatibility."""
    return _start_job(subject_name, subject_dir, config, PIPELINE_STEPS, "full", music_path)

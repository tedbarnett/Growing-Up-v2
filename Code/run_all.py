#!/usr/bin/env python3
"""
Convenience script to run the full pipeline in sequence.

Usage:
    python Code/run_all.py
    python Code/run_all.py --crossfade     # Use crossfade instead of morph
    python Code/run_all.py --skip-detect    # Skip face detection (reuse manifest)
    python Code/run_all.py --skip-to-morph  # Skip to morphing step
"""

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT, OUTPUT_DIR, load_config

VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"
SCRIPTS_DIR = PROJECT_ROOT / "Code"


def run_step(script_name, extra_args=None):
    """Run a pipeline script and check for errors."""
    script_path = SCRIPTS_DIR / script_name
    cmd = [str(VENV_PYTHON), str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nERROR: {script_name} failed with exit code {result.returncode}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run full pipeline")
    parser.add_argument("--crossfade", action="store_true",
                        help="Use crossfade instead of Delaunay morph")
    parser.add_argument("--skip-detect", action="store_true",
                        help="Skip face detection (reuse existing manifest)")
    parser.add_argument("--skip-to-morph", action="store_true",
                        help="Skip to morph rendering step")
    parser.add_argument("--reference", nargs="*",
                        help="Reference photos for face detection")
    parser.add_argument("--add-year-labels", action="store_true",
                        help="Add year overlay to final video")
    args = parser.parse_args()

    print("Growing Up Ted - Full Pipeline")
    print("=" * 60)

    if not args.skip_detect and not args.skip_to_morph:
        detect_args = []
        if args.reference:
            detect_args.extend(["--reference"] + args.reference)
        run_step("01_detect_faces.py", detect_args)

    if not args.skip_to_morph:
        run_step("02_align_faces.py")
        run_step("03_sort_images.py")

    morph_args = []
    if args.crossfade:
        morph_args.append("--crossfade")
    run_step("04_render_morph.py", morph_args)

    encode_args = []
    if args.add_year_labels:
        encode_args.append("--add-year-labels")
    run_step("05_encode_video.py", encode_args)

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print(f"\nOpen the video:")
    config = load_config()
    subject = config.get("subject_name", "Video")
    print(f'  open "{OUTPUT_DIR / f"Growing Up - {subject}.mp4"}"')


if __name__ == "__main__":
    main()

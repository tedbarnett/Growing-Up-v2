#!/usr/bin/env python3
"""
Convert all JPEG images to PNG format.
Skips images that already have a PNG counterpart.
After conversion, preserves the original's mtime and moves the JPEG to deleted/.

Usage:
    python Code/00_convert_to_png.py
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import IMAGES_DIR


def _preserve_dates(src: Path, dst: Path):
    """Copy mtime from src to dst and attempt to copy macOS creation date via SetFile."""
    # Preserve mtime
    stat = src.stat()
    os.utime(dst, (stat.st_atime, stat.st_mtime))
    # Attempt macOS creation date via SetFile (best-effort)
    try:
        # Get creation date from source
        result = subprocess.run(
            ["GetFileInfo", "-d", str(src)],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            subprocess.run(
                ["SetFile", "-d", result.stdout.strip(), str(dst)],
                capture_output=True, timeout=5,
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # SetFile/GetFileInfo not available


def main():
    # Find all JPEG files
    jpegs = sorted([
        f for f in IMAGES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in ('.jpeg', '.jpg')
    ])

    # Find existing PNG stems
    png_stems = set(
        f.stem for f in IMAGES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() == '.png'
    )

    need_convert = [f for f in jpegs if f.stem not in png_stems]
    already_have = [f for f in jpegs if f.stem in png_stems]

    print(f"Total JPEG files: {len(jpegs)}")
    print(f"Already have PNG counterpart: {len(already_have)}")
    print(f"Need conversion: {len(need_convert)}")

    # Create deleted/ subfolder for originals
    deleted_dir = IMAGES_DIR / "deleted"
    deleted_dir.mkdir(exist_ok=True)

    if not need_convert:
        print("\nAll JPEGs already have PNG versions. Nothing to do.")
        # Move already-converted originals to deleted/ if still present
        moved = 0
        for jpeg_path in already_have:
            dest = deleted_dir / jpeg_path.name
            if not dest.exists():
                shutil.move(str(jpeg_path), str(dest))
                moved += 1
        if moved:
            print(f"Moved {moved} already-converted JPEG originals to deleted/")
        return

    print(f"\nConverting {len(need_convert)} JPEGs to PNG...")
    converted = 0
    failed = 0

    for jpeg_path in need_convert:
        png_path = jpeg_path.with_suffix('.png')
        try:
            img = Image.open(jpeg_path)
            # Preserve EXIF data if available
            exif = img.info.get('exif')
            if exif:
                img.save(png_path, 'PNG', exif=exif)
            else:
                img.save(png_path, 'PNG')
            # Preserve filesystem dates from original
            _preserve_dates(jpeg_path, png_path)
            # Move original to deleted/
            shutil.move(str(jpeg_path), str(deleted_dir / jpeg_path.name))
            converted += 1
            print(f"  [{converted}/{len(need_convert)}] {jpeg_path.name} -> {png_path.name}")
        except Exception as e:
            print(f"  FAILED: {jpeg_path.name}: {e}")
            failed += 1

    # Also move already-converted originals
    moved = 0
    for jpeg_path in already_have:
        dest = deleted_dir / jpeg_path.name
        if not dest.exists():
            shutil.move(str(jpeg_path), str(dest))
            moved += 1

    print(f"\nDone! Converted: {converted}, Failed: {failed}")
    if moved:
        print(f"Moved {moved} already-converted JPEG originals to deleted/")
    print(f"\nTotal PNG files now: {len(png_stems) + converted}")


if __name__ == "__main__":
    main()

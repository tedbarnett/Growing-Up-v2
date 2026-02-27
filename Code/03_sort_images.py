#!/usr/bin/env python3
"""
Step 5: Sort aligned face images chronologically.

Extracts dates from filenames and EXIF data, then creates an ordered
sequence in the manifest.

Usage:
    python Code/03_sort_images.py [--manual-overrides overrides.json]
"""

import argparse
import re
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import PROJECT_ROOT, IMAGES_DIR, ALIGNED_DIR, load_manifest, save_manifest


def extract_year_from_filename(filename):
    """
    Try to extract a year from the filename.

    Handles patterns like:
        "1963 Ted as baby.png" -> 1963
        "Photo scan 1968.jpg" -> 1968
        "1971_00039_s_w18alpyxpwb0039.png" -> 1971
        "IMG_1234.png" -> None (no year info)
        "2008-06-15 vacation.jpg" -> 2008
    """
    name = Path(filename).stem

    # Pattern 1: Year at the start of filename
    m = re.match(r'^(\d{4})\b', name)
    if m:
        year = int(m.group(1))
        if 1940 <= year <= 2030:
            return year, extract_date_detail(name, year)

    # Pattern 2: Year anywhere in the filename
    years = re.findall(r'\b(19\d{2}|20[0-2]\d)\b', name)
    if years:
        year = int(years[0])
        return year, extract_date_detail(name, year)

    return None, None


def extract_date_detail(name, year):
    """
    Try to extract month/day in addition to year for finer sorting.

    Returns a sortable string like "1963-01-01" or "1963-06-15"
    """
    # Try full date patterns
    # YYYY-MM-DD
    m = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', name)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # YYYY_MM_DD
    m = re.search(r'(\d{4})_(\d{1,2})_(\d{1,2})', name)
    if m and 1 <= int(m.group(2)) <= 12:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    # Just year - default to mid-year for sorting
    return f"{year:04d}-06-15"


def extract_exif_date(filepath):
    """Try to extract date from EXIF data."""
    try:
        from PIL import Image
        from PIL.ExifTags import Base as ExifBase

        img = Image.open(filepath)
        exif_data = img._getexif()
        if exif_data:
            # DateTimeOriginal (36867) or DateTime (306)
            for tag in [36867, 306, 36868]:
                if tag in exif_data:
                    date_str = exif_data[tag]
                    # Format: "YYYY:MM:DD HH:MM:SS"
                    try:
                        dt = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                        return dt.year, dt.strftime("%Y-%m-%d")
                    except (ValueError, TypeError):
                        pass
    except Exception:
        pass
    return None, None


def sort_manifest(manifest, manual_overrides=None):
    """
    Sort manifest entries chronologically.

    Returns:
        list of (filename, sort_date, year, source) tuples
    """
    entries = []

    for filename, data in manifest.items():
        if "aligned_file" not in data:
            continue  # Skip entries without aligned images

        year = None
        sort_date = None
        source = "unknown"

        # Check manual overrides first
        if manual_overrides and filename in manual_overrides:
            override = manual_overrides[filename]
            year = override.get("year")
            sort_date = override.get("date", f"{year:04d}-06-15" if year else None)
            source = "manual"

        # Try filename
        if year is None:
            year, sort_date = extract_year_from_filename(filename)
            if year:
                source = "filename"

        # Try EXIF
        if year is None:
            img_path = IMAGES_DIR / filename
            if img_path.exists():
                year, sort_date = extract_exif_date(img_path)
                if year:
                    source = "exif"

        # Default: put unknowns at the end
        if year is None:
            year = 9999
            sort_date = "9999-06-15"
            source = "unknown"

        entries.append((filename, sort_date, year, source))

    # Sort by date string (which handles year + month + day)
    entries.sort(key=lambda x: (x[1] or "9999-06-15", x[0]))

    return entries


def main():
    parser = argparse.ArgumentParser(description="Sort images chronologically")
    parser.add_argument("--manual-overrides", type=str, default=None,
                        help="JSON file with manual date overrides")
    args = parser.parse_args()

    manifest = load_manifest()
    if not manifest:
        print("ERROR: manifest.json not found or empty.")
        print("Run 01_detect_faces.py and 02_align_faces.py first.")
        sys.exit(1)

    # Load manual overrides if provided
    overrides = None
    if args.manual_overrides:
        with open(args.manual_overrides) as f:
            overrides = json.load(f)
        print(f"Loaded {len(overrides)} manual overrides.")

    # Sort
    sorted_entries = sort_manifest(manifest, overrides)

    # Display results
    print(f"\nChronological order ({len(sorted_entries)} images):")
    print("-" * 70)

    unknown_count = 0
    year_counts = {}

    for i, (filename, sort_date, year, source) in enumerate(sorted_entries):
        if year == 9999:
            unknown_count += 1
        else:
            year_counts[year] = year_counts.get(year, 0) + 1
        print(f"  {i+1:3d}. [{sort_date or '????-??-??'}] {filename}  ({source})")

    # Store order in manifest
    sequence = []
    for i, (filename, sort_date, year, source) in enumerate(sorted_entries):
        manifest[filename]["sort_order"] = i
        manifest[filename]["sort_date"] = sort_date
        manifest[filename]["sort_year"] = year if year != 9999 else None
        manifest[filename]["date_source"] = source
        sequence.append(filename)

    # Save ordered sequence as a separate key
    manifest["__sequence__"] = sequence
    save_manifest(manifest)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Total images: {len(sorted_entries)}")
    print(f"  Dated: {len(sorted_entries) - unknown_count}")
    print(f"  Unknown date: {unknown_count}")

    if year_counts:
        min_year = min(year_counts.keys())
        max_year = max(year_counts.keys())
        print(f"  Year range: {min_year} - {max_year}")
        print(f"\n  Images per decade:")
        decades = {}
        for y, c in year_counts.items():
            decade = (y // 10) * 10
            decades[decade] = decades.get(decade, 0) + c
        for decade in sorted(decades.keys()):
            bar = "#" * decades[decade]
            print(f"    {decade}s: {decades[decade]:3d} {bar}")

    if unknown_count > 0:
        print(f"\n  WARNING: {unknown_count} images have no date.")
        print("  These will appear at the END of the video.")
        print("  Create a manual overrides JSON to fix this:")
        print('    {"filename.jpg": {"year": 1975, "date": "1975-06-15"}}')

    print(f"\nManifest updated with sort order.")
    print(f"Next step: python Code/04_render_morph.py")


if __name__ == "__main__":
    main()

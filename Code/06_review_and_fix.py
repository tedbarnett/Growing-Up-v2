#!/usr/bin/env python3
"""
Review and fix pipeline: de-duplicate images, filter bad detections,
generate a contact sheet for visual review, and apply exclusions.

Usage:
    python Code/06_review_and_fix.py --contact-sheet   # Generate visual review
    python Code/06_review_and_fix.py --dedupe           # Remove .jpeg/.png duplicates
    python Code/06_review_and_fix.py --filter            # Apply quality filters
    python Code/06_review_and_fix.py --all               # Do everything
"""

import argparse
import json
import sys
import math
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    PROJECT_ROOT, ALIGNED_DIR, OUTPUT_DIR, load_config, load_manifest,
    save_manifest
)


def find_duplicates(manifest):
    """
    Find .jpeg/.png duplicate pairs.
    Returns list of filenames to remove (keeps .png over .jpeg).
    """
    # Group by stem (filename without extension)
    stems = defaultdict(list)
    for filename in manifest:
        if filename.startswith("__"):
            continue
        stem = Path(filename).stem
        stems[stem].append(filename)

    to_remove = []
    for stem, files in stems.items():
        if len(files) > 1:
            # Keep .png, remove .jpeg/.jpg
            pngs = [f for f in files if f.lower().endswith('.png')]
            others = [f for f in files if not f.lower().endswith('.png')]

            if pngs:
                # Keep the first .png, remove all others
                to_remove.extend(others)
                to_remove.extend(pngs[1:])  # Keep only first .png if multiple
            else:
                # No .png, keep first file
                to_remove.extend(others[1:])

    return to_remove


def filter_bad_detections(manifest, config):
    """
    Filter out images with bad face detections:
    - Low similarity scores (wrong person)
    - Very small faces (blurry when zoomed)
    - Misaligned eyes (landmark detection failed)

    Returns list of filenames to exclude with reasons.
    """
    output_size = config.get("output_size", 1024)
    eye_left_target = config.get("eye_left_target", [0.35, 0.40])
    eye_right_target = config.get("eye_right_target", [0.65, 0.40])

    # Target eye positions in pixels
    target_lx = eye_left_target[0] * output_size
    target_ly = eye_left_target[1] * output_size
    target_rx = eye_right_target[0] * output_size
    target_ry = eye_right_target[1] * output_size
    target_eye_dist = math.sqrt((target_rx - target_lx)**2 + (target_ry - target_ly)**2)

    exclusions = []

    for filename, entry in manifest.items():
        if filename.startswith("__"):
            continue
        if "aligned_file" not in entry:
            continue

        # Skip images that were manually reviewed and accepted
        if entry.get("manually_reviewed") and not entry.get("excluded"):
            continue

        reasons = []

        # 1. Check similarity score
        sim = entry.get("similarity", 0)
        if sim < 0.35:
            reasons.append(f"low_similarity({sim:.2f})")

        # 2. Check face size relative to image
        bbox = entry.get("bbox", [0, 0, 0, 0])
        img_size = entry.get("image_size", [1, 1])
        face_w = bbox[2] - bbox[0]
        face_h = bbox[3] - bbox[1]
        face_area = face_w * face_h
        img_area = img_size[0] * img_size[1]
        face_ratio = face_area / img_area if img_area > 0 else 0

        # Also check absolute face pixel size
        face_pixels = max(face_w, face_h)
        if face_pixels < 80:
            reasons.append(f"tiny_face({face_pixels:.0f}px)")

        # 3. Check eye alignment quality (after alignment)
        aligned_iris_l = entry.get("aligned_iris_left")
        aligned_iris_r = entry.get("aligned_iris_right")

        if aligned_iris_l and aligned_iris_r:
            # Check how far eyes are from target positions
            l_err = math.sqrt((aligned_iris_l[0] - target_lx)**2 +
                              (aligned_iris_l[1] - target_ly)**2)
            r_err = math.sqrt((aligned_iris_r[0] - target_rx)**2 +
                              (aligned_iris_r[1] - target_ry)**2)

            # Allow up to 10% of eye distance as error
            max_err = target_eye_dist * 0.15
            if l_err > max_err or r_err > max_err:
                reasons.append(f"eye_misalign(L:{l_err:.0f},R:{r_err:.0f})")

            # Check if eyes are at reasonable distance from each other
            actual_dist = math.sqrt(
                (aligned_iris_r[0] - aligned_iris_l[0])**2 +
                (aligned_iris_r[1] - aligned_iris_l[1])**2
            )
            dist_ratio = actual_dist / target_eye_dist if target_eye_dist > 0 else 0
            if dist_ratio < 0.5 or dist_ratio > 1.5:
                reasons.append(f"bad_eye_dist({dist_ratio:.2f}x)")

        if reasons:
            exclusions.append((filename, reasons))

    return exclusions


def generate_contact_sheet(manifest, output_path, cols=15, thumb_size=128):
    """
    Generate a contact sheet of all aligned faces for visual review.
    Marks excluded/low-quality images with red borders.
    """
    sequence = manifest.get("__sequence__", [])
    if not sequence:
        # Use all entries
        sequence = [k for k in manifest if not k.startswith("__") and "aligned_file" in manifest[k]]

    # Filter to entries with aligned files
    valid = []
    for filename in sequence:
        entry = manifest.get(filename, {})
        if "aligned_file" in entry:
            valid.append(filename)

    if not valid:
        print("No aligned images found!")
        return

    rows = math.ceil(len(valid) / cols)
    sheet_w = cols * thumb_size
    sheet_h = rows * (thumb_size + 20)  # Extra space for labels
    sheet = np.ones((sheet_h, sheet_w, 3), dtype=np.uint8) * 40  # Dark gray bg

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.3
    font_thickness = 1

    for i, filename in enumerate(valid):
        entry = manifest[filename]
        aligned_path = ALIGNED_DIR / entry["aligned_file"]

        row = i // cols
        col = i % cols
        x = col * thumb_size
        y = row * (thumb_size + 20)

        # Load and resize aligned face
        img = cv2.imread(str(aligned_path))
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_size, thumb_size))

        # Color-code border by quality
        sim = entry.get("similarity", 0)
        excluded = entry.get("excluded", False)

        if excluded:
            border_color = (0, 0, 255)  # Red = excluded
            border_w = 3
        elif sim < 0.35:
            border_color = (0, 0, 200)  # Dark red = very low similarity
            border_w = 2
        elif sim < 0.50:
            border_color = (0, 165, 255)  # Orange = low similarity
            border_w = 2
        elif sim == 0.0:
            border_color = (0, 0, 255)  # Red = fallback (no match)
            border_w = 2
        else:
            border_color = (0, 200, 0)  # Green = good match
            border_w = 1

        cv2.rectangle(thumb, (0, 0), (thumb_size - 1, thumb_size - 1),
                       border_color, border_w)

        # Place thumbnail
        sheet[y:y + thumb_size, x:x + thumb_size] = thumb

        # Add year label below
        year = entry.get("sort_year", "?")
        label = str(year) if year else "?"
        cv2.putText(sheet, label, (x + 2, y + thumb_size + 14),
                     font, font_scale, (200, 200, 200), font_thickness)

        # Add index number
        cv2.putText(sheet, str(i + 1), (x + 2, y + 12),
                     font, font_scale, (255, 255, 0), font_thickness)

    # Save
    cv2.imwrite(str(output_path), sheet)
    print(f"Contact sheet saved: {output_path}")
    print(f"  {len(valid)} faces, {cols} columns x {rows} rows")
    print(f"  Green border = good match, Orange = uncertain, Red = bad/excluded")

    return output_path


def apply_exclusions(manifest, to_exclude):
    """Mark files as excluded in manifest and remove from sequence."""
    exclude_set = set(to_exclude)

    for filename in exclude_set:
        if filename in manifest:
            manifest[filename]["excluded"] = True

    # Update sequence
    if "__sequence__" in manifest:
        manifest["__sequence__"] = [
            f for f in manifest["__sequence__"]
            if f not in exclude_set
        ]

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Review and fix face detections")
    parser.add_argument("--contact-sheet", action="store_true",
                        help="Generate contact sheet for visual review")
    parser.add_argument("--dedupe", action="store_true",
                        help="Remove .jpeg/.png duplicates from sequence")
    parser.add_argument("--filter", action="store_true",
                        help="Apply quality filters (similarity, face size, alignment)")
    parser.add_argument("--all", action="store_true",
                        help="Run all fixes")
    parser.add_argument("--exclude", nargs="*",
                        help="Manually exclude specific filenames")
    parser.add_argument("--sim-threshold", type=float, default=0.35,
                        help="Minimum similarity score (default: 0.35)")
    parser.add_argument("--min-face-pixels", type=int, default=80,
                        help="Minimum face size in pixels (default: 80)")
    args = parser.parse_args()

    if args.all:
        args.dedupe = True
        args.filter = True
        args.contact_sheet = True

    config = load_config()
    manifest = load_manifest()

    if not manifest:
        print("ERROR: No manifest found. Run 01_detect_faces.py first.")
        sys.exit(1)

    total_before = len([f for f in manifest.get("__sequence__", [])
                        if not manifest.get(f, {}).get("excluded")])
    all_exclusions = set()

    # Step 1: De-duplicate
    if args.dedupe:
        print("=" * 60)
        print("DE-DUPLICATING .jpeg/.png pairs...")
        dupes = find_duplicates(manifest)
        print(f"  Found {len(dupes)} duplicates to remove:")
        for d in sorted(dupes)[:20]:
            print(f"    - {d}")
        if len(dupes) > 20:
            print(f"    ... and {len(dupes) - 20} more")
        all_exclusions.update(dupes)

    # Step 2: Quality filter
    if args.filter:
        print("\n" + "=" * 60)
        print("FILTERING bad detections...")
        exclusions = filter_bad_detections(manifest, config)
        print(f"  Found {len(exclusions)} problematic images:")
        for filename, reasons in sorted(exclusions):
            reason_str = ", ".join(reasons)
            print(f"    - {filename}: {reason_str}")
        all_exclusions.update(f for f, _ in exclusions)

    # Step 3: Manual exclusions
    if args.exclude:
        print(f"\n  Adding {len(args.exclude)} manual exclusions")
        all_exclusions.update(args.exclude)

    # Apply all exclusions
    if all_exclusions:
        print(f"\n{'='*60}")
        print(f"APPLYING {len(all_exclusions)} total exclusions...")
        manifest = apply_exclusions(manifest, all_exclusions)
        save_manifest(manifest)

        total_after = len(manifest.get("__sequence__", []))
        print(f"  Before: {total_before} images")
        print(f"  After: {total_after} images")
        print(f"  Removed: {total_before - total_after}")

    # Step 4: Contact sheet
    if args.contact_sheet:
        print(f"\n{'='*60}")
        print("GENERATING contact sheet...")
        OUTPUT_DIR.mkdir(exist_ok=True)
        sheet_path = OUTPUT_DIR / "contact_sheet.png"
        generate_contact_sheet(manifest, sheet_path)
        print(f"\nOpen it with:")
        print(f'  open "{sheet_path}"')

    # Summary
    final_count = len(manifest.get("__sequence__", []))
    print(f"\n{'='*60}")
    print(f"SUMMARY: {final_count} images in final sequence")
    print(f"\nTo re-render: python Code/04_render_morph.py --crossfade")
    print(f"Then encode: python Code/05_encode_video.py --add-year-labels")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 3+4: Align and crop faces using eye positions from the manifest.

Applies an affine transform to each image so that the pupils are at
fixed positions, then crops to a square output.

Usage:
    python Code/02_align_faces.py [--size 1024] [--preview]
"""

import argparse
import sys
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    PROJECT_ROOT, IMAGES_DIR, ALIGNED_DIR, load_config, load_manifest, save_manifest,
    compute_affine_transform, apply_affine_transform
)


def transform_landmarks(landmarks, M):
    """Apply affine transform M to a list of [x,y] landmark points."""
    pts = np.array(landmarks, dtype=np.float64)
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])  # Nx3
    transformed = (M @ pts_h.T).T  # Nx2
    return transformed.tolist()


def fix_red_eye(image, iris_center, radius=None):
    """
    Remove red-eye at a given iris position by desaturating red pixels
    in a small circular region around the pupil.

    Parameters:
        image: BGR uint8 image
        iris_center: [x, y] position of the iris
        radius: pixel radius to check (auto-calculated if None)
    """
    h, w = image.shape[:2]
    cx, cy = int(iris_center[0]), int(iris_center[1])

    if radius is None:
        # Estimate pupil radius as ~2% of image width
        radius = max(int(w * 0.02), 5)

    # Define bounding box
    x1 = max(0, cx - radius)
    y1 = max(0, cy - radius)
    x2 = min(w, cx + radius)
    y2 = min(h, cy + radius)

    roi = image[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return image

    # Detect red pixels: high red channel, low green and blue
    b, g, r = roi[:, :, 0].astype(float), roi[:, :, 1].astype(float), roi[:, :, 2].astype(float)

    # Red-eye has high red relative to green and blue
    denom = (g + b + 1)
    red_ratio = r / denom
    is_red = (red_ratio > 0.8) & (r > 80)

    # Create circular mask
    yy, xx = np.ogrid[y1:y2, x1:x2]
    circle_mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2

    # Combine: pixel must be red AND within the circle
    fix_mask = is_red & circle_mask

    if not np.any(fix_mask):
        return image

    # Replace red channel with average of green and blue (desaturate)
    avg = ((g + b) / 2).astype(np.uint8)
    roi[:, :, 2] = np.where(fix_mask, avg, roi[:, :, 2])
    image[y1:y2, x1:x2] = roi

    return image


def align_single_face(img_path, entry, config):
    """
    Load an image, align the face, crop, and return the result
    along with transformed landmarks.

    Returns:
        (aligned_image, transformed_landmarks_68, transformed_iris_left, transformed_iris_right)
        or None if failed
    """
    output_size = config["output_size"]
    eye_left_target = config["eye_left_target"]
    eye_right_target = config["eye_right_target"]

    # Load image at full resolution
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    # Scale coordinates from detection space to full-resolution space.
    # 01_detect_faces.py resizes images to max_dim=2048 before detection,
    # so manifest coordinates may be in a smaller coordinate space.
    actual_h, actual_w = img.shape[:2]
    stored_size = entry.get("image_size", [actual_w, actual_h])
    stored_w, stored_h = stored_size[0], stored_size[1]
    scale_x = actual_w / stored_w if stored_w > 0 else 1.0
    scale_y = actual_h / stored_h if stored_h > 0 else 1.0

    # Get eye positions (prefer iris, fall back to landmark averages)
    left_eye = entry.get("iris_left")
    right_eye = entry.get("iris_right")

    if left_eye is None or right_eye is None:
        lm = entry["landmarks_68"]
        left_eye = np.mean(lm[36:42], axis=0).tolist()
        right_eye = np.mean(lm[42:48], axis=0).tolist()

    # Scale eye positions to full resolution
    left_eye = [left_eye[0] * scale_x, left_eye[1] * scale_y]
    right_eye = [right_eye[0] * scale_x, right_eye[1] * scale_y]

    # Scale all landmarks to full resolution
    landmarks_scaled = [
        [p[0] * scale_x, p[1] * scale_y] for p in entry["landmarks_68"]
    ]

    # Compute affine transform
    M = compute_affine_transform(
        left_eye, right_eye,
        output_size, eye_left_target, eye_right_target
    )

    # Apply transform
    aligned = apply_affine_transform(img, M, output_size)

    # Transform landmarks too (for morphing later)
    lm_transformed = transform_landmarks(landmarks_scaled, M)
    iris_left_t = transform_landmarks([left_eye], M)[0]
    iris_right_t = transform_landmarks([right_eye], M)[0]

    return aligned, lm_transformed, iris_left_t, iris_right_t


def main():
    parser = argparse.ArgumentParser(description="Align and crop faces")
    parser.add_argument("--size", type=int, default=None,
                        help="Output size in pixels (default: from config)")
    parser.add_argument("--preview", action="store_true",
                        help="Show preview of first 5 aligned faces")
    args = parser.parse_args()

    config = load_config()
    if args.size:
        config["output_size"] = args.size

    manifest = load_manifest()
    if not manifest:
        print("ERROR: manifest.json not found or empty.")
        print("Run 01_detect_faces.py first.")
        sys.exit(1)

    ALIGNED_DIR.mkdir(exist_ok=True)
    output_size = config["output_size"]
    total = len(manifest)
    succeeded = 0
    failed = 0

    print(f"Aligning {total} faces to {output_size}x{output_size}...")
    print(f"Eye targets: left={config['eye_left_target']}, right={config['eye_right_target']}")

    for i, (filename, entry) in enumerate(manifest.items()):
        print(f"[{i+1}/{total}] {filename}...", end=" ")

        img_path = IMAGES_DIR / filename
        if not img_path.exists():
            print("SKIP (file not found)")
            failed += 1
            continue

        result = align_single_face(img_path, entry, config)
        if result is None:
            print("SKIP (could not load)")
            failed += 1
            continue

        aligned, lm_transformed, iris_left_t, iris_right_t = result

        # Save aligned image as PNG for quality
        out_name = Path(filename).stem + ".png"
        out_path = ALIGNED_DIR / out_name
        cv2.imwrite(str(out_path), aligned)

        # Update manifest with aligned landmark positions
        entry["aligned_file"] = out_name
        entry["aligned_landmarks_68"] = [
            [round(p[0], 1), round(p[1], 1)] for p in lm_transformed
        ]
        entry["aligned_iris_left"] = [round(v, 1) for v in iris_left_t]
        entry["aligned_iris_right"] = [round(v, 1) for v in iris_right_t]

        succeeded += 1
        print("OK")

    save_manifest(manifest)

    print(f"\n{'='*60}")
    print(f"RESULTS: {succeeded}/{total} faces aligned successfully")
    print(f"Failed: {failed}")
    print(f"Output: {ALIGNED_DIR}/")

    if args.preview and succeeded > 0:
        print("\nShowing preview of first 5 aligned faces...")
        preview_files = sorted(ALIGNED_DIR.glob("*.png"))[:5]
        for pf in preview_files:
            img = cv2.imread(str(pf))
            # Draw crosshairs at eye target positions
            lx = int(config["eye_left_target"][0] * output_size)
            ly = int(config["eye_left_target"][1] * output_size)
            rx = int(config["eye_right_target"][0] * output_size)
            ry = int(config["eye_right_target"][1] * output_size)
            cv2.drawMarker(img, (lx, ly), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.drawMarker(img, (rx, ry), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            cv2.imshow(f"Aligned: {pf.name}", img)

        print("Press any key to close preview windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"\nNext step: python Code/03_sort_images.py")


if __name__ == "__main__":
    main()

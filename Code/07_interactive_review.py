#!/usr/bin/env python3
"""
Interactive face review: shows uncertain/failed face detections and lets
you pick the correct face or skip the image.

For each uncertain image, displays all detected faces numbered.
You type the face number to select, or 's' to skip, or 'q' to quit.

Usage:
    python Code/07_interactive_review.py
    python Code/07_interactive_review.py --threshold 0.5   # Review faces below this similarity
    python Code/07_interactive_review.py --review-all       # Review every single image
"""

import argparse
import json
import sys
import types
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    PROJECT_ROOT, IMAGES_DIR, ALIGNED_DIR, load_config, load_manifest, save_manifest,
    compute_affine_transform, apply_affine_transform, IMAGE_EXTENSIONS
)

# Mediapipe landmarks mapping (same as 01_detect_faces.py)
MEDIAPIPE_TO_68 = {
    0: 10, 1: 338, 2: 297, 3: 332, 4: 284,
    5: 251, 6: 389, 7: 356, 8: 454, 9: 323,
    10: 361, 11: 288, 12: 397, 13: 365, 14: 379,
    15: 378, 16: 400,
    17: 70, 18: 63, 19: 105, 20: 66, 21: 107,
    22: 336, 23: 296, 24: 334, 25: 293, 26: 300,
    27: 168, 28: 197, 29: 5, 30: 4,
    31: 75, 32: 97, 33: 2, 34: 326, 35: 305,
    36: 33, 37: 160, 38: 158, 39: 133, 40: 153, 41: 144,
    42: 362, 43: 385, 44: 387, 45: 263, 46: 373, 47: 380,
    48: 61, 49: 40, 50: 37, 51: 0, 52: 267, 53: 270,
    54: 291, 55: 321, 56: 314, 57: 17, 58: 84, 59: 91,
    60: 78, 61: 82, 62: 13, 63: 312, 64: 308,
    65: 317, 66: 14, 67: 87,
}
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473


def init_models():
    """Initialize insightface and mediapipe."""
    # Stub insightface cython
    for mod_name in [
        'insightface.thirdparty.face3d',
        'insightface.thirdparty.face3d.mesh',
        'insightface.thirdparty.face3d.mesh.cython',
        'insightface.thirdparty.face3d.mesh.cython.mesh_core_cython',
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    from insightface.app import FaceAnalysis
    import mediapipe as mp

    print("Loading insightface...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("Loading mediapipe...")
    model_path = str(Path(__file__).resolve().parent / "face_landmarker_v2_with_blendshapes.task")
    vision = mp.tasks.vision
    opts = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        num_faces=10,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    landmarker = vision.FaceLandmarker.create_from_options(opts)

    return app, landmarker


def load_image(path, max_dim=2048):
    """Load image, resize if needed."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def extract_landmarks(landmarker, img_rgb, face_bbox=None):
    """Extract mediapipe landmarks for a face region."""
    import mediapipe as mp

    h, w = img_rgb.shape[:2]
    if face_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        crop = img_rgb[y1:y2, x1:x2]
        ox, oy = x1, y1
    else:
        crop = img_rgb
        ox, oy = 0, 0

    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return None

    crop_c = np.ascontiguousarray(crop, dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_c)
    results = landmarker.detect(mp_image)

    if not results.face_landmarks:
        return None

    face_lm = results.face_landmarks[0]
    all_lm = [[lm.x * cw + ox, lm.y * ch + oy] for lm in face_lm]

    lm68 = []
    for i in range(68):
        idx = MEDIAPIPE_TO_68[i]
        lm68.append(all_lm[idx] if idx < len(all_lm) else [0, 0])

    iris_l = all_lm[LEFT_IRIS_CENTER] if len(all_lm) > LEFT_IRIS_CENTER else None
    iris_r = all_lm[RIGHT_IRIS_CENTER] if len(all_lm) > RIGHT_IRIS_CENTER else None

    return {"landmarks_68": lm68, "iris_left": iris_l, "iris_right": iris_r}


def show_faces_for_review(img_bgr, faces, filename, current_idx=None):
    """
    Display image with numbered face boxes. Returns user's choice.

    Returns:
        int: face index chosen (0-based)
        -1: skip this image
        -2: quit
    """
    display = img_bgr.copy()
    h, w = display.shape[:2]

    # Scale for display
    max_display = 800
    scale = min(max_display / w, max_display / h, 1.0)
    if scale < 1.0:
        display = cv2.resize(display, (int(w * scale), int(h * scale)))

    # Draw face boxes with numbers
    for i, face in enumerate(faces):
        bbox = (face.bbox * scale).astype(int)
        color = (0, 255, 0) if i == current_idx else (0, 165, 255)
        thickness = 3 if i == current_idx else 2
        cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

        # Face number label
        label = f"#{i+1}"
        if i == current_idx:
            label += " (auto)"
        label_pos = (bbox[0], bbox[1] - 10)
        cv2.putText(display, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                     0.8, color, 2)

        # Similarity score
        sim = getattr(face, '_review_sim', None)
        if sim is not None:
            sim_label = f"sim:{sim:.2f}"
            cv2.putText(display, sim_label, (bbox[0], bbox[3] + 20),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Title
    cv2.putText(display, filename, (10, 30),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, "Enter face # (or 's'=skip, 'q'=quit)", (10, 60),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Face Review", display)

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            return -2
        elif key == ord('s'):
            return -1
        elif key == ord('\r') or key == ord('\n'):
            # Enter = accept current auto-selection
            if current_idx is not None:
                return current_idx
            continue
        elif ord('1') <= key <= ord('9'):
            idx = key - ord('1')
            if idx < len(faces):
                return idx
        elif key == 27:  # ESC
            return -2


def align_and_save(img_bgr, face, landmarks, filename, config):
    """Align face and save to aligned directory."""
    output_size = config["output_size"]
    eye_left_target = config["eye_left_target"]
    eye_right_target = config["eye_right_target"]

    if landmarks["iris_left"] and landmarks["iris_right"]:
        left_eye = landmarks["iris_left"]
        right_eye = landmarks["iris_right"]
    else:
        lm = landmarks["landmarks_68"]
        left_eye = np.mean(lm[36:42], axis=0).tolist()
        right_eye = np.mean(lm[42:48], axis=0).tolist()

    M = compute_affine_transform(left_eye, right_eye, output_size,
                                  eye_left_target, eye_right_target)
    aligned = apply_affine_transform(img_bgr, M, output_size)

    # Transform landmarks
    pts = np.array(landmarks["landmarks_68"], dtype=np.float64)
    ones = np.ones((pts.shape[0], 1))
    pts_h = np.hstack([pts, ones])
    transformed = (M @ pts_h.T).T

    iris_l_t = (M @ np.array([left_eye[0], left_eye[1], 1.0])).tolist()
    iris_r_t = (M @ np.array([right_eye[0], right_eye[1], 1.0])).tolist()

    out_name = Path(filename).stem + ".png"
    out_path = ALIGNED_DIR / out_name
    cv2.imwrite(str(out_path), aligned)

    return {
        "aligned_file": out_name,
        "aligned_landmarks_68": [[round(p[0], 1), round(p[1], 1)] for p in transformed.tolist()],
        "aligned_iris_left": [round(v, 1) for v in iris_l_t],
        "aligned_iris_right": [round(v, 1) for v in iris_r_t],
        "iris_left": [round(v, 1) for v in left_eye],
        "iris_right": [round(v, 1) for v in right_eye],
    }


def main():
    parser = argparse.ArgumentParser(description="Interactive face review")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Review faces below this similarity (default: 0.50)")
    parser.add_argument("--review-all", action="store_true",
                        help="Review every image, not just uncertain ones")
    parser.add_argument("--excluded-only", action="store_true",
                        help="Only review previously excluded images")
    args = parser.parse_args()

    config = load_config()
    manifest = load_manifest()
    app, landmarker = init_models()

    # Build reference embedding from high-confidence images
    print("\nBuilding reference embedding from high-confidence matches...")
    embeddings = []
    for filename, entry in manifest.items():
        if filename.startswith("__"):
            continue
        sim = entry.get("similarity", 0)
        if sim >= 0.65:  # Only use high-confidence matches
            # Re-detect to get embedding
            img_path = IMAGES_DIR / filename
            if not img_path.exists():
                continue
            img = load_image(img_path)
            if img is None:
                continue
            faces = app.get(img)
            if faces:
                bbox = entry.get("bbox", [0, 0, 0, 0])
                # Find face closest to stored bbox
                best = min(faces, key=lambda f: abs(f.bbox[0] - bbox[0]) + abs(f.bbox[1] - bbox[1]))
                emb = best.embedding / np.linalg.norm(best.embedding)
                embeddings.append(emb)
                if len(embeddings) >= 20:
                    break

    if embeddings:
        ref_embedding = np.mean(embeddings, axis=0)
        ref_embedding = ref_embedding / np.linalg.norm(ref_embedding)
        print(f"  Built from {len(embeddings)} high-confidence images")
    else:
        print("ERROR: No high-confidence images to build reference!")
        sys.exit(1)

    # Determine which images to review
    to_review = []
    for filename, entry in sorted(manifest.items()):
        if filename.startswith("__"):
            continue
        if args.excluded_only and not entry.get("excluded"):
            continue
        if args.review_all:
            to_review.append(filename)
        elif entry.get("similarity", 0) < args.threshold or entry.get("excluded"):
            to_review.append(filename)

    print(f"\n{len(to_review)} images to review")
    if not to_review:
        print("Nothing to review!")
        return

    reviewed = 0
    accepted = 0
    skipped = 0

    for filename in to_review:
        img_path = IMAGES_DIR / filename
        if not img_path.exists():
            continue

        img = load_image(img_path)
        if img is None:
            continue

        # Detect all faces
        faces = app.get(img)
        if not faces:
            print(f"  {filename}: No faces detected, skipping")
            skipped += 1
            continue

        # Score all faces against reference
        best_idx = None
        best_sim = -1
        for i, face in enumerate(faces):
            emb = face.embedding / np.linalg.norm(face.embedding)
            sim = float(np.dot(emb, ref_embedding))
            face._review_sim = sim
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        print(f"\n  {filename}: {len(faces)} faces, best sim={best_sim:.2f}")

        # Show for review
        choice = show_faces_for_review(img, faces, filename, best_idx)

        if choice == -2:  # Quit
            print("Quitting review...")
            break
        elif choice == -1:  # Skip
            manifest[filename]["excluded"] = True
            skipped += 1
            print(f"  -> Skipped (excluded)")
        else:
            # User chose a face
            chosen_face = faces[choice]
            reviewed += 1
            accepted += 1

            # Get landmarks for chosen face
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bbox = chosen_face.bbox
            bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
            expanded = [bbox[0] - bw * 0.3, bbox[1] - bh * 0.3,
                        bbox[2] + bw * 0.3, bbox[3] + bh * 0.3]

            lm_data = extract_landmarks(landmarker, img_rgb, expanded)
            if lm_data is None:
                lm_data = extract_landmarks(landmarker, img_rgb)

            if lm_data is None:
                print(f"  -> Landmarks failed, skipping")
                manifest[filename]["excluded"] = True
                skipped += 1
                continue

            # Update manifest
            manifest[filename]["bbox"] = [float(v) for v in chosen_face.bbox]
            emb = chosen_face.embedding / np.linalg.norm(chosen_face.embedding)
            manifest[filename]["similarity"] = float(np.dot(emb, ref_embedding))
            manifest[filename]["manually_reviewed"] = True
            manifest[filename]["excluded"] = False
            manifest[filename]["landmarks_68"] = [
                [round(p[0], 1), round(p[1], 1)] for p in lm_data["landmarks_68"]
            ]
            manifest[filename]["iris_left"] = [round(v, 1) for v in lm_data["iris_left"]] if lm_data["iris_left"] else None
            manifest[filename]["iris_right"] = [round(v, 1) for v in lm_data["iris_right"]] if lm_data["iris_right"] else None
            manifest[filename]["num_faces_in_image"] = len(faces)

            # Re-align and save
            align_result = align_and_save(img, chosen_face, lm_data, filename, config)
            manifest[filename].update(align_result)

            # Un-exclude and add back to sequence if needed
            seq = manifest.get("__sequence__", [])
            if filename not in seq:
                # Insert in correct chronological position
                sort_date = manifest[filename].get("sort_date", "9999")
                inserted = False
                for i, sf in enumerate(seq):
                    sd = manifest.get(sf, {}).get("sort_date", "9999")
                    if sort_date < sd:
                        seq.insert(i, filename)
                        inserted = True
                        break
                if not inserted:
                    seq.append(filename)
                manifest["__sequence__"] = seq

            print(f"  -> Accepted face #{choice+1}, re-aligned")

    cv2.destroyAllWindows()

    # Save
    save_manifest(manifest)

    print(f"\n{'='*60}")
    print(f"REVIEW COMPLETE")
    print(f"  Reviewed: {reviewed + skipped}")
    print(f"  Accepted: {accepted}")
    print(f"  Skipped: {skipped}")
    print(f"  Sequence: {len(manifest.get('__sequence__', []))} images")
    print(f"\nTo re-render: python Code/04_render_morph.py --crossfade")
    print(f"Then encode:  python Code/05_encode_video.py --add-year-labels")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 1+2: Detect faces in all images, identify Ted using insightface
recognition, and extract facial landmarks using mediapipe FaceMesh.

Usage:
    python Code/01_detect_faces.py [--reference img1.jpg img2.jpg ...]

If no reference images are provided, the script will display detected faces
and ask you to confirm which is Ted to build the reference embedding.
"""

import argparse
import json
import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Add scripts dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    PROJECT_ROOT, IMAGES_DIR, load_config, load_manifest, save_manifest,
    get_image_files, IMAGE_EXTENSIONS
)


def init_insightface():
    """Initialize insightface face analysis model."""
    # Workaround: insightface's mask_renderer imports a Cython module
    # that may not be compiled for this architecture. Stub it out.
    import types
    for mod_name in [
        'insightface.thirdparty.face3d',
        'insightface.thirdparty.face3d.mesh',
        'insightface.thirdparty.face3d.mesh.cython',
        'insightface.thirdparty.face3d.mesh.cython.mesh_core_cython',
    ]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


# Path to the downloaded mediapipe face landmarker model
FACE_LANDMARKER_MODEL = str(
    Path(__file__).resolve().parent / "face_landmarker_v2_with_blendshapes.task"
)


def init_mediapipe():
    """Initialize mediapipe FaceLandmarker (new tasks API)."""
    import mediapipe as mp
    vision = mp.tasks.vision
    base_opts = mp.tasks.BaseOptions

    options = vision.FaceLandmarkerOptions(
        base_options=base_opts(model_asset_path=FACE_LANDMARKER_MODEL),
        num_faces=10,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)
    return landmarker


# Mediapipe landmark indices that correspond roughly to dlib's 68 points
# We'll extract the key 68 points for compatibility with our morphing pipeline
# Plus we use iris landmarks for precise eye alignment
MEDIAPIPE_TO_68 = {
    # Jawline (0-16)
    0: 10, 1: 338, 2: 297, 3: 332, 4: 284,
    5: 251, 6: 389, 7: 356, 8: 454, 9: 323,
    10: 361, 11: 288, 12: 397, 13: 365, 14: 379,
    15: 378, 16: 400,
    # Right eyebrow (17-21)
    17: 70, 18: 63, 19: 105, 20: 66, 21: 107,
    # Left eyebrow (22-26)
    22: 336, 23: 296, 24: 334, 25: 293, 26: 300,
    # Nose bridge (27-30)
    27: 168, 28: 197, 29: 5, 30: 4,
    # Nose bottom (31-35)
    31: 75, 32: 97, 33: 2, 34: 326, 35: 305,
    # Right eye (36-41)
    36: 33, 37: 160, 38: 158, 39: 133, 40: 153, 41: 144,
    # Left eye (42-47)
    42: 362, 43: 385, 44: 387, 45: 263, 46: 373, 47: 380,
    # Outer lip (48-59)
    48: 61, 49: 40, 50: 37, 51: 0, 52: 267, 53: 270,
    54: 291, 55: 321, 56: 314, 57: 17, 58: 84, 59: 91,
    # Inner lip (60-67)
    60: 78, 61: 82, 62: 13, 63: 312, 64: 308,
    65: 317, 66: 14, 67: 87,
}

# Iris center landmarks in mediapipe (with refine_landmarks=True)
LEFT_IRIS_CENTER = 468  # Left iris center
RIGHT_IRIS_CENTER = 473  # Right iris center


def extract_mediapipe_landmarks(landmarker, img_rgb, face_bbox=None):
    """
    Run mediapipe FaceLandmarker on an image and extract landmarks.

    Parameters:
        landmarker: mediapipe FaceLandmarker instance (new tasks API)
        img_rgb: RGB image (numpy array)
        face_bbox: optional (x1, y1, x2, y2) to crop before detection

    Returns:
        dict with 'landmarks_68' (list of [x,y]), 'iris_left', 'iris_right',
        'all_landmarks' (full 478 points), or None if no face found
    """
    import mediapipe as mp

    h, w = img_rgb.shape[:2]

    if face_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in face_bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        crop = img_rgb[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    else:
        crop = img_rgb
        offset_x, offset_y = 0, 0

    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return None

    # Convert to mediapipe Image
    # Ensure contiguous uint8 array
    crop_contiguous = np.ascontiguousarray(crop, dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_contiguous)

    results = landmarker.detect(mp_image)
    if not results.face_landmarks:
        return None

    # Use the first detected face
    face_lm = results.face_landmarks[0]

    # Extract all landmarks (478 points)
    all_landmarks = []
    for lm in face_lm:
        all_landmarks.append([
            lm.x * cw + offset_x,
            lm.y * ch + offset_y
        ])

    # Extract 68-point subset
    landmarks_68 = []
    for i in range(68):
        mp_idx = MEDIAPIPE_TO_68[i]
        if mp_idx < len(all_landmarks):
            landmarks_68.append(all_landmarks[mp_idx])
        else:
            landmarks_68.append([0, 0])

    # Extract iris centers (indices 468 and 473 if available)
    iris_left = all_landmarks[LEFT_IRIS_CENTER] if len(all_landmarks) > LEFT_IRIS_CENTER else None
    iris_right = all_landmarks[RIGHT_IRIS_CENTER] if len(all_landmarks) > RIGHT_IRIS_CENTER else None

    return {
        "landmarks_68": landmarks_68,
        "iris_left": iris_left,
        "iris_right": iris_right,
        "all_landmarks": all_landmarks
    }


def load_image(path, max_dim=2048):
    """Load an image, resize if very large, return RGB."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_reference_embedding(app, reference_paths):
    """
    Build a reference face embedding for Ted from reference photos.

    Parameters:
        app: insightface FaceAnalysis instance
        reference_paths: list of image paths

    Returns:
        numpy array: average embedding vector
    """
    embeddings = []

    for path in reference_paths:
        img = load_image(path)
        if img is None:
            print(f"  WARNING: Could not load reference image: {path}")
            continue

        faces = app.get(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if not faces:
            print(f"  WARNING: No face found in reference image: {path}")
            continue

        # Use the largest face in reference photos
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embeddings.append(largest.embedding)
        print(f"  Got embedding from: {Path(path).name}")

    if not embeddings:
        print("ERROR: No valid embeddings from reference photos!")
        sys.exit(1)

    # Average all reference embeddings
    avg_embedding = np.mean(embeddings, axis=0)
    # Normalize
    avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
    return avg_embedding


def find_ted_face(faces, reference_embedding, tolerance=0.6):
    """
    Find the face that best matches Ted's reference embedding.

    Parameters:
        faces: list of insightface face objects
        reference_embedding: Ted's reference embedding vector
        tolerance: max cosine distance (lower = stricter)

    Returns:
        (best_face, similarity_score) or (None, 0) if no match
    """
    if not faces:
        return None, 0.0

    best_face = None
    best_sim = -1

    for face in faces:
        emb = face.embedding / np.linalg.norm(face.embedding)
        sim = np.dot(emb, reference_embedding)
        if sim > best_sim:
            best_sim = sim
            best_face = face

    # Convert similarity to distance-like metric for threshold comparison
    # cosine similarity > (1 - tolerance) means match
    if best_sim >= (1 - tolerance):
        return best_face, float(best_sim)
    else:
        return None, float(best_sim)


def auto_select_references(app, image_files, num_refs=5):
    """
    Automatically select reference photos by finding images with
    exactly one clear, large face (likely solo portraits of Ted).
    """
    candidates = []

    print("Auto-selecting reference photos (looking for clear solo portraits)...")
    for img_path in image_files:
        img = load_image(img_path)
        if img is None:
            continue

        faces = app.get(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if len(faces) == 1:
            face = faces[0]
            area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            img_area = img.shape[0] * img.shape[1]
            face_ratio = area / img_area if img_area > 0 else 0

            # Prefer images where the face is large and detection is confident
            if face_ratio > 0.02 and face.det_score > 0.8:
                candidates.append((img_path, face, face_ratio, face.det_score))

    # Sort by face size ratio * detection confidence
    candidates.sort(key=lambda x: x[2] * x[3], reverse=True)

    if len(candidates) < 3:
        print(f"WARNING: Only found {len(candidates)} clear solo portraits.")
        print("You may want to manually specify reference photos.")

    selected = candidates[:num_refs]
    print(f"Selected {len(selected)} reference photos:")
    for path, face, ratio, score in selected:
        print(f"  {path.name} (face ratio: {ratio:.1%}, confidence: {score:.2f})")

    return selected


def process_images(app, face_mesh, image_files, reference_embedding, config):
    """Process all images: detect Ted's face and extract landmarks."""
    manifest = load_manifest()
    tolerance = config.get("face_recognition_tolerance", 0.6)
    total = len(image_files)
    succeeded = 0
    failed = []

    for i, img_path in enumerate(image_files):
        filename = img_path.name
        print(f"[{i+1}/{total}] Processing: {filename}...", end=" ")

        img_rgb = load_image(img_path)
        if img_rgb is None:
            print("SKIP (could not load)")
            failed.append({"file": filename, "reason": "could not load"})
            continue

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Detect faces with insightface
        faces = app.get(img_bgr)

        if not faces:
            print("SKIP (no faces detected)")
            failed.append({"file": filename, "reason": "no faces detected"})
            continue

        # Find Ted
        ted_face, similarity = find_ted_face(faces, reference_embedding, tolerance)

        if ted_face is None:
            # Try with lower threshold for baby/child photos
            ted_face, similarity = find_ted_face(faces, reference_embedding, tolerance + 0.2)
            if ted_face is None:
                # Last resort: use the largest face
                ted_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                similarity = 0.0
                print(f"LOW MATCH (using largest face, sim={similarity:.2f})", end=" ")

        # Expand bounding box for better landmark detection
        bbox = ted_face.bbox
        bw = bbox[2] - bbox[0]
        bh = bbox[3] - bbox[1]
        expanded_bbox = [
            bbox[0] - bw * 0.3,
            bbox[1] - bh * 0.3,
            bbox[2] + bw * 0.3,
            bbox[3] + bh * 0.3
        ]

        # Extract landmarks with mediapipe
        lm_data = extract_mediapipe_landmarks(face_mesh, img_rgb, expanded_bbox)

        if lm_data is None:
            # Try without bbox crop
            lm_data = extract_mediapipe_landmarks(face_mesh, img_rgb)

        if lm_data is None:
            print("SKIP (landmarks failed)")
            failed.append({"file": filename, "reason": "landmark detection failed"})
            continue

        # Use iris centers if available, otherwise fall back to eye landmark averages
        if lm_data["iris_left"] and lm_data["iris_right"]:
            left_eye = lm_data["iris_left"]
            right_eye = lm_data["iris_right"]
        else:
            lm68 = lm_data["landmarks_68"]
            left_eye = np.mean(lm68[36:42], axis=0).tolist()
            right_eye = np.mean(lm68[42:48], axis=0).tolist()

        # Store in manifest
        manifest[filename] = {
            "bbox": [float(v) for v in ted_face.bbox],
            "similarity": float(similarity),
            "num_faces_in_image": len(faces),
            "landmarks_68": [[round(p[0], 1), round(p[1], 1)] for p in lm_data["landmarks_68"]],
            "iris_left": [round(v, 1) for v in left_eye],
            "iris_right": [round(v, 1) for v in right_eye],
            "image_size": [img_rgb.shape[1], img_rgb.shape[0]],
        }

        succeeded += 1
        print(f"OK (sim={similarity:.2f}, faces={len(faces)})")

        # Save periodically
        if (i + 1) % 20 == 0:
            save_manifest(manifest)
            print(f"  [Saved manifest: {succeeded} faces so far]")

    # Final save
    save_manifest(manifest)

    # Report
    print(f"\n{'='*60}")
    print(f"RESULTS: {succeeded}/{total} images processed successfully")
    print(f"Failed: {len(failed)}")
    if failed:
        print("\nFailed images:")
        for f in failed:
            print(f"  {f['file']}: {f['reason']}")

    # Save failed list
    failed_path = PROJECT_ROOT / "failed_images.json"
    with open(failed_path, "w") as f:
        json.dump(failed, f, indent=2)
    print(f"\nFailed list saved to: {failed_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Detect Ted's face in all photos")
    parser.add_argument("--reference", nargs="*", help="Reference photo paths for Ted")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="Face matching tolerance (0-1, lower=stricter)")
    args = parser.parse_args()

    config = load_config()
    if args.tolerance is not None:
        config["face_recognition_tolerance"] = args.tolerance

    print("Initializing face detection models...")
    app = init_insightface()
    face_mesh = init_mediapipe()

    image_files = get_image_files()
    print(f"Found {len(image_files)} images in project folder.")

    # Build reference embedding
    if args.reference:
        ref_paths = [Path(p) if Path(p).is_absolute() else IMAGES_DIR / p
                     for p in args.reference]
        print(f"\nBuilding reference embedding from {len(ref_paths)} photos...")
        ref_embedding = build_reference_embedding(app, ref_paths)
    else:
        # Auto-select reference photos
        selected = auto_select_references(app, image_files)
        if not selected:
            print("ERROR: Could not auto-select reference photos.")
            print("Please provide reference photos with --reference flag.")
            sys.exit(1)

        # Build embedding from auto-selected photos
        embeddings = []
        for path, face, _, _ in selected:
            emb = face.embedding / np.linalg.norm(face.embedding)
            embeddings.append(emb)
        ref_embedding = np.mean(embeddings, axis=0)
        ref_embedding = ref_embedding / np.linalg.norm(ref_embedding)

    print("\nProcessing all images...")
    manifest = process_images(app, face_mesh, image_files, ref_embedding, config)

    print(f"\nDone! Manifest saved with {len(manifest)} entries.")
    print("Next step: python Code/02_align_faces.py")


if __name__ == "__main__":
    main()

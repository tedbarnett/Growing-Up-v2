"""
Shared utility functions for the Growing Up Ted face morph pipeline.
"""

import json
import os
import math
import numpy as np
import cv2
from pathlib import Path

# Project root (parent of Code/)
PROJECT_ROOT = Path(os.environ.get("GROWUP_PROJECT_ROOT", str(Path(__file__).resolve().parent.parent)))
CONFIG_PATH = Path(os.environ.get("GROWUP_CONFIG_PATH", str(PROJECT_ROOT / "config.json")))
MANIFEST_PATH = Path(os.environ.get("GROWUP_MANIFEST_PATH", str(PROJECT_ROOT / "manifest.json")))
ALIGNED_DIR = Path(os.environ.get("GROWUP_ALIGNED_DIR", str(PROJECT_ROOT / "aligned")))
FRAMES_DIR = Path(os.environ.get("GROWUP_FRAMES_DIR", str(PROJECT_ROOT / "frames")))
OUTPUT_DIR = Path(os.environ.get("GROWUP_OUTPUT_DIR", str(PROJECT_ROOT / "Output")))

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".heic", ".bmp"}


def load_config():
    """Load config.json and return as dict."""
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def load_manifest():
    """Load manifest.json if it exists, otherwise return empty dict."""
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    """Save manifest dict to manifest.json."""
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def get_images_dir(subject_name=None):
    """Return the source images directory for the given subject.
    If subject_name is not provided, reads it from config.json.
    """
    if subject_name is None:
        config = load_config()
        subject_name = config.get("subject_name", "Ted")
    return PROJECT_ROOT / "Images" / subject_name


IMAGES_DIR = Path(os.environ.get("GROWUP_IMAGES_DIR", str(get_images_dir())))


def get_image_files():
    """Return sorted list of image file paths in the Images directory.
    When both .jpeg and .png exist for the same stem, only includes the .png.
    """
    all_files = {}
    for f in IMAGES_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            stem = f.stem
            ext = f.suffix.lower()
            if stem in all_files:
                # Prefer .png over .jpeg/.jpg
                existing_ext = all_files[stem].suffix.lower()
                if ext == '.png' and existing_ext in ('.jpeg', '.jpg'):
                    all_files[stem] = f
                # Otherwise keep existing (first .png wins)
            else:
                all_files[stem] = f
    return sorted(all_files.values(), key=lambda p: p.name.lower())


def compute_eye_centers(landmarks):
    """
    Given 68-point facial landmarks (as list of [x,y] pairs),
    compute the center of each eye.

    Left eye: landmarks 36-41
    Right eye: landmarks 42-47

    Returns: (left_eye_center, right_eye_center) as numpy arrays
    """
    landmarks = np.array(landmarks)
    left_eye = landmarks[36:42].mean(axis=0)
    right_eye = landmarks[42:48].mean(axis=0)
    return left_eye, right_eye


def compute_affine_transform(left_eye, right_eye, output_size,
                              eye_left_target, eye_right_target):
    """
    Compute the 2x3 affine transformation matrix that maps the input
    image so that:
      - left_eye lands at eye_left_target (as fraction of output_size)
      - right_eye lands at eye_right_target (as fraction of output_size)

    Parameters:
        left_eye: (x, y) of left eye in source image
        right_eye: (x, y) of right eye in source image
        output_size: int, size of square output
        eye_left_target: [frac_x, frac_y] target position as fraction
        eye_right_target: [frac_x, frac_y] target position as fraction

    Returns:
        M: 2x3 affine transform matrix
    """
    # Target pixel positions
    target_left = np.array([eye_left_target[0] * output_size,
                            eye_left_target[1] * output_size])
    target_right = np.array([eye_right_target[0] * output_size,
                             eye_right_target[1] * output_size])

    # Source positions
    src_left = np.array(left_eye, dtype=np.float64)
    src_right = np.array(right_eye, dtype=np.float64)

    # Compute rotation angle
    src_delta = src_right - src_left
    tgt_delta = target_right - target_left

    src_angle = math.atan2(src_delta[1], src_delta[0])
    tgt_angle = math.atan2(tgt_delta[1], tgt_delta[0])
    angle = tgt_angle - src_angle

    # Compute scale
    src_dist = np.linalg.norm(src_delta)
    tgt_dist = np.linalg.norm(tgt_delta)
    scale = tgt_dist / src_dist if src_dist > 0 else 1.0

    # Build rotation matrix around left eye, then translate
    cos_a = math.cos(angle) * scale
    sin_a = math.sin(angle) * scale

    # Transform: rotate+scale around src_left, then translate to target_left
    M = np.array([
        [cos_a, -sin_a, target_left[0] - cos_a * src_left[0] + sin_a * src_left[1]],
        [sin_a,  cos_a, target_left[1] - sin_a * src_left[0] - cos_a * src_left[1]]
    ], dtype=np.float64)

    return M


def apply_affine_transform(img, M, output_size):
    """Apply affine transform M to img, producing output_size x output_size result."""
    return cv2.warpAffine(img, M, (output_size, output_size),
                          flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT_101)


def rect_contains(rect, point):
    """Check if a point is inside a rectangle."""
    if point[0] < rect[0] or point[0] > rect[2]:
        return False
    if point[1] < rect[1] or point[1] > rect[3]:
        return False
    return True


def compute_delaunay_triangles(rect, points):
    """
    Compute Delaunay triangulation for a set of points within a rectangle.

    Parameters:
        rect: (x, y, w, h) bounding rectangle
        points: list of (x, y) tuples

    Returns:
        list of triangle index triples [(i1, i2, i3), ...]
    """
    subdiv = cv2.Subdiv2D(rect)
    point_dict = {}

    for i, p in enumerate(points):
        # Clamp points to be within rect
        px = max(rect[0], min(rect[0] + rect[2] - 1, p[0]))
        py = max(rect[1], min(rect[1] + rect[3] - 1, p[1]))
        subdiv.insert((float(px), float(py)))
        # Map coords to index (use rounded coords as key)
        point_dict[(round(px), round(py))] = i

    triangle_list = subdiv.getTriangleList()
    triangles = []
    r = (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])

    for t in triangle_list:
        pts = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
        if all(rect_contains(r, p) for p in pts):
            indices = []
            for p in pts:
                # Find closest point index
                key = (round(p[0]), round(p[1]))
                if key in point_dict:
                    indices.append(point_dict[key])
                else:
                    # Find nearest point
                    min_dist = float('inf')
                    min_idx = 0
                    for k, v in point_dict.items():
                        d = (k[0] - p[0])**2 + (k[1] - p[1])**2
                        if d < min_dist:
                            min_dist = d
                            min_idx = v
                    indices.append(min_idx)
            if len(set(indices)) == 3:  # Valid triangle
                triangles.append(tuple(indices))

    return triangles


def warp_triangle(img1, img2, t1, t2):
    """
    Warp the triangle region from img1 (defined by t1) into img2 (at t2).

    Parameters:
        img1: source image
        img2: destination image (modified in place)
        t1: list of 3 (x,y) points in source
        t2: list of 3 (x,y) points in destination
    """
    # Bounding rectangles
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by bounding rect origin
    t1_offset = [(p[0] - r1[0], p[1] - r1[1]) for p in t1]
    t2_offset = [(p[0] - r2[0], p[1] - r2[1]) for p in t2]

    # Create mask for destination triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_offset), (1.0, 1.0, 1.0), 16, 0)

    # Extract source rectangle
    x1, y1, w1, h1 = r1
    x1 = max(0, x1)
    y1 = max(0, y1)
    img1_rect = img1[y1:y1 + h1, x1:x1 + w1]

    if img1_rect.shape[0] == 0 or img1_rect.shape[1] == 0:
        return
    if r2[2] == 0 or r2[3] == 0:
        return

    # Affine transform from source triangle to destination triangle
    warp_mat = cv2.getAffineTransform(
        np.float32(t1_offset), np.float32(t2_offset)
    )
    warped = cv2.warpAffine(
        img1_rect, warp_mat, (r2[2], r2[3]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
    )

    # Apply mask and copy to destination
    x2, y2, w2, h2 = r2
    x2 = max(0, x2)
    y2 = max(0, y2)

    # Ensure we don't exceed image bounds
    h_avail = min(h2, img2.shape[0] - y2)
    w_avail = min(w2, img2.shape[1] - x2)
    if h_avail <= 0 or w_avail <= 0:
        return

    warped_region = warped[:h_avail, :w_avail]
    mask_region = mask[:h_avail, :w_avail]
    dest_region = img2[y2:y2 + h_avail, x2:x2 + w_avail]

    img2[y2:y2 + h_avail, x2:x2 + w_avail] = (
        dest_region * (1 - mask_region) + warped_region * mask_region
    )


def morph_faces(img1, img2, landmarks1, landmarks2, alpha, output_size):
    """
    Morph between two aligned face images using Delaunay triangulation.

    Parameters:
        img1: first face image (numpy array, float32, 0-1 range)
        img2: second face image (numpy array, float32, 0-1 range)
        landmarks1: 68 landmark points for img1
        landmarks2: 68 landmark points for img2
        alpha: blend factor (0.0 = img1, 1.0 = img2)
        output_size: size of output image

    Returns:
        morphed image (numpy array, float32)
    """
    # Interpolate landmarks
    pts1 = np.array(landmarks1, dtype=np.float32)
    pts2 = np.array(landmarks2, dtype=np.float32)
    pts_morph = (1 - alpha) * pts1 + alpha * pts2

    # Add corner and edge points for full image coverage
    boundary_pts = [
        (0, 0), (output_size // 2, 0), (output_size - 1, 0),
        (0, output_size // 2), (output_size - 1, output_size // 2),
        (0, output_size - 1), (output_size // 2, output_size - 1),
        (output_size - 1, output_size - 1)
    ]

    pts1_full = list(map(tuple, pts1.tolist())) + boundary_pts
    pts2_full = list(map(tuple, pts2.tolist())) + boundary_pts
    pts_morph_full = list(map(tuple, pts_morph.tolist())) + boundary_pts

    # Compute Delaunay on the morphed (intermediate) points
    rect = (0, 0, output_size, output_size)
    triangles = compute_delaunay_triangles(rect, pts_morph_full)

    # Create output image
    img_morph = np.zeros_like(img1)

    for tri_indices in triangles:
        i, j, k = tri_indices

        # Get triangle vertices in each image
        t1 = [pts1_full[i], pts1_full[j], pts1_full[k]]
        t2 = [pts2_full[i], pts2_full[j], pts2_full[k]]
        t_morph = [pts_morph_full[i], pts_morph_full[j], pts_morph_full[k]]

        # Warp both images to the morphed triangle
        warped1 = np.zeros_like(img_morph)
        warped2 = np.zeros_like(img_morph)

        warp_triangle(img1, warped1, t1, t_morph)
        warp_triangle(img2, warped2, t2, t_morph)

        # Create triangle mask
        mask = np.zeros(img_morph.shape[:2], dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(t_morph), 1.0, 16, 0)
        mask = mask[:, :, np.newaxis]

        # Blend and apply
        blended = (1 - alpha) * warped1 + alpha * warped2
        img_morph = img_morph * (1 - mask) + blended * mask

    return img_morph


def add_boundary_landmarks(landmarks, output_size):
    """
    Add 8 boundary points to a set of 68 facial landmarks
    for full-image Delaunay triangulation.
    """
    boundary = [
        [0, 0], [output_size // 2, 0], [output_size - 1, 0],
        [0, output_size // 2], [output_size - 1, output_size // 2],
        [0, output_size - 1], [output_size // 2, output_size - 1],
        [output_size - 1, output_size - 1]
    ]
    return landmarks + boundary

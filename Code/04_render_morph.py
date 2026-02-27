#!/usr/bin/env python3
"""
Step 6: Render morph transitions between consecutive aligned faces.

Uses Delaunay triangulation to warp and blend faces, producing
smooth morphing frames.

Usage:
    python Code/04_render_morph.py [--hold 15] [--morph 30] [--fps 30]
    python Code/04_render_morph.py --crossfade   # Simple crossfade instead of morph
    python Code/04_render_morph.py --crossfade --debug-labels  # With filename overlay
    python Code/04_render_morph.py --crossfade --vignette      # With oval vignette
"""

import argparse
import json
import sys
import numpy as np
import cv2
from pathlib import Path
from datetime import date

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    PROJECT_ROOT, ALIGNED_DIR, FRAMES_DIR, load_config, load_manifest,
    morph_faces
)


def compute_age_label(birthdate_str, photo_date_str):
    """
    Compute a human-readable age label from birthdate and photo date.

    Returns:
        "Newborn" if < 1 month
        "X months" if < 1 year
        "Age X" if >= 1 year
    """
    if not birthdate_str or not photo_date_str:
        return None
    try:
        birth = date.fromisoformat(birthdate_str)
        # photo_date might be just YYYY-MM-DD or partial
        parts = photo_date_str.split("-")
        if len(parts) >= 3:
            photo = date(int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) >= 2:
            photo = date(int(parts[0]), int(parts[1]), 15)
        else:
            photo = date(int(parts[0]), 6, 15)
    except (ValueError, IndexError):
        return None

    if photo < birth:
        return None

    # Calculate months difference
    months = (photo.year - birth.year) * 12 + (photo.month - birth.month)
    if photo.day < birth.day:
        months -= 1

    if months < 1:
        return "Newborn"
    elif months < 12:
        return f"{months} months" if months != 1 else "1 month"
    else:
        years = months // 12
        return f"Age {years}"


def init_segmenter():
    """Initialize mediapipe selfie segmentation (tasks API)."""
    import mediapipe as mp
    model_path = str(Path(__file__).resolve().parent / "selfie_segmenter.tflite")
    options = mp.tasks.vision.ImageSegmenterOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
        output_category_mask=True,
    )
    return mp.tasks.vision.ImageSegmenter.create_from_options(options)


def darken_background(img_bgr_float, segmenter, darken_factor=0.5):
    """
    Darken background pixels using selfie segmentation.

    Args:
        img_bgr_float: BGR float32 image, 0-1 range
        segmenter: mediapipe ImageSegmenter instance
        darken_factor: brightness multiplier for background (0=black, 1=unchanged)

    Returns:
        Image with darkened background, same format as input
    """
    import mediapipe as mp
    img_rgb = cv2.cvtColor((img_bgr_float * 255).astype(np.uint8), cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                        data=np.ascontiguousarray(img_rgb))
    result = segmenter.segment(mp_image)
    mask = result.confidence_masks[0].numpy_view().squeeze()

    # Smooth edges slightly for cleaner transitions
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    mask_3d = mask[:, :, np.newaxis]

    # Person stays at full brightness, background darkened
    return img_bgr_float * (mask_3d + (1 - mask_3d) * darken_factor)


def render_crossfade(img1, img2, alpha):
    """Simple crossfade between two images."""
    return ((1 - alpha) * img1 + alpha * img2).astype(np.float32)


def create_vignette_mask(size, face_scale=0.70):
    """
    Create an oval vignette mask that frames the whole head.

    Parameters:
        size: output image size (square)
        face_scale: how much of the frame the oval fills (0.7 = 70%)

    Returns:
        mask: float32 array (size x size), 1.0 inside oval, fading to 0.0 outside
    """
    mask = np.zeros((size, size), dtype=np.float32)
    center = (size // 2, int(size * 0.50))  # Centered vertically to show full head
    # Oval axes: wide enough for ears, tall enough for top of head to chin
    axis_x = int(size * face_scale * 0.48)
    axis_y = int(size * face_scale * 0.58)

    # Draw filled ellipse
    cv2.ellipse(mask, center, (axis_x, axis_y), 0, 0, 360, 1.0, -1)

    # Blur for soft edge - wide feathering to hide background
    blur_size = int(size * 0.12) | 1  # Ensure odd
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    return mask[:, :, np.newaxis]  # Add channel dim for broadcasting


def apply_vignette(frame_uint8, vignette_mask):
    """Apply vignette mask to a uint8 frame, black outside the oval."""
    frame_f = frame_uint8.astype(np.float32) / 255.0
    result = frame_f * vignette_mask
    return (result * 255).astype(np.uint8)


def _load_year_font(size):
    """Load a nice bold system font for the age label. Cached after first call."""
    if not hasattr(_load_year_font, '_cache'):
        from PIL import ImageFont
        font_size = int(size * 0.066)  # 20% larger than previous 0.055
        # Try bold macOS fonts in order of preference
        for font_name, font_index in [
            ("/System/Library/Fonts/Helvetica.ttc", 1),   # Bold variant
            ("/System/Library/Fonts/HelveticaNeue.ttc", 1),
            ("/Library/Fonts/Arial Bold.ttf", 0),
            ("/Library/Fonts/Arial.ttf", 0),
        ]:
            try:
                _load_year_font._cache = ImageFont.truetype(font_name, font_size, index=font_index)
                return _load_year_font._cache
            except (OSError, IOError):
                continue
        _load_year_font._cache = ImageFont.load_default()
    return _load_year_font._cache


def add_age_label(frame_uint8, label, opacity=1.0):
    """Add age label in a large, bold font centered horizontally near the bottom."""
    from PIL import Image, ImageDraw
    h, w = frame_uint8.shape[:2]
    if not label or opacity <= 0:
        return frame_uint8

    font = _load_year_font(h)

    # Convert to PIL for nice text rendering
    pil_img = Image.fromarray(cv2.cvtColor(frame_uint8, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Get text bounding box
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Position: centered horizontally, near bottom with extra padding
    x = (w - text_w) // 2
    margin = int(h * 0.04)  # ~10px more than previous 0.03
    y = h - text_h - margin

    # Apply opacity
    shadow_alpha = int(opacity * 255)
    text_alpha = int(opacity * 255)

    # Draw with shadow for readability on any background
    draw.text((x + 2, y + 2), label, font=font, fill=(0, 0, 0, shadow_alpha))
    draw.text((x, y), label, font=font, fill=(text_alpha, text_alpha, text_alpha))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def add_debug_label(frame_uint8, filename, year, seq_idx, total):
    """Overlay filename text on the frame for debugging."""
    h, w = frame_uint8.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Filename at very bottom in small text
    label = f"#{seq_idx+1}/{total}: {filename}"
    font_scale = 0.4
    thickness = 1

    cv2.rectangle(frame_uint8, (0, h - 22), (w, h), (0, 0, 0), -1)
    cv2.putText(frame_uint8, label, (5, h - 6), font, font_scale,
                (180, 180, 180), thickness, cv2.LINE_AA)

    return frame_uint8


def render_title_card(output_size, subject_name, fps, frames_dir):
    """
    Render a title card: name on black screen (2s), then fade to black (1s).

    Returns the number of frames rendered.
    """
    from PIL import Image, ImageDraw, ImageFont

    hold_duration = int(2 * fps)    # 2 seconds of name display
    fade_duration = int(1 * fps)    # 1 second fade to black
    total = hold_duration + fade_duration

    # Load a large font for the title
    font_size = int(output_size * 0.10)
    font = None
    for font_name in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    # Measure text
    tmp_img = Image.new("RGB", (output_size, output_size), (0, 0, 0))
    draw = ImageDraw.Draw(tmp_img)
    bbox = draw.textbbox((0, 0), subject_name, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (output_size - text_w) // 2
    y = (output_size - text_h) // 2

    frame_num = 0
    for i in range(total):
        pil_img = Image.new("RGB", (output_size, output_size), (0, 0, 0))
        draw = ImageDraw.Draw(pil_img)

        if i < hold_duration:
            alpha = 255
        else:
            # Fade out
            progress = (i - hold_duration) / fade_duration
            alpha = int(255 * (1.0 - progress))

        draw.text((x, y), subject_name, font=font, fill=(alpha, alpha, alpha))
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        frame_path = frames_dir / f"frame_{frame_num:06d}.png"
        cv2.imwrite(str(frame_path), frame_bgr)
        frame_num += 1

    print(f"  Title card: {total} frames ({total/fps:.1f}s)")
    return frame_num


def render_end_title_card(output_size, subject_name, fps, frames_dir, start_frame_num):
    """
    Render an end title card: fade in name from black (1s), then hold (2s).
    Total: 3 seconds.

    Returns the next frame number.
    """
    from PIL import Image, ImageDraw, ImageFont

    fade_in_duration = int(1 * fps)
    hold_duration = int(2 * fps)
    total = fade_in_duration + hold_duration

    # Load a large font for the title
    font_size = int(output_size * 0.10)
    font = None
    for font_name in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/Library/Fonts/Arial.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    # Measure text
    tmp_img = Image.new("RGB", (output_size, output_size), (0, 0, 0))
    draw = ImageDraw.Draw(tmp_img)
    bbox = draw.textbbox((0, 0), subject_name, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (output_size - text_w) // 2
    y = (output_size - text_h) // 2

    frame_num = start_frame_num
    for i in range(total):
        pil_img = Image.new("RGB", (output_size, output_size), (0, 0, 0))
        draw = ImageDraw.Draw(pil_img)

        if i < fade_in_duration:
            alpha = int(255 * ((i + 1) / fade_in_duration))
        else:
            alpha = 255

        draw.text((x, y), subject_name, font=font, fill=(alpha, alpha, alpha))
        frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        frame_path = frames_dir / f"frame_{frame_num:06d}.png"
        cv2.imwrite(str(frame_path), frame_bgr)
        frame_num += 1

    print(f"  End title card: {total} frames ({total/fps:.1f}s)")
    return frame_num


def scale_and_shift_image(img, scale, shift_y, output_size):
    """
    Scale image around center and shift vertically.

    Parameters:
        img: float32 image (0-1 range)
        scale: scale factor (0.9 = 90% size, shrinks face)
        shift_y: vertical pixel shift (positive = move image down)
        output_size: frame size in pixels
    """
    if scale == 1.0 and shift_y == 0:
        return img
    cx, cy = output_size / 2, output_size / 2
    M = np.float32([
        [scale, 0, cx * (1 - scale)],
        [0, scale, cy * (1 - scale) + shift_y]
    ])
    return cv2.warpAffine(img, M, (output_size, output_size),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def render_sequence(manifest, config, use_crossfade=False,
                    debug_labels=False, use_vignette=False, face_scale=0.70,
                    darken_bg=False, darken_amount=0.5,
                    image_scale=1.0, image_shift_y=0):
    """
    Render all morph frames for the full sequence.

    Parameters:
        manifest: loaded manifest dict
        config: loaded config dict
        use_crossfade: if True, use simple crossfade instead of Delaunay morph
        debug_labels: if True, overlay filename/year on each frame
        use_vignette: if True, apply oval vignette around face
        face_scale: scale of face within frame (smaller = more border)
        darken_bg: if True, darken background behind face before vignette
        darken_amount: brightness of background (0=black, 1=unchanged, default 0.5)
        image_scale: scale the face image (0.9 = 90%, makes face smaller in frame)
        image_shift_y: shift face image down by this many pixels
    """
    sequence = manifest.get("__sequence__", [])
    if not sequence:
        print("ERROR: No sequence found in manifest.")
        print("Run 03_sort_images.py first.")
        sys.exit(1)

    # Filter to only entries with aligned images
    valid_sequence = []
    for filename in sequence:
        entry = manifest.get(filename, {})
        if "aligned_file" in entry:
            aligned_path = ALIGNED_DIR / entry["aligned_file"]
            if aligned_path.exists():
                valid_sequence.append(filename)

    if len(valid_sequence) < 2:
        print(f"ERROR: Need at least 2 aligned images, found {len(valid_sequence)}")
        sys.exit(1)

    print(f"Rendering morph sequence for {len(valid_sequence)} images...")

    hold_frames = config.get("hold_frames", 15)
    morph_frames = config.get("morph_frames", 30)
    output_size = config.get("output_size", 1024)
    fps = config.get("fps", 30)
    subject_name = config.get("subject_name", "")
    birthdate = config.get("subject_birthdate", "")

    total_frames = len(valid_sequence) * hold_frames + (len(valid_sequence) - 1) * morph_frames
    print(f"  Hold: {hold_frames} frames ({hold_frames/fps:.1f}s)")
    print(f"  Morph: {morph_frames} frames ({morph_frames/fps:.1f}s)")
    print(f"  Total: ~{total_frames} frames ({total_frames/fps:.0f}s at {fps}fps)")
    print(f"  Mode: {'crossfade' if use_crossfade else 'Delaunay morph'}")
    if debug_labels:
        print(f"  Debug labels: ON")
    if use_vignette:
        print(f"  Vignette: ON (face_scale={face_scale})")
    if birthdate:
        print(f"  Age labels: ON (birthdate={birthdate})")
    if darken_bg:
        print(f"  Darken background: ON (amount={darken_amount})")
    if image_scale != 1.0 or image_shift_y != 0:
        print(f"  Image transform: scale={image_scale}, shift_y={image_shift_y}px")

    FRAMES_DIR.mkdir(exist_ok=True)

    # Clear old frames to prevent contamination from previous renders
    old_frames = sorted(FRAMES_DIR.glob("frame_*.png"))
    if old_frames:
        print(f"  Clearing {len(old_frames)} old frames...")
        for f in old_frames:
            f.unlink()

    # Pre-compute vignette mask if needed
    vignette_mask = None
    if use_vignette:
        vignette_mask = create_vignette_mask(output_size, face_scale)

    # Initialize segmenter if darkening background
    segmenter = None
    if darken_bg:
        print("  Loading selfie segmenter...")
        segmenter = init_segmenter()

    frame_num = 0

    # Render title card if subject name is set
    if subject_name:
        frame_num = render_title_card(output_size, subject_name, fps, FRAMES_DIR)

    for seq_idx in range(len(valid_sequence)):
        filename = valid_sequence[seq_idx]
        entry = manifest[filename]
        aligned_path = ALIGNED_DIR / entry["aligned_file"]

        # Load current image
        img = cv2.imread(str(aligned_path)).astype(np.float32) / 255.0
        if img.shape[0] != output_size or img.shape[1] != output_size:
            img = cv2.resize(img, (output_size, output_size))
        if segmenter:
            img = darken_background(img, segmenter, darken_amount)
        img = scale_and_shift_image(img, image_scale, image_shift_y, output_size)

        year = entry.get("sort_year", "?")
        sort_date = entry.get("sort_date", str(year) if year != "?" else "")
        age_label = compute_age_label(birthdate, sort_date) if birthdate else str(year)
        print(f"  [{seq_idx+1}/{len(valid_sequence)}] {filename} ({age_label or year})", end="")

        # Hold frames (show this face for a moment)
        for h in range(hold_frames):
            frame_path = FRAMES_DIR / f"frame_{frame_num:06d}.png"
            frame_out = (img * 255).astype(np.uint8)
            if use_vignette:
                frame_out = apply_vignette(frame_out, vignette_mask)
            if age_label:
                frame_out = add_age_label(frame_out, age_label)
            if debug_labels:
                frame_out = add_debug_label(frame_out, filename, year,
                                            seq_idx, len(valid_sequence))
            cv2.imwrite(str(frame_path), frame_out)
            frame_num += 1

        print(f" -> {hold_frames} hold frames", end="")

        # Morph to next image (if not the last)
        if seq_idx < len(valid_sequence) - 1:
            next_filename = valid_sequence[seq_idx + 1]
            next_entry = manifest[next_filename]
            next_path = ALIGNED_DIR / next_entry["aligned_file"]

            img_next = cv2.imread(str(next_path)).astype(np.float32) / 255.0
            if img_next.shape[0] != output_size or img_next.shape[1] != output_size:
                img_next = cv2.resize(img_next, (output_size, output_size))
            if segmenter:
                img_next = darken_background(img_next, segmenter, darken_amount)
            img_next = scale_and_shift_image(img_next, image_scale, image_shift_y, output_size)

            if use_crossfade:
                # Simple crossfade
                for m in range(morph_frames):
                    alpha = (m + 1) / (morph_frames + 1)
                    frame = render_crossfade(img, img_next, alpha)
                    frame_path = FRAMES_DIR / f"frame_{frame_num:06d}.png"
                    frame_out = (frame * 255).astype(np.uint8)
                    if use_vignette:
                        frame_out = apply_vignette(frame_out, vignette_mask)
                    if age_label:
                        frame_out = add_age_label(frame_out, age_label)
                    if debug_labels:
                        frame_out = add_debug_label(
                            frame_out, f"{filename} -> {next_filename}",
                            year, seq_idx, len(valid_sequence))
                    cv2.imwrite(str(frame_path), frame_out)
                    frame_num += 1
            else:
                # Delaunay morph
                lm1 = entry.get("aligned_landmarks_68")
                lm2 = next_entry.get("aligned_landmarks_68")

                if lm1 is None or lm2 is None:
                    # Fall back to crossfade if landmarks missing
                    print(" [crossfade fallback]", end="")
                    for m in range(morph_frames):
                        alpha = (m + 1) / (morph_frames + 1)
                        frame = render_crossfade(img, img_next, alpha)
                        frame_path = FRAMES_DIR / f"frame_{frame_num:06d}.png"
                        frame_out = (frame * 255).astype(np.uint8)
                        if use_vignette:
                            frame_out = apply_vignette(frame_out, vignette_mask)
                        if age_label:
                            frame_out = add_age_label(frame_out, age_label)
                        if debug_labels:
                            frame_out = add_debug_label(
                                frame_out, f"{filename} -> {next_filename}",
                                year, seq_idx, len(valid_sequence))
                        cv2.imwrite(str(frame_path), frame_out)
                        frame_num += 1
                else:
                    for m in range(morph_frames):
                        alpha = (m + 1) / (morph_frames + 1)
                        try:
                            frame = morph_faces(
                                img, img_next, lm1, lm2,
                                alpha, output_size
                            )
                        except Exception as e:
                            # Fall back to crossfade on error
                            frame = render_crossfade(img, img_next, alpha)

                        frame_path = FRAMES_DIR / f"frame_{frame_num:06d}.png"
                        frame_out = (frame * 255).astype(np.uint8)
                        if use_vignette:
                            frame_out = apply_vignette(frame_out, vignette_mask)
                        if age_label:
                            frame_out = add_age_label(frame_out, age_label)
                        if debug_labels:
                            frame_out = add_debug_label(
                                frame_out, f"{filename} -> {next_filename}",
                                year, seq_idx, len(valid_sequence))
                        cv2.imwrite(str(frame_path), frame_out)
                        frame_num += 1

            print(f" + {morph_frames} morph frames")
        else:
            # Last image: add extra hold frames at the end
            for h in range(hold_frames):
                frame_path = FRAMES_DIR / f"frame_{frame_num:06d}.png"
                frame_out = (img * 255).astype(np.uint8)
                if use_vignette:
                    frame_out = apply_vignette(frame_out, vignette_mask)
                if age_label:
                    frame_out = add_age_label(frame_out, age_label)
                if debug_labels:
                    frame_out = add_debug_label(frame_out, filename, year,
                                                seq_idx, len(valid_sequence))
                cv2.imwrite(str(frame_path), frame_out)
                frame_num += 1
            print(f" + {hold_frames} end hold frames", end="")

            # Fade to black over 3 seconds (image + age text fade together)
            fade_to_black_frames = int(3 * fps)
            for f_idx in range(fade_to_black_frames):
                brightness = 1.0 - ((f_idx + 1) / fade_to_black_frames)
                frame_out = (img * 255).astype(np.uint8)
                if use_vignette:
                    frame_out = apply_vignette(frame_out, vignette_mask)
                if age_label:
                    frame_out = add_age_label(frame_out, age_label)
                # Fade entire frame (image + vignette + text) to black
                frame_out = (frame_out.astype(np.float32) * brightness).astype(np.uint8)
                frame_path = FRAMES_DIR / f"frame_{frame_num:06d}.png"
                cv2.imwrite(str(frame_path), frame_out)
                frame_num += 1
            print(f" + {fade_to_black_frames} fade-to-black frames")

    # Render end title card (fade in name from black, hold 3s)
    if subject_name:
        frame_num = render_end_title_card(output_size, subject_name, fps, FRAMES_DIR, frame_num)

    # Write timing info for video encoder (music delay, etc.)
    title_card_duration = 3.0 if subject_name else 0.0  # 2s hold + 1s fade
    render_info = {
        "music_delay_s": title_card_duration,
        "total_frames": frame_num,
        "fps": fps,
    }
    render_info_path = FRAMES_DIR / "render_info.json"
    with open(render_info_path, "w") as f:
        json.dump(render_info, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: {frame_num} frames rendered to {FRAMES_DIR}/")
    print(f"Duration: {frame_num/fps:.1f}s at {fps}fps")
    print(f"\nNext step: python Code/05_encode_video.py")


def main():
    parser = argparse.ArgumentParser(description="Render morph transitions")
    parser.add_argument("--hold", type=int, default=None,
                        help="Hold frames per image")
    parser.add_argument("--morph", type=int, default=None,
                        help="Morph transition frames between images")
    parser.add_argument("--fps", type=int, default=None,
                        help="Target frame rate")
    parser.add_argument("--crossfade", action="store_true",
                        help="Use simple crossfade instead of Delaunay morph")
    parser.add_argument("--start", type=int, default=None,
                        help="Start from this image index (0-based)")
    parser.add_argument("--end", type=int, default=None,
                        help="End at this image index (exclusive)")
    parser.add_argument("--debug-labels", action="store_true",
                        help="Overlay filename/year on each frame for debugging")
    parser.add_argument("--vignette", action="store_true",
                        help="Apply oval vignette around face")
    parser.add_argument("--face-scale", type=float, default=0.70,
                        help="Face scale within frame (default: 0.70, smaller = more border)")
    parser.add_argument("--darken-bg", action="store_true",
                        help="Darken background behind face using selfie segmentation")
    parser.add_argument("--darken-amount", type=float, default=0.5,
                        help="Background brightness (0=black, 1=unchanged, default=0.5)")
    parser.add_argument("--step", type=int, default=1,
                        help="Use every Nth image (default: 1 = all images, 20 = every 20th)")
    parser.add_argument("--image-scale", type=float, default=0.90,
                        help="Scale face image within frame (default: 0.90, smaller = more headroom)")
    parser.add_argument("--image-shift-y", type=float, default=20,
                        help="Shift face image down by pixels (default: 20)")
    args = parser.parse_args()

    config = load_config()
    manifest = load_manifest()

    if args.hold is not None:
        config["hold_frames"] = args.hold
    if args.morph is not None:
        config["morph_frames"] = args.morph
    if args.fps is not None:
        config["fps"] = args.fps

    # Handle start/end subset
    if args.start is not None or args.end is not None:
        sequence = manifest.get("__sequence__", [])
        start = args.start or 0
        end = args.end or len(sequence)
        manifest["__sequence__"] = sequence[start:end]
        print(f"Processing subset: images {start} to {end}")

    # Handle step (every Nth image)
    if args.step > 1:
        sequence = manifest.get("__sequence__", [])
        manifest["__sequence__"] = sequence[::args.step]
        print(f"Using every {args.step}th image: {len(manifest['__sequence__'])} of {len(sequence)}")

    render_sequence(manifest, config, use_crossfade=args.crossfade,
                    debug_labels=args.debug_labels,
                    use_vignette=args.vignette,
                    face_scale=args.face_scale,
                    darken_bg=args.darken_bg,
                    darken_amount=args.darken_amount,
                    image_scale=args.image_scale,
                    image_shift_y=args.image_shift_y)


if __name__ == "__main__":
    main()

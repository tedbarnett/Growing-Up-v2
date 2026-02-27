#!/usr/bin/env python3
"""
Step 7: Encode rendered frames into a final MP4 video using FFmpeg.

Usage:
    python Code/05_encode_video.py [--fps 30] [--music path/to/music.mp3]
    python Code/05_encode_video.py --add-year-labels
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    PROJECT_ROOT, FRAMES_DIR, OUTPUT_DIR, load_config, load_manifest
)


def count_frames():
    """Count the number of rendered frames."""
    frames = sorted(FRAMES_DIR.glob("frame_*.png"))
    return len(frames)


def encode_video(config, output_name="growing_up_ted.mp4", music_paths=None,
                 crf=18, preset="slow"):
    """
    Encode frames to MP4 using FFmpeg.

    Parameters:
        config: config dict
        output_name: output filename
        music_paths: optional list of paths to music file(s)
        crf: constant rate factor (lower = better quality, 18 is visually lossless)
        preset: encoding speed/quality tradeoff
    """
    fps = config.get("fps", 30)
    output_path = OUTPUT_DIR / output_name

    OUTPUT_DIR.mkdir(exist_ok=True)

    frame_count = count_frames()
    if frame_count == 0:
        print("ERROR: No frames found in frames/ directory.")
        print("Run 04_render_morph.py first.")
        sys.exit(1)

    duration = frame_count / fps
    print(f"Encoding {frame_count} frames at {fps}fps ({duration:.1f}s)...")

    # Normalize music_paths
    if not music_paths:
        music_paths = []

    # Read render timing info (written by 04_render_morph.py)
    render_info_path = FRAMES_DIR / "render_info.json"
    music_delay_s = 0.0
    if render_info_path.exists():
        try:
            with open(render_info_path, "r") as f:
                render_info = json.load(f)
            music_delay_s = render_info.get("music_delay_s", 0.0)
        except (json.JSONDecodeError, IOError):
            pass

    # Build FFmpeg command
    input_pattern = str(FRAMES_DIR / "frame_%06d.png")

    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-framerate", str(fps),
        "-i", input_pattern,
    ]

    if len(music_paths) == 1:
        # Single music file: loop to fill video duration
        cmd.extend(["-stream_loop", "-1", "-i", str(music_paths[0])])
    elif len(music_paths) > 1:
        # Multiple music files: add each as a separate input
        for mp in music_paths:
            cmd.extend(["-i", str(mp)])

    cmd.extend([
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",  # Compatibility
        "-movflags", "+faststart",  # Web-optimized
    ])

    if len(music_paths) == 1:
        # Single file: simple audio filter chain
        audio_filters = []
        if music_delay_s > 0:
            delay_ms = int(music_delay_s * 1000)
            audio_filters.append(f"adelay={delay_ms}|{delay_ms}")
        fade_out_start = max(0, duration - 3.0)
        audio_filters.append(f"afade=t=out:st={fade_out_start:.2f}:d=3")

        cmd.extend([
            "-af", ",".join(audio_filters),
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",  # End when shortest stream ends
        ])
    elif len(music_paths) > 1:
        # Multiple files: concat audio inputs, then apply delay + fade
        n = len(music_paths)
        concat_inputs = "".join(f"[{i+1}:a]" for i in range(n))
        filter_parts = [f"{concat_inputs}concat=n={n}:v=0:a=1[acat]"]

        delay_fade = []
        if music_delay_s > 0:
            delay_ms = int(music_delay_s * 1000)
            delay_fade.append(f"adelay={delay_ms}|{delay_ms}")
        fade_out_start = max(0, duration - 3.0)
        delay_fade.append(f"afade=t=out:st={fade_out_start:.2f}:d=3")

        filter_parts.append(f"[acat]{','.join(delay_fade)}[aout]")

        cmd.extend([
            "-filter_complex", ";".join(filter_parts),
            "-map", "0:v", "-map", "[aout]",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
        ])

    cmd.append(str(output_path))

    print(f"Running: {' '.join(cmd[:10])}...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FFmpeg error:\n{result.stderr[-1000:]}")
        sys.exit(1)

    # Get output file size
    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"SUCCESS!")
    print(f"  Output: {output_path}")
    print(f"  Size: {size_mb:.1f} MB")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Resolution: {config.get('output_size', 1024)}x{config.get('output_size', 1024)}")
    print(f"  FPS: {fps}")

    return output_path


def add_year_overlay(config, manifest):
    """
    Create a version with year labels overlaid on the video.

    This generates a text file for FFmpeg's drawtext filter with
    timestamps for each year change.
    """
    sequence = manifest.get("__sequence__", [])
    if not sequence:
        print("ERROR: No sequence in manifest.")
        return

    fps = config.get("fps", 30)
    hold_frames = config.get("hold_frames", 15)
    morph_frames = config.get("morph_frames", 30)
    frames_per_image = hold_frames + morph_frames
    output_size = config.get("output_size", 1024)

    # Build drawtext filter segments
    segments = []
    current_year = None

    for i, filename in enumerate(sequence):
        entry = manifest.get(filename, {})
        year = entry.get("sort_year")
        if year and year != current_year:
            current_year = year
            start_time = i * frames_per_image / fps
            # Show year for the duration of this image + morph
            end_time = start_time + frames_per_image / fps
            segments.append((start_time, end_time, str(year)))

    if not segments:
        print("No year data found for overlay.")
        return

    # Build complex FFmpeg drawtext filter
    input_video = OUTPUT_DIR / "growing_up_ted.mp4"
    output_video = OUTPUT_DIR / "growing_up_ted_with_years.mp4"

    if not input_video.exists():
        print(f"ERROR: {input_video} not found. Run basic encoding first.")
        return

    # Create drawtext filter string
    filters = []
    for start, end, year in segments:
        filters.append(
            f"drawtext=text='{year}':"
            f"fontsize={output_size // 20}:"
            f"fontcolor=white:"
            f"borderw=3:"
            f"bordercolor=black:"
            f"x=(w-text_w)/2:"
            f"y=h-{output_size // 10}:"
            f"enable='between(t,{start:.2f},{end:.2f})'"
        )

    filter_str = ",".join(filters)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vf", filter_str,
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_video)
    ]

    print(f"Adding year labels to video...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"FFmpeg error:\n{result.stderr[-500:]}")
        return

    size_mb = output_video.stat().st_size / (1024 * 1024)
    print(f"Output with year labels: {output_video} ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Encode video from frames")
    parser.add_argument("--fps", type=int, default=None, help="Frame rate")
    parser.add_argument("--music", type=str, nargs="*", default=None,
                        help="Path(s) to background music file(s)")
    parser.add_argument("--crf", type=int, default=18,
                        help="Quality (0-51, lower=better, default=18)")
    parser.add_argument("--preset", default="slow",
                        choices=["ultrafast", "fast", "medium", "slow", "veryslow"],
                        help="Encoding speed/quality tradeoff")
    parser.add_argument("--add-year-labels", action="store_true",
                        help="Create a second version with year overlays")
    parser.add_argument("--output", type=str, default="growing_up_ted.mp4",
                        help="Output filename")
    args = parser.parse_args()

    config = load_config()
    if args.fps:
        config["fps"] = args.fps

    # Encode base video
    music_paths = args.music if args.music else None
    output_path = encode_video(
        config,
        output_name=args.output,
        music_paths=music_paths,
        crf=args.crf,
        preset=args.preset
    )

    # Optionally add year labels
    if args.add_year_labels:
        manifest = load_manifest()
        add_year_overlay(config, manifest)

    print(f"\nAll done! Open the video:")
    print(f"  open \"{output_path}\"")


if __name__ == "__main__":
    main()

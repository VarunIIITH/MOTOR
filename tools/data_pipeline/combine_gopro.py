#!/usr/bin/env python3
import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from moviepy import VideoFileClip, concatenate_videoclips
import argparse

# ---------------------------- Utils ---------------------------- #
def find_telemetry_file(video_filename: str, telemetry_folder: Path):
    """
    Finds the corresponding telemetry file for a given video filename.
    e.g., video 'GX010406.MP4' matches telemetry 'GX010406*.csv'.
    """
    base_name = Path(video_filename).stem
    matches = list(telemetry_folder.glob(f"{base_name}*.csv"))
    return matches[0] if matches else None


def combine_and_process_telemetry(input_folder: Path,
                                  video_files,
                                  csv_output_path: Path,
                                  force: bool = False):
    """
    Processes, combines, and interpolates telemetry data for a list of video files.
    Skips if csv_output_path exists and not force.
    """
    if csv_output_path.exists() and not force:
        print(f"✅ Telemetry already exists: {csv_output_path.name} (skip).")
        return

    all_telemetry_dfs = []
    total_frames = 0
    video_fps = None

    print("\n🚀 Processing files for telemetry and metadata...")
    for video_filename in video_files:
        video_path = input_folder / video_filename

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"⚠️ Warning: Could not open video file '{video_path}'. Skipping.")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if video_fps is None:
            video_fps = fps
        elif video_fps != fps:
            print(f"⚠️ Warning: Mismatched FPS in {video_filename} ({fps}) vs expected ({video_fps}).")

        total_frames += frame_count
        print(f"  - Reading '{video_filename}': {frame_count} frames at {fps:.2f} FPS.")
        cap.release()

        telemetry_path = find_telemetry_file(video_filename, input_folder)
        if telemetry_path is None:
            print(f"⚠️ Warning: No matching telemetry file found for '{video_filename}'.")
            continue

        try:
            telemetry_df = pd.read_csv(telemetry_path)
            all_telemetry_dfs.append(telemetry_df)
            print(f"  - Loaded telemetry: '{telemetry_path.name}'")
        except Exception as e:
            print(f"❌ Error reading telemetry file {telemetry_path}: {e}")

    if not all_telemetry_dfs:
        print("\n❌ Error: No telemetry data could be loaded. Telemetry file will not be created.")
        return

    combined_telemetry_df = pd.concat(all_telemetry_dfs, ignore_index=True)

    # Expect a 'date' column -> adjust if your CSV header differs
    combined_telemetry_df['datetime'] = pd.to_datetime(combined_telemetry_df['date'])
    combined_telemetry_df = combined_telemetry_df.sort_values(by='datetime').reset_index(drop=True)

    start_time = combined_telemetry_df['datetime'].iloc[0]
    combined_telemetry_df['time_seconds'] = (combined_telemetry_df['datetime'] - start_time).dt.total_seconds()

    frame_timestamps = np.arange(total_frames) / video_fps
    output_data = {'frame_number': np.arange(total_frames, dtype=int),
                   'timestamp_sec': frame_timestamps}

    columns_to_interpolate = [
        'GPS (Lat.) [deg]', 'GPS (Long.) [deg]', 'GPS (Alt.) [m]',
        'GPS (2D speed) [m/s]', 'GPS (3D speed) [m/s]', 'precision'
    ]
    xp = combined_telemetry_df['time_seconds'].values
    for col in columns_to_interpolate:
        if col in combined_telemetry_df.columns:
            fp = combined_telemetry_df[col].values
            output_data[col] = np.interp(frame_timestamps, xp, fp)

    output_df = pd.DataFrame(output_data)
    print(f"\n📊 Saving combined telemetry to: {csv_output_path}")
    csv_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(csv_output_path, index=False)
    print("✅ Telemetry combination complete!")


def combine_gopro_videos(input_folder: Path,
                         video_output_path: Path,
                         video_files,
                         fps_out: int = 30,
                         force: bool = False):
    """
    Concatenates GoPro videos and writes a single output file.
    Skips if video_output_path exists and not force.
    """
    if video_output_path.exists() and not force:
        print(f"✅ Combined video already exists: {video_output_path.name} (skip).")
        return

    print("\n✅ Videos to combine:")
    for vf in video_files:
        print(f" - {vf}")

    clips = []
    for filename in video_files:
        file_path = input_folder / filename
        try:
            clip = VideoFileClip(str(file_path))
            clips.append(clip)
        except Exception as e:
            print(f"❌ Error loading clip {filename}: {e}")

    if not clips:
        print("❌ No video clips were successfully loaded. Aborting video combination.")
        return

    final_clip = concatenate_videoclips(clips, method="compose")

    print(f"🚀 Saving combined video to: {video_output_path}")
    video_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_clip.write_videofile(str(video_output_path), fps=fps_out)

    # Clean up
    for clip in clips:
        clip.close()
    final_clip.close()

    print("✅ Video combination complete!")


# ---------------------------- Main ---------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Combine GoPro videos & telemetry with skip-if-exists logic.")
    parser.add_argument("input_folder", help="Path to folder containing GoPro videos + telemetry csvs.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing combined outputs.")
    parser.add_argument("--skip-video", action="store_true",
                        help="Skip video concatenation step.")
    parser.add_argument("--skip-telemetry", action="store_true",
                        help="Skip telemetry combination step.")
    parser.add_argument("--fps", type=int, default=30,
                        help="Output FPS for the combined video (default: 30).")
    parser.add_argument("--base-output-dir", default="",
                        help="Base directory where combined outputs are saved.")
    args = parser.parse_args()

    input_folder = Path(args.input_folder).resolve()
    if not input_folder.is_dir():
        print("❌ The provided path is not a valid directory.")
        sys.exit(1)

    # Collect videos
    video_files = sorted([f for f in os.listdir(input_folder)
                          if f.lower().endswith((".mp4", ".mov"))])
    if not video_files:
        print("❌ No .mp4/.mov files found in the folder.")
        sys.exit(1)

    # Mirror your original relative path logic
    base_output_dir = Path(args.base_output_dir)
    try:
        relative_path_start = Path(input_folder).parents[3]  # 4 levels up
        relative_path = input_folder.relative_to(relative_path_start)
        output_dir = base_output_dir / relative_path
    except Exception:
        output_dir = base_output_dir / input_folder.name

    video_output_filename = f"{input_folder.name}_combined_output.mp4"
    csv_output_filename = f"{input_folder.name}_combined_telemetry.csv"

    video_output_path = output_dir / video_output_filename
    csv_output_path = output_dir / csv_output_filename

    # 1. Telemetry
    if not args.skip_telemetry:
        combine_and_process_telemetry(input_folder, video_files, csv_output_path, force=args.force)
    else:
        print("⏭️ Skipping telemetry step (per flag).")

    # 2. Video
    if not args.skip_video:
        combine_gopro_videos(input_folder, video_output_path, video_files, fps_out=args.fps, force=args.force)
    else:
        print("⏭️ Skipping video step (per flag).")


if __name__ == "__main__":
    main()

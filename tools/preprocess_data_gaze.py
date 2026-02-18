#!/usr/bin/env python3
"""Extract gaze video clips from annotated timestamps.

Same as preprocess_data.py but for the gaze camera stream.

Usage:
    python preprocess_data_gaze.py \
        --excel annotations.xlsx \
        --video gaze_video.mp4 \
        --output_dir clips_gaze/ \
        --metadata_file gaze_metadata.json \
        --video_id 01
"""

import os
import json
import argparse
import pandas as pd
from moviepy import VideoFileClip


def parse_time_to_sec(t_str):
    parts = [float(p) for p in t_str.split(":")]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = parts
        return m * 60 + s
    elif len(parts) == 1:
        return parts[0]
    raise ValueError(f"Invalid time format: {t_str}")


def extract_clips(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_excel(args.excel, engine="openpyxl")
    df_video = df[df["Video ID"].astype(str) == args.video_id]
    video = VideoFileClip(args.video)
    metadata = []

    for i, (_, row) in enumerate(df_video.iterrows()):
        ts = row["Timestamp"]
        if not (isinstance(ts, str) and "-" in ts):
            print(f"Skipping row {i}: invalid timestamp '{ts}'")
            continue

        start_str, end_str = [p.strip() for p in ts.split("-")]
        start_sec = parse_time_to_sec(start_str)
        end_sec = parse_time_to_sec(end_str)
        if end_sec <= start_sec:
            continue

        clip_name = f"{args.video_id}_clip_{i:03d}.mp4"
        clip_path = os.path.join(args.output_dir, clip_name)
        clip = video.subclipped(start_sec, end_sec)
        clip.write_videofile(clip_path, codec="libx264", audio=False)

        metadata.append({
            "video_id": args.video_id, "clip_index": i,
            "clip_filename": clip_name, "start_seconds": start_sec,
            "end_seconds": end_sec, "duration_seconds": end_sec - start_sec,
            "event": row["Event"],
        })

    with open(args.metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Extracted {len(metadata)} gaze clips. Metadata: {args.metadata_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract gaze clips from annotated video")
    parser.add_argument("--excel", required=True, help="Path to annotation Excel file")
    parser.add_argument("--video", required=True, help="Path to gaze input video")
    parser.add_argument("--output_dir", required=True, help="Directory for output clips")
    parser.add_argument("--metadata_file", required=True, help="Path for metadata JSON")
    parser.add_argument("--video_id", required=True, help="Video ID to filter by")
    extract_clips(parser.parse_args())

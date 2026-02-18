#!/usr/bin/env python3
"""QR-code-based multi-video time synchronization.

Supports 3 or 4 video inputs. Detects QR codes to find matching timestamps,
merges frame indices via nearest-time alignment, and concatenates synced videos.

Usage:
    # 4 videos (Aria/Pupil + 3 GoPros):
    python time_sync.py vid1.mp4 vid2.mp4 vid3.mp4 vid4.mp4

    # 3 videos:
    python time_sync.py vid1.mp4 vid2.mp4 vid3.mp4
"""

import sys
import os
import re
import datetime
import time

import av
import cv2
import numpy as np
import pandas as pd
from pyzbar import pyzbar


def detect_qr_code(frame):
    """Detect and decode QR codes in a frame."""
    decoded = pyzbar.decode(frame)
    if decoded:
        return decoded[0].data.decode("utf-8")
    return None


def extract_unix_timestamp(qr_data):
    """Extract UNIX timestamp from QR code data containing a datetime string."""
    match = re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}", qr_data)
    if match:
        dt = datetime.datetime.strptime(match.group(), "%Y-%m-%d %H:%M:%S")
        return time.mktime(dt.timetuple())
    return None


def get_video_timestamp(vid_path, verbose=True):
    """Scan initial frames of a video for a QR code with a timestamp."""
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    timestamp = None

    for i in range(int(fps * 30)):  # scan first 30 seconds
        ret, frame = cap.read()
        if not ret:
            break
        qr = detect_qr_code(frame)
        if qr:
            ts = extract_unix_timestamp(qr)
            if ts:
                timestamp = ts - (i / fps)
                if verbose:
                    print(f"  QR found at frame {i}: {qr} → start_ts={timestamp:.3f}")
                break

    cap.release()
    return timestamp


def get_video_times(vid_path, start_timestamp):
    """Generate per-frame timestamps for a video."""
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    timestamps = [start_timestamp + i / fps for i in range(n_frames)]
    df = pd.DataFrame(timestamps, columns=["timestamp"])
    out_path = vid_path[:-4] + "_timestamps.csv"
    df.to_csv(out_path, index=False)
    return out_path


def merge_timestamps(timefiles, outfile):
    """Merge timestamp files via nearest-time alignment (ref = first video)."""
    dfs = []
    for i, tf in enumerate(timefiles):
        df = pd.read_csv(tf)
        col = f"timestamp{i+1}"
        df[col] = df["timestamp"] if "timestamp" in df.columns else df.iloc[:, 0]
        df[f"index{i+1}"] = df.index
        dfs.append(df[[col, f"index{i+1}"]])

    ref = dfs[0]
    merged = ref.copy()
    for i in range(1, len(dfs)):
        merged = pd.merge_asof(
            merged.sort_values(f"timestamp1"),
            dfs[i].sort_values(f"timestamp{i+1}"),
            left_on=f"timestamp1",
            right_on=f"timestamp{i+1}",
            direction="backward",
        )

    idx_cols = [f"index{i+1}" for i in range(len(dfs))]
    merged[idx_cols].to_csv(outfile, index=False)
    print(f"  Merged timestamps saved to: {outfile}")
    return outfile


def concat_videos(video_paths, timestamp_file, outfile):
    """Concatenate time-synced frames from multiple videos side by side."""
    merged = pd.read_csv(timestamp_file)
    n_vids = len(video_paths)

    caps = [cv2.VideoCapture(v) for v in video_paths]
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    heights = [int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)) for c in caps]
    widths = [int(c.get(cv2.CAP_PROP_FRAME_WIDTH)) for c in caps]

    # resize all to same height
    target_h = min(heights)
    out_w = sum(int(w * target_h / h) for w, h in zip(widths, heights))

    writer = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, target_h))

    for _, row in merged.iterrows():
        frames = []
        for i in range(n_vids):
            idx = int(row[f"index{i+1}"])
            caps[i].set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = caps[i].read()
            if not ret:
                frame = np.zeros((target_h, widths[i], 3), dtype=np.uint8)
            h, w = frame.shape[:2]
            new_w = int(w * target_h / h)
            frame = cv2.resize(frame, (new_w, target_h))
            frames.append(frame)

        combined = np.hstack(frames)
        writer.write(combined)

    writer.release()
    for c in caps:
        c.release()
    print(f"  Output video: {outfile}")


def save_time_synced_videos(video_paths, timestamp_file, output_paths):
    """Save individual time-synced videos."""
    merged = pd.read_csv(timestamp_file)
    n_vids = len(video_paths)

    caps = [cv2.VideoCapture(v) for v in video_paths]
    fps = int(caps[0].get(cv2.CAP_PROP_FPS))
    writers = []
    for i, c in enumerate(caps):
        h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
        writers.append(cv2.VideoWriter(output_paths[i], cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)))

    for _, row in merged.iterrows():
        for i in range(n_vids):
            idx = int(row[f"index{i+1}"])
            caps[i].set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = caps[i].read()
            if ret:
                writers[i].write(frame)

    for c in caps:
        c.release()
    for w in writers:
        w.release()
    print(f"  Synced videos saved: {output_paths}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python time_sync.py vid1.mp4 vid2.mp4 vid3.mp4 [vid4.mp4]")
        sys.exit(1)

    videos = sys.argv[1:]
    n_vids = len(videos)
    print(f"Synchronizing {n_vids} videos...")

    # Get start timestamps via QR detection
    timestamps = []
    for v in videos:
        print(f"\nScanning: {v}")
        ts = get_video_timestamp(v)
        if ts is None:
            print(f"  WARNING: No QR code found in {v}")
        timestamps.append(ts)

    # Generate per-frame timestamp files
    timefiles = []
    for v, ts in zip(videos, timestamps):
        if ts is not None:
            tf = get_video_times(v, ts)
            timefiles.append(tf)

    if len(timefiles) < n_vids:
        print("ERROR: Could not get timestamps for all videos.")
        sys.exit(1)

    # Merge timestamps
    merged_path = "merged_timestamps.csv"
    merge_timestamps(timefiles, merged_path)

    # Concatenate
    print("\nConcatenating videos...")
    concat_videos(videos, merged_path, "final_synced_output.mp4")

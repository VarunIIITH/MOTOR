#!/usr/bin/env python3
"""
Create a gaze‑patch video (300×300 crops) or a full‑frame video with a gaze marker
directly from an undistorted, already‑synced Pupil/Aria video and its gaze CSV/XLSX.

Required gaze columns:
    frame, gaze_x_undist, gaze_y_undist

New:
- Robust patch extraction (no copyMakeBorder crashes at borders)
- Configurable Y offset (default +50 px) and optional X offset
- Full-frame mode with a gaze circle overlay
"""

import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import argparse

# ----------------------------------------------------------------------
# Robust 300×300 gaze‑centred patch (no copyMakeBorder)
# ----------------------------------------------------------------------
def extract_patch(frame, cx, cy, size=300, blank=False):
    """
    Returns a (size x size x 3) patch centered at (cx, cy).
    Handles borders safely by pasting the overlapping region into a black canvas.
    """
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    if blank or frame is None or np.isnan(cx) or np.isnan(cy):
        return canvas

    h, w = frame.shape[:2]
    half = size // 2

    # Desired patch box in source coords
    x1, y1 = int(round(cx)) - half, int(round(cy)) - half
    x2, y2 = x1 + size, y1 + size

    # Overlap with source image
    src_x1 = max(x1, 0)
    src_y1 = max(y1, 0)
    src_x2 = min(x2, w)
    src_y2 = min(y2, h)

    # If no overlap, return black canvas
    if src_x1 >= src_x2 or src_y1 >= src_y2:
        return canvas

    # Corresponding destination coords on canvas
    dst_x1 = src_x1 - x1  # shift because canvas starts at (0,0)
    dst_y1 = src_y1 - y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    # Paste
    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]
    return canvas


def draw_gaze_circle(frame, cx, cy, radius=50, thickness=5, color=(0, 0, 255)):
    """Draw a circle marker if the point lies within frame bounds."""
    if frame is None or np.isnan(cx) or np.isnan(cy):
        return frame
    h, w = frame.shape[:2]
    x, y = int(round(cx)), int(round(cy))
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(frame, (x, y), int(radius), color, int(thickness))
    return frame


def create_gaze_video(
    gaze_file,           # CSV / Excel with frame, gaze_x_undist, gaze_y_undist
    undist_video,        # already-synced MP4
    output_video,        # destination MP4
    mode="patch",        # "patch" or "full"
    patch_size=300,
    y_offset=50,         # +50 px down by default (OpenCV y grows downward)
    x_offset=0,          # optional horizontal offset (default 0)
    draw_radius=50,      # only used in full mode
    draw_thickness=5,    # only used in full mode
    fps=None             # default: use source-video FPS
):
    # ---------- Load gaze file ----------
    ext = os.path.splitext(gaze_file)[1].lower()
    if ext in {".xls", ".xlsx"}:
        df = pd.read_excel(gaze_file)
    else:
        df = pd.read_csv(gaze_file)

    required = {"frame", "gaze_x_undist", "gaze_y_undist"}
    if not required.issubset(df.columns):
        missing = ", ".join(required - set(df.columns))
        raise ValueError(f"Gaze file missing required columns: {missing}")

    df = df.sort_values("frame").reset_index(drop=True)

    # ---------- Open video I/O ----------
    cap = cv2.VideoCapture(undist_video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {undist_video}")

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    if mode == "patch":
        out_w, out_h = patch_size, patch_size
    elif mode == "full":
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_w, out_h = src_w, src_h
    else:
        raise ValueError("mode must be one of: 'patch', 'full'")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (out_w, out_h))

    # ---------- Iterate through gaze rows ----------
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Writing {mode} video"):
        fidx = int(row["frame"])
        cx   = float(row["gaze_x_undist"]) + float(x_offset)
        cy   = float(row["gaze_y_undist"]) + float(y_offset)

        # random access the source frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            # write black frame when out of bounds
            if mode == "patch":
                out = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
            else:
                out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            writer.write(out)
            continue

        if mode == "patch":
            out = extract_patch(frame, cx, cy, size=patch_size, blank=False)
        else:
            out = frame.copy()
            out = draw_gaze_circle(out, cx, cy, radius=draw_radius, thickness=draw_thickness)

        writer.write(out)

    cap.release()
    writer.release()
    print(f"\n✅ Done. Saved {len(df)}-frame {mode} video → {output_video}")


def parse_args():
    p = argparse.ArgumentParser(description="Create gaze-centered patch video or full-frame video with gaze overlay.")
    p.add_argument("--gaze-file", required=True, help="CSV/XLSX with columns: frame,gaze_x_undist,gaze_y_undist")
    p.add_argument("--video", required=True, help="Undistorted, already-synced input video (MP4).")
    p.add_argument("--out", required=True, help="Output MP4 path.")
    p.add_argument("--mode", choices=["patch", "full"], default="patch", help="Output mode.")
    p.add_argument("--patch-size", type=int, default=300, help="Patch size (only for --mode patch).")
    p.add_argument("--y-offset", type=float, default=50.0, help="Y-axis pixel offset applied to gaze (positive = down).")
    p.add_argument("--x-offset", type=float, default=0.0, help="X-axis pixel offset applied to gaze (positive = right).")
    p.add_argument("--draw-radius", type=int, default=50, help="Circle radius (full mode).")
    p.add_argument("--draw-thickness", type=int, default=5, help="Circle thickness (full mode).")
    p.add_argument("--fps", type=float, default=None, help="Output FPS (default: source FPS).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_gaze_video(
        gaze_file     = args.gaze_file,
        undist_video  = args.video,
        output_video  = args.out,
        mode          = args.mode,
        patch_size    = args.patch_size,
        y_offset      = args.y_offset,
        x_offset      = args.x_offset,
        draw_radius   = args.draw_radius,
        draw_thickness= args.draw_thickness,
        fps           = args.fps
    )

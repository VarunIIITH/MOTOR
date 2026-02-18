#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gaze_transfer_homography.py

Transfer gaze from a REFERENCE (Aria) video to one or more TARGET videos using
per-frame homography (SIFT+FLANN). Supports pixel offsets, configurable circle
overlay, debug snapshots, and writes 1-minute segment clips into a 'clips'
subfolder next to each output video.

CSV schema (REFERENCE pixel coordinates):
    frame,gaze_x_undist,gaze_y_undist

Homography convention:
    H_target_to_ref maps target -> reference.
To map gaze from reference -> target, use inv(H_target_to_ref).
"""

import os
import cv2
import argparse
import numpy as np
import pandas as pd
import pathlib
from typing import Optional, Tuple, List

# ------------------------------- CSV Loader ------------------------------ #

def load_gaze_csv_strict(gaze_csv: str) -> pd.DataFrame:
    df = pd.read_csv(gaze_csv)
    required = ["frame", "gaze_x_undist", "gaze_y_undist"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Gaze CSV must have columns {required}, missing {missing}")

    df = df[required].copy()
    df["frame"] = df["frame"].astype(int)
    df = df.sort_values("frame").reset_index(drop=True)
    df = df.rename(columns={
        "gaze_x_undist": "x",
        "gaze_y_undist": "y",
    })
    return df

def gaze_for_frame(df_gaze: pd.DataFrame, frame_idx: int) -> Optional[Tuple[float, float]]:
    row = df_gaze[df_gaze["frame"] == frame_idx]
    if len(row) == 0:
        return None
    x = float(row.iloc[0]["x"])
    y = float(row.iloc[0]["y"])
    if np.isnan(x) or np.isnan(y):
        return None
    return (x, y)

# -------------------------- Homography Utilities ------------------------- #

def compute_homography_sift(frame_ref: np.ndarray,
                            frame_tgt: np.ndarray,
                            sift, flann,
                            lowe_ratio: float = 0.7,
                            ransac_thresh: float = 5.0,
                            max_size: int = 960) -> Optional[np.ndarray]:
    def scale_frame(img):
        h, w = img.shape[:2]
        scale = 1.0
        if max(h, w) > max_size:
            scale = max_size / float(max(h, w))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img, scale

    img_ref, s_ref = scale_frame(frame_ref)
    img_tgt, s_tgt = scale_frame(frame_tgt)

    kp1, des1 = sift.detectAndCompute(img_ref, None)  # reference
    kp2, des2 = sift.detectAndCompute(img_tgt, None)  # target

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None

    # Match target->reference (query: target, train: reference)
    matches = flann.knnMatch(des2, des1, k=2)
    good = []
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < lowe_ratio * n.distance:
            good.append(m)
    if len(good) < 10:
        return None

    pts_tgt = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Unscale back to original resolution
    if abs(s_ref - 1.0) > 1e-6 or abs(s_tgt - 1.0) > 1e-6:
        pts_tgt /= s_tgt
        pts_ref /= s_ref

    H, _ = cv2.findHomography(pts_tgt, pts_ref, cv2.RANSAC, ransac_thresh)
    return H

def map_point_ref_to_target(xy_ref: Tuple[float, float], H_t2r: Optional[np.ndarray]) -> Optional[Tuple[int, int]]:
    if H_t2r is None:
        return None
    try:
        H_inv = np.linalg.inv(H_t2r)
    except np.linalg.LinAlgError:
        return None

    x, y = xy_ref
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H_inv @ p
    if abs(q[2]) < 1e-9:
        return None
    qx = q[0] / q[2]
    qy = q[1] / q[2]
    return int(round(qx)), int(round(qy))

def draw_gaze(frame: np.ndarray,
              xy_t: Optional[Tuple[int, int]],
              radius: int = 8,
              color=(0, 0, 255),
              thickness: int = -1):
    if xy_t is None:
        return
    x, y = xy_t
    h, w = frame.shape[:2]
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(frame, (x, y), radius, color, thickness)

# ------------------------------ Misc Utils ------------------------------- #

def ensure_dir(p: str) -> str:
    path = pathlib.Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)

def make_segment_path(out_path: str, seg_idx: int) -> str:
    """
    Put segments into '<out_dir>/clips/'. Name pattern: <stem>_partXXX.mp4
    Example:
      out_path = /A/B/gaze_transfered_helmet.mp4
      segment   = /A/B/clips/gaze_transfered_helmet_part001.mp4
    """
    out_p = pathlib.Path(out_path)
    clips_dir = out_p.with_name("clips")  # sibling 'clips' folder alongside output file
    clips_dir.mkdir(parents=True, exist_ok=True)
    stem = out_p.stem
    suffix = out_p.suffix
    return str(clips_dir / f"{stem}_part{seg_idx:03d}{suffix}")

# ----------------------------- Main Pipeline ----------------------------- #

class VideoHomographyGaze:
    def __init__(self,
                 ref_video: str,
                 targets: List[str],
                 outs: List[str],
                 gaze_csv: str,
                 ref_offset: int = 0,
                 target_offsets: Optional[List[int]] = None,
                 stride: int = 1,
                 ransac_thresh: float = 5.0,
                 draw_radius: int = 8,
                 draw_thickness: int = -1,
                 lowe_ratio: float = 0.7,
                 max_size: int = 960,
                 homography_fallback_identity: bool = True,
                 debug_frames: Optional[List[int]] = None,
                 debug_dir: Optional[str] = None,
                 x_offset: int = 0,
                 y_offset: int = 0,
                 make_segments: bool = False,
                 segment_minutes: int = 1):
        """
        Writes full videos, and when make_segments=True also writes fixed-length
        segments into a 'clips' folder next to each output.
        """
        assert len(targets) == len(outs), "Provide one --out_video per --target_video"
        if target_offsets is None:
            target_offsets = [0] * len(targets)
        assert len(target_offsets) == len(targets), "--target_offsets must match target videos"

        self.ref_video = ref_video
        self.targets = targets
        self.outs = outs
        self.ref_offset = max(0, int(ref_offset))
        self.target_offsets = [max(0, int(o)) for o in target_offsets]
        self.stride = max(1, int(stride))
        self.ransac_thresh = float(ransac_thresh)
        self.draw_radius = int(draw_radius)
        self.draw_thickness = int(draw_thickness)
        self.lowe_ratio = float(lowe_ratio)
        self.max_size = int(max_size)
        self.h_fallback_identity = bool(homography_fallback_identity)
        self.x_offset = int(x_offset)
        self.y_offset = int(y_offset)

        # debug
        self.debug_frames = set(int(f) for f in (debug_frames or []))
        if debug_dir is None:
            base = os.path.dirname(self.outs[0]) if self.outs else "."
            debug_dir = os.path.join(base, "debug")
        self.debug_dir = ensure_dir(debug_dir)

        # segments
        self.make_segments = bool(make_segments)
        self.segment_minutes = max(1, int(segment_minutes))

        # Features
        self.sift = cv2.SIFT_create()
        index_params = dict(algorithm=1, trees=5)  # KD-Tree
        search_params = dict(checks=64)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

        # Data
        self.df_gaze = load_gaze_csv_strict(gaze_csv)

        # Reference
        self.cap_ref = cv2.VideoCapture(ref_video)
        if not self.cap_ref.isOpened():
            raise RuntimeError(f"Cannot open reference video: {ref_video}")
        self.ref_fps = self.cap_ref.get(cv2.CAP_PROP_FPS) or 30.0

        # Targets + writers
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.caps_t: List[cv2.VideoCapture] = []
        self.writers_full: List[cv2.VideoWriter] = []
        self.segment_writers: List[Optional[cv2.VideoWriter]] = []
        self.frames_in_segment: List[int] = []
        self.seg_idx: List[int] = []
        self.frames_per_segment: List[int] = []
        self.t_fps: List[float] = []
        self.t_size: List[Tuple[int, int]] = []

        for tpath, opath in zip(self.targets, self.outs):
            # Ensure the output directory exists
            ensure_dir(os.path.dirname(opath) or ".")

            cap = cv2.VideoCapture(tpath)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open target video: {tpath}")
            fps = cap.get(cv2.CAP_PROP_FPS) or self.ref_fps
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # full writer
            writer_full = cv2.VideoWriter(opath, fourcc, fps, (w, h))
            if not writer_full.isOpened():
                raise RuntimeError(f"Cannot open writer: {opath}")

            # segment writer (lazy init on first frame)
            seg_writer = None
            frames_per_seg = int(round(float(fps) * 60 * self.segment_minutes))

            self.caps_t.append(cap)
            self.writers_full.append(writer_full)
            self.segment_writers.append(seg_writer)
            self.frames_in_segment.append(0)
            self.seg_idx.append(1)
            self.frames_per_segment.append(max(frames_per_seg, 1))
            self.t_fps.append(float(fps))
            self.t_size.append((w, h))

    def close(self):
        # reference
        if self.cap_ref is not None:
            self.cap_ref.release()
        # targets
        for cap in self.caps_t:
            cap.release()
        # full writers
        for w in self.writers_full:
            w.release()
        # segment writers
        for sw in self.segment_writers:
            if sw is not None:
                sw.release()

    def _seek_offsets(self):
        if self.ref_offset > 0:
            self.cap_ref.set(cv2.CAP_PROP_POS_FRAMES, self.ref_offset)
        for cap, off in zip(self.caps_t, self.target_offsets):
            if off > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, off)

    def _save_debug_images(self,
                           ref_abs_idx: int,
                           frame_ref: np.ndarray,
                           gaze_xy_ref: Optional[Tuple[float, float]],
                           frames_tgt: List[np.ndarray],
                           last_H_t2r: List[Optional[np.ndarray]]):
        if ref_abs_idx not in self.debug_frames:
            return

        # Reference (green)
        ref_dbg = frame_ref.copy()
        if gaze_xy_ref is not None:
            cv2.circle(ref_dbg,
                       (int(round(gaze_xy_ref[0])), int(round(gaze_xy_ref[1]))),
                       self.draw_radius, (0, 255, 0), self.draw_thickness)
        cv2.imwrite(os.path.join(self.debug_dir, f"ref_frame_{ref_abs_idx:06d}.png"), ref_dbg)

        # Each target (red)
        for ti, (fr_t, H_t2r) in enumerate(zip(frames_tgt, last_H_t2r)):
            tgt_dbg = fr_t.copy()
            if gaze_xy_ref is not None and H_t2r is not None:
                xy_t = map_point_ref_to_target(gaze_xy_ref, H_t2r)
                if xy_t is not None:
                    cv2.circle(tgt_dbg, (xy_t[0], xy_t[1]),
                               self.draw_radius, (0, 0, 255), self.draw_thickness)
            out_name = f"tgt{ti}_frame_{ref_abs_idx:06d}.png"
            cv2.imwrite(os.path.join(self.debug_dir, out_name), tgt_dbg)

    def _ensure_segment_writer(self, ti: int):
        if self.segment_writers[ti] is not None:
            return
        seg_path = make_segment_path(self.outs[ti], self.seg_idx[ti])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w, h = self.t_size[ti]
        fps = self.t_fps[ti]
        writer = cv2.VideoWriter(seg_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open segment writer: {seg_path}")
        self.segment_writers[ti] = writer
        self.frames_in_segment[ti] = 0

    def _rotate_segment_writer_if_needed(self, ti: int):
        """
        If current segment reached frames_per_segment, close and start next.
        """
        if self.frames_in_segment[ti] >= self.frames_per_segment[ti]:
            # close existing writer
            if self.segment_writers[ti] is not None:
                self.segment_writers[ti].release()
                self.segment_writers[ti] = None
            # next segment index
            self.seg_idx[ti] += 1
            # open next
            self._ensure_segment_writer(ti)

    def run(self):
        try:
            self._seek_offsets()

            # Cache last successful homography per target
            last_H_t2r: List[Optional[np.ndarray]] = [None for _ in self.caps_t]

            global_idx = 0
            while True:
                ok_ref, frame_ref = self.cap_ref.read()
                if not ok_ref:
                    break

                # Absolute reference frame index (POS_FRAMES points to next)
                ref_abs_idx = int(self.cap_ref.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                # Gaze in reference pixels (+ optional offsets)
                gaze_xy_ref = gaze_for_frame(self.df_gaze, ref_abs_idx)
                if gaze_xy_ref is not None:
                    gx, gy = gaze_xy_ref
                    gaze_xy_ref = (gx + self.x_offset, gy + self.y_offset)

                # Read one frame from each target
                frames_tgt: List[np.ndarray] = []
                for cap in self.caps_t:
                    ok_t, frame_t = cap.read()
                    if not ok_t:
                        # End when any target ends (to keep alignment)
                        return
                    frames_tgt.append(frame_t)

                # Recompute homography on stride
                if global_idx % self.stride == 0:
                    for ti in range(len(self.caps_t)):
                        H = compute_homography_sift(
                            frame_ref, frames_tgt[ti],
                            self.sift, self.flann,
                            lowe_ratio=self.lowe_ratio,
                            ransac_thresh=self.ransac_thresh,
                            max_size=self.max_size
                        )
                        if H is not None:
                            last_H_t2r[ti] = H
                        elif last_H_t2r[ti] is None and self.h_fallback_identity:
                            last_H_t2r[ti] = np.eye(3, dtype=np.float64)

                # Overlay gaze and write to full + segment writers
                for ti, (fr_t, H_t2r, writer_full) in enumerate(zip(frames_tgt, last_H_t2r, self.writers_full)):
                    xy_t = None
                    if gaze_xy_ref is not None and H_t2r is not None:
                        xy_t = map_point_ref_to_target(gaze_xy_ref, H_t2r)

                    draw_gaze(fr_t, xy_t, radius=self.draw_radius, thickness=self.draw_thickness)

                    # write full video
                    writer_full.write(fr_t)

                    # write segment video(s)
                    if self.make_segments:
                        if self.segment_writers[ti] is None:
                            self._ensure_segment_writer(ti)
                        self.segment_writers[ti].write(fr_t)
                        self.frames_in_segment[ti] += 1
                        self._rotate_segment_writer_if_needed(ti)

                # Debug images on selected frames
                self._save_debug_images(ref_abs_idx, frame_ref, gaze_xy_ref, frames_tgt, last_H_t2r)

                global_idx += 1

        finally:
            self.close()

# --------------------------------- CLI ---------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Transfer gaze via per-frame homography; write full videos and 1-minute segments into 'clips' folders.")
    p.add_argument("--ref_video", required=True, help="Path to reference (Aria) video.")
    p.add_argument("--gaze_csv", required=True, help="CSV with columns: frame,gaze_x_undist,gaze_y_undist.")
    p.add_argument("--target_video", nargs="+", required=True, help="One or more target videos (Helmet/Front).")
    p.add_argument("--out_video", nargs="+", required=True, help="Output video path(s), one per target.")
    p.add_argument("--ref_offset", type=int, default=0, help="Frame offset for reference video.")
    p.add_argument("--target_offsets", nargs="+", type=int, default=None,
                   help="Per-target frame offsets; same length/order as --target_video.")
    p.add_argument("--stride", type=int, default=1, help="Recompute homography every N frames (>=1).")
    p.add_argument("--ransac_thresh", type=float, default=5.0, help="RANSAC reprojection threshold (pixels).")
    p.add_argument("--draw_radius", type=int, default=8, help="Gaze circle radius (pixels).")
    p.add_argument("--draw_thickness", type=int, default=-1, help="Circle thickness (-1 for filled).")
    p.add_argument("--lowe_ratio", type=float, default=0.7, help="Lowe ratio test threshold.")
    p.add_argument("--max_size", type=int, default=960, help="Max dimension for SIFT processing.")
    p.add_argument("--no_identity_fallback", action="store_true",
                   help="Disable identity fallback when homography fails at start.")
    p.add_argument("--debug_frames", nargs="+", type=int, default=[100, 500, 1000],
                   help="Reference frame indices at which to save debug images.")
    p.add_argument("--debug_dir", type=str, default=None,
                   help="Directory to save debug images. Defaults to <first out_video dir>/debug.")
    p.add_argument("--x_offset", type=int, default=0, help="Pixel offset in x for gaze coords (applied in reference).")
    p.add_argument("--y_offset", type=int, default=0, help="Pixel offset in y for gaze coords (applied in reference).")
    p.add_argument("--make_segments", action="store_true", help="Also write fixed-length segment clips.")
    p.add_argument("--segment_minutes", type=int, default=1, help="Length of each segment clip in minutes (default 1).")
    return p.parse_args()

def main():
    args = parse_args()

    if len(args.target_video) != len(args.out_video):
        raise ValueError("Number of --target_video must match number of --out_video.")

    if args.target_offsets is None:
        target_offsets = [0] * len(args.target_video)
    else:
        if len(args.target_offsets) != len(args.target_video):
            raise ValueError("--target_offsets length must match --target_video.")
        target_offsets = args.target_offsets

    v = VideoHomographyGaze(
        ref_video=args.ref_video,
        targets=args.target_video,
        outs=args.out_video,
        gaze_csv=args.gaze_csv,
        ref_offset=args.ref_offset,
        target_offsets=target_offsets,
        stride=args.stride,
        ransac_thresh=args.ransac_thresh,
        draw_radius=args.draw_radius,
        draw_thickness=args.draw_thickness,
        lowe_ratio=args.lowe_ratio,
        max_size=args.max_size,
        homography_fallback_identity=not args.no_identity_fallback,
        debug_frames=args.debug_frames,
        debug_dir=args.debug_dir,
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        make_segments=args.make_segments,
        segment_minutes=args.segment_minutes
    )
    print("Starting gaze transfer...")
    v.run()
    print("Done.")

if __name__ == "__main__":
    main()

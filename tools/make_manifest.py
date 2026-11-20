#!/usr/bin/env python3
"""Build the training manifest from the MOTOR Hugging Face dataset.

Usage:
    python tools/make_manifest.py --data_root /path/to/MOTOR --task behavior
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.labels import BEHAVIOR_CLASSES, LEGALITY_CLASSES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Build MOTOR train/val manifest")
    parser.add_argument("--data_root", required=True, help="Local copy of the Hugging Face dataset")
    parser.add_argument("--task", choices=["behavior", "legality"], default="behavior")
    parser.add_argument("--out", default="data/train_val_manifest.csv")
    parser.add_argument("--gaze_dir", default=None, help="Gaze-centred clips as <ride>/<clip>.mp4")
    args = parser.parse_args()

    ann = pd.read_csv(os.path.join(args.data_root, "annotations.csv"))
    ann = ann[ann["Split"].isin(["train", "val"])]
    classes = BEHAVIOR_CLASSES if args.task == "behavior" else LEGALITY_CLASSES
    col = "Event" if args.task == "behavior" else "Legality"
    keep = ann[col].isin(classes)
    if (~keep).any():
        print(f"Skipping {int((~keep).sum())} clips with {col}: {sorted(ann.loc[~keep, col].unique())}")
    ann = ann[keep]

    rows = []
    for _, r in ann.iterrows():
        cid = r["Clip ID"]
        ride = cid.split("_")[0]
        row = {
            "clip_id": cid,
            "label": classes.index(r[col]),
            "path_video": os.path.join(args.data_root, "clips", ride, "front", f"{cid}.mp4"),
            "path_telemetry": os.path.join(args.data_root, "clips", ride, "telemetry", f"{cid}.csv"),
            "split": r["Split"],
        }
        if args.gaze_dir:
            row["path_gaze"] = os.path.join(args.gaze_dir, ride, f"{cid}.mp4")
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{args.out}: {(df.split == 'train').sum()} train | {(df.split == 'val').sum()} val | {len(classes)} classes")


if __name__ == "__main__":
    main()

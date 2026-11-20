"""Telemetry parsing and preprocessing helpers."""

import json
import ast
import numpy as np
import pandas as pd


def pad_or_trunc(vec, target_len):
    """Pad with zeros or truncate a vector to target_len."""
    vec = list(vec)
    if len(vec) >= target_len:
        return vec[:target_len]
    return vec + [0.0] * (target_len - len(vec))


def resample_bins(t, x, n_bins):
    """Average a signal into n_bins equal-time bins over the clip."""
    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    if len(x) == 0:
        return np.zeros(n_bins, dtype=np.float32)
    if len(x) == 1 or t[-1] <= t[0]:
        return np.full(n_bins, x.mean(), dtype=np.float32)
    edges = np.linspace(t[0], t[-1], n_bins + 1)
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, n_bins - 1)
    sums = np.bincount(idx, weights=x, minlength=n_bins)
    cnts = np.bincount(idx, minlength=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    if np.isnan(out).any():
        out = np.interp(centers, centers[~np.isnan(out)], out[~np.isnan(out)])
    return out.astype(np.float32)


def telemetry_vector(csv_path, n_bins=64):
    """Speed and lean (gyro_z) from a per-clip telemetry CSV, n_bins each."""
    df = pd.read_csv(csv_path)
    t = df["timestamp_s"].to_numpy()
    speed = resample_bins(t, df["speed_2d_mps"].to_numpy(), n_bins)
    lean = resample_bins(t, df["gyro_z"].to_numpy(), n_bins)
    return np.concatenate([speed, lean])


def _try_parse_vec(x):
    """Parse a list of floats from JSON/literal string."""
    if isinstance(x, (list, tuple, np.ndarray)):
        try:
            return [float(v) for v in x]
        except Exception:
            return []
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    s_clean = " ".join(s.split())
    try:
        val = json.loads(s_clean)
        if isinstance(val, (list, tuple)):
            return [float(v) for v in val]
    except Exception:
        pass
    try:
        val = ast.literal_eval(s_clean)
        if isinstance(val, (list, tuple)):
            return [float(v) for v in val]
    except Exception:
        pass
    return []


def parse_telemetry_vec(row, tele_len=128):
    """Extract the telemetry vector for a DataFrame row.

    Uses the per-clip telemetry CSV in 'path_telemetry', falls back to
    'telemetry_vec_64' or 'speed_vec_32' + 'gyro_vec_32'.
    """
    path = row.get("path_telemetry", None)
    if isinstance(path, str) and path:
        return telemetry_vector(path, n_bins=tele_len // 2).tolist()

    vec = _try_parse_vec(row.get("telemetry_vec_64", []))
    if len(vec) != tele_len:
        s32 = _try_parse_vec(row.get("speed_vec_32", []))
        g32 = _try_parse_vec(row.get("gyro_vec_32", []))
        if len(s32) == 32 and len(g32) == 32:
            vec = list(s32) + list(g32)
    return pad_or_trunc(vec, tele_len)

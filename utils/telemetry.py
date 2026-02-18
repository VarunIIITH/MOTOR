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


def parse_telemetry_vec(row, tele_len=256):
    """Extract and pad/truncate telemetry from a DataFrame row.

    Tries 'telemetry_vec_64' first, falls back to concatenating
    'speed_vec_32' + 'gyro_vec_32'.
    """
    vec = _try_parse_vec(row.get("telemetry_vec_64", []))
    if len(vec) != tele_len:
        s32 = _try_parse_vec(row.get("speed_vec_32", []))
        g32 = _try_parse_vec(row.get("gyro_vec_32", []))
        if len(s32) == 32 and len(g32) == 32:
            vec = list(s32) + list(g32)
    return pad_or_trunc(vec, tele_len)

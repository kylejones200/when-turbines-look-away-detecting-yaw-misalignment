"""Yaw misalignment statistics (degrees)."""

from __future__ import annotations

import numpy as np


def _rem_euclid(a: float, m: float) -> float:
    r = a % m
    if r < 0:
        r += m
    return r


def yaw_misalignment_stats(yaw: np.ndarray, wind_dir: np.ndarray) -> tuple[float, float]:
    y_arr = np.asarray(yaw, dtype=float)
    w_arr = np.asarray(wind_dir, dtype=float)
    errs = []
    for y, w in zip(y_arr, w_arr):
        e = _rem_euclid(y - w, 360.0)
        if e > 180.0:
            e -= 360.0
        if e < -180.0:
            e += 360.0
        errs.append(abs(e))
    if not errs:
        return 0.0, 0.0
    return float(sum(errs) / len(errs)), float(max(errs))

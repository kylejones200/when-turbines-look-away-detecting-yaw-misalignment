#!/usr/bin/env python3
"""Python vs Rust kernel benchmark."""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from compute_kernel import yaw_misalignment_stats  # noqa: E402

def main() -> None:
    n = 5000
    yaw = np.ascontiguousarray(np.arange(n) * 0.7 % 360.0)
    wind_dir = np.ascontiguousarray((np.arange(n) * 0.5 + 10.0) % 360.0)
    t0 = time.perf_counter()
    for _ in range(200):
        yaw_misalignment_stats(yaw, wind_dir)
    py_s = time.perf_counter() - t0
    try:
        import when_turbines_look_away_detecting_yaw_misalignment_rs as rs
    except ImportError:
        print("Build: maturin develop --release -m rust/py/Cargo.toml")
        print(f"Python {py_s:.3f}s")
        return
    rs_s = rs.bench_kernel_py(yaw, wind_dir, 10000)
    print(f"Python {py_s:.3f}s Rust {rs_s:.3f}s speedup {py_s / max(rs_s, 1e-9):.1f}x")
    py_m, py_x = yaw_misalignment_stats(yaw, wind_dir)
    rs_m, rs_x = rs.yaw_misalignment_stats_py(yaw, wind_dir)
    np.testing.assert_allclose(py_m, rs_m, rtol=1e-10)
    np.testing.assert_allclose(py_x, rs_x, rtol=1e-10)
    print("Correctness: OK")

if __name__ == "__main__":
    main()

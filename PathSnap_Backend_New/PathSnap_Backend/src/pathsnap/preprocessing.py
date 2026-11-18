"""GPS preprocessing: one row per ORIGINAL GPS observation, nothing deleted.

What it adds to each observation (all distances in metres, times in seconds):
  obs_idx        original row number in the .track file (the permanent observation ID)
  x, y           UTM metres (projection chosen per trip, see geo.LocalProjector)
  dt_prev_s      time since previous observation (NaN for the first)
  dt_next_s      time to next observation (NaN for the last)
  step_m         straight-line distance from the previous observation (NaN for the first)
  speed_mps      step_m / dt_prev_s (NaN if undefined)
  move_bearing   smoothed direction of travel, degrees (NaN when the vehicle is not moving
                 enough for a direction to be meaningful)
  bearing_valid  True when move_bearing is defined
  feat_bearing, feat_bearing_valid
                 heading used by the FEATURES (labels/evaluation always use move_bearing). With
                 feature_bearing_mode "fixed" (default) these are copies of move_bearing/bearing_valid.
                 With "adaptive", the centred window grows (3 s -> up to feature_bearing_max_half_window_s)
                 until the displacement reaches feature_bearing_min_disp_m, so slow movers whose 1 Hz
                 steps are smaller than the GPS noise still get a direction from a long-enough baseline;
                 the heading is valid only if that displacement is reached.
  is_stationary  part of a "stay": all points within R metres of each other's anchor for >= W seconds
  is_spike       GPS jump: implausibly fast both into and out of this point
  gap_before     dt_prev_s > gap threshold     gap_after   dt_next_s > gap threshold
  segment_id     increments after every "long" gap; useful for sequence decoding later

Why smoothing for direction: at 1 Hz a car moves ~10 m per step while GPS noise is ~5 m, so a
single-step heading is noisy. We use the displacement between the points ~W seconds before
and after (centred window), and only trust it when that displacement is large enough.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .geo import LocalProjector, bearing_deg

DEFAULTS = {
    "sort_by_time": True,
    "gap_s": 10.0,
    "long_gap_s": 60.0,
    "stationary_window_s": 30.0,
    "stationary_max_disp_m": 15.0,
    "bearing_half_window_s": 3.0,
    "bearing_min_disp_m": 5.0,
    "spike_speed_mps": 70.0,
    "feature_bearing_mode": "fixed",          # "fixed" (= move_bearing) or "adaptive"
    "feature_bearing_min_disp_m": 20.0,       # adaptive: baseline length that makes noise small vs displacement
    "feature_bearing_max_half_window_s": 30.0,
}

ADAPTIVE_HALF_WINDOWS_S = (3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0)


def adaptive_bearing(x: np.ndarray, y: np.ndarray, t: np.ndarray, min_disp: float, max_hw: float) -> tuple[np.ndarray, np.ndarray]:
    """Heading from the smallest centred window (from ADAPTIVE_HALF_WINDOWS_S, <= max_hw) whose
    displacement is >= min_disp. NaN / invalid where no such window exists."""
    n = len(t)
    brg = np.full(n, np.nan); ok = np.zeros(n, bool)
    for hw in ADAPTIVE_HALF_WINDOWS_S:
        if hw > max_hw:
            break
        jb = np.clip(np.searchsorted(t, t - hw), 0, n - 1)
        jf = np.clip(np.searchsorted(t, t + hw, side="right") - 1, 0, n - 1)
        dx, dy = x[jf] - x[jb], y[jf] - y[jb]
        new = (~ok) & (np.hypot(dx, dy) >= min_disp)
        brg[new] = bearing_deg(dx[new], dy[new]); ok |= new
    return brg, ok


def detect_stays(x: np.ndarray, y: np.ndarray, t: np.ndarray, radius_m: float, min_duration_s: float) -> np.ndarray:
    """Stay-point detection: mark every run of consecutive points that all stay within
    `radius_m` of the run's first point for at least `min_duration_s` seconds.

    Robust to GPS jitter (a parked car's position wanders a few metres every second) and has
    no edge effects at the start/end of a stop. Runs in linear time: after a stay is found,
    the scan continues after its last point.
    """
    n = len(t)
    stay = np.zeros(n, bool)
    i = 0
    chunk = 64
    while i < n - 1:
        j = i                                   # last index still within radius of anchor i
        while j < n - 1:
            hi = min(n, j + 1 + chunk)
            far = np.hypot(x[j + 1:hi] - x[i], y[j + 1:hi] - y[i]) >= radius_m
            if far.any():
                j = j + int(np.argmax(far))
                break
            j = hi - 1
        if t[j] - t[i] >= min_duration_s:
            stay[i:j + 1] = True
            i = j + 1
        else:
            i += 1
    return stay


def preprocess_track(track: pd.DataFrame, cfg: dict | None = None,
                     projector: LocalProjector | None = None) -> tuple[pd.DataFrame, LocalProjector, dict]:
    """Return (observations dataframe, projector used, small report dict)."""
    c = {**DEFAULTS, **(cfg or {})}
    df = track[["lon", "lat", "t"]].copy()
    df.insert(0, "obs_idx", np.arange(len(df), dtype=np.int64))
    report = {"n_obs": int(len(df)), "reordered": False, "duplicate_timestamps": 0}
    if len(df) == 0:
        raise ValueError("Track has no valid observations.")

    t_raw = df["t"].to_numpy()
    if c["sort_by_time"] and np.any(np.diff(t_raw) < 0):
        # Stable sort keeps original order for equal times; obs_idx still points to the raw row.
        df = df.sort_values("t", kind="mergesort").reset_index(drop=True)
        report["reordered"] = True
    t = df["t"].to_numpy(float)
    report["duplicate_timestamps"] = int((np.diff(t) == 0).sum())

    proj = projector or LocalProjector.for_points(df["lon"].to_numpy(), df["lat"].to_numpy())
    x, y = proj.to_xy(df["lon"].to_numpy(), df["lat"].to_numpy())
    df["x"], df["y"] = x, y
    n = len(df)

    dt = np.diff(t)
    step = np.hypot(np.diff(x), np.diff(y))
    df["dt_prev_s"] = np.r_[np.nan, dt]
    df["dt_next_s"] = np.r_[dt, np.nan]
    df["step_m"] = np.r_[np.nan, step]
    with np.errstate(divide="ignore", invalid="ignore"):
        spd = np.where(dt > 0, step / np.where(dt > 0, dt, 1), np.nan)
    df["speed_mps"] = np.r_[np.nan, spd]

    # Smoothed direction from the points ~half-window before and after.
    hw = c["bearing_half_window_s"]
    # first observation at/after t-hw, and last observation at/before t+hw
    jb = np.clip(np.searchsorted(t, t - hw), 0, n - 1)
    jf = np.clip(np.searchsorted(t, t + hw, side="right") - 1, 0, n - 1)
    ddx, ddy = x[jf] - x[jb], y[jf] - y[jb]
    disp = np.hypot(ddx, ddy)
    valid = disp >= c["bearing_min_disp_m"]
    df["move_bearing"] = np.where(valid, bearing_deg(ddx, ddy), np.nan)
    df["bearing_valid"] = valid
    if c.get("feature_bearing_mode", "fixed") == "adaptive":
        fb, fv = adaptive_bearing(x, y, t, float(c["feature_bearing_min_disp_m"]), float(c["feature_bearing_max_half_window_s"]))
    else:
        fb, fv = df["move_bearing"].to_numpy(float), valid
    df["feat_bearing"] = fb
    df["feat_bearing_valid"] = fv

    # Stationary = part of a stay (see detect_stays). Per-step speed is NOT used because GPS
    # jitter makes a parked car look like it moves a few metres every second.
    df["is_stationary"] = detect_stays(x, y, t, c["stationary_max_disp_m"], c["stationary_window_s"])

    sp_in = df["speed_mps"].to_numpy()
    sp_out = np.r_[spd, np.nan]
    with np.errstate(invalid="ignore"):
        df["is_spike"] = (sp_in > c["spike_speed_mps"]) & (sp_out > c["spike_speed_mps"])
    with np.errstate(invalid="ignore"):
        df["gap_before"] = df["dt_prev_s"].to_numpy() > c["gap_s"]
        df["gap_after"] = df["dt_next_s"].to_numpy() > c["gap_s"]
        long_gap = df["dt_prev_s"].to_numpy() > c["long_gap_s"]
    df["segment_id"] = np.cumsum(long_gap).astype(np.int32)

    report.update(
        utm_epsg=proj.epsg,
        stationary_frac=float(df["is_stationary"].mean()),
        spikes=int(df["is_spike"].sum()),
        gaps=int(df["gap_before"].sum()),
        long_gaps=int(long_gap.sum()),
        bearing_valid_frac=float(valid.mean()),
    )
    return df, proj, report

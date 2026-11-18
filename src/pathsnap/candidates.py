"""Spatial candidate generation.

For every GPS observation we ask the R-tree (shapely STRtree) for road segments within a
radius, compute the exact point-to-segment distance, rank by distance and keep the top K.

Radius expansion: we start with a small radius (cheap, enough for most points). Points with
fewer than `min_candidates` arcs are re-queried with the next, larger radius, and so on up to
the maximum. Points with nothing even at the maximum radius get zero candidates and are
recorded as such (they are never dropped).

Directed arcs are kept separate: a two-way street gives two candidates (one per direction)
at the same distance; direction features decide between them later.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import shapely

from .geo import point_segment_distance
from .road_network import RoadNetwork

DEFAULTS = {"radii_m": [50.0, 100.0, 200.0], "max_candidates": 24, "min_candidates": 2}


def generate_candidates(px: np.ndarray, py: np.ndarray, net: RoadNetwork, cfg: dict | None = None):
    """Return (candidates dataframe, per-observation info dataframe).

    candidates columns: obs_row, arc_id, dist_m, proj_frac, dist_rank, radius_m
    obs info columns:   obs_row, n_candidates, radius_used_m, radius_expanded, nearest_dist_m
    `obs_row` is the row position of the observation in the arrays passed in.
    """
    c = {**DEFAULTS, **(cfg or {})}
    radii = sorted(float(r) for r in c["radii_m"])
    k, min_c = int(c["max_candidates"]), int(c["min_candidates"])
    n = len(px)
    pts = shapely.points(px, py)
    pending = np.arange(n)
    radius_used = np.full(n, radii[-1])
    parts = []
    for level, r in enumerate(radii):
        if len(pending) == 0:
            break
        qi, ti = net.tree.query(pts[pending], predicate="dwithin", distance=r)
        obs = pending[qi]
        arc = net.valid_ids[ti]
        d, frac = point_segment_distance(px[obs], py[obs], net.x0[arc], net.y0[arc], net.x1[arc], net.y1[arc])
        keep = d <= r                      # dwithin is inclusive; guard against float edge cases
        obs, arc, d, frac = obs[keep], arc[keep], d[keep], frac[keep]
        counts = np.bincount(obs, minlength=n)[pending]
        last = level == len(radii) - 1
        done_mask = (counts >= min_c) | last
        done = pending[done_mask]
        radius_used[done] = r
        sel = np.isin(obs, done)
        parts.append(pd.DataFrame({"obs_row": obs[sel], "arc_id": arc[sel], "dist_m": d[sel],
                                   "proj_frac": frac[sel], "radius_m": r}))
        pending = pending[~done_mask]

    cand = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["obs_row", "arc_id", "dist_m", "proj_frac", "radius_m"])
    cand = cand.drop_duplicates(["obs_row", "arc_id"])
    # Rank by distance; exact ties (e.g. the two directions of one street) broken by arc_id so
    # the result is deterministic. Distances are rounded to 1 micrometre for tie detection.
    cand["_d"] = np.round(cand["dist_m"].to_numpy(float), 6)
    cand = cand.sort_values(["obs_row", "_d", "arc_id"], kind="mergesort").drop(columns="_d")
    cand["dist_rank"] = cand.groupby("obs_row").cumcount().astype(np.int32)
    cand = cand[cand["dist_rank"] < k].reset_index(drop=True)
    cand["obs_row"] = cand["obs_row"].astype(np.int64)
    cand["arc_id"] = cand["arc_id"].astype(np.int64)

    n_c = np.bincount(cand["obs_row"].to_numpy(), minlength=n)
    nearest = np.full(n, np.nan)
    first = cand[cand["dist_rank"] == 0]
    nearest[first["obs_row"].to_numpy()] = first["dist_m"].to_numpy()
    info = pd.DataFrame({"obs_row": np.arange(n), "n_candidates": n_c.astype(np.int32),
                         "radius_used_m": radius_used, "radius_expanded": radius_used > radii[0],
                         "nearest_dist_m": nearest})
    return cand, info


def rank_of_arc(px: np.ndarray, py: np.ndarray, target_arc: np.ndarray, net: RoadNetwork) -> tuple[np.ndarray, np.ndarray]:
    """For each point, distance to `target_arc` and its rank among ALL indexed arcs by distance
    (same tie-break as generate_candidates). Used to measure coverage for any radius / K
    without regenerating candidates. Invalid target (e.g. -1) gives NaN / -1.
    """
    n = len(px)
    dist = np.full(n, np.nan)
    rank = np.full(n, -1, dtype=np.int64)
    ok = (target_arc >= 0) & (target_arc < net.n_arcs)
    ok[ok] &= net.valid[target_arc[ok]]
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return dist, rank
    a = target_arc[idx]
    d_t, _ = point_segment_distance(px[idx], py[idx], net.x0[a], net.y0[a], net.x1[a], net.y1[a])
    dist[idx] = d_t
    qi, ti = net.tree.query(shapely.points(px[idx], py[idx]), predicate="dwithin", distance=d_t + 1e-6)
    arc = net.valid_ids[ti]
    o = idx[qi]
    d, _ = point_segment_distance(px[o], py[o], net.x0[arc], net.y0[arc], net.x1[arc], net.y1[arc])
    dr, dt = np.round(d, 6), np.round(dist[o], 6)
    better = (dr < dt) | ((dr == dt) & (arc < target_arc[o]))
    rank[idx] = np.bincount(o[better], minlength=n)[idx]
    return dist, rank

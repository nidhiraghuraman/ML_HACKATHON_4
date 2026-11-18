"""Candidate features. Identical code runs for training, student-testing and the API.

No feature uses the reference route. Everything is computed from the GPS track, the trip's
road network and the candidate set, all of which are available at inference time.
Missing values are kept as NaN (and listed in FEATURE_SCHEMA) so Phase 3 can impute them
explicitly. No transformation is fitted here, so there is nothing that could leak test data.

Deliberately NOT used as model features (kept only as information columns):
  - raw x / y / lon / lat: every trip is in a different city; absolute position cannot
    generalise to unseen trips.
  - absolute headings (arc_bearing_deg, move_bearing): "north" means nothing on its own;
    only the DIFFERENCE between GPS heading and road heading is informative.
  - node / arc IDs: arbitrary per-trip numbering.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .geo import angle_diff_deg
from .road_network import RoadNetwork

# name -> (unit, level, missing allowed, description)
FEATURE_SCHEMA: dict[str, tuple[str, str, bool, str]] = {
    # distance
    "dist_m": ("m", "candidate", False, "Distance from the GPS point to the closest point on the arc"),
    "dist_rank": ("rank", "candidate", False, "Rank of the arc by distance among this point's candidates (0 = nearest)"),
    "dist_minus_nearest_m": ("m", "candidate", False, "dist_m minus the distance of the nearest candidate (0 for the nearest)"),
    "is_nearest": ("0/1", "candidate", False, "1 if this is the nearest candidate (dist_rank == 0); explicit flag for linear models"),
    "proj_frac": ("fraction", "candidate", False, "Where the closest point lies along the arc: 0 = start node, 1 = end node (0/1 also mean 'beyond the end')"),
    "arc_length_m": ("m", "candidate", False, "Length of the arc"),
    # direction
    "bearing_diff_deg": ("deg", "candidate", True, "Angle between GPS direction of travel and arc direction, 0-180 (0 = same way, 180 = wrong way on a two-way street). NaN when the GPS heading is not reliable (vehicle not moving)"),
    "bearing_valid": ("0/1", "observation", False, "1 if the GPS heading is reliable for this point"),
    # road / connectivity
    "arc_has_twin": ("0/1", "candidate", False, "1 if the reverse arc exists (two-way road): direction must decide"),
    "from_node_degree": ("count", "candidate", False, "Number of arcs touching the arc's start node (in + out)"),
    "to_node_degree": ("count", "candidate", False, "Number of arcs touching the arc's end node (in + out)"),
    "connects_prev": ("0/1", "candidate", True, "1 if this arc equals, or directly follows, a candidate of the previous GPS point. NaN for the first point or if the previous point has no candidates"),
    "connects_next": ("0/1", "candidate", True, "1 if this arc equals, or directly precedes, a candidate of the next GPS point. NaN for the last point or if the next point has no candidates"),
    # movement (same value for all candidates of one observation)
    "speed_mps": ("m/s", "observation", True, "GPS speed from the previous point. NaN for the first point"),
    "step_m": ("m", "observation", True, "Distance from the previous GPS point. NaN for the first point"),
    "dt_prev_s": ("s", "observation", True, "Time since the previous GPS point. NaN for the first point"),
    "dt_next_s": ("s", "observation", True, "Time until the next GPS point. NaN for the last point"),
    "is_stationary": ("0/1", "observation", False, "1 if the vehicle moved < threshold metres over the stationary window"),
    "is_spike": ("0/1", "observation", False, "1 if the point is an implausibly fast jump in and out (GPS spike)"),
    "gap_before": ("0/1", "observation", False, "1 if dt_prev_s exceeds the gap threshold"),
    "gap_after": ("0/1", "observation", False, "1 if dt_next_s exceeds the gap threshold"),
    # candidate context
    "n_candidates": ("count", "observation", False, "Number of candidates kept for this GPS point"),
    "radius_expanded": ("0/1", "observation", False, "1 if the search radius had to be widened to find enough candidates"),
}
FEATURE_COLUMNS = list(FEATURE_SCHEMA)


def _connects(rows: np.ndarray, arcs: np.ndarray, frm: np.ndarray, to: np.ndarray, n_obs: int,
              n_cand_per_obs: np.ndarray, base: int, direction: int) -> np.ndarray:
    """direction=-1: link to previous observation; +1: link to next observation."""
    nb = rows + direction
    if direction < 0:
        # this arc equals a prev candidate, or this arc's FROM node == a prev candidate's TO node
        same = np.isin(nb * base + arcs, rows * base + arcs)
        link = np.isin(nb * base + frm, rows * base + to)
    else:
        # this arc equals a next candidate, or this arc's TO node == a next candidate's FROM node
        same = np.isin(nb * base + arcs, rows * base + arcs)
        link = np.isin(nb * base + to, rows * base + frm)
    res = (same | link).astype(float)
    nb_ok = (nb >= 0) & (nb < n_obs)
    nb_has = np.zeros(len(rows), bool)
    nb_has[nb_ok] = n_cand_per_obs[nb[nb_ok]] > 0
    res[~nb_has] = np.nan
    return res


def compute_features(obs: pd.DataFrame, cand: pd.DataFrame, info: pd.DataFrame, net: RoadNetwork) -> pd.DataFrame:
    """Return `cand` with feature columns (and information columns) added."""
    out = cand.copy()
    rows = out["obs_row"].to_numpy()
    arcs = out["arc_id"].to_numpy()
    frm, to = net.arc_from[arcs], net.arc_to[arcs]
    out["from_node"], out["to_node"] = frm, to

    nearest = info["nearest_dist_m"].to_numpy()[rows]
    out["dist_minus_nearest_m"] = out["dist_m"].to_numpy() - nearest
    out["is_nearest"] = (out["dist_rank"].to_numpy() == 0).astype(np.int8)
    out["arc_length_m"] = net.length_m[arcs]
    out["arc_bearing_deg"] = net.bearing[arcs]

    mb = obs["move_bearing"].to_numpy(float)[rows]
    bv = obs["bearing_valid"].to_numpy(bool)[rows]
    diff = np.where(bv, angle_diff_deg(mb, net.bearing[arcs]), np.nan)
    out["bearing_diff_deg"] = diff
    out["move_bearing"] = mb

    out["arc_has_twin"] = net.has_twin[arcs].astype(np.int8)
    deg = net.node_in_deg + net.node_out_deg
    out["from_node_degree"] = deg[frm].astype(np.int32)
    out["to_node_degree"] = deg[to].astype(np.int32)

    n_obs = len(obs)
    ncand = info["n_candidates"].to_numpy()
    base = max(net.n_arcs, net.n_nodes) + 1
    out["connects_prev"] = _connects(rows, arcs, frm, to, n_obs, ncand, base, -1)
    out["connects_next"] = _connects(rows, arcs, frm, to, n_obs, ncand, base, +1)

    for col in ("speed_mps", "step_m", "dt_prev_s", "dt_next_s"):
        out[col] = obs[col].to_numpy(float)[rows]
    for col in ("bearing_valid", "is_stationary", "is_spike", "gap_before", "gap_after"):
        out[col] = obs[col].to_numpy(bool)[rows].astype(np.int8)
    out["n_candidates"] = ncand[rows].astype(np.int32)
    out["radius_expanded"] = info["radius_expanded"].to_numpy(bool)[rows].astype(np.int8)
    out["radius_used_m"] = info["radius_used_m"].to_numpy()[rows]
    return out


def schema_records() -> list[dict]:
    return [{"feature": k, "unit": u, "level": lvl, "missing_allowed": m, "description": d}
            for k, (u, lvl, m, d) in FEATURE_SCHEMA.items()]

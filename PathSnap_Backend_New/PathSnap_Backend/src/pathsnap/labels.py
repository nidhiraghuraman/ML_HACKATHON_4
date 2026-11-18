"""Reference-route alignment and candidate labels (training / development data only).

PROBLEM
  .route lists the arcs driven, in order, but does not say WHICH GPS observation belongs to
  WHICH route arc. A track has ~5x more rows than its route, so row numbers cannot be matched.

METHOD: monotone dynamic-programming alignment (a DTW-style alignment)
  States are ROUTE POSITIONS j = 0..M-1 (not arc IDs), so a route that drives the same arc
  twice has two different positions and the two passes cannot be confused.
  Each observation i is assigned one position j(i) such that positions never go backwards in
  time: j(i) >= j(i-1). Positions may be skipped (short arcs between 1 Hz samples, GPS gaps).
  The assignment minimises the total cost
      sum_i [ emission(i, j(i)) ] + eta * (route metres advanced)
  emission(i, j) = min(d_ij, cap)^2 / (2 sigma^2) + w_heading * (1 - cos(heading diff)) / 2
    d_ij       = distance (m) from observation i to route arc at position j
    cap        = distance cap so far-off points (parked off-road, spikes) cannot drag the path
    heading    = only when the GPS heading is valid (vehicle moving)
    eta        = tiny cost per metre advanced: prefers not to jump ahead without evidence
  Solved exactly with a prefix-minimum recurrence, O(N * M) time:
      C_i(j) = e_i(j) + eta*c_j + min_{j' <= j} [ C_{i-1}(j') - eta*c_j' ]
  where c_j is the cumulative route length at position j. Backtracking gives j(i).

LABELS
  obs_status (per observation):
    aligned      clear assignment
    ambiguous    another route arc near position j(i) (+/- window) is within `ambiguity_margin_m`
                 of the assigned one, typically a point sitting on the node shared by two
                 consecutive route arcs. Both are plausible, so we do not invent certainty.
    unalignable  assigned distance > `unalign_dist_m`, or stationary with distance >
                 `stationary_unalign_dist_m` (e.g. parked off-road), or the route is empty.
  label (per candidate):
     1  positive    the arc at the aligned position (if it was among the candidates)
     0  negative    any other candidate
    -1  ambiguous   a plausible alternative (see above) or an exact duplicate of the positive
                    (same from/to nodes, different arc ID: indistinguishable geometrically)
    -2  unalignable all candidates of an unalignable observation
ROUTE BREAKS: alignment does not require route arcs to be connected, so a break does not
  break the alignment. Observations aligned next to a break are flagged `near_route_break`.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .geo import point_segment_distance, angle_diff_deg
from .road_network import RoadNetwork

DEFAULTS = {
    "sigma_m": 10.0,
    "cap_m": 50.0,
    "w_heading": 3.0,
    "eta_per_m": 0.001,
    "unalign_dist_m": 40.0,
    "stationary_unalign_dist_m": 20.0,
    "ambiguity_margin_m": 2.0,
    "ambiguity_window": 2,
    "row_chunk": 1024,
}

LABEL_POS, LABEL_NEG, LABEL_AMB, LABEL_UNALIGN = 1, 0, -1, -2


def _emission_chunk(px, py, brg, brg_ok, net, r_arcs, c):
    d, _ = point_segment_distance(px[:, None], py[:, None], net.x0[r_arcs][None, :], net.y0[r_arcs][None, :],
                                  net.x1[r_arcs][None, :], net.y1[r_arcs][None, :])
    e = np.minimum(d, c["cap_m"]) ** 2 / (2 * c["sigma_m"] ** 2)
    diff = angle_diff_deg(brg[:, None], net.bearing[r_arcs][None, :])
    pen = c["w_heading"] * (1 - np.cos(np.radians(diff))) / 2
    pen = np.where(brg_ok[:, None] & np.isfinite(pen), pen, 0.0)
    return e + pen, d


def align_to_route(obs: pd.DataFrame, route: np.ndarray, net: RoadNetwork, cfg: dict | None = None) -> pd.DataFrame:
    """Return per-observation alignment: route_pos, route_arc, align_dist_m, align_margin_m,
    near_route_break, obs_status. Rows are in the same order as `obs`."""
    c = {**DEFAULTS, **(cfg or {})}
    n = len(obs)
    out = pd.DataFrame({"route_pos": np.full(n, -1, np.int32), "route_arc": np.full(n, -1, np.int64),
                        "align_dist_m": np.full(n, np.nan), "align_margin_m": np.full(n, np.nan),
                        "near_route_break": np.zeros(n, bool), "obs_status": np.full(n, "unalignable", object),
                        "route_progress_m": np.full(n, np.nan), "naive_route_arc": np.full(n, -1, np.int64)})
    if route is None or len(route) == 0 or n == 0:
        return out
    route = np.asarray(route, np.int64)
    ok = (route >= 0) & (route < net.n_arcs)
    ok[ok] &= net.valid[route[ok]]
    pos_ids = np.flatnonzero(ok)                 # usable route positions (original numbering)
    if len(pos_ids) == 0:
        return out
    r_arcs = route[pos_ids]
    M = len(r_arcs)
    cum = np.r_[0.0, np.cumsum(net.length_m[r_arcs])[:-1]]
    eta = c["eta_per_m"]

    px, py = obs["x"].to_numpy(float), obs["y"].to_numpy(float)
    brg = obs["move_bearing"].to_numpy(float)
    brg_ok = obs["bearing_valid"].to_numpy(bool)

    bp = np.zeros((n, M), dtype=np.uint16 if M < 65535 else np.int32)
    ar = np.arange(M)
    prev = None
    naive = np.empty(n, np.int64)          # nearest route arc ignoring order (for validation only)
    step = int(c["row_chunk"])
    for s in range(0, n, step):
        e_chunk, d_chunk = _emission_chunk(px[s:s + step], py[s:s + step], brg[s:s + step], brg_ok[s:s + step], net, r_arcs, c)
        naive[s:s + step] = r_arcs[np.argmin(d_chunk, axis=1)]
        for k in range(e_chunk.shape[0]):
            i = s + k
            if prev is None:
                prev = e_chunk[k] + eta * cum
                continue
            a = prev - eta * cum
            pm = np.minimum.accumulate(a)
            bp[i] = np.maximum.accumulate(np.where(a == pm, ar, 0))
            prev = e_chunk[k] + pm + eta * cum
    j = np.empty(n, np.int64)
    j[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        j[i - 1] = bp[i, j[i]]

    # distances to assigned and nearby route positions (window) for ambiguity
    w = int(c["ambiguity_window"])
    a_arc = r_arcs[j]
    d_assigned, frac_assigned = point_segment_distance(px, py, net.x0[a_arc], net.y0[a_arc], net.x1[a_arc], net.y1[a_arc])
    margin = np.full(n, np.inf)
    for off in range(-w, w + 1):
        if off == 0:
            continue
        jj = j + off
        inside = (jj >= 0) & (jj < M)
        jj_c = np.clip(jj, 0, M - 1)
        alt = r_arcs[jj_c]
        d_alt, _ = point_segment_distance(px, py, net.x0[alt], net.y0[alt], net.x1[alt], net.y1[alt])
        diff_arc = alt != a_arc
        m = np.where(inside & diff_arc, d_alt - d_assigned, np.inf)
        margin = np.minimum(margin, m)

    # route breaks: consecutive usable positions whose arcs do not connect
    brk = np.zeros(M, bool)
    if M > 1:
        disc = net.arc_to[r_arcs[:-1]] != net.arc_from[r_arcs[1:]]
        brk[:-1] |= disc
        brk[1:] |= disc

    stationary = obs["is_stationary"].to_numpy(bool)
    unalign = (d_assigned > c["unalign_dist_m"]) | (stationary & (d_assigned > c["stationary_unalign_dist_m"]))
    ambiguous = ~unalign & (margin <= c["ambiguity_margin_m"])
    status = np.where(unalign, "unalignable", np.where(ambiguous, "ambiguous", "aligned"))

    out["route_pos"] = pos_ids[j].astype(np.int32)
    out["route_arc"] = a_arc
    out["align_dist_m"] = d_assigned
    out["align_margin_m"] = np.where(np.isfinite(margin), margin, np.nan)
    out["near_route_break"] = brk[j]
    out["obs_status"] = status
    out["route_progress_m"] = cum[j] + frac_assigned * net.length_m[a_arc]   # metres along the route
    out["naive_route_arc"] = naive
    return out


def ambiguous_alternatives(obs_align: pd.DataFrame, obs: pd.DataFrame, route: np.ndarray, net: RoadNetwork,
                           cfg: dict | None = None) -> set[tuple[int, int]]:
    """Set of (obs_row, arc_id) pairs that are plausible alternatives to the aligned arc."""
    c = {**DEFAULTS, **(cfg or {})}
    alts: set[tuple[int, int]] = set()
    if route is None or len(route) == 0:
        return alts
    route = np.asarray(route, np.int64)
    w = int(c["ambiguity_window"])
    px, py = obs["x"].to_numpy(float), obs["y"].to_numpy(float)
    rows = np.flatnonzero(obs_align["obs_status"].to_numpy() == "ambiguous")
    for i in rows:
        p = int(obs_align["route_pos"].iat[i])
        base = float(obs_align["align_dist_m"].iat[i])
        a0 = int(obs_align["route_arc"].iat[i])
        for q in range(max(0, p - w), min(len(route), p + w + 1)):
            a = int(route[q])
            if a == a0 or a < 0 or a >= net.n_arcs or not net.valid[a]:
                continue
            d, _ = point_segment_distance(px[i], py[i], net.x0[a], net.y0[a], net.x1[a], net.y1[a])
            if d - base <= c["ambiguity_margin_m"]:
                alts.add((int(i), a))
    return alts


def label_candidates(cand: pd.DataFrame, obs_align: pd.DataFrame, obs: pd.DataFrame, route, net: RoadNetwork,
                     cfg: dict | None = None) -> np.ndarray:
    """Return an int8 label per candidate row (see module docstring)."""
    rows = cand["obs_row"].to_numpy()
    arcs = cand["arc_id"].to_numpy()
    status = obs_align["obs_status"].to_numpy()[rows]
    pos_arc = obs_align["route_arc"].to_numpy()[rows]
    lab = np.full(len(cand), LABEL_NEG, np.int8)
    lab[arcs == pos_arc] = LABEL_POS
    # exact geometric duplicates of the positive arc (same from & to node, different id)
    same_geom = (net.arc_from[arcs] == net.arc_from[np.maximum(pos_arc, 0)]) & \
                (net.arc_to[arcs] == net.arc_to[np.maximum(pos_arc, 0)]) & (arcs != pos_arc) & (pos_arc >= 0)
    lab[same_geom] = LABEL_AMB
    alts = ambiguous_alternatives(obs_align, obs, route, net, cfg)
    if alts:
        key = pd.MultiIndex.from_arrays([rows, arcs])
        is_alt = key.isin(list(alts))
        lab[is_alt & (lab != LABEL_POS)] = LABEL_AMB
    lab[status == "unalignable"] = LABEL_UNALIGN
    return lab

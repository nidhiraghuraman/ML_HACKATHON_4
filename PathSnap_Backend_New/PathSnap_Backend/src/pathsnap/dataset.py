"""Per-trip pipeline: load -> preprocess -> road network -> candidates -> features [-> labels].

`process_trip(..., with_labels=False)` is exactly what the API will run at inference time.
With labels it also aligns the reference route and measures candidate coverage.
Each trip is processed in isolation: its own projection, road network and spatial index.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd

from .io import load_trip
from .preprocessing import preprocess_track
from .road_network import build_road_network
from .candidates import generate_candidates, rank_of_arc
from .features import compute_features
from .labels import align_to_route, label_candidates

OBS_INFO_COLS = ["n_candidates", "radius_used_m", "radius_expanded", "nearest_dist_m"]


def process_trip(trip_dir: Path, partition: str, cfg: dict, with_labels: bool = True, return_context: bool = False):
    """Return (observations table, candidate table, report[, context]).

    context = {"trip": Trip, "net": RoadNetwork}, used by diagnostics and maps."""
    p2 = cfg["phase2"]
    timings = {}
    t0 = time.perf_counter()
    trip = load_trip(trip_dir, partition, load_route=with_labels)
    timings["load_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    obs, proj, pre_rep = preprocess_track(trip.track, p2["preprocessing"])
    timings["preprocess_s"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    net = build_road_network(trip.nodes, trip.arcs, proj)
    timings["network_s"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    px, py = obs["x"].to_numpy(), obs["y"].to_numpy()
    cand, info = generate_candidates(px, py, net, p2["candidates"])
    timings["candidates_s"] = time.perf_counter() - t1

    t1 = time.perf_counter()
    cand = compute_features(obs, cand, info, net)
    timings["features_s"] = time.perf_counter() - t1

    obs = obs.copy()
    obs.insert(1, "seq", np.arange(len(obs), dtype=np.int64))      # time-order position
    for col in OBS_INFO_COLS:
        obs[col] = info[col].to_numpy()

    report = {"trip_id": trip.trip_id, "partition": partition, **{f"pre_{k}": v for k, v in pre_rep.items()},
              **{f"net_{k}": v for k, v in net.report.items()},
              "n_candidates_total": int(len(cand)), "zero_candidate_obs": int((info["n_candidates"] == 0).sum()),
              "expanded_obs": int(info["radius_expanded"].sum())}

    if with_labels:
        t1 = time.perf_counter()
        al = align_to_route(obs, trip.route, net, p2["labels"])
        for col in al.columns:
            obs[col] = al[col].to_numpy()
        cand["label"] = label_candidates(cand, al, obs, trip.route, net, p2["labels"])
        # coverage: is the aligned arc among the candidates, and at what rank among ALL arcs?
        pos = cand[cand["label"] == 1]
        obs["positive_in_candidates"] = False
        obs.loc[pos["obs_row"].to_numpy(), "positive_in_candidates"] = True
        obs["positive_rank"] = -1
        obs.loc[pos["obs_row"].to_numpy(), "positive_rank"] = pos["dist_rank"].to_numpy()
        tgt = np.where(obs["obs_status"].to_numpy() != "unalignable", obs["route_arc"].to_numpy(), -1)
        d_t, r_t = rank_of_arc(px, py, tgt, net)
        obs["true_arc_dist_m"], obs["true_arc_rank_all"] = d_t, r_t
        st = obs["obs_status"].value_counts().to_dict()
        timings["labels_s"] = time.perf_counter() - t1
        route = trip.route
        report.update(
            route_arcs=int(len(route)), route_unique=int(len(np.unique(route))),
            route_positions_hit=int(obs.loc[obs.obs_status != "unalignable", "route_pos"].nunique()),
            obs_aligned=int(st.get("aligned", 0)), obs_ambiguous=int(st.get("ambiguous", 0)),
            obs_unalignable=int(st.get("unalignable", 0)),
            cand_pos=int((cand.label == 1).sum()), cand_neg=int((cand.label == 0).sum()),
            cand_amb=int((cand.label == -1).sum()), cand_unalign=int((cand.label == -2).sum()),
        )
    else:
        cand["label"] = np.int8(-9)          # unknown (inference)

    cand.insert(0, "trip_id", trip.trip_id)
    cand.insert(1, "partition", partition)
    cand.insert(2, "obs_idx", obs["obs_idx"].to_numpy()[cand["obs_row"].to_numpy()])
    cand.insert(3, "t", obs["t"].to_numpy()[cand["obs_row"].to_numpy()])
    cand = cand.rename(columns={"obs_row": "seq"})
    if with_labels:
        for col in ("obs_status", "route_pos", "align_dist_m"):
            cand[col] = obs[col].to_numpy()[cand["seq"].to_numpy()]
    obs.insert(0, "trip_id", trip.trip_id)
    obs.insert(1, "partition", partition)
    report.update({k: round(v, 4) for k, v in timings.items()})
    report["total_s"] = round(time.perf_counter() - t0, 4)
    if return_context:
        return obs, cand, report, {"trip": trip, "net": net}
    return obs, cand, report


# Documentation for every non-feature column (features are documented in features.FEATURE_SCHEMA).
CANDIDATE_COLUMN_DOCS = {
    "trip_id": "Trip folder name, e.g. 00000001. Group key for cross-validation",
    "partition": "train or student_test (never mixed in one file)",
    "obs_idx": "Original row number of the GPS observation in the .track file",
    "t": "GPS timestamp (s), copied unchanged from .track",
    "seq": "Position of the observation in time order within the trip (0..n-1)",
    "arc_id": "Candidate arc ID = row number in this trip's .arcs file",
    "from_node": "Arc start node ID (row in .nodes)",
    "to_node": "Arc end node ID (row in .nodes)",
    "radius_m": "Search radius at which this candidate was found",
    "radius_used_m": "Final search radius used for this observation",
    "arc_bearing_deg": "Arc heading (deg, 0 = grid north). Information only, not a feature",
    "move_bearing": "Smoothed GPS heading (deg). Information only, not a feature",
    "label": "1 positive, 0 negative, -1 ambiguous, -2 observation unalignable, -9 unknown (no reference route)",
    "obs_status": "aligned / ambiguous / unalignable (observation-level alignment outcome)",
    "route_pos": "Aligned position in the .route file (row number), -1 if none",
    "align_dist_m": "Distance from the observation to its aligned route arc (m)",
}
OBS_COLUMN_DOCS = {
    "trip_id": "Trip folder name", "partition": "train or student_test",
    "obs_idx": "Original .track row", "seq": "Time-order position",
    "lon": "Original longitude", "lat": "Original latitude", "t": "Original timestamp (s)",
    "x": "UTM easting (m)", "y": "UTM northing (m)",
    "dt_prev_s": "Seconds since previous point", "dt_next_s": "Seconds to next point",
    "step_m": "Metres from previous point", "speed_mps": "Speed from previous point (m/s)",
    "move_bearing": "Smoothed heading (deg), NaN if not moving", "bearing_valid": "Heading reliable",
    "is_stationary": "Stationary flag", "is_spike": "GPS spike flag",
    "gap_before": "dt_prev_s > gap threshold", "gap_after": "dt_next_s > gap threshold",
    "segment_id": "Increments after each long gap",
    "n_candidates": "Candidates kept", "radius_used_m": "Final search radius", "radius_expanded": "Radius widened",
    "nearest_dist_m": "Distance to nearest candidate (NaN if none)",
    "route_pos": "Aligned route position (-1 none)", "route_arc": "Aligned route arc ID (-1 none)",
    "align_dist_m": "Distance to aligned arc (m)", "align_margin_m": "Distance gap to the closest alternative route arc nearby (m)",
    "near_route_break": "Aligned next to a route break", "obs_status": "aligned / ambiguous / unalignable",
    "route_progress_m": "Metres along the reference route at the aligned point",
    "naive_route_arc": "Nearest route arc ignoring order (validation only)",
    "positive_in_candidates": "Aligned arc is among the candidates", "positive_rank": "Its dist_rank (-1 if absent)",
    "true_arc_dist_m": "Distance to aligned arc (for coverage study)",
    "true_arc_rank_all": "Rank of aligned arc among ALL arcs by distance (coverage study)",
}

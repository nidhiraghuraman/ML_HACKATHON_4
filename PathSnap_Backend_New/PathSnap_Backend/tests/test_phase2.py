"""Phase 2 tests on small synthetic examples (no real dataset needed).

Run from the project folder:  python -m pytest tests -v
Synthetic roads are defined in metres around (lon 10, lat 50) and converted to lon/lat with the
same projector the pipeline uses, so distances in the tests are exact metres.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap.config import load_config
from pathsnap.geo import LocalProjector, haversine_m, bearing_deg, angle_diff_deg, point_segment_distance
from pathsnap.preprocessing import preprocess_track
from pathsnap.road_network import build_road_network
from pathsnap.candidates import generate_candidates, rank_of_arc
from pathsnap.features import compute_features, FEATURE_COLUMNS
from pathsnap.labels import align_to_route, label_candidates, LABEL_POS, LABEL_NEG, LABEL_AMB, LABEL_UNALIGN
from pathsnap.dataset import process_trip, CANDIDATE_COLUMN_DOCS
from pathsnap.diagnostics import check_trip

PROJ = LocalProjector(10.0, 50.0)
X0, Y0 = PROJ.to_xy(10.0, 50.0)


def net_from_xy(node_xy, arcs):
    """Build a RoadNetwork from node coordinates in metres relative to the reference point."""
    node_xy = np.asarray(node_xy, float)
    lon, lat = PROJ.to_lonlat(X0 + node_xy[:, 0], Y0 + node_xy[:, 1])
    nodes = np.c_[lon, lat]
    return build_road_network(nodes, np.asarray(arcs, np.int64), PROJ), nodes


def track_from_xy(xy, t):
    xy = np.asarray(xy, float)
    lon, lat = PROJ.to_lonlat(X0 + xy[:, 0], Y0 + xy[:, 1])
    return pd.DataFrame({"lon": lon, "lat": lat, "t": np.asarray(t, float)})


def prep(xy, t):
    obs, _, _ = preprocess_track(track_from_xy(xy, t), projector=PROJ)
    return obs


def east_road(n_seg=3, seg_len=100.0, two_way=True):
    """Straight east-west road: nodes 0..n_seg along y = 0."""
    nodes = [[i * seg_len, 0.0] for i in range(n_seg + 1)]
    arcs = [[i, i + 1] for i in range(n_seg)]
    if two_way:
        arcs += [[i + 1, i] for i in range(n_seg)]
    return nodes, arcs


# ----------------------------------------------------------------------------- geometry
def test_projection_distance_matches_haversine():
    lon = np.array([10.0, 10.01]); lat = np.array([50.0, 50.005])
    x, y = PROJ.to_xy(lon, lat)
    d_proj = np.hypot(x[1] - x[0], y[1] - y[0])
    d_hav = haversine_m(lon[0], lat[0], lon[1], lat[1])
    assert abs(d_proj - d_hav) / d_hav < 0.005
    lo, la = PROJ.to_lonlat(x, y)
    assert np.allclose(lo, lon, atol=1e-9) and np.allclose(la, lat, atol=1e-9)


def test_utm_zone_selection():
    assert LocalProjector(10.0, 50.0).epsg == 32632       # zone 32 north
    assert LocalProjector(-58.4, -34.6).epsg == 32721     # Buenos Aires, zone 21 south


def test_bearing_and_difference():
    assert np.allclose(bearing_deg([0, 1, 0, -1], [1, 0, -1, 0]), [0, 90, 180, 270])
    assert angle_diff_deg(350, 10) == pytest.approx(20)
    assert angle_diff_deg(0, 180) == pytest.approx(180)
    assert angle_diff_deg(90, 90) == pytest.approx(0)
    assert np.isnan(angle_diff_deg(np.nan, 10))


def test_point_segment_distance():
    d, f = point_segment_distance(np.array([5.0, 15.0]), np.array([3.0, 0.0]), 0.0, 0.0, 10.0, 0.0)
    assert d[0] == pytest.approx(3) and f[0] == pytest.approx(0.5)
    assert d[1] == pytest.approx(5) and f[1] == pytest.approx(1.0)      # beyond the end -> end node


def test_road_network_construction_and_invalid_arcs():
    nodes = [[0, 0], [100, 0], [100, 100]]
    arcs = [[0, 1], [1, 0], [1, 2], [2, 7], [2, 2]]     # arc 3: missing node 7; arc 4: zero length
    net, _ = net_from_xy(nodes, arcs)
    assert net.n_arcs == 5
    assert net.report["invalid_node_ref"] == 1 and net.report["zero_length"] == 1
    assert list(net.valid) == [True, True, True, False, False]
    assert set(net.valid_ids) == {0, 1, 2}
    assert net.length_m[0] == pytest.approx(100, abs=1e-6)
    assert net.bearing[0] == pytest.approx(90, abs=1e-6)      # east
    assert net.bearing[1] == pytest.approx(270, abs=1e-6)     # west (direction preserved)
    assert net.bearing[2] == pytest.approx(0, abs=1e-6)       # north
    assert list(net.has_twin[:3]) == [True, True, False]
    assert net.arc_from[2] == 1 and net.arc_to[2] == 2         # ids preserved


# ----------------------------------------------------------------------------- candidates
def test_candidate_ranking_ties_and_k():
    nodes = [[0, 0], [100, 0], [0, 10], [100, 10], [0, 30], [100, 30]]
    arcs = [[0, 1], [1, 0], [2, 3], [4, 5]]
    net, _ = net_from_xy(nodes, arcs)
    px, py = np.array([X0 + 50.0]), np.array([Y0 + 2.0])
    cand, info = generate_candidates(px, py, net, {"radii_m": [50], "max_candidates": 3, "min_candidates": 1})
    assert list(cand.arc_id) == [0, 1, 2]                 # twins tie at 2 m -> arc_id order; 30 m road cut by K=3
    assert list(cand.dist_rank) == [0, 1, 2]
    assert cand.dist_m.iloc[0] == pytest.approx(2, abs=1e-6) and cand.dist_m.iloc[2] == pytest.approx(8, abs=1e-6)
    assert info.n_candidates.iat[0] == 3 and not info.radius_expanded.iat[0]
    assert not cand.duplicated(["obs_row", "arc_id"]).any()


def test_radius_expansion_and_no_candidates():
    net, _ = net_from_xy([[0, 0], [100, 0]], [[0, 1]])
    px = np.array([X0 + 50.0, X0 + 50.0]); py = np.array([Y0 + 80.0, Y0 + 1000.0])
    cand, info = generate_candidates(px, py, net, {"radii_m": [50, 100], "max_candidates": 5, "min_candidates": 1})
    assert info.n_candidates.tolist() == [1, 0]
    assert info.radius_expanded.tolist() == [True, True]
    assert info.radius_used_m.iat[0] == 100
    assert cand.radius_m.iat[0] == 100 and cand.dist_m.iat[0] == pytest.approx(80, abs=1e-6)
    assert np.isnan(info.nearest_dist_m.iat[1])          # recorded, not dropped


def test_rank_of_arc_matches_generator():
    nodes = [[0, 0], [100, 0], [0, 10], [100, 10]]
    net, _ = net_from_xy(nodes, [[0, 1], [1, 0], [2, 3]])
    px, py = np.array([X0 + 50.0]), np.array([Y0 + 2.0])
    d, r = rank_of_arc(px, py, np.array([2]), net)
    assert d[0] == pytest.approx(8, abs=1e-6) and r[0] == 2


# ----------------------------------------------------------------------------- preprocessing
def test_preprocessing_stationary_gap_spike_and_order():
    # 40 s parked at x=0 (small jitter), then driving east at 10 m/s, a 120 s gap, then a spike
    t = list(range(40)) + list(range(40, 60)) + [180, 181, 182, 183, 184]
    xy = [[np.sin(i) * 2, np.cos(i) * 2] for i in range(40)] + [[(i - 39) * 10, 0] for i in range(40, 60)]
    xy += [[400, 0], [410, 0], [5000, 0], [430, 0], [440, 0]]
    tr = track_from_xy(xy, t)
    shuffled = tr.iloc[::-1].reset_index(drop=True)                  # reverse order on purpose
    obs, _, rep = preprocess_track(shuffled, projector=PROJ)
    assert rep["reordered"] and len(obs) == len(tr)                  # nothing deleted
    assert np.all(np.diff(obs.t) > 0)
    assert obs.obs_idx.iat[0] == len(tr) - 1                         # original row id kept
    assert obs.is_stationary.iloc[5:30].all()                        # parked part
    assert not obs.is_stationary.iloc[45:58].any()                   # moving part
    i_gap = int(np.flatnonzero(obs.t.to_numpy() == 180)[0])
    assert obs.gap_before.iat[i_gap] and obs.dt_prev_s.iat[i_gap] == 121
    assert obs.segment_id.iat[i_gap] == obs.segment_id.iat[i_gap - 1] + 1
    i_spk = int(np.flatnonzero(obs.t.to_numpy() == 182)[0])
    assert obs.is_spike.iat[i_spk] and obs.is_spike.sum() == 1
    assert np.isnan(obs.dt_prev_s.iat[0]) and np.isnan(obs.dt_next_s.iat[-1])
    assert obs.move_bearing.iloc[45:55].round().eq(90).all()          # driving east


# ----------------------------------------------------------------------------- features
def test_features_direction_and_connectivity():
    nodes, arcs = east_road(n_seg=2, seg_len=100)                    # arcs 0,1 east; 2,3 west
    net, _ = net_from_xy(nodes, arcs)
    obs = prep([[20 + 10 * i, 3] for i in range(15)], range(15))     # driving east, 3 m north of road
    cand, info = generate_candidates(obs.x.to_numpy(), obs.y.to_numpy(), net, {"radii_m": [50], "max_candidates": 4, "min_candidates": 1})
    f = compute_features(obs, cand, info, net)
    row = f[(f.obs_row == 5)].set_index("arc_id")
    assert row.loc[0, "bearing_diff_deg"] == pytest.approx(0, abs=1)       # eastbound arc
    assert row.loc[2, "bearing_diff_deg"] == pytest.approx(180, abs=1)     # westbound twin
    assert row.loc[0, "arc_has_twin"] == 1
    assert row.loc[0, "dist_m"] == pytest.approx(3, abs=1e-6)
    assert f[f.obs_row == 0].connects_prev.isna().all()                    # first point: explicitly missing
    assert row.loc[0, "connects_prev"] == 1                                # same arc as previous point
    assert set(FEATURE_COLUMNS) <= set(f.columns)


# ----------------------------------------------------------------------------- labels
def _labels(nodes, arcs, route, xy, t, cfg=None):
    net, _ = net_from_xy(nodes, arcs)
    obs = prep(xy, t)
    cand, info = generate_candidates(obs.x.to_numpy(), obs.y.to_numpy(), net, {"radii_m": [60, 200], "max_candidates": 8, "min_candidates": 1})
    al = align_to_route(obs, np.asarray(route), net, cfg)
    lab = label_candidates(cand, al, obs, np.asarray(route), net, cfg)
    return obs, cand, al, lab, net


def test_alignment_follows_route_in_order():
    nodes, arcs = east_road(n_seg=3, seg_len=100)
    xy = [[5 + 10 * i, 2] for i in range(29)]
    obs, cand, al, lab, _ = _labels(nodes, arcs, [0, 1, 2], xy, range(29))
    assert (np.diff(al.route_pos) >= 0).all()
    assert al.route_arc.iat[0] == 0 and al.route_arc.iat[-1] == 2
    assert al.route_arc.iat[15] == 1
    pos = cand[lab == LABEL_POS]
    assert pos.obs_row.is_unique                                     # at most one positive per point
    # the westbound twin is never positive
    assert not np.isin(pos.arc_id, [3, 4, 5]).any()
    assert ((lab == LABEL_NEG) & np.isin(cand.arc_id, [3, 4, 5])).any()


def test_repeated_route_arcs_use_positions():
    # square loop driven twice: route repeats arcs 0,1,2,3
    nodes = [[0, 0], [100, 0], [100, 100], [0, 100]]
    arcs = [[0, 1], [1, 2], [2, 3], [3, 0]]
    lap = [[x, 0] for x in range(5, 100, 10)] + [[100, y] for y in range(5, 100, 10)] + \
          [[x, 100] for x in range(95, 0, -10)] + [[0, y] for y in range(95, 0, -10)]
    xy = lap + lap
    _, _, al, _, _ = _labels(nodes, arcs, [0, 1, 2, 3, 0, 1, 2, 3], xy, range(len(xy)))
    n = len(lap)
    assert al.route_pos.iloc[:n].max() <= 3                          # first lap -> positions 0..3
    assert al.route_pos.iloc[n + 5:].min() >= 4                      # second lap -> positions 4..7
    assert set(al.route_arc.iloc[n + 5:]) <= {0, 1, 2, 3}


def test_route_break_is_flagged_not_fatal():
    nodes, arcs = east_road(n_seg=3, seg_len=100, two_way=False)
    xy = [[5 + 10 * i, 2] for i in range(10)] + [[205 + 10 * i, 2] for i in range(9)]
    _, _, al, _, _ = _labels(nodes, arcs, [0, 2], xy, range(19))     # arc 1 missing -> break
    assert al.near_route_break.all()
    assert set(al.route_arc) == {0, 2}
    assert (al.obs_status != "unalignable").all()


def test_ambiguous_point_on_shared_node():
    nodes, arcs = east_road(n_seg=2, seg_len=100, two_way=False)
    xy = [[50, 1], [80, 1], [100, 1], [120, 1], [150, 1]]            # third point on the shared node
    obs, cand, al, lab, _ = _labels(nodes, arcs, [0, 1], xy, range(5))
    assert al.obs_status.iat[2] == "ambiguous"
    c = cand[cand.obs_row == 2].assign(label=lab[cand.obs_row.to_numpy() == 2])
    assert (c.label == LABEL_POS).sum() == 1 and (c.label == LABEL_AMB).sum() == 1
    assert al.obs_status.iat[0] == "aligned"


def test_unalignable_far_and_stationary_points():
    nodes, arcs = east_road(n_seg=2, seg_len=100, two_way=False)
    # moving along the road, then parked 30 m off-road for 60 s, then 70 m off-road while moving
    xy = [[10 * i, 2] for i in range(10)] + [[100, 30]] * 60 + [[110 + 10 * i, 70] for i in range(5)]
    t = list(range(10)) + list(range(10, 70)) + list(range(70, 75))
    obs, cand, al, lab, _ = _labels(nodes, arcs, [0, 1], xy, t)
    parked = slice(20, 60)
    assert obs.is_stationary.iloc[parked].all()
    assert (al.obs_status.iloc[parked] == "unalignable").all()       # 30 m > 20 m stationary cut-off
    assert (al.obs_status.iloc[-5:] == "unalignable").all()          # 70 m > 40 m cut-off
    assert (lab[np.isin(cand.obs_row, np.arange(20, 60))] == LABEL_UNALIGN).all()
    assert (al.obs_status.iloc[:8] != "unalignable").all()


def test_long_gap_skips_route_positions():
    nodes, arcs = east_road(n_seg=6, seg_len=100, two_way=False)
    xy = [[5 + 10 * i, 2] for i in range(10)] + [[505 + 10 * i, 2] for i in range(9)]
    t = list(range(10)) + list(range(200, 209))                       # 190 s gap, arcs 1-4 not observed
    obs, _, al, _, _ = _labels(nodes, arcs, list(range(6)), xy, t)
    assert obs.gap_before.iat[10]
    assert al.route_arc.iat[9] == 0 and al.route_arc.iat[10] == 5
    assert (al.obs_status != "unalignable").all()


# ----------------------------------------------------------------------------- full pipeline / schema / isolation
def _write_trip(folder: Path, trip_id: str, node_xy, arcs, route, xy, t):
    d = folder / trip_id
    d.mkdir(parents=True)
    tr = track_from_xy(xy, t)
    np.savetxt(d / f"{trip_id}.track", tr[["lon", "lat", "t"]].to_numpy(), fmt="%.7f")
    lon, lat = PROJ.to_lonlat(X0 + np.asarray(node_xy, float)[:, 0], Y0 + np.asarray(node_xy, float)[:, 1])
    np.savetxt(d / f"{trip_id}.nodes", np.c_[lon, lat], fmt="%.7f")
    np.savetxt(d / f"{trip_id}.arcs", np.asarray(arcs), fmt="%d")
    np.savetxt(d / f"{trip_id}.route", np.asarray(route), fmt="%d")
    return d


def test_process_trip_schema_and_trip_isolation(tmp_path):
    cfg = load_config()
    nodes_a, arcs_a = east_road(n_seg=3, seg_len=100)
    nodes_b = [[0, 0], [0, 100], [0, 200]]
    arcs_b = [[0, 1], [1, 2]]                                         # different network, fewer arcs
    da = _write_trip(tmp_path / "train", "00000001", nodes_a, arcs_a, [0, 1, 2], [[5 + 10 * i, 2] for i in range(29)], range(29))
    db = _write_trip(tmp_path / "train", "00000002", nodes_b, arcs_b, [0, 1], [[2, 5 + 10 * i] for i in range(19)], range(19))
    results = {}
    for d in (da, db):
        obs, cand, rep, ctx = process_trip(d, "train", cfg, with_labels=True, return_context=True)
        checks = check_trip(obs, cand, ctx, with_labels=True)
        assert all(checks.values()), checks
        assert set(FEATURE_COLUMNS) <= set(cand.columns)
        assert set(CANDIDATE_COLUMN_DOCS) <= set(cand.columns)
        assert cand.trip_id.unique().tolist() == [d.name]
        assert cand.arc_id.max() < len(ctx["trip"].arcs)
        results[d.name] = (obs, cand)
    # trip B has only arcs 0 and 1: no candidate may carry trip A's arc ids 2..5
    assert results["00000002"][1].arc_id.max() <= 1
    # inference mode needs no route and yields unknown labels
    obs, cand, rep = process_trip(da, "train", cfg, with_labels=False)
    assert (cand.label == -9).all() and "obs_status" not in cand.columns

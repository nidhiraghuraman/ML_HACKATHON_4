"""Phase 5 (HMM / Viterbi) tests on small synthetic road networks (no real dataset needed).

Run from the project folder:  python -m pytest tests -v
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from pathsnap import hmm as H
from pathsnap.geo import point_segment_distance
from test_phase2 import net_from_xy, X0, Y0

# main street y=0: arcs 0,1,2 eastbound (nodes 0->1->2->3), 3,4,5 westbound twins
# parallel one-way street y=30: nodes 4->5->6, arcs 6,7 eastbound
# separate island road far away: nodes 7->8, arc 8 (not connected)
NODES = [[0, 0], [100, 0], [200, 0], [300, 0], [0, 30], [150, 30], [300, 30], [5000, 5000], [5100, 5000]]
ARCS = [[0, 1], [1, 2], [2, 3], [1, 0], [2, 1], [3, 2], [4, 5], [5, 6], [7, 8]]


@pytest.fixture(scope="module")
def net():
    n, _ = net_from_xy(NODES, ARCS)
    return n


def make_inputs(net, xy, t, cands, scores):
    """obs (x, y, t, seq) and candidate rows (seq, arc_id, proj_frac, dist_m, dist_rank) for given
    candidate arc lists per point; scores = probability per candidate (same nesting)."""
    xy = np.asarray(xy, float)
    obs = pd.DataFrame({"seq": np.arange(len(xy)), "x": X0 + xy[:, 0], "y": Y0 + xy[:, 1], "t": np.asarray(t, float)})
    rows = []
    for i, (arcs, sc) in enumerate(zip(cands, scores)):
        for a, s in zip(arcs, sc):
            d, f = point_segment_distance(obs.x[i], obs.y[i], net.x0[a], net.y0[a], net.x1[a], net.y1[a])
            rows.append({"seq": i, "arc_id": a, "proj_frac": float(f), "dist_m": float(d), "score": s})
    c = pd.DataFrame(rows)
    c["dist_rank"] = c.groupby("seq").dist_m.rank(method="first").astype(int) - 1
    return obs, c


def decode(net, obs, c, emission="model", **cfg):
    paths = H.NetworkPaths(net)
    st = H.build_states(obs, c, c.score.to_numpy(), emission, cfg)
    steps = H.transition_inputs(st, paths, cfg)
    arcs, chain = H.viterbi(st, steps, cfg)
    return arcs, chain, st, paths


# ----------------------------------------------------------------------------- shortest paths
def test_directed_shortest_paths(net):
    p = H.NetworkPaths(net)
    dist, _ = p.from_node(0, 1000)
    assert dist[3] == pytest.approx(300, abs=1e-6)
    assert p.path_arcs(0, 3, 1000) == [0, 1, 2]
    assert p.path_arcs(3, 0, 1000) == [5, 4, 3]                # uses the westbound twins
    assert p.path_arcs(6, 4, 1000) is None                     # one-way street cannot be driven backwards
    assert p.path_arcs(0, 3, 150) is None                      # beyond the search limit


def test_route_distance_same_arc_and_backtracking(net):
    p = H.NetworkPaths(net)
    assert p.route_distance(0, 0.2, 0, 0.7, 1000, 15) == pytest.approx(50, abs=1e-6)
    assert p.route_distance(0, 0.5, 0, 0.4, 1000, 15) == 0.0                     # 10 m back = GPS noise
    assert p.route_distance(0, 0.5, 1, 0.5, 1000, 15) == pytest.approx(100, abs=1e-6)
    assert np.isinf(p.route_distance(0, 0.5, 8, 0.5, 1000, 15))                  # disconnected island


# ----------------------------------------------------------------------------- decoding
def test_viterbi_fixes_isolated_wrong_direction(net):
    xs = [20 + 10 * k for k in range(15)]
    xy = [[x, 2] for x in xs]
    cands, scores = [], []
    for k, x in enumerate(xs):
        e = 0 if x < 100 else 1
        w = 3 if x < 100 else 4                               # westbound twin of the same street
        cands.append([e, w]); scores.append([0.3, 0.7] if k == 7 else [0.7, 0.3])
    obs, c = make_inputs(net, xy, range(15), cands, scores)
    arcs, chain, st, paths = decode(net, obs, c)
    clf = H.classifier_only(st)
    assert clf[7] in (3, 4)                                   # classifier alone picks the wrong direction
    assert arcs[7] in (0, 1) and set(arcs) <= {0, 1}          # HMM keeps the eastbound route
    assert H.continuity(arcs, st, paths, {}) == 1.0
    assert H.continuity(clf, st, paths, {}) < 1.0


def test_gap_and_empty_point_start_new_chains(net):
    xy = [[20, 2], [30, 2], [40, 2], [50, 2], [60, 2]]
    t = [0, 1, 2, 200, 201]                                   # 198 s gap before point 3
    cands = [[0], [0], [], [0], [0]]                          # point 2 has no candidates
    obs, c = make_inputs(net, xy, t, cands, [[0.9]] * 2 + [[]] + [[0.9]] * 2)
    arcs, chain, _, _ = decode(net, obs, c)
    assert arcs[2] == -1 and chain[2] == -1
    assert chain[0] == chain[1] and chain[3] == chain[4] and chain[1] != chain[3]


def test_disconnected_candidates_do_not_crash(net):
    xy = [[50, 2], [5050, 5002]]                              # second point on the island road
    obs, c = make_inputs(net, xy, [0, 1], [[0], [8]], [[0.9], [0.9]])
    arcs, chain, _, _ = decode(net, obs, c)
    assert list(arcs) == [0, 8] and chain[0] != chain[1]      # impossible transition -> new chain


def test_parallel_road_rejected_by_transition(net):
    # points on the main street; at one point the parallel one-way street looks slightly better
    xs = [20 + 20 * k for k in range(12)]
    cands, scores = [], []
    for k, x in enumerate(xs):
        main = 0 if x < 100 else (1 if x < 200 else 2)
        par = 6 if x < 150 else 7
        cands.append([main, par]); scores.append([0.45, 0.55] if k == 5 else [0.8, 0.2])
    obs, c = make_inputs(net, [[x, 5] for x in xs], range(0, 24, 2), cands, scores)
    arcs, _, _, _ = decode(net, obs, c)
    assert arcs[5] in (0, 1, 2)                                # jumping 30 m sideways and back is not plausible


# ----------------------------------------------------------------------------- route reconstruction and metrics
def test_route_reconstruction_fills_skipped_arcs(net):
    xy = [[50, 2], [250, 2]]                                   # skips arc 1 entirely
    obs, c = make_inputs(net, xy, [0, 10], [[0], [2]], [[0.9], [0.9]])
    arcs, _, st, paths = decode(net, obs, c)
    route, breaks = H.reconstruct_route(arcs, st, paths, {})
    assert route == [0, 1, 2] and breaks == 0
    m = H.route_metrics(route, np.array([0, 1, 2]), net.length_m)
    assert m["arc_f1"] == 1.0 and m["route_mismatch"] == pytest.approx(0.0)


def test_route_metrics_hand_values(net):
    m = H.route_metrics([0, 1, 6], np.array([0, 1, 2]), net.length_m)
    assert m["arc_precision"] == pytest.approx(2 / 3) and m["arc_recall"] == pytest.approx(2 / 3)
    L = net.length_m
    assert m["route_mismatch"] == pytest.approx((L[6] + L[2]) / (L[0] + L[1] + L[2]))


def test_point_accuracy_and_lenient(net):
    obs = pd.DataFrame({"seq": [0, 1, 2], "obs_status": ["aligned", "aligned", "ambiguous"], "route_arc": [0, 1, 1]})
    pa = H.point_accuracy(np.array([0, 4, 2]), obs, alt={2: {2}})
    assert pa["point_acc"] == 0.5 and pa["amb_lenient_acc"] == 1.0


def test_distance_emission_and_no_labels_needed(net):
    xy = [[20 + 10 * k, 3] for k in range(6)]
    obs, c = make_inputs(net, xy, range(6), [[0, 3, 6]] * 6, [[0, 0, 0]] * 6)
    assert "label" not in c and "obs_status" not in obs          # pure inference inputs
    arcs, _, _, _ = decode(net, obs, c, emission="distance")
    assert set(arcs) == {0} or set(arcs) == {3}                  # stays on the main street
    assert 6 not in set(arcs)

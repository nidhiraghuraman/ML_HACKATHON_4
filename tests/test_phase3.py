"""Phase 3 tests on small synthetic data (no real dataset needed).

Run from the project folder:  python -m pytest tests -v
"""
from __future__ import annotations
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathsnap import modeling as M
from pathsnap import metrics as Q
from pathsnap.features import FEATURE_COLUMNS

PARAMS = {
    "logistic_regression": {"C": 1.0, "max_iter": 500},
    "knn": {"n_neighbors": 5, "weights": "uniform"},
    "decision_tree": {"max_depth": 4, "min_samples_leaf": 2},
    "random_forest": {"n_estimators": 10, "max_depth": 4, "min_samples_leaf": 2, "max_samples": 0.8},
    "hist_gradient_boosting": {"max_iter": 20, "learning_rate": 0.1, "max_leaf_nodes": 15},
}


def synthetic_candidates(n_trips=12, obs_per_trip=30, cands=4, seed=0) -> pd.DataFrame:
    """Each GPS point has `cands` candidates; the positive is usually the closest, with a small
    heading difference. All schema features exist; some values are NaN on purpose."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_trips):
        for s in range(obs_per_trip):
            pos = rng.integers(cands)
            dists = np.sort(rng.uniform(1, 40, cands))
            order = rng.permutation(cands)
            for r in range(cands):
                is_pos = order[r] == pos
                rows.append({"trip_id": f"T{t:02d}", "seq": s, "arc_id": 100 * t + r, "dist_rank": r,
                             "dist_m": dists[r] * (0.5 if is_pos else 1.0),
                             "bearing_diff_deg": rng.uniform(0, 20) if is_pos else rng.uniform(0, 180),
                             "label": int(is_pos), "obs_status": "aligned"})
    df = pd.DataFrame(rows)
    for f in FEATURE_COLUMNS:
        if f not in df:
            df[f] = rng.normal(size=len(df))
    df.loc[df.sample(frac=0.1, random_state=seed).index, "bearing_diff_deg"] = np.nan   # missing headings
    df.loc[df.seq == 0, "speed_mps"] = np.nan
    return df


# ----------------------------------------------------------------------------- labels / features
def test_label_filtering_excludes_ambiguous_and_unalignable():
    df = pd.DataFrame({"trip_id": ["A"] * 6, "seq": [0, 0, 0, 1, 1, 2], "label": [1, 0, -1, -2, -2, 0]})
    out, rep = M.filter_training_rows(df)
    assert sorted(out.label.unique()) == [0, 1]
    assert rep["rows_kept"] == 3 and rep["rows_kept_positive"] == 1 and rep["rows_kept_negative"] == 2
    assert rep["rows_excluded_ambiguous"] == 1 and rep["rows_excluded_unalignable_observation"] == 2
    assert rep["obs_with_unalignable_observation_rows"] == 1
    assert (out.label != -1).all()                     # ambiguous never turned into a negative


def test_invalid_labels_are_counted_not_kept():
    df = pd.DataFrame({"trip_id": ["A"] * 3, "seq": [0, 1, 2], "label": [1, 7, 0]})
    out, rep = M.filter_training_rows(df)
    assert rep["rows_excluded_invalid_label"] == 1 and len(out) == 2


def test_feature_selection_uses_schema_and_excludes_identifiers():
    df = synthetic_candidates(2, 2)
    feats = M.feature_columns(df.columns)
    assert feats == list(FEATURE_COLUMNS)
    for bad in ("trip_id", "seq", "arc_id", "obs_idx", "label", "route_pos", "obs_status", "align_dist_m"):
        assert bad not in feats
    with pytest.raises(ValueError):
        M.assert_no_leakage(feats + ["arc_id"])
    with pytest.raises(ValueError):
        M.assert_no_leakage(feats + ["route_pos"])
    with pytest.raises(ValueError):
        M.feature_columns([c for c in df.columns if c != "dist_m"])     # schema feature missing


def test_ranking_observations_requires_one_positive_and_aligned():
    df = pd.DataFrame({"trip_id": ["A"] * 7, "seq": [0, 0, 1, 1, 2, 2, 3], "dist_rank": [0, 1, 0, 1, 0, 1, 0],
                       "label": [1, 0, 0, 0, 1, -1, 1], "obs_status": ["aligned"] * 4 + ["ambiguous"] * 2 + ["aligned"]})
    r = M.ranking_observations(df, "aligned")
    assert set(r.seq) == {0, 3}                        # seq 1 has no positive, seq 2 is ambiguous


# ----------------------------------------------------------------------------- folds
def test_trip_folds_are_disjoint_complete_and_reproducible():
    df = synthetic_candidates(n_trips=20, obs_per_trip=5)
    f1 = M.make_trip_folds(df, 5, seed=42)
    f2 = M.make_trip_folds(df, 5, seed=42)
    assert f1 == f2                                    # reproducible
    seen = []
    for f in f1:
        assert not set(f["train_trips"]) & set(f["val_trips"])
        assert set(f["train_trips"]) | set(f["val_trips"]) == set(df.trip_id)
        seen += f["val_trips"]
    assert sorted(seen) == sorted(df.trip_id.unique())  # each trip validated exactly once


def test_assert_disjoint_raises_on_overlap():
    with pytest.raises(AssertionError):
        M.assert_disjoint(["T1", "T2"], ["T2", "T3"])


def test_sample_whole_observations_keeps_points_intact_and_is_reproducible():
    df = synthetic_candidates(4, 50)
    a = M.sample_whole_observations(df, 200, seed=1)
    b = M.sample_whole_observations(df, 200, seed=1)
    assert a.equals(b) and len(a) <= 200
    assert (a.groupby(["trip_id", "seq"]).size() == 4).all()   # never a partial GPS point


# ----------------------------------------------------------------------------- metrics
def test_candidate_metrics_known_values():
    y = np.array([1, 1, 0, 0, 0])
    pred = np.array([1, 0, 1, 0, 0])
    score = np.array([0.9, 0.4, 0.6, 0.2, 0.1])
    m = Q.candidate_metrics(y, score, pred, is_probability=True)
    assert m["precision"] == pytest.approx(0.5) and m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5) and m["accuracy"] == pytest.approx(0.6)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert 0 <= m["roc_auc"] <= 1 and np.isfinite(m["log_loss"])
    assert np.isnan(Q.candidate_metrics(y, score, pred, is_probability=False)["log_loss"])


def test_ranking_top_k_and_mrr_within_observations():
    # point A: positive ranked 1st; point B: 2nd; point C: 3rd. Point B has much larger scores
    # overall, which must not affect A or C (ranking happens inside each point only).
    df = pd.DataFrame({"trip_id": ["T"] * 9, "seq": [0, 0, 0, 1, 1, 1, 2, 2, 2], "dist_rank": [0, 1, 2] * 3,
                       "label": [1, 0, 0, 0, 1, 0, 0, 0, 1]})
    score = np.array([0.5, 0.2, 0.1, 99, 98, 1, 0.3, 0.2, 0.1])
    m, per = Q.ranking_metrics(df, score)
    assert list(per["rank"]) == [1, 2, 3]
    assert m["top1"] == pytest.approx(1 / 3) and m["top3"] == pytest.approx(1.0)
    assert m["mrr"] == pytest.approx((1 + 1 / 2 + 1 / 3) / 3)
    assert m["n_obs"] == 3


def test_ranking_ties_broken_by_distance_rank():
    df = pd.DataFrame({"trip_id": ["T"] * 2, "seq": [0, 0], "dist_rank": [1, 0], "label": [1, 0]})
    m, _ = Q.ranking_metrics(df, np.array([0.7, 0.7]))
    assert m["top1"] == 0.0                            # tie -> nearer candidate (dist_rank 0) wins


def test_same_seq_in_different_trips_is_not_mixed():
    df = pd.DataFrame({"trip_id": ["A", "A", "B", "B"], "seq": [0, 0, 0, 0], "dist_rank": [0, 1, 0, 1], "label": [1, 0, 0, 1]})
    m, per = Q.ranking_metrics(df, np.array([0.9, 0.1, 0.2, 0.8]))
    assert m["top1"] == 1.0 and list(per["rank"]) == [1, 1]


def test_lenient_top1_accepts_ambiguous_alternative():
    df = pd.DataFrame({"trip_id": ["T"] * 4, "seq": [0, 0, 1, 1], "dist_rank": [0, 1, 0, 1], "label": [-1, 1, 1, 0]})
    r = Q.lenient_top1(df, np.array([0.9, 0.8, 0.9, 0.1]))
    assert r["lenient_top1"] == 1.0 and r["strict_top1"] == 0.5


# ----------------------------------------------------------------------------- baselines
def test_nearest_baseline_picks_dist_rank_zero():
    df = synthetic_candidates(3, 20)
    m, _ = Q.ranking_metrics(df, M.baseline_nearest_score(df))
    expected = df[df.label == 1].dist_rank.eq(0).mean()
    assert m["top1"] == pytest.approx(expected)


def test_formula_baseline_uses_heading_and_ignores_missing():
    df = pd.DataFrame({"dist_m": [3.0, 3.0, 3.0, 5.0], "bearing_diff_deg": [0.0, 180.0, np.nan, np.nan]})
    s = M.baseline_formula_score(df, sigma_m=10, w_heading=3)
    assert s[0] > s[1]                                 # same distance, wrong-way twin loses
    assert s[2] == pytest.approx(-9 / 200)             # NaN heading: distance term only
    assert s[2] > s[3]


# ----------------------------------------------------------------------------- models
@pytest.mark.parametrize("name", list(M.MODEL_FAMILIES))
def test_models_handle_missing_values_and_output_probabilities(name, tmp_path):
    df = synthetic_candidates(6, 30)
    feats = M.feature_columns(df.columns)
    X, y = df[feats].to_numpy(np.float32), df.label.to_numpy()
    assert np.isnan(X).any()                           # missing values present
    model = M.build_model(name, PARAMS[name], seed=0)
    model.fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert p.shape == (len(df),) and np.all((p >= 0) & (p <= 1))
    # serialisation round trip gives the same predictions (random forest averages trees in
    # parallel, so the summation order may differ by ~1e-16; allow a negligible tolerance)
    path = tmp_path / f"{name}.joblib"
    joblib.dump(model, path)
    assert np.allclose(joblib.load(path).predict_proba(X)[:, 1], p, rtol=0, atol=1e-12)


def test_learned_model_beats_chance_on_synthetic_signal():
    df = synthetic_candidates(10, 40, seed=3)
    feats = M.feature_columns(df.columns)
    folds = M.make_trip_folds(df, 2, seed=0)
    f = folds[0]
    tr, va = df[df.trip_id.isin(f["train_trips"])], df[df.trip_id.isin(f["val_trips"])]
    model = M.build_model("hist_gradient_boosting", PARAMS["hist_gradient_boosting"], seed=0).fit(tr[feats].to_numpy(np.float32), tr.label)
    m, _ = Q.ranking_metrics(va, model.predict_proba(va[feats].to_numpy(np.float32))[:, 1])
    assert m["top1"] > 0.5                             # 4 candidates -> chance is 0.25

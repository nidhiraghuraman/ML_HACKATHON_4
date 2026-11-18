"""Phase 4 analysis: probability calibration and error analysis (on out-of-fold predictions).

Calibration asks: "when the model says 0.8, is the candidate correct ~80% of the time?" The HMM in
Phase 5 uses these probabilities as emission scores, so systematic over/under-confidence matters.
  reliability table: probabilities in equal-width bins; per bin mean predicted vs observed share.
  ECE (expected calibration error) = sum over bins of (bin share) * |observed - predicted|.
  Brier score = mean squared difference between probability and 0/1 label.

Error analysis: for every aligned GPS point whose top-ranked candidate is wrong, what KIND of error?
  reverse_direction  predicted arc = the opposite direction of the correct arc (same street)
  adjacent_arc       predicted arc shares a node with the correct arc (junction / next segment)
  other_road         anything else (e.g. a parallel street)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from . import metrics as Q


def reliability_table(y, p, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    b = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows, ece = [], 0.0
    for k in range(n_bins):
        m = b == k
        if not m.any():
            rows.append({"bin": k, "lo": edges[k], "hi": edges[k + 1], "n": 0, "mean_pred": np.nan, "observed": np.nan})
            continue
        mp, ob = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(ob - mp)
        rows.append({"bin": k, "lo": edges[k], "hi": edges[k + 1], "n": int(m.sum()), "mean_pred": mp, "observed": ob})
    return pd.DataFrame(rows), float(ece)


def calibration_summary(y, p, n_bins=10) -> dict:
    table, ece = reliability_table(y, p, n_bins)
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return {"ece": ece, "brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p, labels=[0, 1])), "table": table}


def classify_errors(rank_rows: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """One row per aligned GPS point: predicted (top-1) arc, correct arc and error type.
    rank_rows: candidate rows of aligned points with exactly one positive."""
    r = Q.rank_within_observations(rank_rows.trip_id.to_numpy(), rank_rows.seq.to_numpy(), score, rank_rows.dist_rank.to_numpy())
    top = rank_rows[r == 1].set_index(["trip_id", "seq"])
    true = rank_rows[rank_rows.label == 1].set_index(["trip_id", "seq"]).reindex(top.index)
    out = pd.DataFrame(index=top.index)
    out["pred_arc"], out["true_arc"] = top.arc_id, true.arc_id
    pf, pt, tf, tt = top.from_node.to_numpy(), top.to_node.to_numpy(), true.from_node.to_numpy(), true.to_node.to_numpy()
    correct = top.arc_id.to_numpy() == true.arc_id.to_numpy()
    rev = (pf == tt) & (pt == tf)
    adj = (pf == tf) | (pf == tt) | (pt == tf) | (pt == tt)
    out["error_type"] = np.where(correct, "correct", np.where(rev, "reverse_direction", np.where(adj, "adjacent_arc", "other_road")))
    for c in ("is_stationary", "bearing_valid", "gap_before", "n_candidates", "speed_mps", "arc_has_twin"):
        out[c] = true[c].to_numpy() if c in true else np.nan
    out["true_dist_rank"] = true.dist_rank.to_numpy()
    out["true_dist_m"] = true.dist_m.to_numpy()
    return out.reset_index()


def error_slices(err: pd.DataFrame) -> pd.DataFrame:
    """Top-1 error rate in meaningful groups of GPS points."""
    e = err.assign(wrong=err.error_type != "correct")
    groups = {
        "all points": pd.Series("all", index=e.index),
        "vehicle stationary": e.is_stationary.map({1: "stationary", 0: "moving"}),
        "GPS heading reliable": e.bearing_valid.map({1: "heading reliable", 0: "no reliable heading"}),
        "time gap before point": e.gap_before.map({1: "after a gap", 0: "no gap"}),
        "correct road is two-way": e.arc_has_twin.map({1: "two-way (twin arc exists)", 0: "one-way"}),
        "correct arc's distance rank": pd.cut(e.true_dist_rank, [-1, 0, 1, 3, 1000], labels=["nearest", "2nd", "3rd-4th", "5th+"]),
        "distance to correct arc (m)": pd.cut(e.true_dist_m, [-0.1, 5, 10, 20, 1e9], labels=["0-5", "5-10", "10-20", ">20"]),
        "speed (m/s)": pd.cut(e.speed_mps, [-0.1, 2, 5, 15, 30, 1e9], labels=["0-2", "2-5", "5-15", "15-30", ">30"]),
        "candidates per point": pd.cut(e.n_candidates, [0, 4, 8, 16, 100], labels=["1-4", "5-8", "9-16", "17-24"]),
    }
    rows = []
    for gname, key in groups.items():
        for val, g in e.groupby(key, observed=True):
            rows.append({"slice": gname, "group": str(val), "points": int(len(g)), "share_of_points": len(g) / len(e),
                         "top1_error_rate": float(g.wrong.mean()), "share_of_all_errors": float(g.wrong.sum() / max(e.wrong.sum(), 1))})
    return pd.DataFrame(rows)


def worst_trips(err: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    t = err.assign(wrong=err.error_type != "correct").groupby("trip_id").agg(points=("wrong", "size"), top1_error_rate=("wrong", "mean"))
    types = pd.crosstab(err.trip_id, err.error_type, normalize="index")
    return t.join(types).sort_values("top1_error_rate", ascending=False).head(n).reset_index()

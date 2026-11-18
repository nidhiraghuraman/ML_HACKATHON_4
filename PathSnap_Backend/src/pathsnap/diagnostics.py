"""Phase 2 diagnostics: quality checks, coverage study, label and feature statistics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .features import FEATURE_SCHEMA, FEATURE_COLUMNS


# ----------------------------------------------------------------------------- per-trip checks
def check_trip(obs: pd.DataFrame, cand: pd.DataFrame, ctx: dict, with_labels: bool) -> dict:
    """Hard correctness checks for one trip. Every value should be True."""
    trip, net = ctx["trip"], ctx["net"]
    arcs = cand["arc_id"].to_numpy()
    res = {}
    res["arc_ids_exist"] = bool(((arcs >= 0) & (arcs < len(trip.arcs))).all())
    ok = res["arc_ids_exist"]
    res["arcs_match_this_trip"] = bool(ok and (trip.arcs[arcs, 0] == cand["from_node"].to_numpy()).all()
                                       and (trip.arcs[arcs, 1] == cand["to_node"].to_numpy()).all())
    res["single_trip_id"] = bool(cand["trip_id"].nunique() <= 1 and obs["trip_id"].nunique() == 1
                                 and (cand.empty or cand["trip_id"].iat[0] == trip.trip_id))
    res["all_observations_kept"] = bool(len(obs) == len(trip.track) and obs["obs_idx"].is_unique)
    raw_t = trip.track["t"].to_numpy()
    res["timestamps_preserved"] = bool(np.array_equal(obs["t"].to_numpy(), raw_t[obs["obs_idx"].to_numpy()])
                                       and np.array_equal(cand["t"].to_numpy(), obs["t"].to_numpy()[cand["seq"].to_numpy()]))
    res["no_duplicate_candidates"] = bool(not cand.duplicated(["seq", "arc_id"]).any())
    feats = cand[FEATURE_COLUMNS].to_numpy(float)
    res["no_infinite_features"] = bool(not np.isinf(feats).any())
    must = [k for k, v in FEATURE_SCHEMA.items() if not v[2]]
    res["required_features_not_missing"] = bool(not cand[must].isna().any().any())
    g = cand.groupby("seq")["dist_m"]
    res["rank_matches_distance"] = bool((g.diff().fillna(0) >= -1e-6).all())
    if with_labels:
        pos = cand[cand["label"] == 1]
        route = trip.route
        res["one_positive_max_per_obs"] = bool(not pos["seq"].duplicated().any())
        rp = pos["route_pos"].to_numpy()
        res["positives_are_aligned_route_arcs"] = bool(len(pos) == 0 or (route[rp] == pos["arc_id"].to_numpy()).all())
        res["no_positive_on_unalignable"] = bool((pos["obs_status"] != "unalignable").all())
        res["route_progress_monotone"] = bool((np.diff(obs["route_pos"].to_numpy()) >= 0).all())
    return res


# ----------------------------------------------------------------------------- coverage
def coverage_table(obs: pd.DataFrame, radii, ks) -> pd.DataFrame:
    """Share of alignable observations whose aligned arc would be a candidate for a FIXED
    radius r and top-K (no expansion). Uses the exact distance and global rank of the aligned arc."""
    o = obs[obs["obs_status"] != "unalignable"]
    d, r = o["true_arc_dist_m"].to_numpy(), o["true_arc_rank_all"].to_numpy()
    rows = []
    for rad in radii:
        for k in list(ks) + [10**9]:
            rows.append({"radius_m": rad, "k": "all" if k == 10**9 else k,
                         "coverage": float(np.mean((d <= rad) & (r < k) & (r >= 0)))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- labels
def label_quality(obs: pd.DataFrame, reports: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    """Per-trip label statistics plus human-readable warnings."""
    rows, warnings = [], []
    for (part, tid), o in obs.groupby(["partition", "trip_id"], sort=False):
        n = len(o)
        st = o["obs_status"].value_counts()
        al = o[o["obs_status"] != "unalignable"]
        rep = reports[(reports.trip_id == tid) & (reports.partition == part)].iloc[0]
        prog = al["route_progress_m"].to_numpy()
        tt = al["t"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            dprog, dtt = np.diff(prog), np.diff(tt)
            vprog = np.where(dtt > 0, dprog / dtt, np.inf)
        jumps = int(((vprog > 70) & (dprog > 200)).sum())
        row = {
            "partition": part, "trip_id": tid, "n_obs": n,
            "aligned_frac": st.get("aligned", 0) / n, "ambiguous_frac": st.get("ambiguous", 0) / n,
            "unalignable_frac": st.get("unalignable", 0) / n,
            "median_align_dist_m": float(al["align_dist_m"].median()) if len(al) else np.nan,
            "p95_align_dist_m": float(al["align_dist_m"].quantile(0.95)) if len(al) else np.nan,
            "route_positions_hit_frac": rep["route_positions_hit"] / max(rep["route_arcs"], 1),
            "agree_with_unordered_nearest": float((al["route_arc"] == al["naive_route_arc"]).mean()) if len(al) else np.nan,
            "positive_in_candidates_frac": float(al["positive_in_candidates"].mean()) if len(al) else np.nan,
            "progress_jumps": jumps,
            "near_break_frac": float(al["near_route_break"].mean()) if len(al) else 0.0,
            "stationary_frac": float(o["is_stationary"].mean()),
        }
        rows.append(row)
        if row["unalignable_frac"] > 0.2:
            warnings.append(f"{part} {tid}: {row['unalignable_frac']:.0%} of observations unalignable")
        if row["route_positions_hit_frac"] < 0.7:
            warnings.append(f"{part} {tid}: only {row['route_positions_hit_frac']:.0%} of route positions received a GPS point")
        if row["median_align_dist_m"] > 15:
            warnings.append(f"{part} {tid}: median alignment distance {row['median_align_dist_m']:.1f} m")
        if jumps:
            warnings.append(f"{part} {tid}: {jumps} implausible jumps along the route (> 70 m/s and > 200 m)")
        if row["positive_in_candidates_frac"] < 0.98:
            warnings.append(f"{part} {tid}: aligned arc missing from candidates for {1 - row['positive_in_candidates_frac']:.1%} of alignable points")
    return pd.DataFrame(rows), warnings


# ----------------------------------------------------------------------------- features
def feature_stats(cand: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in FEATURE_COLUMNS:
        v = cand[f].to_numpy(float)
        fin = v[np.isfinite(v)]
        q1, q3 = (np.percentile(fin, [25, 75]) if len(fin) else (np.nan, np.nan))
        iqr = q3 - q1
        out = int(((fin < q1 - 3 * iqr) | (fin > q3 + 3 * iqr)).sum()) if len(fin) and iqr > 0 else 0
        unit, level, miss_ok, _ = FEATURE_SCHEMA[f]
        rows.append({"feature": f, "unit": unit, "level": level, "missing_allowed": miss_ok,
                     "missing": int(np.isnan(v).sum()), "missing_frac": float(np.isnan(v).mean()),
                     "infinite": int(np.isinf(v).sum()),
                     "mean": float(fin.mean()) if len(fin) else np.nan, "std": float(fin.std()) if len(fin) else np.nan,
                     "min": float(fin.min()) if len(fin) else np.nan, "p50": float(np.median(fin)) if len(fin) else np.nan,
                     "p99": float(np.percentile(fin, 99)) if len(fin) else np.nan, "max": float(fin.max()) if len(fin) else np.nan,
                     "outliers_3iqr": out})
    return pd.DataFrame(rows)


def single_feature_auc(cand: pd.DataFrame) -> pd.DataFrame:
    """Ranking power of each feature ALONE for positive vs negative candidates (labels 1 / 0).
    AUC 0.5 = useless on its own, 1.0 / 0.0 = perfectly separates (direction shown separately).
    This is a screening statistic, not a model. NaNs are dropped per feature."""
    d = cand[cand["label"].isin([0, 1])]
    y = d["label"].to_numpy()
    rows = []
    for f in FEATURE_COLUMNS:
        v = d[f].to_numpy(float)
        m = np.isfinite(v)
        if m.sum() < 10 or len(np.unique(y[m])) < 2 or len(np.unique(v[m])) < 2:
            rows.append({"feature": f, "auc": np.nan, "auc_strength": np.nan, "higher_means": "", "n_used": int(m.sum())})
            continue
        auc = roc_auc_score(y[m], v[m])
        rows.append({"feature": f, "auc": auc, "auc_strength": max(auc, 1 - auc),
                     "higher_means": "more likely positive" if auc >= 0.5 else "less likely positive", "n_used": int(m.sum())})
    return pd.DataFrame(rows).sort_values("auc_strength", ascending=False)

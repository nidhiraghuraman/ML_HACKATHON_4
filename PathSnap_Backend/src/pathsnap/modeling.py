"""Phase 3 building blocks: column roles, label filtering, trip-grouped folds, models, baselines.

Column roles are derived from the Phase 2 schema (features.FEATURE_SCHEMA), never typed by hand:
  FEATURES   the 23 inference-time features (none uses the reference route)
  TARGET     "label" (1 positive, 0 negative, -1 ambiguous, -2 unalignable observation)
  GROUP      "trip_id" (cross-validation unit)
  everything else is metadata: kept for ranking and tracing, never given to a model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from .features import FEATURE_COLUMNS

TARGET = "label"
GROUP = "trip_id"
OBS_KEY = ["trip_id", "seq"]
# Columns that must NEVER be model inputs: identifiers (allow memorisation) and route-derived
# columns (reveal the answer). Checked by assert_no_leakage().
ID_COLUMNS = ["trip_id", "partition", "obs_idx", "seq", "arc_id", "from_node", "to_node", "t"]
ROUTE_DERIVED = ["label", "obs_status", "route_pos", "align_dist_m"]
INFO_ONLY = ["radius_m", "radius_used_m", "arc_bearing_deg", "move_bearing"]
FORBIDDEN = set(ID_COLUMNS) | set(ROUTE_DERIVED) | set(INFO_ONLY)

MODEL_FAMILIES = {
    "logistic_regression": "Linear",
    "knn": "Instance-based",
    "decision_tree": "Tree",
    "random_forest": "Bagging ensemble",
    "hist_gradient_boosting": "Boosting ensemble",
}


def feature_columns(df_columns) -> list[str]:
    """Features = Phase 2 schema columns that exist in the data. Raises if any is missing."""
    missing = [f for f in FEATURE_COLUMNS if f not in df_columns]
    if missing:
        raise ValueError(f"Dataset is missing schema features: {missing}")
    feats = list(FEATURE_COLUMNS)
    assert_no_leakage(feats)
    return feats


def assert_no_leakage(feats: list[str]) -> None:
    bad = sorted(set(feats) & FORBIDDEN)
    if bad:
        raise ValueError(f"Forbidden (identifier / route-derived) columns in features: {bad}")


# ----------------------------------------------------------------------------- label filtering
def filter_training_rows(cand: pd.DataFrame, allowed=(0, 1)) -> tuple[pd.DataFrame, dict]:
    """Keep only rows whose label is in `allowed`. Report every exclusion reason (rows and GPS points)."""
    lab = cand[TARGET]
    rep = {"rows_total": int(len(cand)), "obs_total": int(cand[OBS_KEY].drop_duplicates().shape[0])}
    names = {-1: "ambiguous", -2: "unalignable_observation", -9: "unknown"}
    for v, name in names.items():
        m = lab == v
        rep[f"rows_excluded_{name}"] = int(m.sum())
        rep[f"obs_with_{name}_rows"] = int(cand.loc[m, OBS_KEY].drop_duplicates().shape[0])
    invalid = ~lab.isin(list(allowed) + list(names))
    rep["rows_excluded_invalid_label"] = int(invalid.sum())
    keep = lab.isin(list(allowed))
    out = cand[keep].reset_index(drop=True)
    rep["rows_kept"] = int(len(out))
    rep["rows_kept_positive"] = int((out[TARGET] == 1).sum())
    rep["rows_kept_negative"] = int((out[TARGET] == 0).sum())
    rep["obs_kept"] = int(out[OBS_KEY].drop_duplicates().shape[0])
    rep["positive_rate"] = float(out[TARGET].mean()) if len(out) else float("nan")
    return out, rep


def ranking_observations(cand: pd.DataFrame, status: str = "aligned") -> pd.DataFrame:
    """Candidate rows of GPS points that are fair to rank: given status AND exactly one positive.
    All of their candidates are kept (including -1 duplicates), so models are ranked on the full
    candidate set a real prediction would see."""
    c = cand[cand["obs_status"] == status]
    npos = c.assign(_p=(c[TARGET] == 1)).groupby(OBS_KEY)["_p"].transform("sum")
    return c[npos == 1].reset_index(drop=True)


# ----------------------------------------------------------------------------- folds
def make_trip_folds(rows: pd.DataFrame, n_splits: int, seed: int) -> list[dict]:
    """Trip-grouped, label-stratified folds. Every trip is in exactly one validation fold."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    groups = rows[GROUP].to_numpy()
    for k, (tr, va) in enumerate(sgkf.split(np.zeros(len(rows)), rows[TARGET].to_numpy(), groups)):
        tr_trips, va_trips = sorted(set(groups[tr])), sorted(set(groups[va]))
        assert_disjoint(tr_trips, va_trips)
        folds.append({"fold": k, "train_trips": tr_trips, "val_trips": va_trips})
    all_val = [t for f in folds for t in f["val_trips"]]
    assert len(all_val) == len(set(all_val)) == rows[GROUP].nunique(), "each trip must be validated exactly once"
    return folds


def assert_disjoint(train_trips, val_trips) -> None:
    overlap = set(train_trips) & set(val_trips)
    if overlap:
        raise AssertionError(f"Trip leakage: trips in both training and validation: {sorted(overlap)}")


# ----------------------------------------------------------------------------- models
def _imputer():
    # Median imputation + a 0/1 "was missing" column per feature with NaNs. Fitted on training
    # rows of each fold only (inside the Pipeline), so validation data never influences it.
    return SimpleImputer(strategy="median", add_indicator=True)


def build_model(name: str, params: dict, seed: int):
    p = dict(params)
    if name == "logistic_regression":
        return Pipeline([("impute", _imputer()), ("scale", StandardScaler()),
                         ("model", LogisticRegression(C=p["C"], max_iter=p["max_iter"], random_state=seed))])
    if name == "knn":
        return Pipeline([("impute", _imputer()), ("scale", StandardScaler()),
                         ("model", KNeighborsClassifier(n_neighbors=p["n_neighbors"], weights=p["weights"], n_jobs=-1))])
    if name == "decision_tree":
        return Pipeline([("impute", _imputer()),
                         ("model", DecisionTreeClassifier(max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], random_state=seed))])
    if name == "random_forest":
        return Pipeline([("impute", _imputer()),
                         ("model", RandomForestClassifier(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                                          min_samples_leaf=p["min_samples_leaf"], max_samples=p["max_samples"],
                                                          n_jobs=-1, random_state=seed))])
    if name == "hist_gradient_boosting":
        # Handles NaN natively, so no imputer. early_stopping=False: a fixed number of rounds,
        # because sklearn's internal early-stopping split is row-level (would mix trips).
        # Optional keys (used by Phase 4 tuning) default to scikit-learn's own defaults, so the
        # Phase 3 configuration builds exactly the same model as before.
        return HistGradientBoostingClassifier(max_iter=p["max_iter"], learning_rate=p["learning_rate"],
                                              max_leaf_nodes=p["max_leaf_nodes"], min_samples_leaf=p.get("min_samples_leaf", 20),
                                              l2_regularization=p.get("l2_regularization", 0.0), max_depth=p.get("max_depth"),
                                              max_features=p.get("max_features", 1.0), early_stopping=False, random_state=seed)
    raise ValueError(f"unknown model {name}")


def sample_whole_observations(rows: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    """Random GPS points (with ALL their candidates) until about max_rows rows. Used for KNN only."""
    if len(rows) <= max_rows:
        return rows
    keys = rows[OBS_KEY].drop_duplicates()
    keys = keys.sample(frac=1.0, random_state=seed)
    sizes = rows.groupby(OBS_KEY).size().reindex(pd.MultiIndex.from_frame(keys)).to_numpy()
    n = int(np.searchsorted(np.cumsum(sizes), max_rows, side="right"))
    chosen = keys.iloc[:max(n, 1)]
    return rows.merge(chosen, on=OBS_KEY, how="inner")


# ----------------------------------------------------------------------------- baselines
def baseline_nearest_score(df: pd.DataFrame) -> np.ndarray:
    """Higher = better. The nearest candidate (dist_rank 0) wins; ties already broken in Phase 2."""
    return -df["dist_rank"].to_numpy(float)


def baseline_formula_score(df: pd.DataFrame, sigma_m: float, w_heading: float) -> np.ndarray:
    """Fixed hand-made score (not tuned): -dist^2/(2 sigma^2) - w*(1-cos(heading diff))/2.
    The heading term is skipped when the GPS heading is unreliable (NaN)."""
    d = df["dist_m"].to_numpy(float)
    diff = df["bearing_diff_deg"].to_numpy(float)
    pen = np.where(np.isfinite(diff), w_heading * (1 - np.cos(np.radians(diff))) / 2, 0.0)
    return -(d ** 2) / (2 * sigma_m ** 2) - pen

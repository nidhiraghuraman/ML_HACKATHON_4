"""Phase 3 feature analysis.

- Permutation importance (main method): for each CV fold, take that fold's model and a sample of
  its HELD-OUT GPS points, shuffle one feature, and measure how much Top-1 drops. The model never
  saw these trips, and this runs after model selection, so it cannot influence the choice.
- Logistic Regression coefficients (on standardised features): sign and size of a linear effect.
- Impurity importance (trees): how much each feature reduced impurity during training. Known to
  favour continuous features with many split points; reported for comparison only.
None of these prove causation: correlated features share or steal importance from each other.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .metrics import ranking_metrics
from .modeling import OBS_KEY


def permutation_importance_top1(fold_models: list, cand: pd.DataFrame, feats: list[str], obs_per_fold: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drops = {f: [] for f in feats}
    base_scores = []
    for model, fold in fold_models:
        va = cand[cand.in_rank & cand.trip_id.isin(fold["val_trips"])]
        keys = va[OBS_KEY].drop_duplicates()
        if len(keys) > obs_per_fold:
            keys = keys.sample(obs_per_fold, random_state=seed + fold["fold"])
        va = va.merge(keys, on=OBS_KEY).reset_index(drop=True)
        X = va[feats].to_numpy(np.float32)
        base, _ = ranking_metrics(va, model.predict_proba(X)[:, 1])
        base_scores.append(base["top1"])
        for j, f in enumerate(feats):
            Xp = X.copy()
            Xp[:, j] = X[rng.permutation(len(X)), j]
            m, _ = ranking_metrics(va, model.predict_proba(Xp)[:, 1])
            drops[f].append(base["top1"] - m["top1"])
    rows = [{"feature": f, "top1_drop_mean": float(np.mean(v)), "top1_drop_std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
             "folds": len(v)} for f, v in drops.items()]
    out = pd.DataFrame(rows).sort_values("top1_drop_mean", ascending=False).reset_index(drop=True)
    out.attrs["baseline_top1_on_sample"] = float(np.mean(base_scores))
    return out


def _input_names(pipeline, feats):
    """Feature names after the imputer (adds '<name>_missing' indicator columns)."""
    imp = pipeline.named_steps.get("impute")
    if imp is None:
        return list(feats)
    names = list(feats)
    if imp.indicator_ is not None:
        names += [f"{feats[i]}_missing" for i in imp.indicator_.features_]
    return names


def lr_coefficients(pipeline, feats) -> pd.DataFrame:
    names = _input_names(pipeline, feats)
    coef = pipeline.named_steps["model"].coef_.ravel()
    return pd.DataFrame({"feature": names, "coef": coef}).sort_values("coef", key=np.abs, ascending=False).reset_index(drop=True)


def impurity_importance(final_models: dict, feats) -> pd.DataFrame | None:
    rows = []
    for name in ("decision_tree", "random_forest"):
        m = final_models.get(name)
        if m is None:
            continue
        names = _input_names(m, feats)
        for n, v in zip(names, m.named_steps["model"].feature_importances_):
            rows.append({"model": name, "feature": n, "importance": float(v)})
    return pd.DataFrame(rows) if rows else None


def redundancy(rows: pd.DataFrame, feats, seed: int, threshold: float = 0.9) -> pd.DataFrame:
    s = rows.sample(min(len(rows), 200_000), random_state=seed)
    corr = s[feats].corr(method="spearman")
    out = []
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            v = corr.iat[i, j]
            if np.isfinite(v) and abs(v) >= threshold:
                out.append({"feature_a": feats[i], "feature_b": feats[j], "spearman": float(v)})
    return pd.DataFrame(out, columns=["feature_a", "feature_b", "spearman"])

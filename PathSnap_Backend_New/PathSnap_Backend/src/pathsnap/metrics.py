"""Phase 3 metrics.

Candidate level (one row = one GPS point x one candidate arc, labels 0/1 only):
  accuracy, precision, recall, F1 (positive class), macro F1, ROC-AUC, PR-AUC (average precision),
  log loss (only for real probabilities), confusion matrix.

GPS-point level (the map-matching question: "which candidate is the road?"):
  candidates are sorted by score WITHIN each GPS point (never across points); ties are broken by
  dist_rank (nearer first), exactly like inference. rank = position of the positive (1 = top).
  Top-k accuracy = share of points whose positive has rank <= k.
  MRR = mean of 1/rank (1.0 if always first, 0.5 if always second, ...).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, log_loss, confusion_matrix)


def candidate_metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray, is_probability: bool) -> dict:
    """y: 0/1 labels; score: any 'higher = more likely positive' value; pred: 0/1 decisions."""
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "roc_auc": roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan,
        "log_loss": log_loss(y, np.clip(score, 1e-7, 1 - 1e-7), labels=[0, 1]) if is_probability else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return {k: (float(v) if not isinstance(v, int) else v) for k, v in out.items()}


def rank_within_observations(trip: np.ndarray, seq: np.ndarray, score: np.ndarray, dist_rank: np.ndarray) -> np.ndarray:
    """1-based rank of every row inside its GPS point (trip, seq), by score desc then dist_rank asc."""
    tcode = pd.factorize(trip)[0]
    order = np.lexsort((dist_rank, -np.asarray(score, float), seq, tcode))
    t_s, s_s = tcode[order], np.asarray(seq)[order]
    new = np.r_[True, (t_s[1:] != t_s[:-1]) | (s_s[1:] != s_s[:-1])]
    pos = np.arange(len(order))
    start = np.maximum.accumulate(np.where(new, pos, 0))
    rank = np.empty(len(order), np.int64)
    rank[order] = pos - start + 1
    return rank


def ranking_metrics(df: pd.DataFrame, score: np.ndarray, ks=(1, 3, 5)) -> tuple[dict, pd.DataFrame]:
    """df must contain trip_id, seq, dist_rank, label for the GPS points to evaluate (each with
    exactly one label==1 row). Returns (summary dict, per-point table with the positive's rank)."""
    r = rank_within_observations(df["trip_id"].to_numpy(), df["seq"].to_numpy(), score, df["dist_rank"].to_numpy())
    pos = df["label"].to_numpy() == 1
    per = pd.DataFrame({"trip_id": df["trip_id"].to_numpy()[pos], "seq": df["seq"].to_numpy()[pos], "rank": r[pos]})
    out = {f"top{k}": float((per["rank"] <= k).mean()) for k in ks}
    out["mrr"] = float((1.0 / per["rank"]).mean())
    out["n_obs"] = int(len(per))
    return out, per


def lenient_top1(df: pd.DataFrame, score: np.ndarray) -> dict:
    """For AMBIGUOUS GPS points: correct if the top candidate is the positive OR a labelled
    alternative (-1), since both are equally consistent with the route there."""
    r = rank_within_observations(df["trip_id"].to_numpy(), df["seq"].to_numpy(), score, df["dist_rank"].to_numpy())
    top = df[r == 1]
    return {"lenient_top1": float(top["label"].isin([1, -1]).mean()) if len(top) else np.nan,
            "strict_top1": float((top["label"] == 1).mean()) if len(top) else np.nan,
            "n_obs": int(len(top))}


def per_trip_top1(per_point: pd.DataFrame) -> pd.Series:
    return per_point.assign(hit=per_point["rank"] == 1).groupby("trip_id")["hit"].mean()

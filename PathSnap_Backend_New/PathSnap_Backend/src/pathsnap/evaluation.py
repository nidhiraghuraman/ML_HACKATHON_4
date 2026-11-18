"""Shared evaluation helpers (moved from scripts/train_phase3_models.py so Phases 3, 4 and 5 score
models in exactly the same way)."""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import metrics as Q
from . import modeling as M


def add_eval_flags(cand: pd.DataFrame) -> pd.DataFrame:
    """Flags used to pick evaluation rows (never used as features).
    in_fit:  label 0/1 (candidate-level metrics; training rows)
    in_rank: aligned GPS point with exactly one positive (primary ranking metrics)
    in_amb:  ambiguous GPS point with exactly one positive (lenient Top-1)"""
    c = cand
    c["in_fit"] = c.label.isin([0, 1])
    npos = (c.label == 1).groupby([c.trip_id, c.seq]).transform("sum")
    c["in_rank"] = (c.obs_status == "aligned") & (npos == 1)
    c["in_amb"] = (c.obs_status == "ambiguous") & (npos == 1)
    return c


def baseline_scores(df: pd.DataFrame, cfg3: dict) -> dict:
    bf = cfg3["baseline_formula"]
    return {"baseline_nearest": M.baseline_nearest_score(df),
            "baseline_formula": M.baseline_formula_score(df, bf["sigma_m"], bf["w_heading"])}


def evaluate_scores(df: pd.DataFrame, score: np.ndarray, is_prob: bool, threshold: float) -> tuple[dict, pd.DataFrame]:
    """All metrics for one score vector over the rows of df (flags in_fit / in_rank / in_amb)."""
    out = {}
    fit = df.in_fit.to_numpy()
    y = df.label.to_numpy()[fit]
    s = score[fit]
    if is_prob:
        pred = (s >= threshold).astype(int)
    else:
        # a ranking score has no threshold: predict "correct road" for the top candidate of each point
        sub = df[fit]
        r = Q.rank_within_observations(sub.trip_id.to_numpy(), sub.seq.to_numpy(), s, sub.dist_rank.to_numpy())
        pred = (r == 1).astype(int)
    out.update(Q.candidate_metrics(y, s, pred, is_prob))
    rk = df.in_rank.to_numpy()
    rm, per = Q.ranking_metrics(df[rk], score[rk])
    out.update(rm)
    am = df.in_amb.to_numpy()
    la = Q.lenient_top1(df[am], score[am])
    out["amb_lenient_top1"], out["amb_strict_top1"], out["amb_n_obs"] = la["lenient_top1"], la["strict_top1"], la["n_obs"]
    return out, per

"""PHASE 3 — baselines + five model families, trip-grouped cross-validation, model selection.

Run from the project folder (ML_Hackathon) with the venv active:
    python scripts\\train_phase3_models.py                (full experiment, ~15-30 min)
    python scripts\\train_phase3_models.py --report-only  (rebuild the report from saved results)
    python scripts\\train_phase3_models.py --quick        (1-2 min smoke test on 15 trips; NOT real results)

What it does:
  1. load the Phase 2 TRAIN candidates and verify the schema and label values
  2. keep only labels 0/1 for fitting (ambiguous/unalignable excluded and counted)
  3. build 5 trip-grouped folds (no trip ever in train and validation of the same fold)
  4. per fold: evaluate 2 baselines and fit/evaluate 5 models (candidate + GPS-point metrics, runtime)
  5. aggregate, select the model by the rule in config.yaml, SAVE the selection
  6. refit on all 80 train trips, save the selected model
  7. (config flag) independent student-test scores for every model — after selection, never used to choose
  8. feature analysis, plots, report
"""
from __future__ import annotations
import io
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import sklearn

from pathsnap.config import load_config, resolve
from pathsnap import modeling as M
from pathsnap import metrics as Q
from pathsnap import phase3_plots as P

MODELS = list(M.MODEL_FAMILIES)
BASELINES = ["baseline_nearest", "baseline_formula"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def model_bytes(model) -> int:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.tell()


def load_partition(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"Missing {path}. Run Phase 2 first: python scripts\\build_phase2_dataset.py")
    return pd.read_parquet(path)


def verify_schema(cand: pd.DataFrame, partition: str) -> list[str]:
    feats = M.feature_columns(cand.columns)
    needed = M.OBS_KEY + ["arc_id", "dist_rank", "label", "obs_status"]
    miss = [c for c in needed if c not in cand.columns]
    if miss:
        raise ValueError(f"{partition}: missing columns {miss}")
    bad = set(cand.label.unique()) - {1, 0, -1, -2}
    if bad:
        raise ValueError(f"{partition}: unexpected label values {bad}")
    if set(cand.partition.unique()) != {partition}:
        raise ValueError(f"{partition}: file contains other partitions {set(cand.partition.unique())}")
    return feats


def add_eval_flags(cand: pd.DataFrame) -> pd.DataFrame:
    """Flags used to pick evaluation rows (computed once, never used as features)."""
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


def apply_quick_mode(cfg: dict) -> dict:
    """--quick: a ~1-2 minute smoke test (15 trips, 3 folds, small models) written to separate
    folders. Its numbers are NOT results; it only checks that everything runs."""
    c3 = cfg["phase3"]
    c3["output_dir"] = c3["output_dir"] + "_quick"
    c3["artifacts_dir"] = c3["artifacts_dir"] + "_quick"
    c3["cv_folds"] = 3
    c3["knn_train_rows"] = 10000
    c3["permutation_obs_per_fold"] = 1000
    c3["train_eval_obs"] = 2000
    c3["models"]["random_forest"]["n_estimators"] = 20
    c3["models"]["hist_gradient_boosting"]["max_iter"] = 50
    c3["quick_trips"] = 15
    return cfg


def main():
    cfg = load_config()
    if "--quick" in sys.argv:
        cfg = apply_quick_mode(cfg)
    c3 = cfg["phase3"]
    seed = int(cfg["seed"])
    np.random.seed(seed)
    proc = resolve(cfg["phase2"]["processed_dir"])
    out = resolve(c3["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    art = resolve(c3["artifacts_dir"]); art.mkdir(parents=True, exist_ok=True)
    thr = float(c3["threshold"])
    if "--report-only" in sys.argv:
        from pathsnap.phase3_report import rebuild_from_files
        rebuild_from_files(out, cfg, art)
        print(f"Report rebuilt from saved results: {out / 'phase3_report.md'}")
        return
    t_start = time.time()
    plots = []

    # ------------------------------------------------------------------ 1-2 load, verify, filter
    log("Loading Phase 2 training candidates ...")
    cand = load_partition(proc / "candidates_train.parquet")
    feats = verify_schema(cand, "train")
    if c3.get("quick_trips"):
        keep = sorted(cand.trip_id.unique())[: c3["quick_trips"]]
        cand = cand[cand.trip_id.isin(keep)].reset_index(drop=True)
        log(f"QUICK MODE: {len(keep)} trips, 3 folds, small models -> {c3['output_dir']} (not real results)")
    cand = add_eval_flags(cand)
    fit_rows, filt = M.filter_training_rows(cand, tuple(c3["train_labels"]))
    filt["obs_aligned_rankable"] = int(cand.loc[cand.in_rank, M.OBS_KEY].drop_duplicates().shape[0])
    filt["obs_ambiguous_rankable"] = int(cand.loc[cand.in_amb, M.OBS_KEY].drop_duplicates().shape[0])
    filt["obs_aligned_without_positive"] = int(cand.loc[(cand.obs_status == "aligned") & ~cand.in_rank, M.OBS_KEY].drop_duplicates().shape[0])
    filt["trips"] = int(cand.trip_id.nunique())
    (out / "label_filtering.json").write_text(json.dumps(filt, indent=2), encoding="utf-8")
    log(f"  {filt['trips']} trips, {filt['rows_total']:,} rows -> {filt['rows_kept']:,} fit rows "
        f"({filt['rows_kept_positive']:,} positive, {filt['rows_kept_negative']:,} negative); "
        f"excluded {filt['rows_excluded_ambiguous']:,} ambiguous + {filt['rows_excluded_unalignable_observation']:,} unalignable")
    log(f"  {len(feats)} features; identifiers and route-derived columns excluded (checked)")

    # ------------------------------------------------------------------ 3 folds
    folds = M.make_trip_folds(fit_rows, int(c3["cv_folds"]), seed)
    for f in folds:
        tr = fit_rows[fit_rows.trip_id.isin(f["train_trips"])]
        va = fit_rows[fit_rows.trip_id.isin(f["val_trips"])]
        f.update(train_rows=int(len(tr)), val_rows=int(len(va)), train_pos_rate=float(tr.label.mean()),
                 val_pos_rate=float(va.label.mean()), n_train_trips=len(f["train_trips"]), n_val_trips=len(f["val_trips"]))
    (out / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    log(f"  {len(folds)} trip-grouped folds; validation trips per fold: {[f['n_val_trips'] for f in folds]}")

    X_all = cand[feats].to_numpy(np.float32)
    oof = cand[["trip_id", "seq", "arc_id", "dist_rank", "label", "obs_status"]].copy()
    for m in BASELINES + MODELS:
        oof[m] = np.nan
    fold_rows, fold_models = [], {m: [] for m in MODELS}
    per_point_all = {m: [] for m in BASELINES + MODELS}

    # ------------------------------------------------------------------ 4 cross-validation
    for f in folds:
        k = f["fold"]
        M.assert_disjoint(f["train_trips"], f["val_trips"])
        tr_mask = cand.in_fit.to_numpy() & cand.trip_id.isin(f["train_trips"]).to_numpy()
        va_mask = cand.trip_id.isin(f["val_trips"]).to_numpy() & (cand.in_fit | cand.in_rank | cand.in_amb).to_numpy()
        assert not (set(cand.trip_id[tr_mask]) & set(cand.trip_id[va_mask])), "trip leakage"
        va_df = cand[va_mask]
        n_va_obs = va_df[M.OBS_KEY].drop_duplicates().shape[0]
        log(f"Fold {k + 1}/{len(folds)}: {len(f['train_trips'])} train trips ({tr_mask.sum():,} rows), "
            f"{len(f['val_trips'])} validation trips ({va_mask.sum():,} rows)")
        for name, s in baseline_scores(va_df, c3).items():
            met, per = evaluate_scores(va_df, s, False, thr)
            oof.loc[va_mask, name] = s
            per_point_all[name].append(per)
            fold_rows.append({"fold": k, "model": name, **met, "fit_s": 0.0, "predict_s": 0.0, "train_rows": 0})
        y_tr = cand.label.to_numpy()[tr_mask]
        for name in MODELS:
            model = M.build_model(name, c3["models"][name], seed)
            if name == "knn":
                sub = M.sample_whole_observations(cand[tr_mask], int(c3["knn_train_rows"]), seed + k)
                Xf, yf = sub[feats].to_numpy(np.float32), sub.label.to_numpy()
            else:
                Xf, yf = X_all[tr_mask], y_tr
            t0 = time.perf_counter(); model.fit(Xf, yf); fit_s = time.perf_counter() - t0
            t0 = time.perf_counter(); p = model.predict_proba(X_all[va_mask])[:, 1]; pred_s = time.perf_counter() - t0
            met, per = evaluate_scores(va_df, p, True, thr)
            oof.loc[va_mask, name] = p
            per_point_all[name].append(per)
            # training-fold Top-1 on a sample of training GPS points (overfitting evidence)
            trk = cand[cand.in_rank & cand.trip_id.isin(f["train_trips"])]
            keys = trk[M.OBS_KEY].drop_duplicates()
            keys = keys.sample(min(len(keys), int(c3["train_eval_obs"])), random_state=seed + k)
            trk = trk.merge(keys, on=M.OBS_KEY)
            met["train_top1"] = Q.ranking_metrics(trk, model.predict_proba(trk[feats].to_numpy(np.float32))[:, 1])[0]["top1"]
            fold_rows.append({"fold": k, "model": name, **met, "fit_s": fit_s, "predict_s": pred_s, "train_rows": int(len(yf)),
                              "predict_s_per_1k_cand": 1000 * pred_s / va_mask.sum(), "predict_s_per_1k_obs": 1000 * pred_s / n_va_obs})
            fold_models[name].append((model, f))
            log(f"    {name:<24} top1={met['top1']:.4f} mrr={met['mrr']:.4f} f1={met['f1']:.4f} pr_auc={met['pr_auc']:.4f} "
                f"fit {fit_s:.1f}s predict {pred_s:.1f}s")

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(out / "cv_fold_metrics.csv", index=False)
    oof.to_parquet(proc / "phase3_oof_scores.parquet", index=False)

    # ------------------------------------------------------------------ 5 aggregate + select
    metric_cols = ["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc", "log_loss",
                   "top1", "top3", "top5", "mrr", "amb_lenient_top1", "train_top1", "fit_s", "predict_s", "predict_s_per_1k_cand", "predict_s_per_1k_obs"]
    agg = []
    for m, g in folds_df.groupby("model", sort=False):
        row = {"model": m, "family": M.MODEL_FAMILIES.get(m, "Baseline (no learning)")}
        for c in metric_cols:
            v = g[c].dropna() if c in g else pd.Series(dtype=float)
            if len(v):
                row[f"{c}_mean"], row[f"{c}_std"], row[f"{c}_median"] = v.mean(), v.std(ddof=1), v.median()
            else:                                  # e.g. baselines have no training or log loss
                row[f"{c}_mean"] = row[f"{c}_std"] = row[f"{c}_median"] = np.nan
        agg.append(row)
    summary = pd.DataFrame(agg)

    per_trip = {}
    for m, lst in per_point_all.items():
        per_trip[m] = Q.per_trip_top1(pd.concat(lst, ignore_index=True))
    per_trip = pd.DataFrame(per_trip)
    per_trip.to_csv(out / "per_trip_top1.csv")
    summary["trip_macro_top1"] = summary.model.map(per_trip.mean())
    summary["trip_top1_std"] = summary.model.map(per_trip.std())
    summary["trip_top1_min"] = summary.model.map(per_trip.min())
    summary.to_csv(out / "cv_summary.csv", index=False)

    sel = c3["selection"]
    cand_models = summary[summary.model.isin(MODELS)].copy()
    best = cand_models[f"{sel['primary']}_mean"].max()
    tied = cand_models[cand_models[f"{sel['primary']}_mean"] >= best - float(sel["tie_margin"])].copy()
    tied = tied.sort_values([f"{sel['tie_breakers'][0]}_mean", f"{sel['tie_breakers'][1]}_mean"], ascending=[False, True])
    selected = tied.model.iat[0]
    base_top1 = summary.set_index("model").loc["baseline_nearest", "top1_mean"]
    sel_row = summary.set_index("model").loc[selected]
    selection = {
        "rule": f"highest mean CV {sel['primary']}; models within {sel['tie_margin']} of the best are tied; ties broken by "
                f"higher {sel['tie_breakers'][0]}, then lower {sel['tie_breakers'][1]}",
        "best_top1": float(best), "tied_models": tied.model.tolist(), "selected": selected,
        "selected_top1_mean": float(sel_row.top1_mean), "selected_top1_std": float(sel_row.top1_std),
        "nearest_baseline_top1_mean": float(base_top1),
        "beats_nearest_baseline": bool(sel_row.top1_mean > base_top1),
        "beats_baseline_in_every_fold": bool((folds_df[folds_df.model == selected].set_index("fold").top1 >
                                              folds_df[folds_df.model == "baseline_nearest"].set_index("fold").top1).all()),
        "decided_at": time.strftime("%Y-%m-%d %H:%M:%S"), "student_test_used": False,
    }
    (out / "model_selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    log(f"Selected: {selected} (CV Top-1 {sel_row.top1_mean:.4f} ± {sel_row.top1_std:.4f}; nearest baseline {base_top1:.4f}). Saved before any student-test use.")

    # ------------------------------------------------------------------ 6 refit on all training trips
    fit_all = cand.in_fit.to_numpy()
    final = {}
    to_refit = MODELS if c3["evaluate_student_test"] else [selected]
    sizes, refit_s = {}, {}
    for name in to_refit:
        log(f"Refitting {name} on all {cand.trip_id.nunique()} training trips ...")
        model = M.build_model(name, c3["models"][name], seed)
        if name == "knn":
            sub = M.sample_whole_observations(cand[fit_all], int(c3["knn_train_rows"]), seed)
            Xf, yf = sub[feats].to_numpy(np.float32), sub.label.to_numpy()
        else:
            Xf, yf = X_all[fit_all], cand.label.to_numpy()[fit_all]
        t0 = time.perf_counter(); model.fit(Xf, yf); refit_s[name] = time.perf_counter() - t0
        final[name] = model
        sizes[name] = model_bytes(model)
    summary["model_size_mb"] = summary.model.map({k: v / 1e6 for k, v in sizes.items()})
    summary.to_csv(out / "cv_summary.csv", index=False)
    joblib.dump(final[selected], art / "selected_model.joblib")
    meta = {"model": selected, "family": M.MODEL_FAMILIES[selected], "params": c3["models"][selected],
            "feature_columns": feats, "train_labels": c3["train_labels"], "trained_on": f"all {cand.trip_id.nunique()} train trips (labels 0/1)",
            "seed": seed, "sklearn": sklearn.__version__, "python": platform.python_version(),
            "cv_top1_mean": float(sel_row.top1_mean), "size_mb": sizes[selected] / 1e6, "refit_s": refit_s[selected],
            "predict": "model.predict_proba(X[feature_columns])[:, 1] = probability that the candidate is the correct road"}
    (art / "selected_model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ 7 independent student-test check
    student = None
    if c3["evaluate_student_test"]:
        log("Independent student-test evaluation (selection already saved; not used to choose) ...")
        sc = load_partition(proc / "candidates_student_test.parquet")
        verify_schema(sc, "student_test")
        assert not (set(sc.trip_id) & set(cand.trip_id)), "student-test trips overlap training trips"
        sc = add_eval_flags(sc)
        sdf = sc[sc.in_fit | sc.in_rank | sc.in_amb]
        Xs = sdf[feats].to_numpy(np.float32)
        rows = []
        for name, s in baseline_scores(sdf, c3).items():
            met, _ = evaluate_scores(sdf, s, False, thr)
            rows.append({"model": name, **met})
        for name in MODELS:
            p = final[name].predict_proba(Xs)[:, 1]
            met, _ = evaluate_scores(sdf, p, True, thr)
            rows.append({"model": name, **met})
        student = pd.DataFrame(rows)
        student.to_csv(out / "student_test_metrics.csv", index=False)

    # ------------------------------------------------------------------ 8 feature analysis
    log("Feature analysis ...")
    from pathsnap.phase3_analysis import permutation_importance_top1, lr_coefficients, impurity_importance, redundancy
    perm = permutation_importance_top1(fold_models[selected], cand, feats, int(c3["permutation_obs_per_fold"]), seed)
    perm.to_csv(out / "permutation_importance_selected.csv", index=False)
    coef = lr_coefficients(final["logistic_regression"], feats) if "logistic_regression" in final else None
    if coef is not None:
        coef.to_csv(out / "logistic_regression_coefficients.csv", index=False)
    imp = impurity_importance(final, feats)
    if imp is not None:
        imp.to_csv(out / "tree_impurity_importance.csv", index=False)
    red = redundancy(cand[cand.in_fit], feats, seed)
    red.to_csv(out / "feature_redundancy.csv", index=False)
    schema = {"feature_columns": feats, "target": M.TARGET, "group": M.GROUP, "observation_key": M.OBS_KEY,
              "metadata_columns": [c for c in cand.columns if c not in feats and c not in ("in_fit", "in_rank", "in_amb")],
              "forbidden_as_features": sorted(M.FORBIDDEN),
              "label_rules": {"fit": "label in {0, 1}", "excluded": {"-1": "ambiguous", "-2": "unalignable observation"},
                              "ranking_primary": "aligned GPS points with exactly one positive (all their candidates compete)",
                              "ranking_secondary": "ambiguous GPS points, lenient: top candidate labelled 1 or -1"}}
    (out / "phase3_feature_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ plots
    log("Plots ...")
    P.comparison_plots(summary, selected, out, plots)
    P.curves_plot(oof, BASELINES + MODELS, out, plots)
    P.fold_plot(folds_df, out, plots)
    P.trip_plot(per_trip, out, plots)
    sel_oof = oof[oof.label.isin([0, 1])]
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(sel_oof.label, (sel_oof[selected] >= thr).astype(int), labels=[0, 1]).ravel()
    P.confusion_plot({"tn": tn, "fp": fp, "fn": fn, "tp": tp}, selected, thr, out, plots)
    P.importance_plot(perm, selected, coef, out, plots)
    P.runtime_plot(summary, out, plots)
    if student is not None:
        P.student_plot(student, selected, out, plots)

    # ------------------------------------------------------------------ config + report
    elapsed = time.time() - t_start
    exp = {"seed": seed, "phase3_config": c3, "features": feats, "python": platform.python_version(),
           "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__,
           "elapsed_s": round(elapsed, 1), "refit_s": refit_s, "model_size_bytes": sizes,
           "confusion_selected_oof": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}
    (out / "experiment_config.json").write_text(json.dumps(exp, indent=2, default=str), encoding="utf-8")
    from pathsnap.phase3_report import write_report
    write_report(out, cfg, filt, folds, folds_df, summary, per_trip, selection, student, perm, coef, imp, red, exp, plots, meta)

    # ------------------------------------------------------------------ console
    print(f"\n=== PHASE 3 SUMMARY ({len(folds)}-fold trip-grouped CV on {cand.trip_id.nunique()} train trips; aligned GPS points) ===")
    show = summary.set_index("model").reindex(BASELINES + MODELS)
    print(f"  {'model':<24} {'Top-1':>15} {'Top-3':>7} {'MRR':>7} {'F1':>7} {'PR-AUC':>7} {'fit s':>7}")
    for m, r in show.iterrows():
        print(f"  {m:<24} {r.top1_mean:.4f}±{r.top1_std:.4f} {r.top3_mean:7.4f} {r.mrr_mean:7.4f} {r.f1_mean:7.4f} {r.pr_auc_mean:7.4f} {r.fit_s_mean:7.1f}")
    print(f"  selected: {selected}  (saved to {art / 'selected_model.joblib'}, {sizes[selected] / 1e6:.1f} MB)")
    if student is not None:
        s = student.set_index("model")
        print(f"  student-test (independent): {selected} Top-1 {s.loc[selected, 'top1']:.4f} vs nearest {s.loc['baseline_nearest', 'top1']:.4f}")
    print(f"  total time {elapsed / 60:.1f} min. Open {out / 'phase3_report.md'}")


if __name__ == "__main__":
    main()

"""Development-data experiment: adaptive (displacement-based) heading window for the FEATURES.

Motivation: for slow movers (walking pace, stop-and-go) a 1 Hz step is smaller than the GPS noise, so the
fixed +/-3 s heading with a 5 m validity threshold is mostly noise yet marked "reliable". The adaptive mode
grows the centred window until the displacement is >= 20 m (max +/-30 s). One principled setting, not tuned.

Uses ONLY the 80 training trips (5 Phase 3 trip-grouped folds) for the decision, and reports the 10 student-test
trips. The evaluator partition is NOT touched here.

Pre-registered adoption rule (written to decision.json before anything is evaluated):
  adopt iff (1) mean CV Top-1 over the 5 Phase 3 folds improves, AND
            (2) pooled out-of-fold HMM point accuracy on the training trips improves,
  with the final model's hyperparameters and the Phase 5 decoder settings held fixed.

Outputs: results/adaptive_heading/{decision.json, folds.csv, train_hmm_per_trip.csv, student_test.csv}
         artifacts/adaptive_heading/{model.joblib, final_model.json} (refit on all 80 training trips)
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap import hmm as H  # noqa: E402
from pathsnap import inference as I  # noqa: E402
from pathsnap import modeling as M  # noqa: E402
from pathsnap.config import load_config, resolve  # noqa: E402
from pathsnap.dataset import process_trip  # noqa: E402
from pathsnap.evaluation import add_eval_flags, evaluate_scores  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.preprocessing import preprocess_track  # noqa: E402
from pathsnap.road_network import build_road_network  # noqa: E402

SETTING = {"feature_bearing_mode": "adaptive", "feature_bearing_min_disp_m": 20.0, "feature_bearing_max_half_window_s": 30.0}
RULE = ("adopt iff mean CV Top-1 (5 Phase 3 trip-grouped folds) improves AND pooled out-of-fold HMM point accuracy on the "
        "80 training trips improves; model hyperparameters and decoder settings fixed; student-test reported only; "
        "evaluator data not used")
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def build(cfg, part):
    raw = resolve(cfg["data"]["raw_dir"])
    cs = []
    for d in list_trips(raw, part):
        _, cand, _ = process_trip(d, part, cfg, with_labels=True)[:3]
        cs.append(cand)
    return pd.concat(cs, ignore_index=True)


def hmm_train_acc(cfg, ct, score, fixed, prm, ot, raw):
    rows = []
    for d in list_trips(raw, "train"):
        tid = d.name
        trip = load_trip(d, "train")
        _, proj, _ = preprocess_track(trip.track, cfg["phase2"]["preprocessing"])
        net = build_road_network(trip.nodes, trip.arcs, proj)
        paths = H.NetworkPaths(net)
        o = ot[ot.trip_id == tid].sort_values("seq").reset_index(drop=True)
        m = (ct.trip_id == tid).to_numpy()
        c = ct[m]
        st = H.build_states(o, c, score[m], "model", fixed)
        arcs, _ = H.viterbi(st, H.transition_inputs(st, paths, fixed), prm)
        al = o.obs_status.to_numpy() == "aligned"
        rows.append({"trip_id": tid, "aligned": int(al.sum()), "hits": int((arcs[al] == o.route_arc.to_numpy()[al]).sum()),
                     "classifier_hits": int((H.classifier_only(st)[al] == o.route_arc.to_numpy()[al]).sum())})
    return pd.DataFrame(rows)


def main():
    cfg = load_config()
    out = resolve("results/adaptive_heading"); out.mkdir(parents=True, exist_ok=True)
    art = resolve("artifacts/adaptive_heading"); art.mkdir(parents=True, exist_ok=True)
    (out / "decision.json").write_text(json.dumps({"rule": RULE, "setting": SETTING, "status": "running"}, indent=2))
    cfg_a = copy.deepcopy(cfg)
    cfg_a["phase2"]["preprocessing"].update(SETTING)
    proc = resolve(cfg["phase2"]["processed_dir"])
    raw = resolve(cfg["data"]["raw_dir"])
    _, meta = I.load_final_model(cfg, which="phase4")
    feats, params = meta["feature_columns"], meta["params"]
    seed = int(cfg["seed"])

    log("Rebuilding training candidates/features with the adaptive feature heading ...")
    ct_old = pd.read_parquet(proc / "candidates_train.parquet")
    ct = build(cfg_a, "train")
    key = ["trip_id", "seq", "arc_id"]
    assert (ct[key].to_numpy() == ct_old[key].to_numpy()).all(), "candidate set changed"
    assert (ct.label.to_numpy() == ct_old.label.to_numpy()).all(), "labels changed (must not)"
    changed = ~np.isclose(ct.bearing_diff_deg.to_numpy(float), ct_old.bearing_diff_deg.to_numpy(float), equal_nan=True)
    log(f"  same candidates and labels; bearing_diff_deg changed on {changed.mean():.2%} of rows; "
        f"heading valid {ct_old.bearing_valid.mean():.2%} -> {ct.bearing_valid.mean():.2%}")

    folds = json.loads((resolve(cfg["phase3"]["output_dir"]) / "folds.json").read_text())
    old_oof = pd.read_parquet(proc / "phase5_oof_all.parquet")
    assert (old_oof[key].to_numpy() == ct_old[key].to_numpy()).all()
    s_old = old_oof.score.to_numpy()
    X = ct[feats].to_numpy(np.float32); y = ct.label.to_numpy()
    s_new = np.full(len(ct), np.nan)
    ev = add_eval_flags(ct_old.copy())
    frows = []
    for f in folds:
        M.assert_disjoint(f["train_trips"], f["val_trips"])
        tr = ct.label.isin([0, 1]).to_numpy() & ct.trip_id.isin(f["train_trips"]).to_numpy()
        va = ct.trip_id.isin(f["val_trips"]).to_numpy()
        mdl = M.build_model(meta["model"], params, seed).fit(X[tr], y[tr])
        s_new[va] = mdl.predict_proba(X[va])[:, 1]
        dv = ev[va].reset_index(drop=True)
        mo, _ = evaluate_scores(dv, s_old[va], True, 0.5)
        mn, _ = evaluate_scores(dv, s_new[va], True, 0.5)
        frows.append({"fold": f["fold"] + 1, **{f"old_{k}": mo[k] for k in ("top1", "macro_f1", "log_loss", "mrr")},
                      **{f"new_{k}": mn[k] for k in ("top1", "macro_f1", "log_loss", "mrr")}})
        log(f"  fold {f['fold'] + 1}: Top-1 {mo['top1']:.4f} -> {mn['top1']:.4f}   log loss {mo['log_loss']:.4f} -> {mn['log_loss']:.4f}")
    fd = pd.DataFrame(frows); fd.to_csv(out / "folds.csv", index=False)

    log("Out-of-fold HMM on the training trips (decoder settings fixed) ...")
    fixed = {**H.DEFAULTS, **cfg["phase5"]["fixed"]}
    prm = I.load_hmm_params(cfg, which="phase4")
    ot = pd.read_parquet(proc / "observations_train.parquet", columns=["trip_id", "seq", "x", "y", "t", "obs_status", "route_arc"])
    base_cols = ["trip_id", "seq", "arc_id", "proj_frac", "dist_m", "dist_rank"]
    h_old = hmm_train_acc(cfg, ct_old[base_cols], s_old, fixed, prm, ot, raw)
    h_new = hmm_train_acc(cfg, ct[base_cols], s_new, fixed, prm, ot, raw)
    ht = h_old.merge(h_new, on=["trip_id", "aligned"], suffixes=("_old", "_new"))
    ht.to_csv(out / "train_hmm_per_trip.csv", index=False)
    acc = lambda col: float(ht[col].sum() / ht.aligned.sum())
    res = {"cv_top1_old": float(fd.old_top1.mean()), "cv_top1_new": float(fd.new_top1.mean()),
           "cv_folds_improved": int((fd.new_top1 > fd.old_top1).sum()),
           "cv_macro_f1_old": float(fd.old_macro_f1.mean()), "cv_macro_f1_new": float(fd.new_macro_f1.mean()),
           "cv_log_loss_old": float(fd.old_log_loss.mean()), "cv_log_loss_new": float(fd.new_log_loss.mean()),
           "train_hmm_point_acc_old": acc("hits_old"), "train_hmm_point_acc_new": acc("hits_new"),
           "train_classifier_point_acc_old": acc("classifier_hits_old"), "train_classifier_point_acc_new": acc("classifier_hits_new"),
           "train_trips_hmm_better": int((ht.hits_new > ht.hits_old).sum()), "train_trips_hmm_worse": int((ht.hits_new < ht.hits_old).sum())}
    adopt = res["cv_top1_new"] > res["cv_top1_old"] and res["train_hmm_point_acc_new"] > res["train_hmm_point_acc_old"]
    log(f"  CV Top-1 {res['cv_top1_old']:.4f} -> {res['cv_top1_new']:.4f}; train OOF HMM {res['train_hmm_point_acc_old']:.4f} -> "
        f"{res['train_hmm_point_acc_new']:.4f}  => adopt={adopt}")

    log("Refit on all 80 training trips; student-test check (reported only) ...")
    fit = ct.label.isin([0, 1]).to_numpy()
    final = M.build_model(meta["model"], params, seed).fit(X[fit], y[fit])
    joblib.dump(final, art / "model.joblib")
    (art / "final_model.json").write_text(json.dumps({**meta, "path": "artifacts/adaptive_heading/model.joblib",
                                                      "source": "phase4 tuned params, adaptive feature heading",
                                                      "preprocessing_overrides": SETTING}, indent=2))
    old_model, _ = I.load_final_model(cfg, which="phase4")
    st_rows = []
    for d in list_trips(raw, "student_test"):
        route = I.read_reference_route(d)
        for name, mdl, c in (("old", old_model, cfg), ("new", final, cfg_a)):
            r = I.match_trip(d, "student_test", c, mdl, feats, prm, methods=("classifier_only", "model_hmm"))
            e = I.evaluate_match(r, c, route)
            st_rows.append({"trip_id": d.name, "version": name, "aligned": e["model_hmm"]["aligned_points"],
                            "classifier_point_acc": e["classifier_only"]["point_acc"], "hmm_point_acc": e["model_hmm"]["point_acc"],
                            "hmm_arc_f1": e["model_hmm"]["arc_f1"]})
    st = pd.DataFrame(st_rows); st.to_csv(out / "student_test.csv", index=False)
    for v in ("old", "new"):
        s = st[st.version == v]
        res[f"student_hmm_point_acc_{v}"] = float((s.hmm_point_acc * s.aligned).sum() / s.aligned.sum())
        res[f"student_classifier_point_acc_{v}"] = float((s.classifier_point_acc * s.aligned).sum() / s.aligned.sum())
        res[f"student_arc_f1_mean_{v}"] = float(s.hmm_arc_f1.mean())
    (out / "decision.json").write_text(json.dumps({"rule": RULE, "setting": SETTING, "adopted": adopt, "results": res,
                                                   "minutes": (time.time() - T0) / 60}, indent=2))
    log(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

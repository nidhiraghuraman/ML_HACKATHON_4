"""Stage 2 of the adaptive-heading experiment (development data only).

The Phase 5 decoder (beta, w) was tuned on the OLD model's out-of-fold probabilities. Here the identical Phase 5
procedure is re-applied to the adaptive-heading model: out-of-fold probabilities for every training candidate
(Phase 3 folds, fixed tuned hyperparameters), then the Phase 5 grid (config phase5.grid.model_hmm) selected by pooled
out-of-fold point accuracy on the 80 training trips.

Pre-registered rule: the new pipeline (adaptive-heading model + its re-tuned decoder) replaces the current one only if
its pooled training out-of-fold HMM point accuracy exceeds the current pipeline's (results/phase5 summary, train_oof,
model_hmm). The student-test result is reported only. The evaluator partition is not used.

Outputs: results/adaptive_heading/{decoder_grid.csv, stage2_decision.json, student_test_stage2.csv},
         results/adaptive_heading/hmm_params.json, data/processed/adaptive_oof_all.parquet (cache)
"""
from __future__ import annotations

import copy
import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from experiment_adaptive_heading import SETTING, build  # noqa: E402
from pathsnap import hmm as H  # noqa: E402
from pathsnap import inference as I  # noqa: E402
from pathsnap import modeling as M  # noqa: E402
from pathsnap.config import load_config, resolve  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.preprocessing import preprocess_track  # noqa: E402
from pathsnap.road_network import build_road_network  # noqa: E402

T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def main():
    cfg = load_config()
    cfg_a = copy.deepcopy(cfg); cfg_a["phase2"]["preprocessing"].update(SETTING)
    out = resolve("results/adaptive_heading"); proc = resolve(cfg["phase2"]["processed_dir"]); raw = resolve(cfg["data"]["raw_dir"])
    _, meta = I.load_final_model(cfg, which="phase4")
    feats, params, seed = meta["feature_columns"], meta["params"], int(cfg["seed"])
    rule = ("replace the current pipeline only if pooled training out-of-fold HMM point accuracy of (adaptive model + "
            "decoder re-tuned with the Phase 5 grid on training OOF) exceeds the current pipeline's; student-test reported only")
    (out / "stage2_decision.json").write_text(json.dumps({"rule": rule, "status": "running"}, indent=2))
    cache = proc / "adaptive_oof_all.parquet"
    if cache.exists():
        ct = pd.read_parquet(cache); log("cached adaptive OOF")
    else:
        log("Rebuilding training features (adaptive heading) and out-of-fold probabilities ...")
        ct = build(cfg_a, "train")
        folds = json.loads((resolve(cfg["phase3"]["output_dir"]) / "folds.json").read_text())
        X = ct[feats].to_numpy(np.float32); y = ct.label.to_numpy(); s = np.full(len(ct), np.nan)
        for f in folds:
            M.assert_disjoint(f["train_trips"], f["val_trips"])
            tr = ct.label.isin([0, 1]).to_numpy() & ct.trip_id.isin(f["train_trips"]).to_numpy()
            va = ct.trip_id.isin(f["val_trips"]).to_numpy()
            s[va] = M.build_model(meta["model"], params, seed).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
            log(f"  fold {f['fold'] + 1} scored")
        ct = ct[["trip_id", "seq", "arc_id", "proj_frac", "dist_m", "dist_rank", "label"]].assign(score=s)
        ct.to_parquet(cache, index=False)
    fixed = {**H.DEFAULTS, **cfg["phase5"]["fixed"]}
    ot = pd.read_parquet(proc / "observations_train.parquet", columns=["trip_id", "seq", "x", "y", "t", "obs_status", "route_arc"])
    log("Building HMM inputs for the 80 training trips ...")
    T = {}
    for d in list_trips(raw, "train"):
        trip = load_trip(d, "train")
        _, proj, _ = preprocess_track(trip.track, cfg["phase2"]["preprocessing"])
        paths = H.NetworkPaths(build_road_network(trip.nodes, trip.arcs, proj))
        o = ot[ot.trip_id == d.name].sort_values("seq").reset_index(drop=True)
        c = ct[ct.trip_id == d.name]
        st = H.build_states(o, c, c.score.to_numpy(), "model", fixed)
        T[d.name] = (st, H.transition_inputs(st, paths, fixed), o)
    g = cfg["phase5"]["grid"]["model_hmm"]
    rows = []
    for beta, w in itertools.product(g["beta_m"], g["w_emission"]):
        prm = {**fixed, "beta_m": beta, "w_emission": w}
        hits = tot = 0
        for st, steps, o in T.values():
            arcs, _ = H.viterbi(st, steps, prm)
            al = o.obs_status.to_numpy() == "aligned"
            hits += int((arcs[al] == o.route_arc.to_numpy()[al]).sum()); tot += int(al.sum())
        rows.append({"beta_m": beta, "w_emission": w, "point_acc": hits / tot})
        log(f"  beta={beta:<4} w={w:<4} train OOF point_acc={hits / tot:.4f}")
    grid = pd.DataFrame(rows); grid.to_csv(out / "decoder_grid.csv", index=False)
    best = grid.sort_values("point_acc", ascending=False, kind="mergesort").iloc[0]
    prm = {**fixed, "beta_m": float(best.beta_m), "w_emission": float(best.w_emission)}
    cur = pd.read_csv(resolve("results/phase5/summary_metrics.csv"))
    cur_acc = float(cur[(cur.partition == "train_oof") & (cur.method == "model_hmm")].point_acc.iloc[0])
    adopt = float(best.point_acc) > cur_acc
    dist = json.loads(resolve("results/phase5/hmm_params.json").read_text())["distance_hmm"]
    (out / "hmm_params.json").write_text(json.dumps({"model_hmm": prm, "distance_hmm": dist,
                                                     "selected_with": "pooled point accuracy, training trips, out-of-fold probabilities "
                                                                      "of the adaptive-heading model"}, indent=2))
    log(f"best beta={prm['beta_m']} w={prm['w_emission']} -> {best.point_acc:.4f} vs current pipeline {cur_acc:.4f}; adopt={adopt}")
    model = joblib.load(resolve("artifacts/adaptive_heading/model.joblib"))
    old_model, _ = I.load_final_model(cfg, which="phase4"); old_prm = I.load_hmm_params(cfg, which="phase4")
    st_rows = []
    for d in list_trips(raw, "student_test"):
        route = I.read_reference_route(d)
        for name, mdl, c, p in (("current", old_model, cfg, old_prm), ("adaptive_retuned", model, cfg_a, prm)):
            r = I.match_trip(d, "student_test", c, mdl, feats, p, methods=("classifier_only", "model_hmm"))
            e = I.evaluate_match(r, c, route)
            st_rows.append({"trip_id": d.name, "version": name, "aligned": e["model_hmm"]["aligned_points"],
                            "classifier_point_acc": e["classifier_only"]["point_acc"], "hmm_point_acc": e["model_hmm"]["point_acc"],
                            "hmm_arc_f1": e["model_hmm"]["arc_f1"], "continuity": e["model_hmm"]["continuity"]})
    st = pd.DataFrame(st_rows); st.to_csv(out / "student_test_stage2.csv", index=False)
    res = {"best_beta_m": prm["beta_m"], "best_w_emission": prm["w_emission"], "train_oof_point_acc_new": float(best.point_acc),
           "train_oof_point_acc_current": cur_acc}
    for v in ("current", "adaptive_retuned"):
        s = st[st.version == v]
        res[f"student_hmm_point_acc_{v}"] = float((s.hmm_point_acc * s.aligned).sum() / s.aligned.sum())
        res[f"student_arc_f1_mean_{v}"] = float(s.hmm_arc_f1.mean())
    (out / "stage2_decision.json").write_text(json.dumps({"rule": rule, "adopted": adopt, "results": res,
                                                          "minutes": (time.time() - T0) / 60}, indent=2))
    log(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

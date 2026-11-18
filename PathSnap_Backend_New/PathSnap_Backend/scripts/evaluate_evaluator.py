"""Evaluate the trained PathSnap pipeline on the external evaluator dataset (evaluation only).

Nothing is fitted or tuned here: the final model (artifacts/phase4) and decoder settings (results/phase5/hmm_params.json),
both chosen on the training trips, are applied unchanged. The reference .route of each case is read only AFTER the
prediction for that case is complete, to compute metrics.

Usage:  python scripts/evaluate_evaluator.py --which phase4 --tag baseline      (Phase 4 model + Phase 5 decoder)
        python scripts/evaluate_evaluator.py --which final  --tag final         (config final_pipeline)
Outputs: results/evaluator/<tag>/ per_case.csv, aggregate.json, errors.csv, ranking.csv, points.csv.gz, run.json

Metric definitions (point = GPS observation):
  aligned points   points the Phase 2 route alignment assigns to one reference arc (within 40 m; 20 m if stationary)
  point_acc        exact arc-level accuracy of the final decision on aligned points (= observation-level Top-1 of the
                   final output; the guidelines' "Observation-level Top-1 Accuracy")
  amb_lenient_acc  ambiguous points (junction nodes): correct if the aligned arc or a labelled alternative
  top-k, MRR       rank of the aligned arc among the point's candidates by classifier probability (aligned points)
  cand_coverage    share of aligned points whose aligned arc is among the generated candidates
  continuity       share of consecutive point pairs (gap <= 60 s) whose chosen arcs are equal/connected
  arc P/R/F1       reconstructed route vs reference route as arc sets; len_* weighted by arc length
  route_mismatch   (length wrongly added + length missed) / reference length (Newson & Krumm)
  exact_route      reconstructed arc sequence identical to the reference sequence
  decision P/R/F1  candidate rows of aligned points: positive = aligned arc, predicted = chosen arc
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pathsnap import evaluator_data as ED  # noqa: E402
from pathsnap import hmm as H  # noqa: E402
from pathsnap import inference as I  # noqa: E402
from pathsnap.config import load_config, resolve  # noqa: E402
from pathsnap.labels import align_to_route, ambiguous_alternatives  # noqa: E402

METHODS = ("nearest", "classifier_only", "model_hmm")


def classify_error(pred: int, ref: int, cand_arcs: set, beam_arcs: set, arcs: np.ndarray) -> str:
    if pred < 0:
        return "unmatched (no arc output)"
    if ref not in cand_arcs:
        return "candidate miss (reference arc not generated)"
    if ref not in beam_arcs:
        return "beam miss (reference not in HMM top-8)"
    pf, pt = arcs[pred]; rf, rt = arcs[ref]
    if pf == rt and pt == rf:
        return "direction (reverse twin)"
    if len({pf, pt} & {rf, rt}):
        return "adjacent arc (shares a node)"
    return "other road"


def evaluate_case(d: Path, part: str, cfg: dict, model, feats, hp_model: dict, hp_dist: dict) -> dict:
    t0 = time.perf_counter()
    res = I.match_trip(d, part, cfg, model, feats, hp_model, methods=METHODS)          # prediction: no route read
    t_pred = time.perf_counter() - t0
    res_d = I.match_trip(d, part, cfg, model, feats, hp_dist, methods=("distance_hmm",))
    res["methods"]["distance_hmm"] = res_d["methods"]["distance_hmm"]
    route = I.read_reference_route(d)                                                 # evaluation only, after prediction
    ev = I.evaluate_match(res, cfg, route)
    obs, cand, net, trip = res["obs"], res["cand"], res["net"], res["trip"]
    al = align_to_route(obs, route, net, cfg["phase2"]["labels"])
    alts = ambiguous_alternatives(al, obs, route, net, cfg["phase2"]["labels"])
    status = al.obs_status.to_numpy(); ref = al.route_arc.to_numpy()
    seq_of_row = obs.seq.to_numpy()
    # candidate ranking by classifier probability
    cg = {s: g for s, g in cand.groupby("seq", sort=False)}
    st_model = res["methods"]["model_hmm"]["states"]
    hmm_arcs = res["methods"]["model_hmm"]["arcs"]; clf_arcs = res["methods"]["classifier_only"]["arcs"]
    rows, err_rows, dec = [], [], {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for i in range(len(obs)):
        s = int(seq_of_row[i])
        g = cg.get(s)
        cand_arcs = set(g.arc_id.tolist()) if g is not None else set()
        beam = set(st_model.arcs[i].tolist()) if i < len(st_model.arcs) else set()
        rank = np.nan
        if status[i] == "aligned" and g is not None and ref[i] in cand_arcs:
            order = g.sort_values(["prob", "dist_rank"], ascending=[False, True]).arc_id.to_numpy()
            rank = int(np.flatnonzero(order == ref[i])[0]) + 1
        rows.append({"seq": s, "status": status[i], "ref_arc": int(ref[i]), "hmm_arc": int(hmm_arcs[i]), "clf_arc": int(clf_arcs[i]),
                     "nearest_arc": int(res["methods"]["nearest"]["arcs"][i]), "n_cand": len(cand_arcs),
                     "ref_in_cand": ref[i] in cand_arcs, "ref_in_beam": ref[i] in beam, "rank": rank,
                     "align_dist_m": float(al.align_dist_m.iat[i]) if np.isfinite(al.align_dist_m.iat[i]) else np.nan,
                     "stationary": bool(obs.is_stationary.iat[i]), "speed": float(obs.speed_mps.iat[i]) if np.isfinite(obs.speed_mps.iat[i]) else np.nan})
        if status[i] == "aligned":
            for a in cand_arcs:
                y, p = a == ref[i], a == hmm_arcs[i]
                dec["tp" if y and p else "fn" if y else "fp" if p else "tn"] += 1
            for meth, arr in (("model_hmm", hmm_arcs), ("classifier_only", clf_arcs)):
                if arr[i] != ref[i]:
                    err_rows.append({"seq": s, "method": meth, "type": classify_error(int(arr[i]), int(ref[i]), cand_arcs, beam, trip.arcs),
                                     "stationary": bool(obs.is_stationary.iat[i])})
    pts = pd.DataFrame(rows)
    alg = pts[pts.status == "aligned"]
    rk = alg["rank"].to_numpy(float)
    ranking = {"aligned_points": int(len(alg)), "cand_coverage": float(alg.ref_in_cand.mean()) if len(alg) else np.nan,
               "beam_coverage": float(alg.ref_in_beam.mean()) if len(alg) else np.nan,
               "top1": float(np.mean(rk == 1)), "top3": float(np.mean(rk <= 3)), "top5": float(np.mean(rk <= 5)),
               "mrr": float(np.nanmean(np.where(np.isfinite(rk), 1.0 / rk, 0.0)))}
    P = dec["tp"] / max(1, dec["tp"] + dec["fp"]); R = dec["tp"] / max(1, dec["tp"] + dec["fn"])
    pred_route = [int(a) for a in res["methods"]["model_hmm"]["route"]]
    out = {"trip_id": d.name, "n_points": int(len(obs)), "aligned": int((status == "aligned").sum()),
           "ambiguous": int((status == "ambiguous").sum()), "unalignable": int((status == "unalignable").sum()),
           "stationary_share": float(obs.is_stationary.mean()), "ref_route_arcs": int(len(route)), "pred_route_arcs": len(pred_route),
           "exact_route": pred_route == [int(a) for a in route], "decision_precision": P, "decision_recall": R,
           "decision_f1": 2 * P * R / max(1e-12, P + R), "predict_s": t_pred, **{f"rank_{k}": v for k, v in ranking.items()}}
    for m, mv in ev.items():
        for k, v in mv.items():
            out[f"{m}__{k}"] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
    pts.insert(0, "trip_id", d.name)
    errs = pd.DataFrame(err_rows); errs.insert(0, "trip_id", d.name) if len(errs) else None
    return out, pts, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--which", default="final", choices=["final", "phase4"],
                    help="final = config final_pipeline (adaptive heading); phase4 = Phase 4 model + Phase 5 decoder (baseline)")
    ap.add_argument("--hmm-params", default=None, help="JSON with model_hmm/distance_hmm settings (default: Phase 5 file)")
    a = ap.parse_args()
    cfg = load_config()
    part = ED.settings(cfg)["partition"]
    ED.extract(cfg)
    man = ED.validate(cfg)
    if not man["all_ok"]:
        raise SystemExit(f"Evaluator data failed validation: {[t for t in man['trips'] if t['problems']]}")
    model, meta = I.load_final_model(cfg, a.which)
    feats = meta["feature_columns"]
    if a.hmm_params:
        hp = json.loads(Path(a.hmm_params).read_text())
        hp_model = {**H.DEFAULTS, **cfg["phase5"].get("fixed", {}), **hp["model_hmm"]}
        hp_dist = {**H.DEFAULTS, **cfg["phase5"].get("fixed", {}), **hp["distance_hmm"]}
    else:
        hp_model = I.load_hmm_params(cfg, a.which)
        dist = json.loads(resolve("results/phase5/hmm_params.json").read_text())["distance_hmm"]
        hp_dist = {**H.DEFAULTS, **cfg["phase5"].get("fixed", {}), **dist}
    out = resolve(cfg["evaluator"]["output_dir"]) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    cases, allpts, allerr = [], [], []
    for d in ED.trip_dirs(cfg):
        c, p, e = evaluate_case(d, part, cfg, model, feats, hp_model, hp_dist)
        cases.append(c); allpts.append(p); allerr.append(e)
        print(f"{c['trip_id']}: pts {c['n_points']:5d} aligned {c['aligned']:5d} | clf {c['classifier_only__point_acc']:.4f} "
              f"hmm {c['model_hmm__point_acc']:.4f} | F1 {c['model_hmm__arc_f1']:.4f} | cov {c['rank_cand_coverage']:.4f} "
              f"| {c['predict_s']:.1f}s", flush=True)
    pc = pd.DataFrame(cases)
    pts = pd.concat(allpts, ignore_index=True)
    errs = pd.concat([e for e in allerr if len(e)], ignore_index=True)
    pc.to_csv(out / "per_case.csv", index=False)
    pts.to_csv(out / "points.csv.gz", index=False)
    errs.to_csv(out / "errors.csv", index=False)
    al = pts[pts.status == "aligned"]
    agg = {"cases": len(pc), "points": int(len(pts)), "aligned_points": int(len(al)),
           "unalignable_share": float((pts.status == "unalignable").mean()), "ambiguous_share": float((pts.status == "ambiguous").mean()),
           "pooled": {}, "mean_over_cases": {}}
    for m in ("nearest", "classifier_only", "distance_hmm", "model_hmm"):
        col = {"nearest": "nearest_arc", "classifier_only": "clf_arc", "model_hmm": "hmm_arc"}.get(m)
        w = pc["aligned"]
        agg["pooled"][m] = {"point_acc": float((pc[f"{m}__point_acc"] * w).sum() / w.sum())}
        for k in ("point_acc", "amb_lenient_acc", "continuity", "arc_precision", "arc_recall", "arc_f1", "len_f1", "route_mismatch", "route_breaks"):
            if f"{m}__{k}" in pc:
                agg["mean_over_cases"].setdefault(m, {})[k] = float(pc[f"{m}__{k}"].mean())
        if col:
            agg["pooled"][m]["recomputed_from_points"] = float((al[col] == al.ref_arc).mean())
    rk = al["rank"].to_numpy(float)
    agg["ranking_pooled"] = {"cand_coverage": float(al.ref_in_cand.mean()), "beam_coverage": float(al.ref_in_beam.mean()),
                             "top1": float(np.mean(rk == 1)), "top3": float(np.mean(rk <= 3)), "top5": float(np.mean(rk <= 5)),
                             "mrr": float(np.mean(np.where(np.isfinite(rk), 1 / np.where(np.isfinite(rk), rk, 1), 0)))}
    agg["exact_route_cases"] = int(pc.exact_route.sum())
    agg["decision_f1_mean"] = float(pc.decision_f1.mean())
    agg["errors"] = {m: errs[errs.method == m].type.value_counts().to_dict() for m in ("classifier_only", "model_hmm")} if len(errs) else {}
    agg["errors_stationary_share"] = {m: float(errs[errs.method == m].stationary.mean()) for m in ("classifier_only", "model_hmm")} if len(errs) else {}
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    (out / "run.json").write_text(json.dumps({"tag": a.tag, "model": meta.get("path"), "hmm_model": hp_model, "hmm_distance": hp_dist,
                                              "which": a.which,
                                              "preprocessing_overrides": getattr(model, "pathsnap_preprocessing_overrides", {}),
                                              "note": "no fitting or tuning on evaluator data"}, indent=2, default=str))
    print(json.dumps({k: agg[k] for k in ("pooled", "ranking_pooled", "exact_route_cases", "unalignable_share")}, indent=1))


if __name__ == "__main__":
    main()

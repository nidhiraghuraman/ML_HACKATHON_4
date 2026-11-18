"""Write deliverables/EVALUATOR_EVALUATION_REPORT.md. Every table is generated from saved result files:
results/evaluator/{manifest.json, baseline/, final/, figures/}, results/adaptive_heading/*.  Usage: python scripts/build_evaluator_report.py"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
E = R / "evaluator"
man = json.loads((E / "manifest.json").read_text())
B = json.loads((E / "baseline/aggregate.json").read_text()); F = json.loads((E / "final/aggregate.json").read_text())
pb = pd.read_csv(E / "baseline/per_case.csv", dtype={"trip_id": str}).set_index("trip_id")
pf = pd.read_csv(E / "final/per_case.csv", dtype={"trip_id": str}).set_index("trip_id")
folds = pd.read_csv(R / "adaptive_heading/folds.csv")
d1 = json.loads((R / "adaptive_heading/decision.json").read_text()); d2 = json.loads((R / "adaptive_heading/stage2_decision.json").read_text())
mv = pd.read_csv(E / "baseline/movement_stats.csv", dtype={"trip": str})
f4 = lambda v: f"{v:.4f}"
pct = lambda v: f"{100 * v:.2f}%"


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def case_table(p):
    return md(pd.DataFrame({
        "Case": p.index, "GPS pts": p.n_points, "Aligned": p.aligned, "Ambig.": p.ambiguous, "Unalign.": p.unalignable,
        "Nearest": p["nearest__point_acc"].map(f4), "Classifier": p["classifier_only__point_acc"].map(f4),
        "Clf+HMM": p["model_hmm__point_acc"].map(f4), "Ambig. lenient": p["model_hmm__amb_lenient_acc"].map(f4),
        "Top-3": p.rank_top3.map(f4), "MRR": p.rank_mrr.map(f4), "Continuity": p["model_hmm__continuity"].map(f4),
        "Arc P": p["model_hmm__arc_precision"].map(f4), "Arc R": p["model_hmm__arc_recall"].map(f4),
        "Arc F1": p["model_hmm__arc_f1"].map(f4), "Mismatch": p["model_hmm__route_mismatch"].map(f4),
        "Ref / pred arcs": [f"{a} / {b}" for a, b in zip(p.ref_route_arcs, p.pred_route_arcs)],
        "Exact route": p.exact_route.map({True: "yes", False: "no"})}))


def agg_rows(A, p):
    rows = []
    for m, lab in (("nearest", "Nearest road"), ("classifier_only", "Classifier only"), ("distance_hmm", "Distance-only HMM"), ("model_hmm", "Classifier + HMM (deployed)")):
        mo = A["mean_over_cases"][m]
        rows.append({"Method": lab, "Point acc. (pooled)": pct(A["pooled"][m]["point_acc"]), "Point acc. (mean of cases)": pct(mo["point_acc"]),
                     "Cases ≥ 95%": f"{int((p[f'{m}__point_acc'] >= 0.95).sum())}/10", "Ambig. lenient": f4(mo["amb_lenient_acc"]),
                     "Continuity": f4(mo["continuity"]), "Arc P": f4(mo["arc_precision"]), "Arc R": f4(mo["arc_recall"]),
                     "Arc F1": f4(mo["arc_f1"]), "Length F1": f4(mo["len_f1"]), "Route mismatch": f4(mo["route_mismatch"]),
                     "Route breaks (mean)": f"{mo['route_breaks']:.1f}"})
    return md(pd.DataFrame(rows))


def err_table():
    cats = ["direction (reverse twin)", "adjacent arc (shares a node)", "other road", "beam miss (reference not in HMM top-8)",
            "candidate miss (reference arc not generated)", "unmatched (no arc output)"]
    rows = []
    for run, A in (("baseline", B), ("final", F)):
        for m in ("classifier_only", "model_hmm"):
            ec = A["errors"].get(m, {}); tot = sum(ec.values())
            rows.append({"Run": run, "Decision": m.replace("_", " "), "Wrong points": tot,
                         **{c: f"{ec.get(c, 0)} ({100 * ec.get(c, 0) / max(tot, 1):.0f}%)" for c in cats},
                         "Stationary share": pct(A["errors_stationary_share"][m])})
    return md(pd.DataFrame(rows))


rk = lambda A: A["ranking_pooled"]
share15 = lambda p: 100 * (p.aligned * (1 - p.model_hmm__point_acc)).loc['00000015'] / (p.aligned * (1 - p.model_hmm__point_acc)).sum()
movement = mv[mv.part != "train"].copy()
movement["acc"] = movement["hmm_point_acc(dev)"].map(lambda v: "—" if pd.isna(v) else f4(v))
mov_tab = md(pd.DataFrame({"Partition": movement.part, "Trip": movement.trip, "Speed, 30 s net displacement (m/s)": movement.disp_speed_30s,
                           "Speed, 1 s steps (m/s)": movement.step_speed, "Share of 30 s windows < 2 m/s": movement["share_disp<2mps"],
                           "Dev. HMM point acc.": movement.acc}))
slow_train = mv[mv.part == "train"].sort_values("disp_speed_30s").head(5)
fold_tab = md(pd.DataFrame({"Fold": folds.fold, "Top-1 current": folds.old_top1.map(f4), "Top-1 adaptive": folds.new_top1.map(f4),
                            "Macro F1 current": folds.old_macro_f1.map(f4), "Macro F1 adaptive": folds.new_macro_f1.map(f4),
                            "Log loss current": folds.old_log_loss.map(f4), "Log loss adaptive": folds.new_log_loss.map(f4)}))
r1, r2 = d1["results"], d2["results"]
val_tab = md(pd.DataFrame({"Case": [t["trip_id"] for t in man["trips"]], "GPS pts": [t["points"] for t in man["trips"]],
                           "Nodes": [t["nodes"] for t in man["trips"]], "Arcs": [t["arcs"] for t in man["trips"]],
                           "Route arcs": [t["route_len"] for t in man["trips"]],
                           "Route ids in 0..n−1": [pct(t["route_in_range_0based"]) for t in man["trips"]],
                           "Consecutive route arcs connected, 0-based": [pct(t["route_chain_0based"]) for t in man["trips"]],
                           "… if read 1-based": [pct(t["route_chain_1based"]) for t in man["trips"]],
                           "ID / content overlap with dev data": ["no" if not (t["id_overlap"] or t["track_duplicate"]) else "YES" for t in man["trips"]],
                           "Malformed rows": [sum(v["lines"] - v["kept"] for v in t["parse"].values() if "lines" in v) for t in man["trips"]]}))
dB, dF = B["pooled"]["model_hmm"]["point_acc"], F["pooled"]["model_hmm"]["point_acc"]

doc = f"""# PathSnap — Evaluator Dataset Evaluation Report

**Data:** `data/external_evaluator/evaluator_test (1) (1).zip`, 10 cases (`.track`, `.nodes`, `.arcs`, `.route`).

**Scripts:**
- `scripts/evaluate_evaluator.py`: evaluation.
- `scripts/experiment_adaptive_heading.py` and `scripts/experiment_adaptive_decoder.py`: development-data experiments.
- `scripts/build_evaluator_report.py`: this report.

Every number below is read from files in `results/evaluator/` and `results/adaptive_heading/`.

## 1. Summary

| | Baseline (Phase 4 model + Phase 5 decoder) | Final (adopted on development data) |
|---|---|---|
| **Point accuracy, classifier + HMM (pooled, {B['aligned_points']:,} aligned points)** | **{pct(dB)}** | **{pct(dF)}** |
| Point accuracy, mean over the 10 cases | {pct(B['mean_over_cases']['model_hmm']['point_acc'])} | {pct(F['mean_over_cases']['model_hmm']['point_acc'])} |
| Cases ≥ 95% point accuracy | {int((pb.model_hmm__point_acc >= .95).sum())}/10 | {int((pf.model_hmm__point_acc >= .95).sum())}/10 |
| Route arc F1 (mean over cases) | {f4(B['mean_over_cases']['model_hmm']['arc_f1'])} | {f4(F['mean_over_cases']['model_hmm']['arc_f1'])} |
| Route continuity (mean) | {f4(B['mean_over_cases']['model_hmm']['continuity'])} | {f4(F['mean_over_cases']['model_hmm']['continuity'])} |
| Exact route reproduced | {B['exact_route_cases']}/10 | {F['exact_route_cases']}/10 |
| Classifier Top-1 / Top-3 / MRR (pooled) | {f4(rk(B)['top1'])} / {f4(rk(B)['top3'])} / {f4(rk(B)['mrr'])} | {f4(rk(F)['top1'])} / {f4(rk(F)['top3'])} / {f4(rk(F)['mrr'])} |
| Candidate coverage (reference arc generated) | {pct(rk(B)['cand_coverage'])} | {pct(rk(F)['cand_coverage'])} |

**Against the 95% target.** No official evaluator scoring formula exists in the project materials (§4). On the closest official definition, the guidelines' *observation-level Top-1 accuracy* applied to the final decisions, the deployed pipeline reaches **{pct(dF)}** pooled and **{pct(F['mean_over_cases']['model_hmm']['point_acc'])}** as the mean over cases.

95% is **not** reached on every view:
- Cases 00000015 ({pct(pf.loc['00000015','model_hmm__point_acc'])}) and 00000030 ({pct(pf.loc['00000030','model_hmm__point_acc'])}) are below 95%.
- The mean route arc F1 is {f4(F['mean_over_cases']['model_hmm']['arc_f1'])}.
- No case reproduces the reference route exactly.

**Data separation.** Nothing was trained, tuned or selected on the evaluator data. The pipeline change in §7 was chosen by rules fixed in advance and evaluated on the 80 training trips (cross-validation) and reported on the 10 student-test trips. The evaluator was then run once.

That change was prompted by *diagnosing* evaluator case 00000015 (§6). The same weakness was already visible in the development data, and the decision itself used development data only. Keep this in mind when citing the final number: it is not a fully blind number in the strictest sense. The baseline number is.

## 2. Data, file formats and loader

**Format.** The files are identical in format to the training data: headerless, whitespace-separated, and `.track` uses CRLF line endings. They are parsed by the same `io.read_table` / `io.load_trip`:

| File | Columns | IDs |
|---|---|---|
| `.track` | lon, lat, t (s) | GPS row = observation (not linked to node or arc rows) |
| `.nodes` | lon, lat | row index = node_id, 0-based |
| `.arcs` | from_node, to_node | row index = arc_id, 0-based |
| `.route` | arc_id | chronological reference arcs |

**Dedicated loader.** `src/pathsnap/evaluator_data.py`, configured by the `evaluator:` section of `config.yaml`:

- Extracts the ZIP to `data/external_evaluator/evaluator_test/<id>/`. This is outside `data/raw/`, so training code that lists `data/raw/train` can never see these cases.
- Writes only members named `<id>/<id>.<track|nodes|arcs|route>`. Paths and other names are ignored (tested).
- Validates every case (`results/evaluator/manifest.json`).

**Route indexing convention: 0-based, verified.** In every case, all route IDs are in 0..n_arcs−1, and 100% of consecutive route arcs are connected (to-node of arc k = from-node of arc k+1) when read 0-based. Read 1-based, at most 18.5% would be. All files parse with no malformed rows. No case shares an ID or a byte-identical `.track` with the training or student-test data.

{val_tab}

## 3. Compatibility with the pipeline

The unchanged pipeline (preprocessing, STRtree candidates, the 23 features, the model, HMM/Viterbi and route reconstruction) ran on all 10 cases without modification or errors, through the same `inference.match_trip` used by the API.

- **Candidates:** coverage is {pct(rk(B)['cand_coverage'])}, so the reference arc of every aligned point is among the generated candidates. There are no candidate-generation failures.
- **HMM beam:** it contains the reference for {pct(rk(B)['beam_coverage'])} of points.
- **Label coverage:** {pct(B['unalignable_share'])} of GPS points are unalignable (more than 40 m from the route) and {pct(B['ambiguous_share'])} are ambiguous (junction nodes).
- **API check:** serving the evaluator cases with `PATHSNAP_RAW_DIR=data/external_evaluator PATHSNAP_PARTITION=evaluator_test` returns the same per-case numbers as this report. This was checked on cases 00000015, 00000030, 00000087 and 00000094, also from a fresh install of the packaged backend.

## 4. Metrics and the official scoring formula

**Search result.** `Main Guidelines.pdf`, `Models and Training Guidelines.pdf`, `Deliverables.pdf`, `Dataset Exploration.pdf`, `Assessment and Plagiarism Policy.pdf` and the project code were searched. **No official evaluator scoring formula exists.** What the materials do say:
- The API "calculates prediction accuracy using its internal reference data".
- *Observation-level Top-1 accuracy* is defined: "for each GPS observation, select the candidate with the highest predicted score and compare it with the corresponding reference road segment".
- Route continuity and sequence-level performance should be reported.

No formula was invented. All metrics below are reported, and "point accuracy" is treated as the primary metric.

**Metric definitions** (implemented in `inference.evaluate_match` and `scripts/evaluate_evaluator.py`):

- **Aligned points.** The reference `.route` is aligned to the GPS points by the Phase 2 order-preserving alignment, the same code used for training labels. A point is *aligned* if one reference arc is assigned within 40 m (20 m if stationary). It is *ambiguous* if another route arc within ±2 positions lies within 2 m (a junction node), and *unalignable* otherwise.
- **Point accuracy (exact arc-level).** The share of aligned points whose output arc equals the aligned reference arc. *Pooled* = over all points of all cases; *mean of cases* = average of per-case values.
- **Ambiguous (lenient).** Ambiguous points count as correct if the output is the aligned arc or a labelled alternative.
- **Top-k, MRR.** The rank of the aligned arc among the point's candidates, ordered by classifier probability (aligned points).
- **Continuity.** The share of consecutive point pairs (time gap ≤ 60 s) whose arcs are equal or connected by a feasible directed path.
- **Route arc P/R/F1.** The reconstructed route compared with the reference route, as arc sets. *Length F1* weights each arc by its length. *Route mismatch* = (length wrongly added + length missed) / reference length [Newson & Krumm 2009].
- **Route-level accuracy.** Whether the reconstructed arc sequence equals the reference sequence exactly.
- **Decision precision/recall/F1.** Over the candidate rows of aligned points: positive = the aligned arc, predicted = the chosen arc.

## 5. Baseline results (Phase 4 model + Phase 5 decoder, unchanged)

{agg_rows(B, pb)}

Per case (classifier + HMM unless the column says otherwise):

{case_table(pb)}

## 6. Error analysis and bottleneck

![Error types](../results/evaluator/figures/error_types.png)

{err_table()}

**Error types.**
- *Direction:* the output is the reverse twin of the reference arc.
- *Adjacent:* the output shares a node with the reference (the transition to the next arc is early or late).
- *Other road:* e.g. a parallel street.
- *Beam miss:* the reference was a candidate but not among the HMM's top-8 states.
- *Candidate miss:* the reference was never generated. This never happened.
- *Transition errors:* continuity is ≈ 1.0, so the final routes contain no disconnected transitions.
- *Sequence-reconstruction errors* show up as extra arcs: the predicted route is longer than the reference in most cases, which lowers arc precision.

**Bottleneck: one slow case.** In the baseline, case 00000015 holds {share15(pb):.0f}% of all wrong points of the deployed decision. Its GPS inputs (no labels involved) show a trace that moves at walking pace:

- The net displacement over 30 s is 1.30 m/s, with 97% of 30-s windows below 2 m/s.
- The 1 s step speed reads 7.19 m/s because the GPS jitter (~5 m) exceeds the real per-second movement.
- The old preprocessing therefore produced a "reliable" heading (≥ 5 m over ±3 s) that is mostly noise. This is exactly the feature that separates the two directions of a road.

Supporting observation (uses labels, diagnosis only): on correct points of 00000015, 10.6% of the reference arcs point more than 90° away from the GPS movement, against 1–2% on other cases. So its reference is also less consistent with the observed motion.

**The same weakness exists in the development data:**
- The five slowest *training* trips by 30-s displacement ({', '.join(slow_train.trip)}) include the four worst training trips (out-of-fold HMM accuracy 0.67–0.85).
- Student-test trip 00000077 (2.76 m/s) is the worst student-test trip.
- The Phase 4 error analysis had already found "no reliable heading" and "stationary" to be the two largest error sources.

{mov_tab}

## 7. Improvement: adaptive feature heading (decided on development data only)

**Change.** The heading used by the *features* (`feat_bearing`, `feat_bearing_valid`; `src/pathsnap/preprocessing.py`) grows its centred window from ±3 s up to ±30 s until the displacement reaches 20 m. It is marked valid only then.

- **Setting:** one principled value, about 4× the median GPS noise of 4.6 m, chosen without tuning.
- **Unchanged:** labels, evaluation and the HMM still use the original heading, so the training labels and the scoring reference stay the same. This was asserted: the candidates and labels are identical.
- **Default:** `feature_bearing_mode: fixed` reproduces all earlier features exactly. This was checked on 61k rows, and a test covers it.

**Stage 1, model** (rule fixed before running: {d1['rule']}). Tuned hyperparameters unchanged, same 5 trip-grouped folds; the current-model values reproduce Phase 4 exactly:

{fold_tab}

- **Cross-validation:** CV Top-1 {f4(r1['cv_top1_old'])} → **{f4(r1['cv_top1_new'])}**, better in {r1['cv_folds_improved']}/5 folds. Macro F1 {f4(r1['cv_macro_f1_old'])} → {f4(r1['cv_macro_f1_new'])}; log loss {f4(r1['cv_log_loss_old'])} → {f4(r1['cv_log_loss_new'])}.
- **Training out-of-fold:** classifier-only {pct(r1['train_classifier_point_acc_old'])} → {pct(r1['train_classifier_point_acc_new'])}. HMM with the old decoder {pct(r1['train_hmm_point_acc_old'])} → {pct(r1['train_hmm_point_acc_new'])} ({r1['train_trips_hmm_better']} trips better, {r1['train_trips_hmm_worse']} worse).
- **Outcome:** adopted = {d1['adopted']}.

**Stage 2, decoder re-tuned for the new model** (rule: {d2['rule']}). The Phase 5 grid on training out-of-fold probabilities selected β = {r2['best_beta_m']:g} m, w = {r2['best_w_emission']:g}.

- **Training out-of-fold point accuracy:** {pct(r2['train_oof_point_acc_current'])} → **{pct(r2['train_oof_point_acc_new'])}**. The grid is flat: all settings are within 0.06 pp.
- **Student test** (reported only): point accuracy {pct(r2['student_hmm_point_acc_current'])} → {pct(r2['student_hmm_point_acc_adaptive_retuned'])}; mean route F1 {f4(r2['student_arc_f1_mean_current'])} → {f4(r2['student_arc_f1_mean_adaptive_retuned'])}.
- **Outcome:** adopted = {d2['adopted']}.

**Honest size of the effect.** The classifier gains are large. After decoding, the end-to-end gain on development data is small (+0.3 pp on training, flat on the student test). The slowest development trips did **not** improve after decoding (e.g. student-test 00000077: 0.9092 → 0.8949).

**Deployment.** `config.yaml → final_pipeline` points to `artifacts/adaptive_heading/final_model.json` and `results/adaptive_heading/hmm_params.json`. The preprocessing override travels with the model (`preprocessing_overrides`), and `match_trip` applies it automatically, so a model can never receive features built differently from its training. Removing the two `final_pipeline` lines restores the Phase 4/5 pipeline.

## 8. Final results on the evaluator (single run of the adopted pipeline)

![Per-case point accuracy](../results/evaluator/figures/per_case_point_accuracy.png)

{agg_rows(F, pf)}

Per case:

{case_table(pf)}

**Changes against the baseline:**
- Pooled point accuracy {pct(dB)} → {pct(dF)}; ambiguous-lenient accuracy {f4(B['mean_over_cases']['model_hmm']['amb_lenient_acc'])} → {f4(F['mean_over_cases']['model_hmm']['amb_lenient_acc'])}; route arc F1 {f4(B['mean_over_cases']['model_hmm']['arc_f1'])} → {f4(F['mean_over_cases']['model_hmm']['arc_f1'])}.
- Seven cases are at 100% point accuracy.

**Regressions:**
- Case 00000030: {f4(pb.loc['00000030','model_hmm__point_acc'])} → {f4(pf.loc['00000030','model_hmm__point_acc'])}.
- Case 00000023's route arc F1: {f4(pb.loc['00000023','model_hmm__arc_f1'])} → {f4(pf.loc['00000023','model_hmm__arc_f1'])}, with point accuracy still 1.0. The reconstructed route adds {int(pf.loc['00000023','pred_route_arcs'] - pf.loc['00000023','ref_route_arcs'])} arcs.

**Observation not acted on.** On the evaluator, the final *classifier alone* ({pct(F['pooled']['classifier_only']['point_acc'])} pooled) scores higher than classifier + HMM ({pct(dF)}). This is entirely because the HMM degrades slow case 00000015 ({f4(pf.loc['00000015','classifier_only__point_acc'])} → {f4(pf.loc['00000015','model_hmm__point_acc'])}). On development data the HMM is better: training {pct(r2['train_oof_point_acc_new'])} vs {pct(r1['train_classifier_point_acc_new'])}, student test {pct(r2['student_hmm_point_acc_adaptive_retuned'])} vs {pct(r1['student_classifier_point_acc_new'])}.

Switching the deployed decoder because of this evaluator result would be selection on evaluator data, which `Dataset Exploration.pdf` forbids. It was therefore **not** done.

## 9. Remaining causes of error and next steps

**Remaining causes:**
- {share15(pf):.0f}% of the final wrong points are in case 00000015. The main type is adjacent-arc, a transition timing offset along the route, followed by direction errors. Both are consistent with walking-pace movement, where GPS jitter (`d_gps`) dominates the HMM's transition term.
- Extra arcs in reconstructed routes come from shortest-path joining between consecutive matches. They lower arc precision ({f4(F['mean_over_cases']['model_hmm']['arc_precision'])}) more than recall ({f4(F['mean_over_cases']['model_hmm']['arc_recall'])}).

**Next steps.** Each must be decided on training/student-test data before another evaluator run:
1. Speed-aware decoding for slow segments: smooth `d_gps` over a displacement window, or raise the emission weight when the 30-s displacement speed is low.
2. Collapse back-and-forth detours in route reconstruction.

## 10. Reproduce

```bash
source venv/bin/activate
python scripts/evaluate_evaluator.py --which phase4 --tag baseline   # baseline (Phase 4 model + Phase 5 decoder)
python scripts/experiment_adaptive_heading.py                       # stage 1 (training/student-test only)
python scripts/experiment_adaptive_decoder.py                       # stage 2 (training/student-test only)
python scripts/evaluate_evaluator.py --which final --tag final       # adopted pipeline, evaluator
python scripts/evaluator_report_figures.py && python scripts/build_evaluator_report.py
python -m pytest tests -q                                            # 74 passed
PATHSNAP_RAW_DIR=data/external_evaluator PATHSNAP_PARTITION=evaluator_test bash scripts/start_api.sh   # serve the 10 cases
```
"""
out = ROOT / "deliverables" / "EVALUATOR_EVALUATION_REPORT.md"
out.write_text(doc, encoding="utf-8")
print("wrote", out, len(doc), "chars")

"""PHASE 2 — build the candidate-level dataset, labels, diagnostics, plots and maps.

Run from the project folder (ML_Hackathon) with the venv active:
    python scripts\\build_phase2_dataset.py

Steps for every trip of every partition in config.yaml -> phase2.partitions:
  load -> preprocess GPS -> road geometry + spatial index -> candidates -> features
  -> route alignment + labels -> correctness checks
Then diagnostics, plots, Folium maps, Parquet datasets, schemas and phase2_report.md.
Train and student_test are processed and saved separately. No model is trained here.
"""
from __future__ import annotations
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm

from pathsnap.config import load_config, resolve
from pathsnap.io import list_trips
from pathsnap.dataset import process_trip, CANDIDATE_COLUMN_DOCS, OBS_COLUMN_DOCS
from pathsnap.features import FEATURE_COLUMNS, schema_records
from pathsnap import diagnostics as dg
from pathsnap import phase2_plots as pp


def main():
    cfg = load_config()
    p2 = cfg["phase2"]
    np.random.seed(cfg["seed"])
    raw = resolve(cfg["data"]["raw_dir"])
    proc = resolve(p2["processed_dir"]); proc.mkdir(parents=True, exist_ok=True)
    out = resolve(p2["output_dir"]); (out / "maps").mkdir(parents=True, exist_ok=True)
    proc_mem = psutil.Process()
    peak_rss = proc_mem.memory_info().rss
    t_start = time.time()
    map_trips = set(p2["diagnostics"]["map_trips"])
    files_plots, maps_written = [], []
    examples = {}
    all_obs, all_cand, all_rep, checks = {}, {}, [], []

    for part in p2["partitions"]:
        dirs = list_trips(raw, part)
        if not dirs:
            print(f"[warn] no trips for partition '{part}' under {raw}")
            continue
        obs_l, cand_l = [], []
        for d in tqdm(dirs, desc=f"Phase 2 [{part}]", unit="trip"):
            obs, cand, rep, ctx = process_trip(d, part, cfg, with_labels=True, return_context=True)
            chk = dg.check_trip(obs, cand, ctx, with_labels=True)
            checks.append({"partition": part, "trip_id": rep["trip_id"], **chk})
            peak_rss = max(peak_rss, proc_mem.memory_info().rss)
            rep["rss_mb_after"] = round(proc_mem.memory_info().rss / 1e6, 1)
            all_rep.append(rep)
            if rep["trip_id"] in map_trips:
                pp.folium_trip_map(obs, cand, ctx, out / "maps" / f"phase2_trip_{rep['trip_id']}.html",
                                   p2["diagnostics"]["map_max_obs_with_candidates"])
                maps_written.append(f"phase2_trip_{rep['trip_id']}.html")
                for kind, ex in pp.pick_examples(obs, cand, ctx["net"], rep["trip_id"]):
                    examples.setdefault(kind, ex)      # first trip (in processing order) that has each kind
            obs_l.append(obs); cand_l.append(cand)
        all_obs[part] = pd.concat(obs_l, ignore_index=True)
        all_cand[part] = pd.concat(cand_l, ignore_index=True)

    if not all_obs:
        sys.exit("No data processed. Check data/raw/train and data/raw/student_test.")

    # ---- reproducibility: rebuild the first train trip and compare byte-for-byte content
    first = list_trips(raw, p2["partitions"][0])[0]
    _, c2, _ = process_trip(first, p2["partitions"][0], cfg, with_labels=True)
    c1 = all_cand[p2["partitions"][0]]
    c1 = c1[c1.trip_id == first.name].reset_index(drop=True)
    reproducible = bool(pd.util.hash_pandas_object(c1, index=False).equals(pd.util.hash_pandas_object(c2, index=False)))

    # ---- save datasets
    sizes = {}
    for part in all_obs:
        for name, df in (("candidates", all_cand[part]), ("observations", all_obs[part])):
            path = proc / f"{name}_{part}.parquet"
            df.to_parquet(path, index=False)
            sizes[path.name] = round(path.stat().st_size / 1e6, 1)

    rep_df = pd.DataFrame(all_rep)
    rep_df.to_csv(out / "trip_processing_report.csv", index=False)
    chk_df = pd.DataFrame(checks)
    chk_df.to_csv(out / "quality_checks.csv", index=False)
    check_cols = [c for c in chk_df.columns if c not in ("partition", "trip_id")]
    check_summary = {c: bool(chk_df[c].all()) for c in check_cols}
    check_summary["reproducible_rebuild"] = reproducible

    # ---- diagnostics
    tr_obs = all_obs.get("train", next(iter(all_obs.values())))
    tr_cand = all_cand.get("train", next(iter(all_cand.values())))
    cov_frames = []
    for part, o in all_obs.items():
        cv = dg.coverage_table(o, p2["diagnostics"]["coverage_radii_m"], p2["diagnostics"]["coverage_k"])
        cv.insert(0, "partition", part)
        cov_frames.append(cv)
    cov = pd.concat(cov_frames, ignore_index=True)
    cov.to_csv(out / "candidate_coverage.csv", index=False)

    actual_cov = {}
    for part, o in all_obs.items():
        al = o[o.obs_status != "unalignable"]
        topk = {k: float(((al.positive_rank >= 0) & (al.positive_rank < k)).mean()) for k in p2["diagnostics"]["coverage_k"]
                if k <= p2["candidates"]["max_candidates"]}
        actual_cov[part] = {"coverage": float(al.positive_in_candidates.mean()), "topk": topk,
                            "missing": int((~al.positive_in_candidates).sum()), "alignable": int(len(al))}

    lq, warnings = dg.label_quality(pd.concat(all_obs.values(), ignore_index=True), rep_df, p2)
    lq.to_csv(out / "label_quality_per_trip.csv", index=False)
    (out / "label_warnings.txt").write_text("\n".join(warnings) or "no warnings", encoding="utf-8")
    fstats = dg.feature_stats(tr_cand)
    fstats.to_csv(out / "feature_stats_train.csv", index=False)
    auc = dg.single_feature_auc(tr_cand)
    auc.to_csv(out / "feature_single_auc_train.csv", index=False)

    # ---- schemas
    (out / "feature_schema.json").write_text(json.dumps(schema_records(), indent=2), encoding="utf-8")
    ds_schema = {
        "candidates": [{"column": c, "dtype": str(tr_cand[c].dtype),
                        "description": CANDIDATE_COLUMN_DOCS.get(c, "feature (see feature_schema.json)" if c in FEATURE_COLUMNS else "")}
                       for c in tr_cand.columns],
        "observations": [{"column": c, "dtype": str(tr_obs[c].dtype), "description": OBS_COLUMN_DOCS.get(c, "")}
                         for c in tr_obs.columns],
        "feature_columns": FEATURE_COLUMNS,
        "group_column": "trip_id", "label_column": "label",
    }
    (out / "dataset_schema.json").write_text(json.dumps(ds_schema, indent=2), encoding="utf-8")
    schema_ok = all(c in tr_cand.columns for c in FEATURE_COLUMNS) and all(c in tr_cand.columns for c in CANDIDATE_COLUMN_DOCS)
    check_summary["schema_columns_present"] = bool(schema_ok)

    # ---- plots
    pp.coverage_plots(cov[cov.partition == "train"] if "train" in all_obs else cov,
                      {"coverage": actual_cov.get("train", next(iter(actual_cov.values())))["coverage"],
                       "radii": p2["candidates"]["radii_m"], "k": p2["candidates"]["max_candidates"]}, out, files_plots)
    pp.candidate_count_plot(pd.concat(all_obs.values(), ignore_index=True), out, files_plots)
    cand_counts = tr_cand.label.value_counts().reindex([1, 0, -1, -2]).fillna(0).astype(int)
    pp.label_distribution_plot(lq, cand_counts, out, files_plots)
    pp.alignment_distance_plot(tr_obs, out, files_plots, p2)
    corr = pp.feature_plots(tr_cand, auc, out, files_plots, cfg["seed"])
    pp.runtime_plot(rep_df, out, files_plots)
    order = ["stationary", "gap", "ambiguous", "clear"]
    pp.example_panels([examples[k] for k in order if k in examples], out, files_plots)

    elapsed = time.time() - t_start
    meta = {"elapsed_s": round(elapsed, 1), "peak_rss_mb": round(peak_rss / 1e6, 1), "python": platform.python_version(),
            "parquet_mb": sizes, "checks": check_summary, "actual_coverage": actual_cov}
    (out / "phase2_summary.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    from pathsnap.phase2_report import write_report
    write_report(out, cfg, all_obs, all_cand, rep_df, lq, warnings, cov, actual_cov, fstats, auc, corr,
                 check_summary, meta, files_plots, maps_written)

    # ---- console summary
    print("\n=== PHASE 2 SUMMARY ===")
    for part in all_obs:
        o, c = all_obs[part], all_cand[part]
        st = o.obs_status.value_counts()
        print(f"  {part:<13} obs={len(o):>7,}  candidates={len(c):>9,}  "
              f"aligned={st.get('aligned', 0):>7,} ambiguous={st.get('ambiguous', 0):>6,} unalignable={st.get('unalignable', 0):>6,}  "
              f"coverage={actual_cov[part]['coverage']:.2%}")
    bad = [k for k, v in check_summary.items() if not v]
    print(f"  quality checks: {'ALL PASSED' if not bad else 'FAILED: ' + ', '.join(bad)}")
    print(f"  label warnings: {len(warnings)} (see results\\phase2\\label_warnings.txt)")
    print(f"  datasets: {', '.join(f'{k} ({v} MB)' for k, v in sizes.items())}")
    print(f"  time {elapsed:.0f} s, peak memory {peak_rss / 1e6:.0f} MB")
    print("Open results\\phase2\\phase2_report.md first.")


if __name__ == "__main__":
    main()

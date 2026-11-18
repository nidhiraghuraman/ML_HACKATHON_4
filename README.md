# PathSnap — GPS Map-Matching (UE24CS352A Hackathon)

PathSnap takes noisy GPS points from a vehicle and works out which road segments it actually drove on.
Pipeline: preprocessing → candidate roads per GPS point → ML model scores each candidate → HMM + Viterbi picks
the most likely connected route → FastAPI server (Laptop 1) → evaluator UI (Laptop 2).

> Status: **Phases 1-3 complete** (exploration; candidates, features, labels; baseline models and comparison). Later sections are added as each phase is built.

## Project layout

```
ML_Hackathon/
├── config.yaml              all settings (paths, seeds, thresholds)
├── requirements.txt         exact package versions (Python 3.11)
├── src/pathsnap/            the reusable library code
│   ├── config.py            reads config.yaml
│   ├── io.py                loads .track / .nodes / .arcs / .route safely
│   ├── geo.py               lon/lat -> metres (UTM per trip), headings, point-to-segment distance
│   ├── preprocessing.py     GPS flags: gaps, speed, heading, stationary (stay points), spikes
│   ├── road_network.py      directed arc geometry + STRtree spatial index per trip
│   ├── candidates.py        nearby-road search with radius expansion and top-K
│   ├── features.py          candidate features + FEATURE_SCHEMA (single source of truth)
│   ├── labels.py            order-preserving route alignment and candidate labels
│   ├── dataset.py           runs the whole chain for one trip (reused by the API later)
│   ├── diagnostics.py       quality checks, coverage study, label/feature statistics
│   ├── phase2_plots.py      diagnostic plots and Folium maps
│   ├── phase2_report.py     writes results/phase2/phase2_report.md
│   ├── modeling.py          Phase 3: column roles, label filter, trip-grouped folds, 5 models, baselines
│   ├── metrics.py           candidate metrics + within-GPS-point ranking (Top-k, MRR)
│   ├── phase3_analysis.py   permutation importance, coefficients, redundancy
│   ├── phase3_plots.py      comparison, curves, folds, trips, confusion, importance plots
│   └── phase3_report.py     writes results/phase3/phase3_report.md
├── scripts/
│   ├── explore_data.py          Phase 1: dataset report, plots, Folium maps
│   ├── build_phase2_dataset.py  Phase 2: candidate dataset, labels, diagnostics
│   └── train_phase3_models.py   Phase 3: baselines + 5 models, trip-grouped CV, selection
├── tests/                   test_phase2.py (17) + test_phase3.py (20), synthetic data only
├── artifacts/phase3/        selected_model.joblib + selected_model_meta.json
├── data/processed/          Phase 2 Parquet datasets (NOT committed; regenerate)
├── data/raw/                the dataset (NOT committed to Git)
│   ├── train/00000001/...   80 training trips
│   └── student_test/...     10 student-testing trips
└── results/
    ├── exploration/         outputs of Phase 1
    ├── phase2/              outputs of Phase 2
    └── phase3/              outputs of Phase 3
```

## 1. Installation (Windows, CMD, Python 3.11)

```bat
cd /d C:\Users\Nidhi\OneDrive\Desktop\ML_Hackathon
py -3.11 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Every later command assumes you are in `ML_Hackathon` with `(venv)` shown at the start of the prompt.

## 2. Dataset placement

Each trip folder must sit directly under `data\raw\train\` or `data\raw\student_test\`:

```
data\raw\train\00000001\00000001.track
data\raw\train\00000001\00000001.nodes
data\raw\train\00000001\00000001.arcs
data\raw\train\00000001\00000001.route
```

The evaluator-testing trips are not provided. On demo day they go in `data\raw\evaluator_test\<trip_id>\` on Laptop 1 only.

## 3. Phase 1 — Dataset exploration

```bat
python scripts\explore_data.py
```

Takes about 30 seconds. Writes to `results\exploration\`:

| File | Contents |
|---|---|
| `dataset_report.md` | summary of everything below, generated from the computed numbers |
| `trip_inventory.csv` | one row per trip: file sizes, row counts, timestamps, speeds, network size, connectivity, GPS-to-route distances |
| `partition_summary.json` | totals per partition + trip IDs |
| `data_quality_issues.csv` | every anomaly found (trip, issue, value) |
| `sample_rows.txt` | first rows of each file type |
| `*.png` | trip locations, trajectory lengths, sampling intervals, network size, GPS-to-road distances, data-quality counts, example trajectories |
| `maps\trip_<id>.html` | interactive Folium maps (raw GPS, road network, reference route, layer control). Opening them needs an internet connection for the map tiles |

Which Folium trips get drawn is set in `config.yaml` → `exploration.folium_trips`.

## 4. Phase 2 — Candidates, features and labels

```bat
python scripts\build_phase2_dataset.py
python -m pytest tests -v
```

The first command takes about 1-2 minutes and needs about 1.5 GB of memory. It processes every train and student-test trip and finishes with
`quality checks: ALL PASSED`. The second runs 17 tests on synthetic data and must end with `17 passed`.

**What it does, per trip (each trip in isolation):**

1. **Preprocessing** (`preprocessing.py`): converts GPS to UTM metres and adds time gaps, speed, a smoothed heading (from the points 3 s
   before and after), stationary flags (stay-point detection: within 15 m for at least 30 s), spikes and long-gap segments. No point is deleted;
   `obs_idx` is the original `.track` row.
2. **Road geometry** (`road_network.py`): each directed arc becomes a segment with its original `arc_id`, `from_node`, `to_node`, length and
   heading. Invalid arcs are counted and excluded from the index.
3. **Candidates** (`candidates.py`): an STRtree (R-tree) finds arcs within 50 m. If fewer than 2 are found, it widens to 100 m and then 200 m.
   Candidates are ranked by exact distance and the nearest 24 are kept. Radius and K come from the measured coverage study in the report.
4. **Features** (`features.py`): 23 features (distance, rank, heading difference, two-way flag, node degrees, connectivity to
   neighbouring points' candidates, speed, gaps, stationary, candidate count). None uses the reference route. Schema:
   `results/phase2/feature_schema.json`.
5. **Labels** (`labels.py`): GPS points are aligned to **route positions** with order-preserving dynamic programming (positions never go
   backwards, may skip). Candidate labels: `1` positive, `0` negative, `-1` ambiguous (an equally close neighbouring route arc, or a
   duplicate arc), `-2` the observation is unalignable (too far from the route, e.g. parked off-road).

**Outputs:** `data/processed/candidates_{train,student_test}.parquet` (one row per GPS point × candidate arc) and
`observations_{train,student_test}.parquet` (one row per GPS point). In `results/phase2/`: `phase2_report.md` (read first),
`quality_checks.csv`, `candidate_coverage.csv`, `label_quality_per_trip.csv`, `label_warnings.txt`, feature statistics, schemas,
plots, and `maps/phase2_trip_*.html` (candidate roads coloured by label, stationary points, gaps; use the layer box to switch layers).

**Reading the diagnostics:** *coverage* is how often the correct arc is among the candidates (the model can never beat it).
*Unalignable* points carry no training label. *Ambiguous* points keep one positive, but the equally plausible alternative is
not counted as wrong. `label_warnings.txt` lists trips with unusual alignment.

**Label limitations:** the reference route is taken as given (not verified). Only one arc per point is positive. Alignment settings
(`config.yaml` -> `phase2.labels`) were chosen from GPS-noise reasoning, not tuned against an independent truth.

**Into Phase 3:** train on `candidates_train.parquet` rows with `label` 0/1, using the feature columns listed in
`dataset_schema.json` and `trip_id` as the cross-validation group. Student-test files are for evaluation only.

## 5. Phase 3 — Baselines and model comparison

```bat
python scripts\train_phase3_models.py --quick
python scripts\train_phase3_models.py
```

`--quick` is a 1-2 minute smoke test (15 trips, 3 folds, small models, written to `results\phase3_quick`); its numbers are not results.
The full run takes about 20-40 minutes depending on the laptop and prints progress for every fold and model.
`python scripts\train_phase3_models.py --report-only` rebuilds the report from saved results without retraining.

**Experiment:** rows with label 0/1 from the 80 train trips are used for fitting; ambiguous (-1) and unalignable (-2) rows are
excluded and counted. The 23 Phase 2 features are the only inputs: identifiers and route-derived columns are blocked by
`modeling.assert_no_leakage`. 5-fold `StratifiedGroupKFold` grouped by `trip_id` (seed 42) scores every model only on trips
it never saw, with an assertion proving the trip sets are disjoint. Compared: nearest-road and distance+heading baselines,
Logistic Regression, KNN (trained on a documented 60,000-row subset), Decision Tree, Random Forest and Histogram Gradient
Boosting, all with initial, untuned settings (`config.yaml` -> `phase3.models`).

**Metrics:** candidate level (accuracy, precision, recall, F1, macro F1, ROC-AUC, PR-AUC, log loss, confusion matrix) and
GPS-point level (Top-1/3/5, MRR), ranking candidates only within each GPS point, plus per-trip Top-1 and runtime.

**Selection rule** (fixed in `config.yaml` before training): highest mean CV Top-1; models within 0.5 percentage points are
decided by MRR, then prediction speed. The selection is saved to `results\phase3\model_selection.json` *before* the
independent student-test check, which scores every model once for the report and never influences the choice
(`phase3.evaluate_student_test`).

**Outputs:** `results\phase3\phase3_report.md` (read first), `cv_fold_metrics.csv`, `cv_summary.csv`, `per_trip_top1.csv`,
`folds.json`, `label_filtering.json`, `model_selection.json`, `experiment_config.json`, feature-analysis CSVs, plots;
`artifacts\phase3\selected_model.joblib` + `selected_model_meta.json`; out-of-fold scores in
`data\processed\phase3_oof_scores.parquet`.

## Key dataset facts (from Phase 1)

- 80 train + 10 student-test trips, **worldwide** (34°S to 62°N), each with its **own** road network.
- GPS is nominally 1 Hz, but 14 trips have under 90% one-second steps, 23 have gaps over 60 s, and 2 have gaps over an hour.
- Road arcs are **directed**; two-way roads are stored as two arcs (median 95% of arcs have a reverse twin).
- Routes are arc-ID sequences. A route has far fewer rows than the GPS track (median ~5 GPS points per route arc), so GPS rows ≠ route rows.
- Some trips include long stationary periods (parked off-road), where GPS clusters sit 100–240 m from the route.

## Glossary

- **Node**: a point in the road network (a junction or a bend), given as lon/lat.
- **Arc**: a one-way road segment from one node to another. Its row number is its ID.
- **Candidate**: a nearby arc that a GPS point *might* belong to.
- **Feature**: a number describing a candidate (e.g. its distance to the GPS point) that the model uses to judge it.
- **Label**: the answer used for training: 1 if the candidate is the road actually driven, 0 if not.
- **Spatial index (STRtree)**: a tree of bounding boxes that finds nearby roads without checking every road.
- **Data leakage**: letting the model see information it will not have at prediction time (e.g. the reference route). Our features never use it.
- **Model**: a rule learned from examples that turns a candidate's features into a probability of being the correct road.
- **Training / validation**: the model learns from training trips and is scored on validation trips it has never seen.
- **Cross-validation (grouped by trip)**: repeat that 5 times with different held-out trips, so every trip is scored once by a model that never saw it.
- **Top-1 accuracy**: share of GPS points where the highest-scored candidate is the correct road.
- **UTM projection**: converts lon/lat (angles) into metres on a flat grid, so distances are accurate.

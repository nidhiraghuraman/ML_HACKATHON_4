# PathSnap — GPS Map-Matching with ML Candidate Scoring and HMM/Viterbi Decoding

PathSnap turns noisy GPS points into the most likely sequence of **directed road arcs** a vehicle drove.
It does not snap each point to the nearest road. Instead it scores candidate roads with a machine-learning
model and then chooses a coherent, connected route with a Hidden Markov Model (HMM) decoded by Viterbi.

```
Raw GPS trajectory (.track) + trip road network (.nodes, .arcs)
        |
        v
GPS preprocessing ............ UTM metres, time gaps, stationary periods, spikes, smoothed heading   (preprocessing.py)
        |
        v
Candidate road generation .... STRtree spatial index, 50/100/200 m radius expansion, top 24 arcs    (road_network.py, candidates.py)
        |
        v
Feature extraction ........... 23 inference-time features (distance, heading difference, ...)        (features.py)
        |
        v
ML candidate scoring ......... probability that each candidate is the correct road                    (modeling.py; Phases 3-4)
        |
        v
Sequential inference ......... HMM: emissions = log model probability, transitions = directed
        |                      network distance vs GPS distance; Viterbi decoding; chain breaks        (hmm.py; Phase 5)
        v
Route reconstruction ......... shortest directed paths between matched arcs -> connected arc route   (hmm.py)
        |
        v
Evaluation / visualisation ... point accuracy, continuity, route precision/recall/F1, Folium maps    (inference.py, phase5_report.py)
        |
        v
FastAPI (Laptop 1)  <-- HTTP/JSON -->  Evaluator UI (Laptop 2)                                        (api.py, ui/index.html)
```

Training labels come from the reference `.route` files (Phase 2 alignment). Inference never reads them; the
API uses them only server-side, after matching, to compute metrics.

## Project layout

```
ML_Hackathon/
├── config.yaml                  every setting (paths, seeds, thresholds, search spaces, decoder, API)
├── requirements.txt             pinned versions
├── data/raw/{train,student_test[,evaluator_test]}/<trip_id>/   dataset (not committed)
├── data/processed/              Phase 2 Parquet datasets + out-of-fold scores (not committed; regenerated)
├── src/pathsnap/
│   ├── io.py, geo.py, config.py                       loading, projections, config
│   ├── preprocessing.py, road_network.py, candidates.py, features.py, labels.py, dataset.py   Phase 2
│   ├── diagnostics.py, phase2_plots.py, phase2_report.py                                      Phase 2 reports
│   ├── modeling.py, metrics.py, evaluation.py, phase3_*.py                                    Phase 3
│   ├── tuning.py, phase4_analysis.py, phase4_plots.py, phase4_report.py                       Phase 4
│   ├── hmm.py, inference.py, phase5_report.py                                                 Phase 5
│   └── api.py                                                                                 Phase 6
├── scripts/
│   ├── explore_data.py          Phase 1     ├── tune_phase4.py       Phase 4
│   ├── build_phase2_dataset.py  Phase 2     ├── run_phase5_hmm.py    Phase 5
│   ├── train_phase3_models.py   Phase 3     ├── start_api.sh / start_ui.sh   Phases 6-8
│   └── build_final_report.py    Phase 9
├── ui/index.html (+ ui/vendor/leaflet)        Phase 7 evaluator interface
├── docs/DEPLOYMENT.md                         Phase 8 two-laptop guide
├── artifacts/phase3, artifacts/phase4         trained models + metadata
├── results/exploration, phase2 ... phase5     reports, tables, plots, maps
├── logs/api.log                               API request log (not committed)
└── tests/                                     synthetic-data tests for every phase
```

## 1. Environment (Ubuntu, Python 3.12)

All commands run in a terminal, in the project folder, with the virtual environment active:
```bash
cd ~/ML_Hackathon
source venv/bin/activate
pip install -r requirements.txt          # only needed once, or after requirements.txt changes
python -m pytest tests -v                # all tests use small synthetic data
```

## 2. Dataset placement

```
data/raw/train/00000001/00000001.track  .nodes  .arcs  .route
data/raw/student_test/00000000/...
data/raw/evaluator_test/<trip_id>/...    (demo day only, on Laptop 1)
```
From the extracted `Student folder/` in the project root:
```bash
mkdir -p data/raw
cp -r "Student folder/train" "Student folder/student_test" data/raw/
```

## 3. Running the phases (in order)

| Phase | Command | Time (approx.) | Main output |
|---|---|---|---|
| 1 Exploration | `python scripts/explore_data.py` | < 1 min | `results/exploration/dataset_report.md` |
| 2 Candidates, features, labels | `python scripts/build_phase2_dataset.py` | 1-2 min | `results/phase2/phase2_report.md`, `data/processed/*.parquet` |
| 3 Baselines + 5 models | `python scripts/train_phase3_models.py` | ~20-40 min | `results/phase3/phase3_report.md`, `artifacts/phase3/selected_model.joblib` |
| 4 Bayesian optimisation | `python scripts/tune_phase4.py` | ~20-40 min | `results/phase4/phase4_report.md`, `artifacts/phase4/final_model.json` |
| 5 HMM / Viterbi | `python scripts/run_phase5_hmm.py` | ~10-20 min | `results/phase5/phase5_report.md`, `results/phase5/hmm_params.json`, maps |
| 6 API | `bash scripts/start_api.sh` | — | `http://<ip>:8000` |
| 7 UI | `bash scripts/start_ui.sh` or open `ui/index.html` | — | browser |
| 8 Deployment | see `docs/DEPLOYMENT.md` | — | — |
| 9 Report draft | `python scripts/build_final_report.py` | seconds | `results/final_report_draft.md` |

Phases 3 and 4 also have `--quick` (1-3 minute smoke test, separate output folders, not results) and `--report-only`
(rebuild the report from saved results). Phase 5 has `--report-only`.

Read reports in the terminal with `less results/phase5/phase5_report.md`, open plots/maps with
`xdg-open results/phase5/maps/trip_00000091.html` (on WSL: `explorer.exe "$(wslpath -w results/phase5/maps/trip_00000091.html)"`).

## 4. What each phase does

**Phase 1 — exploration.** Inventory, validation and statistics of every file; plots and Folium maps.

**Phase 2 — candidates, features, labels.** Preprocessing, candidate arcs per GPS point, 23 features, and labels from an
order-preserving dynamic-programming alignment of GPS points to route positions (positive / negative / ambiguous / unalignable).

**Phase 3 — model comparison.** Nearest-road and distance+heading baselines and five scikit-learn model families
(Logistic Regression, KNN, Decision Tree, Random Forest, HistGradientBoosting) on 5-fold **trip-grouped** cross-validation
(no trip is ever in training and validation at once; asserted). Candidate metrics (accuracy, precision, recall, F1, macro F1,
ROC-AUC, PR-AUC, log loss) and per-GPS-point ranking metrics (Top-1/3/5, MRR). Selection rule fixed in `config.yaml` before training.

**Phase 4 — Bayesian optimisation.** Optuna TPE search of the selected model on training trips only; tuned vs untuned on the same
Phase 3 folds; adoption only if CV Top-1 improves; calibration check (reliability, ECE) and error analysis.

**Phase 5 — HMM/Viterbi.** States = top candidates per point; emission = model log-probability; transition = |directed network
distance − GPS distance| / β; impossible moves (unreachable, too fast) excluded; new chain after long gaps or dead ends; route
reconstructed with shortest directed paths. Decoder settings tuned on training trips with out-of-fold probabilities; evaluated on the
student-test trips through the same inference code the API uses. Compares nearest road, classifier only, distance-only HMM and
classifier + HMM.

**Phase 6 — API** (`src/pathsnap/api.py`): `GET /health`, `GET /trips`, `POST /predict {"trip_id": "..."}`, `GET /metrics`.
Interactive documentation at `http://<ip>:8000/docs`. Example:
```bash
curl -s http://localhost:8000/trips | python -m json.tool
curl -s -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"trip_id": "00000091", "include_points": false}' | python -m json.tool | less
```

**Phase 7 — UI** (`ui/index.html`): API address field (never hard-coded), connection test, trip list, loading state, errors,
map of raw GPS and predicted route, server-side metrics, inference time, predicted arc IDs.

**Phase 8 — deployment**: `docs/DEPLOYMENT.md` (IP address, firewall, WSL port forwarding, hotspot fallback, troubleshooting).

**Phase 9 — report**: `scripts/build_final_report.py` fills the Deliverables.pdf structure with the real numbers;
sections that need the team's own words are marked `[TEAM: ...]`.

## 5. Using the saved model directly

```bash
python - <<'EOF'
import sys; sys.path.insert(0, "src")
from pathsnap.config import load_config, resolve
from pathsnap import inference as I
cfg = load_config()
model, meta = I.load_final_model(cfg)
res = I.match_trip(resolve("data/raw/student_test/00000091"), "student_test", cfg, model, meta["feature_columns"], I.load_hmm_params(cfg))
r = res["methods"]["model_hmm"]
print(len(res["obs"]), "GPS points ->", len(r["route"]), "route arcs; first arcs:", r["route"][:10])
EOF
```

## 6. Known limitations

- Labels and point accuracy are measured against the Phase 2 alignment, which treats the reference route as correct.
- The evaluator-test partition is not part of the provided data; its results are produced by the API on the day.
- KNN is trained on a documented subset (brute-force distance cost); the Bayesian search uses a subset of GPS points (final
  comparison uses all data).
- Route metrics compare arc sets (order is not scored); the HMM beam keeps 8 candidates per point.
- Map backgrounds in the UI and Folium maps need internet; routes, metrics and the API do not.

## Glossary

- **Arc**: a one-way road segment between two nodes; its row in `.arcs` is its ID. Two-way streets are two arcs.
- **Candidate**: a nearby arc that a GPS point might belong to.
- **Feature / label**: numbers describing a candidate / whether it is the correct road (1) or not (0).
- **Trip-grouped cross-validation**: the model is always scored on whole trips it never saw (prevents leakage).
- **Top-1 accuracy**: share of GPS points whose highest-scored candidate is the correct arc.
- **Hyperparameter / Bayesian optimisation**: settings chosen before training / choosing the next settings to try from earlier results.
- **HMM**: hidden states = candidate arcs; emissions = how well a point fits an arc; transitions = how plausible moving between arcs is.
- **Viterbi**: dynamic programming that finds the single most probable sequence of states.

## React frontend and deliverables

- `frontend/`: React + TypeScript + Vite evaluator UI. Setup: `cd frontend && npm ci && cp .env.example .env` (set `VITE_API_BASE_URL`), then `npm run dev`. See `frontend/README.md`.
- `scripts/export_evaluation_summary.py`: copies the saved Phase 2-5 results into `artifacts/evaluation_summary.json` (served at `GET /evaluation`).
- `scripts/build_deliverables.sh`: builds `deliverables/PathSnap_Backend.zip` and `deliverables/PathSnap_Frontend.zip`.
- `deliverables/TWO_LAPTOP_SETUP.md`, `API_DOCUMENTATION.md`, `PROJECT_COMPLETION_REPORT.md`.

## Evaluator dataset (external, evaluation only)

Put the evaluator ZIP in `data/external_evaluator/`. The loader (`src/pathsnap/evaluator_data.py`) extracts it to `data/external_evaluator/evaluator_test/<id>/`, outside `data/raw/`, and validates formats, the 0-based route-arc indexing and the separation from training data.

```bash
python scripts/evaluate_evaluator.py --which final  --tag final      # deployed pipeline (config final_pipeline)
python scripts/evaluate_evaluator.py --which phase4 --tag baseline   # Phase 4 model + Phase 5 decoder
PATHSNAP_RAW_DIR=data/external_evaluator PATHSNAP_PARTITION=evaluator_test bash scripts/start_api.sh
```

Results are in `results/evaluator/`; the report is `deliverables/EVALUATOR_EVALUATION_REPORT.md`. Nothing is trained or tuned on evaluator data.

# PathSnap Backend (Laptop 1)

FastAPI server that map-matches noisy GPS trajectories onto the road network:
preprocessing → candidate road arcs → tuned HistGradientBoosting candidate scorer → HMM/Viterbi decoding → connected route.

## Contents
| Path | What it is |
|---|---|
| `src/pathsnap/` | Python package (pipeline + API in `api.py`) |
| `artifacts/phase4/` | Final tuned model (`tuned_model.joblib`) + `final_model.json` (params, feature list) |
| `artifacts/phase3/` | Phase 3 selected (untuned) model — fallback only |
| `artifacts/evaluation_summary.json` | Saved experiment results served at `GET /evaluation` |
| `results/phase5/hmm_params.json` | Tuned HMM decoder settings |
| `data/raw/student_test/` | 10 student-test trips served by default |
| `config.yaml`, `requirements.txt`, `.env.example` | Settings and dependencies |
| `scripts/start_api.sh` | Start script |
| `tests/` | Pipeline (Phase 2), HMM and API tests (synthetic data, no training data needed) |

No training data, virtual environment, caches or logs are included.

## Run (Ubuntu / WSL, Python 3.11 or 3.12)
```bash
cd PathSnap_Backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional: edit PORT / partition / CORS
bash scripts/start_api.sh       # listens on 0.0.0.0:8000
```
Check from the same machine: `curl http://127.0.0.1:8000/health` → `"status":"ok","model_loaded":true`.

Run the tests: `python -m pytest tests -q`

## Evaluator trips
Copy the evaluator folder so the trips are at `data/raw/evaluator_test/<trip_id>/<trip_id>.track|.nodes|.arcs`, then:
```bash
PATHSNAP_PARTITION=evaluator_test bash scripts/start_api.sh
```
Trips can also be uploaded from the frontend (`POST /upload`). If a trip folder contains a `.route` file and `PATHSNAP_EVALUATE=1`,
accuracy is computed on the server; the route itself is never sent to clients and never used as model input.

See `TWO_LAPTOP_SETUP.md` and `API_DOCUMENTATION.md` (in the deliverables folder) for networking and endpoints.

#!/usr/bin/env bash
# Build deliverables/PathSnap_Backend.zip and deliverables/PathSnap_Frontend.zip from the project.
# Usage: bash scripts/build_deliverables.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd); OUT="$ROOT/deliverables"; TMP=$(mktemp -d)
mkdir -p "$OUT"
for f in artifacts/phase4/final_model.json artifacts/phase4/tuned_model.joblib artifacts/evaluation_summary.json results/phase5/hmm_params.json artifacts/adaptive_heading/model.joblib artifacts/adaptive_heading/final_model.json results/adaptive_heading/hmm_params.json; do
  [ -f "$f" ] || { echo "missing $f - run the phases / export_evaluation_summary.py first"; exit 1; }
done

B="$TMP/PathSnap_Backend"
mkdir -p "$B/src" "$B/scripts" "$B/artifacts/phase3" "$B/artifacts/phase4" "$B/artifacts/adaptive_heading" "$B/results/phase5" "$B/results/adaptive_heading" "$B/data/raw" "$B/data/external_evaluator" "$B/tests"
cp -r src/pathsnap "$B/src/"; find "$B" -name '__pycache__' -prune -exec rm -rf {} +
cp config.yaml requirements.txt .env.example "$B/"
cp packaging/BACKEND_README.md "$B/README.md"
cp scripts/start_api.sh "$B/scripts/"
cp artifacts/phase4/final_model.json artifacts/phase4/tuned_model.joblib "$B/artifacts/phase4/"
cp artifacts/phase3/selected_model.joblib artifacts/phase3/selected_model_meta.json "$B/artifacts/phase3/"
cp artifacts/evaluation_summary.json "$B/artifacts/"
cp results/phase5/hmm_params.json "$B/results/phase5/"
cp artifacts/adaptive_heading/model.joblib artifacts/adaptive_heading/final_model.json "$B/artifacts/adaptive_heading/"
cp results/adaptive_heading/hmm_params.json "$B/results/adaptive_heading/"
[ -d data/external_evaluator/evaluator_test ] && cp -r data/external_evaluator/evaluator_test "$B/data/external_evaluator/"
cp scripts/evaluate_evaluator.py "$B/scripts/"
cp -rL data/raw/student_test "$B/data/raw/"
cp tests/test_phase2.py tests/test_api.py tests/test_phase5.py tests/test_evaluator_integration.py "$B/tests/"
[ -f tests/conftest.py ] && cp tests/conftest.py "$B/tests/"
rm -f "$OUT/PathSnap_Backend.zip"; (cd "$TMP" && zip -qr "$OUT/PathSnap_Backend.zip" PathSnap_Backend)

F="$TMP/PathSnap_Frontend"
mkdir -p "$F"; (cd frontend && tar --exclude=node_modules --exclude=dist --exclude=.env --exclude=.env.local --exclude='*.tsbuildinfo' -cf - .) | tar -xf - -C "$F"
rm -f "$OUT/PathSnap_Frontend.zip"; (cd "$TMP" && zip -qr "$OUT/PathSnap_Frontend.zip" PathSnap_Frontend)
rm -rf "$TMP"
ls -la "$OUT"/*.zip

"""End-to-end map matching for one trip (used by Phase 5 evaluation and the Phase 6 API).

match_trip():   raw trip folder -> Phase 2 preprocessing + candidates + features (no labels)
                -> model probabilities -> HMM/Viterbi -> matched arc per GPS point + connected route.
                The reference .route file is NOT read here.
evaluate_match(): compares a match result with the reference route (server-side only): pointwise
                accuracy against the Phase 2 route alignment, continuity, route precision/recall/F1
                and route mismatch.
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from .config import resolve, PROJECT_ROOT
from .dataset import process_trip
from .labels import align_to_route, ambiguous_alternatives
from . import hmm as H


def load_final_model(cfg: dict):
    """The model chosen in Phase 4 (artifacts/phase4/final_model.json), falling back to Phase 3."""
    p4 = resolve(cfg["phase4"]["artifacts_dir"]) / "final_model.json"
    if p4.exists():
        meta = json.loads(p4.read_text(encoding="utf-8"))
        path = PROJECT_ROOT / meta["path"]
    else:
        p3 = resolve(cfg["phase3"]["artifacts_dir"])
        meta = json.loads((p3 / "selected_model_meta.json").read_text(encoding="utf-8"))
        path = p3 / "selected_model.joblib"
        meta["path"] = str(path.relative_to(PROJECT_ROOT))
    return joblib.load(path), meta


def load_hmm_params(cfg: dict) -> dict:
    """Decoder settings tuned in Phase 5 (results/phase5/hmm_params.json), else the defaults."""
    p = resolve(cfg["phase5"]["output_dir"]) / "hmm_params.json"
    params = dict(H.DEFAULTS)
    params.update(cfg["phase5"].get("fixed", {}))
    if p.exists():
        params.update(json.loads(p.read_text(encoding="utf-8"))["model_hmm"])
    return params


def match_trip(trip_dir: Path, partition: str, cfg: dict, model, feats: list[str], hmm_params: dict,
               methods=("model_hmm",)) -> dict:
    """Map-match one trip. `methods` may include: model_hmm (final system), classifier_only,
    distance_hmm, nearest. Returns per-method matched arcs, routes and timings."""
    t0 = time.perf_counter()
    obs, cand, rep, ctx = process_trip(Path(trip_dir), partition, cfg, with_labels=False, return_context=True)
    t_pre = time.perf_counter() - t0
    t1 = time.perf_counter()
    prob = model.predict_proba(cand[feats].to_numpy(np.float32))[:, 1] if len(cand) else np.zeros(0)
    t_model = time.perf_counter() - t1
    paths = H.NetworkPaths(ctx["net"])
    out = {"trip_id": ctx["trip"].trip_id, "obs": obs, "cand": cand.assign(prob=prob), "net": ctx["net"], "trip": ctx["trip"],
           "paths": paths, "timing": {"preprocess_candidates_features_s": t_pre, "model_s": t_model}, "methods": {}}
    states = {}
    for m in methods:
        t2 = time.perf_counter()
        kind = "distance" if m in ("distance_hmm", "nearest") else "model"
        if kind not in states:
            st = H.build_states(obs, cand, prob, kind, hmm_params)
            states[kind] = (st, None)
        st, steps = states[kind]
        if m in ("model_hmm", "distance_hmm"):
            if steps is None:
                steps = H.transition_inputs(st, paths, hmm_params)
                states[kind] = (st, steps)
            arcs, chain = H.viterbi(st, steps, hmm_params)
        else:
            arcs, chain = H.classifier_only(st), np.full(len(obs), -1)
        route, breaks = H.reconstruct_route(arcs, st, paths, hmm_params)
        out["methods"][m] = {"arcs": arcs, "chain": chain, "route": route, "route_breaks": breaks, "states": st,
                             "decode_s": time.perf_counter() - t2}
    out["timing"]["total_s"] = time.perf_counter() - t0
    return out


def read_reference_route(trip_dir: Path) -> np.ndarray | None:
    """The trip's .route file (server-side only), or None if the trip has none."""
    trip_dir = Path(trip_dir)
    f = trip_dir / f"{trip_dir.name}.route"
    if not f.exists():
        return None
    from .io import read_table
    df, _ = read_table(f, ["arc_id"])
    return df["arc_id"].to_numpy(np.int64)


def evaluate_match(result: dict, cfg: dict, route: np.ndarray) -> dict:
    """Server-side evaluation of every method in `result` against the reference route.
    The route is used only here, after matching, and is never returned to a client."""
    obs = result["obs"]
    al = align_to_route(obs, route, result["net"], cfg["phase2"]["labels"])
    alts = ambiguous_alternatives(al, obs, route, result["net"], cfg["phase2"]["labels"])
    alt = {}
    for row, arc in alts:
        alt.setdefault(int(obs.seq.iat[row]), set()).add(arc)
    obs_eval = pd.DataFrame({"seq": obs.seq.to_numpy(), "obs_status": al.obs_status.to_numpy(), "route_arc": al.route_arc.to_numpy()})
    metrics = {}
    for m, r in result["methods"].items():
        pa = H.point_accuracy(r["arcs"], obs_eval, alt)
        cont = H.continuity(r["arcs"], r["states"], result["paths"], cfg["phase5"].get("fixed", {}))
        rm = H.route_metrics(r["route"], route, result["net"].length_m)
        metrics[m] = {**pa, "continuity": cont, **rm, "route_breaks": r["route_breaks"]}
    return metrics


def route_lonlat(net_nodes: np.ndarray, arcs_table: np.ndarray, route: list[int]) -> list[list[list[float]]]:
    """Route as a list of [lat, lon] polylines (one per arc) for maps."""
    return [[[float(net_nodes[arcs_table[a, 0], 1]), float(net_nodes[arcs_table[a, 0], 0])],
             [float(net_nodes[arcs_table[a, 1], 1]), float(net_nodes[arcs_table[a, 1], 0])]] for a in route]

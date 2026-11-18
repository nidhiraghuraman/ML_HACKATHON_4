"""PathSnap FastAPI backend (Laptop 1).

Start (project folder, venv active):
    bash scripts/start_api.sh            # = uvicorn pathsnap.api:app --app-dir src --host 0.0.0.0 --port 8000
Interactive docs: http://<ip>:8000/docs     Endpoint reference: docs/API_DOCUMENTATION.md

GET  /health                 status, model loaded, partition, trip count
GET  /model                  final model, parameters, features, decoder settings
GET  /dataset                partitions/trips available on this server (+ dataset statistics if bundled)
GET  /evaluation             saved experiment results (Phases 3-5) for the evaluation dashboard
GET  /trips                  trip IDs of the served partition (no reference data)
GET  /preview                ?trip_id=... or ?upload_id=...  raw trajectory preview (no ML)
POST /upload                 multipart upload of a trip (.zip, or .track + .nodes + .arcs [+ .route]),
                             or a .track plus network_trip_id; validated and stored under uploads/
POST /match                  {"trip_id"| "upload_id", "include_points", "include_candidates"} full pipeline
POST /predict                {"trip_id"} same as /match for a server trip (kept for the evaluator flow)
GET  /results                results computed since the server started
GET  /metrics                server-side metrics of evaluated trips and their mean

Reference routes (.route) are read only on the server, only after matching, only to compute metrics.
They are never part of a response.
"""
from __future__ import annotations
import io
import json
import logging
import os
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import load_config, resolve, PROJECT_ROOT
from .features import FEATURE_SCHEMA
from .io import list_trips, read_table
from . import inference as I

log = logging.getLogger("pathsnap.api")
ALLOWED_EXT = {"track", "nodes", "arcs", "route"}
MAX_UPLOAD_BYTES = 60 * 1024 * 1024
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


# ----------------------------------------------------------------------------- schemas
class MatchRequest(BaseModel):
    trip_id: str | None = Field(None, description="Server trip folder name, e.g. 00000091")
    upload_id: str | None = Field(None, description="ID returned by POST /upload")
    include_points: bool = Field(True, description="Matched arc, probability and top candidates of every GPS point")
    include_candidates: bool = Field(True, description="Candidate road geometry for a sample of GPS points (for the map)")


class PredictRequest(BaseModel):
    trip_id: str = Field(..., examples=["00000091"])
    include_points: bool = True


class CandidateInfo(BaseModel):
    arc_id: int
    prob: float = Field(..., description="Model probability that this arc is the correct road")
    dist_m: float


class PointResult(BaseModel):
    seq: int
    t: float
    lon: float
    lat: float
    arc_id: int = Field(..., description="Matched directed arc after HMM/Viterbi (-1 = unmatched)")
    prob: float | None = Field(None, description="Model probability of the matched arc")
    classifier_arc_id: int = Field(..., description="Highest-probability arc without sequence information")
    chain: int
    top: list[CandidateInfo] = Field(default_factory=list, description="Top-3 candidates by probability")


class MatchResponse(BaseModel):
    source: str = Field(..., description="trip or upload")
    id: str
    n_points: int
    matched_points: int
    chains: int
    route_arc_ids: list[int] = Field(..., description="Reconstructed route: connected directed arc IDs")
    route_breaks: int
    route_length_m: float
    route_geometry: list[list[list[float]]] = Field(..., description="Reconstructed route, one [[lat,lon],[lat,lon]] segment per arc")
    matched_geometry: list[list[list[float]]] = Field(..., description="Arcs matched to GPS points (before path filling)")
    classifier_route_geometry: list[list[list[float]]] = Field(..., description="Baseline: route from classifier-only matches")
    candidate_geometry: list[dict] = Field(default_factory=list, description="Candidate arcs of sampled points: {seq, arc_id, prob, chosen, line}")
    gps_geometry: list[list[float]]
    points: list[PointResult] | None = None
    stats: dict[str, float]
    timing_s: dict[str, float]
    warnings: list[str]
    metrics: dict[str, float] | None = None
    evaluated: bool
    model: str


class PredictResponse(MatchResponse):
    trip_id: str


class UploadResponse(BaseModel):
    upload_id: str
    files: list[str]
    n_points: int
    duration_s: float
    median_dt_s: float
    n_nodes: int
    n_arcs: int
    has_route: bool
    bbox: dict[str, float]
    gps_geometry: list[list[float]]
    warnings: list[str]


# ----------------------------------------------------------------------------- helpers
def _seg(nodes, arcs, a):
    return [[float(nodes[arcs[a, 0], 1]), float(nodes[arcs[a, 0], 0])], [float(nodes[arcs[a, 1], 1]), float(nodes[arcs[a, 1], 0])]]


def _validate_trip_folder(d: Path, need_network: bool = True) -> tuple[dict, list[str]]:
    """Parse and check a trip folder; raise HTTPException(422) for unusable input, return warnings."""
    tid = d.name
    warns: list[str] = []
    info: dict = {}
    tables = {}
    for kind, cols in (("track", ["lon", "lat", "t"]), ("nodes", ["lon", "lat"]), ("arcs", ["from_node", "to_node"]), ("route", ["arc_id"])):
        f = d / f"{tid}.{kind}"
        if not f.exists():
            if kind in ("track",) or (need_network and kind in ("nodes", "arcs")):
                raise HTTPException(422, f"Missing .{kind} file")
            continue
        try:
            df, rep = read_table(f, cols)
        except Exception as e:
            raise HTTPException(422, f".{kind} file could not be read as whitespace-separated numbers: {e}")
        bad = rep["lines"] - rep["kept"]
        if rep["kept"] == 0:
            raise HTTPException(422, f".{kind} file has no valid rows (expected {len(cols)} numeric columns: {', '.join(cols)})")
        if bad and kind in ("nodes", "arcs"):
            raise HTTPException(422, f".{kind} file has {bad} malformed rows; row numbers are IDs, so it cannot be used")
        if bad:
            warns.append(f".{kind}: {bad} malformed rows ignored")
        tables[kind] = df
    tr = tables["track"]
    if len(tr) < 2:
        raise HTTPException(422, "The track needs at least 2 GPS points")
    if ((tr.lon.abs() > 180) | (tr.lat.abs() > 90)).any():
        raise HTTPException(422, "Track has coordinates outside longitude [-180, 180] / latitude [-90, 90]; columns must be lon lat t")
    t = tr.t.to_numpy()
    dt = np.diff(t)
    if (dt < 0).any():
        warns.append("Timestamps are not in order; they will be sorted")
    if (dt == 0).any():
        warns.append(f"{int((dt == 0).sum())} duplicate timestamps")
    if len(dt) and np.nanmax(dt) > 60:
        warns.append(f"Largest time gap {np.nanmax(dt):.0f} s: the route will be decoded in separate chains")
    nodes = tables.get("nodes")
    arcs = tables.get("arcs")
    if nodes is not None and arcs is not None:
        n = len(nodes)
        a = arcs.to_numpy(np.int64)
        badref = int(((a < 0) | (a >= n)).sum())
        if badref:
            warns.append(f"{badref} arc endpoints reference missing nodes; those arcs are ignored")
        pad = 0.005                                                   # ~500 m margin around the network's bounding box
        inside = ((tr.lon >= nodes.lon.min() - pad) & (tr.lon <= nodes.lon.max() + pad) &
                  (tr.lat >= nodes.lat.min() - pad) & (tr.lat <= nodes.lat.max() + pad)).mean()
        if inside < 0.5:
            raise HTTPException(422, f"Only {inside:.0%} of GPS points lie inside the road network's area: track and network do not match")
        if inside < 0.95:
            warns.append(f"{1 - inside:.0%} of GPS points lie outside the road network's area")
        info.update(n_nodes=n, n_arcs=len(arcs))
    info.update(n_points=len(tr), duration_s=float(t.max() - t.min()), median_dt_s=float(np.median(dt)) if len(dt) else 0.0,
                bbox={"lon_min": float(tr.lon.min()), "lon_max": float(tr.lon.max()), "lat_min": float(tr.lat.min()), "lat_max": float(tr.lat.max())},
                has_route="route" in tables, track=tr)
    return info, warns


# ----------------------------------------------------------------------------- app factory
def create_app(cfg: dict | None = None, model=None, model_meta: dict | None = None, hmm_params: dict | None = None,
               raw_dir: Path | None = None, partition: str | None = None, evaluate: bool | None = None,
               upload_dir: Path | None = None) -> FastAPI:
    """Build the app. Arguments override config/artifacts (used by the tests)."""
    cfg = cfg or load_config()
    api_cfg = cfg.get("api", {})
    partition = partition or os.environ.get("PATHSNAP_PARTITION") or api_cfg.get("partition", "student_test")
    raw_dir = Path(raw_dir or os.environ.get("PATHSNAP_RAW_DIR") or resolve(cfg["data"]["raw_dir"]))
    upload_dir = Path(upload_dir or os.environ.get("PATHSNAP_UPLOAD_DIR") or (PROJECT_ROOT / "uploads"))
    if evaluate is None:
        evaluate = os.environ.get("PATHSNAP_EVALUATE", "1" if api_cfg.get("evaluate", True) else "0") == "1"
    if model is None:
        model, model_meta = I.load_final_model(cfg)
    hmm_params = hmm_params or I.load_hmm_params(cfg)
    feats = model_meta["feature_columns"]
    model_name = model_meta.get("path", "model")
    origins = [o.strip() for o in os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",") if o.strip()]

    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        for h in (logging.StreamHandler(), logging.FileHandler(PROJECT_ROOT / "logs" / "api.log", encoding="utf-8")):
            h.setFormatter(fmt); log.addHandler(h)

    app = FastAPI(title="PathSnap map-matching API", version="2.0",
                  description="ML candidate scoring + HMM/Viterbi map matching on directed road networks. Reference routes stay on the server.")
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])
    state = {"cache": {}, "metrics": {}, "results": {}}
    eval_path = resolve("artifacts") / "evaluation_summary.json"

    def trip_dirs() -> dict[str, Path]:
        return {d.name: d for d in list_trips(raw_dir, partition)}

    def upload_path(uid: str) -> Path:
        if not ID_RE.match(uid or ""):
            raise HTTPException(422, "Invalid upload_id")
        d = upload_dir / uid
        if not d.is_dir():
            raise HTTPException(404, f"Unknown upload '{uid}'")
        return d

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        client = request.client.host if request.client else "?"
        response = await call_next(request)
        log.info(f"{client} {request.method} {request.url.path} -> {response.status_code} in {1000 * (time.perf_counter() - t0):.0f} ms")
        return response

    # ------------------------------------------------------------------ info endpoints
    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": True, "model": model_name, "partition": partition, "n_trips": len(trip_dirs()),
                "evaluation_enabled": evaluate, "upload_enabled": True,
                "hmm": {k: hmm_params[k] for k in ("beta_m", "w_emission", "max_states")}}

    @app.get("/model")
    def model_info():
        ev = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
        return {"model": model_name, "family": model_meta.get("model", type(model).__name__), "estimator": type(model).__name__,
                "params": model_meta.get("params"), "source": model_meta.get("source"),
                "features": [{"name": k, "unit": v[0], "level": v[1], "description": v[3]} for k, v in FEATURE_SCHEMA.items() if k in feats],
                "decoder": {k: hmm_params[k] for k in ("max_states", "beta_m", "w_emission", "vmax_mps", "gap_break_s", "max_route_m")},
                "selection": ev.get("selection"), "tuning": ev.get("tuning")}

    @app.get("/dataset")
    def dataset():
        parts = []
        for p in sorted(x.name for x in raw_dir.iterdir() if x.is_dir()) if raw_dir.exists() else []:
            parts.append({"partition": p, "trips": len(list_trips(raw_dir, p)), "served": p == partition})
        ev = json.loads(eval_path.read_text(encoding="utf-8")) if eval_path.exists() else {}
        return {"raw_dir": str(raw_dir), "partitions": parts, "statistics": ev.get("dataset")}

    @app.get("/evaluation")
    def evaluation():
        if not eval_path.exists():
            raise HTTPException(404, "No saved evaluation results (artifacts/evaluation_summary.json). Run scripts/export_evaluation_summary.py")
        return json.loads(eval_path.read_text(encoding="utf-8"))

    @app.get("/trips")
    def trips():
        out = []
        for tid, d in sorted(trip_dirs().items()):
            track = d / f"{tid}.track"
            n = sum(1 for line in open(track, "rb") if line.strip()) if track.exists() else 0
            out.append({"trip_id": tid, "n_points": n})
        return {"partition": partition, "trips": out}

    @app.get("/preview")
    def preview(trip_id: str | None = Query(None), upload_id: str | None = Query(None)):
        if bool(trip_id) == bool(upload_id):
            raise HTTPException(422, "Give exactly one of trip_id or upload_id")
        d = upload_path(upload_id) if upload_id else trip_dirs().get(trip_id)
        if d is None:
            raise HTTPException(404, f"Unknown trip '{trip_id}'")
        info, warns = _validate_trip_folder(d)
        tr = info.pop("track")
        step = max(1, len(tr) // 3000)
        info.pop("has_route")                                          # do not reveal reference availability details
        return {"id": d.name, **info, "gps_geometry": tr[["lat", "lon"]].to_numpy()[::step].round(7).tolist(), "warnings": warns}

    # ------------------------------------------------------------------ upload
    @app.post("/upload", response_model=UploadResponse)
    async def upload(files: list[UploadFile] = File(..., description=".zip with a trip's files, or .track/.nodes/.arcs[/.route] files"),
                     network_trip_id: str | None = Form(None, description="Use this server trip's road network for an uploaded .track")):
        uid = "u" + uuid.uuid4().hex[:10]
        d = upload_dir / uid
        d.mkdir(parents=True, exist_ok=True)
        try:
            total, saved = 0, {}
            for f in files:
                data = await f.read()
                total += len(data)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"Upload larger than {MAX_UPLOAD_BYTES // 2**20} MB")
                name = Path(f.filename or "").name
                ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
                if ext == "zip":
                    try:
                        zf = zipfile.ZipFile(io.BytesIO(data))
                    except zipfile.BadZipFile:
                        raise HTTPException(422, f"{name} is not a valid zip file")
                    for zi in zf.infolist():
                        zname = Path(zi.filename).name                 # never trust paths inside the zip
                        zext = zname.rsplit(".", 1)[-1].lower() if "." in zname else ""
                        if zi.is_dir() or zext not in ALLOWED_EXT or zname.startswith("."):
                            continue
                        if zi.file_size > MAX_UPLOAD_BYTES:
                            raise HTTPException(413, f"{zname} is too large")
                        if zext in saved:
                            raise HTTPException(422, f"The zip contains more than one .{zext} file")
                        (d / f"{uid}.{zext}").write_bytes(zf.read(zi))
                        saved[zext] = zname
                elif ext in ALLOWED_EXT:
                    if ext in saved:
                        raise HTTPException(422, f"More than one .{ext} file uploaded")
                    (d / f"{uid}.{ext}").write_bytes(data)
                    saved[ext] = name
                else:
                    raise HTTPException(422, f"Unsupported file '{name}': use .zip, .track, .nodes, .arcs or .route")
            if "track" not in saved:
                raise HTTPException(422, "No .track file found in the upload")
            if not {"nodes", "arcs"} <= set(saved):
                if not network_trip_id:
                    raise HTTPException(422, "A road network is needed: upload .nodes and .arcs, or choose network_trip_id")
                src = trip_dirs().get(network_trip_id)
                if src is None:
                    raise HTTPException(404, f"Unknown network_trip_id '{network_trip_id}'")
                for k in ("nodes", "arcs"):
                    shutil.copy(src / f"{network_trip_id}.{k}", d / f"{uid}.{k}")
                    saved[k] = f"{network_trip_id}.{k} (server network)"
            info, warns = _validate_trip_folder(d)
        except HTTPException:
            shutil.rmtree(d, ignore_errors=True)
            raise
        tr = info.pop("track")
        step = max(1, len(tr) // 3000)
        log.info(f"upload {uid}: files {list(saved.values())}, {info['n_points']} points")
        return UploadResponse(upload_id=uid, files=[f"{k}: {v}" for k, v in saved.items()], gps_geometry=tr[["lat", "lon"]].to_numpy()[::step].round(7).tolist(),
                              warnings=warns, **info)

    # ------------------------------------------------------------------ matching
    def run_match(key: str, d: Path, part_label: str, source: str, include_points: bool, include_candidates: bool) -> MatchResponse:
        cached = state["cache"].get(key)
        if cached is None:
            try:
                res = I.match_trip(d, part_label, cfg, model, feats, hmm_params, methods=("model_hmm", "classifier_only"))
            except HTTPException:
                raise
            except Exception as e:
                log.exception(f"match failed for {key}")
                raise HTTPException(500, f"Processing failed: {type(e).__name__}: {e}")
            metrics = None
            if evaluate:
                route = I.read_reference_route(d)                     # server-side only
                if route is not None and len(route):
                    m = I.evaluate_match(res, cfg, route)
                    metrics = {k: float(v) for k, v in m["model_hmm"].items() if v is not None and np.isfinite(v)}
                    metrics.update({f"classifier_only_{k}": float(m["classifier_only"][k]) for k in ("point_acc", "continuity", "arc_f1")
                                    if np.isfinite(m["classifier_only"].get(k, np.nan))})
                    state["metrics"][key] = metrics
            cached = (res, metrics)
            state["cache"][key] = cached
            log.info(f"{key}: {len(res['obs'])} points matched in {res['timing']['total_s']:.2f} s"
                     + (f"; point_acc={metrics.get('point_acc', float('nan')):.4f} route_f1={metrics.get('arc_f1', float('nan')):.4f}" if metrics else ""))
        res, metrics = cached
        r, rc = res["methods"]["model_hmm"], res["methods"]["classifier_only"]
        obs, trip, net, cand = res["obs"], res["trip"], res["net"], res["cand"]
        nodes, arcs = trip.nodes, trip.arcs
        lat, lon = obs.lat.to_numpy(), obs.lon.to_numpy()
        n = len(obs)
        # per-point probabilities and top candidates
        c = cand[["seq", "arc_id", "prob", "dist_m"]].sort_values(["seq", "prob"], ascending=[True, False])
        top3 = c.groupby("seq").head(3)
        top_by_seq = {s: g for s, g in top3.groupby("seq")}
        prob_lookup = dict(zip(zip(c.seq.to_numpy(), c.arc_id.to_numpy()), c.prob.to_numpy()))
        chosen_prob = np.array([prob_lookup.get((i, int(a)), np.nan) if a >= 0 else np.nan for i, a in enumerate(r["arcs"])])
        points = None
        if include_points:
            points = []
            for i in range(n):
                g = top_by_seq.get(i)
                top = [] if g is None else [CandidateInfo(arc_id=int(a), prob=float(p), dist_m=float(dd)) for a, p, dd in zip(g.arc_id, g.prob, g.dist_m)]
                points.append(PointResult(seq=int(obs.seq.iat[i]), t=float(obs.t.iat[i]), lon=float(lon[i]), lat=float(lat[i]), arc_id=int(r["arcs"][i]),
                                          prob=None if not np.isfinite(chosen_prob[i]) else float(chosen_prob[i]),
                                          classifier_arc_id=int(rc["arcs"][i]), chain=int(r["chain"][i]), top=top))
        cand_geo = []
        if include_candidates:
            for i in np.unique(np.linspace(0, n - 1, min(n, 120)).astype(int)):
                g = top_by_seq.get(int(i))
                if g is None:
                    continue
                for a, p in zip(g.arc_id, g.prob):
                    cand_geo.append({"seq": int(i), "arc_id": int(a), "prob": float(p), "chosen": bool(int(a) == int(r["arcs"][i])), "line": _seg(nodes, arcs, int(a))})
        matched_unique = list(dict.fromkeys(int(a) for a in r["arcs"] if a >= 0))
        step = max(1, n // 3000)
        warns = []
        no_cand = int((r["arcs"] < 0).sum())
        if no_cand:
            warns.append(f"{no_cand} GPS points had no candidate road within the search radius")
        chains = int(r["chain"].max() + 1) if n and r["chain"].max() >= 0 else 0
        if chains > 1:
            warns.append(f"Decoded in {chains} separate chains (time gaps or impossible transitions)")
        if r["route_breaks"]:
            warns.append(f"{r['route_breaks']} places where consecutive matches could not be connected")
        low = float(np.nanmean(chosen_prob < 0.5)) if np.isfinite(chosen_prob).any() else 0.0
        if low > 0.2:
            warns.append(f"{low:.0%} of matched points have model probability below 0.5 (low confidence)")
        cps = cand.groupby("seq").size()
        stats = {"mean_candidates_per_point": float(cps.mean()) if len(cps) else 0.0, "mean_matched_prob": float(np.nanmean(chosen_prob)) if np.isfinite(chosen_prob).any() else 0.0,
                 "low_confidence_share": low, "hmm_changed_share": float(np.mean(r["arcs"] != rc["arcs"])) if n else 0.0,
                 "matched_arcs": float(len(matched_unique)), "duration_s": float(obs.t.max() - obs.t.min()) if n else 0.0}
        out = dict(source=source, id=d.name, n_points=n, matched_points=int((r["arcs"] >= 0).sum()), chains=chains,
                   route_arc_ids=[int(a) for a in r["route"]], route_breaks=int(r["route_breaks"]),
                   route_length_m=float(sum(net.length_m[a] for a in r["route"])),
                   route_geometry=I.route_lonlat(nodes, arcs, r["route"]), matched_geometry=I.route_lonlat(nodes, arcs, matched_unique),
                   classifier_route_geometry=I.route_lonlat(nodes, arcs, rc["route"]), candidate_geometry=cand_geo,
                   gps_geometry=[[float(a), float(b)] for a, b in zip(lat[::step], lon[::step])], points=points, stats=stats,
                   timing_s={k: float(v) for k, v in res["timing"].items()} | {"decode_s": float(r["decode_s"])},
                   warnings=warns, metrics=metrics, evaluated=metrics is not None, model=model_name)
        state["results"][key] = {"source": source, "id": d.name, "n_points": n, "route_arcs": len(r["route"]), "chains": chains,
                                 "total_s": res["timing"]["total_s"], "evaluated": metrics is not None,
                                 "point_acc": (metrics or {}).get("point_acc"), "arc_f1": (metrics or {}).get("arc_f1"), "at": time.strftime("%H:%M:%S")}
        return out

    @app.post("/match", response_model=MatchResponse)
    def match(req: MatchRequest):
        if bool(req.trip_id) == bool(req.upload_id):
            raise HTTPException(422, "Give exactly one of trip_id or upload_id")
        if req.trip_id:
            d = trip_dirs().get(req.trip_id)
            if d is None:
                raise HTTPException(404, f"Unknown trip '{req.trip_id}' in partition '{partition}'")
            return MatchResponse(**run_match(f"trip:{req.trip_id}", d, partition, "trip", req.include_points, req.include_candidates))
        d = upload_path(req.upload_id)
        return MatchResponse(**run_match(f"upload:{req.upload_id}", d, "upload", "upload", req.include_points, req.include_candidates))

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        d = trip_dirs().get(req.trip_id)
        if d is None:
            raise HTTPException(404, f"Unknown trip '{req.trip_id}' in partition '{partition}'")
        return PredictResponse(trip_id=req.trip_id, **run_match(f"trip:{req.trip_id}", d, partition, "trip", req.include_points, False))

    @app.get("/results")
    def results():
        return {"results": list(state["results"].values())}

    @app.get("/metrics")
    def metrics():
        per = state["metrics"]
        keys = ["point_acc", "continuity", "arc_precision", "arc_recall", "arc_f1", "route_mismatch"]
        mean = {k: float(np.mean([m[k] for m in per.values() if k in m])) for k in keys if any(k in m for m in per.values())}
        return {"partition": partition, "evaluated_trips": len(per), "per_trip": per, "mean": mean}

    return app


def _default_app():
    try:
        return create_app()
    except FileNotFoundError as e:
        a = FastAPI(title="PathSnap (not ready)")

        @a.get("/health")
        def health():
            return {"status": "error", "model_loaded": False, "detail": f"Model artifacts missing ({e}). Run Phases 3-5 first."}
        return a


app = _default_app()

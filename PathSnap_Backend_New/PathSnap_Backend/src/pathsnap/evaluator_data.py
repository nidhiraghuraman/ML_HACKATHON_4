"""Dedicated loader for the external evaluator dataset (kept separate from data/raw/train and student_test).

Layout (default, see config.yaml -> evaluator):
    data/external_evaluator/<anything>.zip            the ZIP as received
    data/external_evaluator/evaluator_test/<trip>/    extracted by this module: <trip>.track/.nodes/.arcs[/.route]

File formats are the same as the training data and are parsed by the same `io.load_trip`:
    .track  lon lat t          (whitespace separated, no header; CRLF tolerated)
    .nodes  lon lat            row index = node_id (0-based)
    .arcs   from_node to_node  row index = arc_id (0-based)
    .route  arc_id             chronological reference arcs, 0-based arc row indices (verified by `validate`)

Separation rules enforced here:
  * extraction only writes files named <digits>.<track|nodes|arcs|route> inside a folder of the same id;
    every other archive member (paths, other names) is ignored, so the archive cannot write elsewhere;
  * trips whose id exists in a training / student-test folder, or whose .track is byte-identical to one there,
    are reported as overlap (they would not be unseen);
  * .route files are read only by `validate` (index/connectivity checks) and by evaluation code after
    prediction; nothing in this module feeds them to preprocessing, features or models.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np

from .config import resolve
from .io import load_trip

MEMBER_RE = re.compile(r"^(\d+)\.(track|nodes|arcs|route)$")
REQUIRED = ("track", "nodes", "arcs")


def settings(cfg: dict) -> dict:
    ev = cfg.get("evaluator", {})
    return {"dir": resolve(ev.get("dir", "data/external_evaluator")), "partition": ev.get("partition", "evaluator_test"),
            "zip_glob": ev.get("zip_glob", "*.zip")}


def extract(cfg: dict, zip_path: Path | None = None, overwrite: bool = False) -> Path:
    """Extract the evaluator ZIP into <dir>/<partition>/<trip>/ and return that partition folder."""
    s = settings(cfg)
    out = s["dir"] / s["partition"]
    if zip_path is None:
        zips = sorted(s["dir"].glob(s["zip_glob"]))
        if not zips:
            if out.exists():
                return out
            raise FileNotFoundError(f"No evaluator ZIP in {s['dir']} and no extracted folder {out}")
        zip_path = zips[0]
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = Path(info.filename).name
            m = MEMBER_RE.match(name)
            parent = Path(info.filename).parent.name
            if not m or parent != m.group(1):
                continue  # ignore anything that is not <id>/<id>.<kind>
            dest = out / m.group(1) / name
            if dest.exists() and not overwrite:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                dst.write(src.read())
    return out


def trip_dirs(cfg: dict) -> list[Path]:
    s = settings(cfg)
    pdir = s["dir"] / s["partition"]
    return sorted(p for p in pdir.iterdir() if p.is_dir()) if pdir.exists() else []


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def validate(cfg: dict) -> dict:
    """Format, indexing-convention and separation checks for every evaluator trip. Returns a manifest."""
    raw = resolve(cfg["data"]["raw_dir"])
    dev_ids, dev_hashes = set(), set()
    for part in cfg["data"]["partitions"]:
        for d in sorted((raw / part).iterdir()) if (raw / part).exists() else []:
            dev_ids.add(d.name)
            f = d / f"{d.name}.track"
            if f.exists():
                dev_hashes.add(_md5(f))
    trips = []
    for d in trip_dirs(cfg):
        files = {k: (d / f"{d.name}.{k}").exists() for k in (*REQUIRED, "route")}
        rec = {"trip_id": d.name, "files": files, "problems": []}
        if not all(files[k] for k in REQUIRED):
            rec["problems"].append("missing required file")
            trips.append(rec)
            continue
        t = load_trip(d, settings(cfg)["partition"], load_route=True)
        n_arcs, n_nodes = len(t.arcs), len(t.nodes)
        rec.update(points=int(len(t.track)), nodes=n_nodes, arcs=n_arcs,
                   parse={k: v for k, v in t.parse_report.items() if isinstance(v, dict)},
                   id_overlap=d.name in dev_ids, track_duplicate=_md5(d / f"{d.name}.track") in dev_hashes)
        if t.arcs.size and (t.arcs.min() < 0 or t.arcs.max() >= n_nodes):
            rec["problems"].append("arc references a node outside .nodes")
        if t.route is not None and len(t.route):
            r = t.route.astype(np.int64)
            in_range = (r >= 0) & (r < n_arcs)
            rec["route_len"] = int(len(r))
            rec["route_in_range_0based"] = float(in_range.mean())
            ok = r[in_range]

            def chain(ids):
                return float(np.mean(t.arcs[ids[:-1], 1] == t.arcs[ids[1:], 0])) if len(ids) > 1 else float("nan")
            rec["route_chain_0based"] = chain(ok)
            r1 = r - 1
            rec["route_chain_1based"] = chain(r1[(r1 >= 0) & (r1 < n_arcs)])
            if rec["route_in_range_0based"] < 1:
                rec["problems"].append("route ids outside 0..n_arcs-1")
        if rec["id_overlap"] or rec["track_duplicate"]:
            rec["problems"].append("overlaps the development data")
        trips.append(rec)
    return {"partition": settings(cfg)["partition"], "dir": str(settings(cfg)["dir"]), "n_trips": len(trips), "trips": trips,
            "all_ok": all(not t["problems"] for t in trips)}


def write_manifest(cfg: dict, path: Path) -> dict:
    man = validate(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(man, indent=2), encoding="utf-8")
    return man

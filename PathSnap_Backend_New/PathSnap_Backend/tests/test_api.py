"""Phase 6 API tests with synthetic trips and a tiny model (no real dataset or artifacts needed).

Run from the project folder:  python -m pytest tests -v
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from pathsnap.config import load_config
from pathsnap.features import FEATURE_COLUMNS
from pathsnap import hmm as H
from pathsnap.api import create_app
from test_phase2 import _write_trip, east_road


class DistanceModel:
    """Tiny stand-in model: probability falls with distance and with heading difference."""
    def predict_proba(self, X):
        d = X[:, FEATURE_COLUMNS.index("dist_m")]
        b = np.nan_to_num(X[:, FEATURE_COLUMNS.index("bearing_diff_deg")], nan=45.0)
        p = 1 / (1 + np.exp(0.3 * d + 0.03 * b - 2))
        return np.column_stack([1 - p, p])


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    nodes, arcs = east_road(n_seg=4, seg_len=100)                 # arcs 0-3 eastbound, 4-7 westbound
    xy = [[5 + 10 * i, 3] for i in range(38)]
    _write_trip(raw / "student_test", "00000001", nodes, arcs, [0, 1, 2, 3], xy, range(38))
    _write_trip(raw / "student_test", "00000002", nodes, arcs, [7, 6, 5, 4], [[395 - 10 * i, -3] for i in range(38)], range(38))
    cfg = load_config()
    app = create_app(cfg=cfg, model=DistanceModel(), model_meta={"feature_columns": FEATURE_COLUMNS, "path": "test-model"},
                     hmm_params={**H.DEFAULTS, **cfg["phase5"]["fixed"]}, raw_dir=raw, partition="student_test", evaluate=True,
                     upload_dir=tmp_path_factory.mktemp("uploads"))
    c = TestClient(app)
    c.raw = raw
    return c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "ok" and j["n_trips"] == 2 and j["partition"] == "student_test"


def test_trips_list_has_no_reference_data(client):
    j = client.get("/trips").json()
    assert [t["trip_id"] for t in j["trips"]] == ["00000001", "00000002"]
    assert all(set(t) == {"trip_id", "n_points"} for t in j["trips"])


def test_predict_matches_direction_and_returns_metrics(client):
    r = client.post("/predict", json={"trip_id": "00000001"})
    assert r.status_code == 200
    j = r.json()
    assert j["n_points"] == 38 and j["matched_points"] == 38
    assert j["route_arc_ids"] == [0, 1, 2, 3]                     # eastbound arcs in order, connected
    assert j["route_breaks"] == 0 and j["route_length_m"] == pytest.approx(400, abs=1e-3)
    assert j["evaluated"] and j["metrics"]["arc_f1"] == pytest.approx(1.0)
    assert len(j["points"]) == 38 and {p["arc_id"] for p in j["points"]} <= {0, 1, 2, 3}
    assert len(j["route_geometry"]) == 4 and j["timing_s"]["total_s"] > 0
    j2 = client.post("/predict", json={"trip_id": "00000002", "include_points": False}).json()
    assert j2["route_arc_ids"] == [7, 6, 5, 4] and j2["points"] is None   # westbound trip -> westbound arcs


def test_response_never_contains_reference_route(client):
    j = client.post("/predict", json={"trip_id": "00000001"}).json()
    assert not any(k in j for k in ("route", "reference_route", "labels", "route_pos", "obs_status"))
    allowed = {"aligned_points", "point_acc", "ambiguous_points", "amb_lenient_acc", "matched_share", "continuity",
               "arc_precision", "arc_recall", "arc_f1", "len_precision", "len_recall", "len_f1", "route_mismatch", "route_breaks"}
    assert set(j["metrics"]) <= allowed | {f"classifier_only_{k}" for k in ("point_acc", "continuity", "arc_f1")}


def test_unknown_trip_and_bad_request(client):
    assert client.post("/predict", json={"trip_id": "99999999"}).status_code == 404
    assert client.post("/predict", json={}).status_code == 422             # missing trip_id


def test_metrics_endpoint_aggregates(client):
    client.post("/predict", json={"trip_id": "00000001"})
    j = client.get("/metrics").json()
    assert j["evaluated_trips"] >= 1 and "arc_f1" in j["mean"]


def test_evaluation_can_be_disabled(tmp_path):
    nodes, arcs = east_road(n_seg=2, seg_len=100)
    _write_trip(tmp_path / "evaluator_test", "00000009", nodes, arcs, [0, 1], [[5 + 10 * i, 3] for i in range(19)], range(19))
    cfg = load_config()
    app = create_app(cfg=cfg, model=DistanceModel(), model_meta={"feature_columns": FEATURE_COLUMNS, "path": "m"},
                     hmm_params={**H.DEFAULTS, **cfg["phase5"]["fixed"]}, raw_dir=tmp_path, partition="evaluator_test", evaluate=False,
                     upload_dir=tmp_path / "uploads")
    j = TestClient(app).post("/predict", json={"trip_id": "00000009"}).json()
    assert j["evaluated"] is False and j["metrics"] is None and j["route_arc_ids"] == [0, 1]


# ----------------------------------------------------------------------------- new endpoints
import io
import zipfile


def _files(raw, tid, kinds=("track", "nodes", "arcs")):
    d = raw / "student_test" / tid
    return [("files", (f"{tid}.{k}", (d / f"{tid}.{k}").read_bytes(), "text/plain")) for k in kinds]


def test_info_endpoints(client):
    m = client.get("/model").json()
    assert m["model"] == "test-model" and len(m["features"]) == len(FEATURE_COLUMNS) and "beta_m" in m["decoder"]
    d = client.get("/dataset").json()
    assert {"partition": "student_test", "trips": 2, "served": True} in d["partitions"]
    p = client.get("/preview", params={"trip_id": "00000001"}).json()
    assert p["n_points"] == 38 and len(p["gps_geometry"]) == 38 and "has_route" not in p


def test_upload_separate_files_then_match(client):
    r = client.post("/upload", files=_files(client.raw, "00000001", ("track", "nodes", "arcs", "route")))
    assert r.status_code == 200, r.text
    u = r.json()
    assert u["upload_id"].startswith("u") and u["n_points"] == 38 and u["n_arcs"] == 8 and u["has_route"]
    j = client.post("/match", json={"upload_id": u["upload_id"]}).json()
    assert j["source"] == "upload" and j["route_arc_ids"] == [0, 1, 2, 3] and j["evaluated"]
    assert j["points"][5]["top"] and 0 <= j["points"][5]["prob"] <= 1 and j["candidate_geometry"]
    assert any(cg["chosen"] for cg in j["candidate_geometry"])
    assert client.get("/results").json()["results"]


def test_upload_zip_with_folder_paths(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for k in ("track", "nodes", "arcs"):
            z.write(client.raw / "student_test" / "00000002" / f"00000002.{k}", f"../../evil/dir/00000002.{k}")
        z.writestr("notes.txt", "ignored")
    r = client.post("/upload", files=[("files", ("trip.zip", buf.getvalue(), "application/zip"))])
    assert r.status_code == 200, r.text
    j = client.post("/match", json={"upload_id": r.json()["upload_id"], "include_points": False}).json()
    assert j["route_arc_ids"] == [7, 6, 5, 4] and j["evaluated"] is False      # no .route uploaded -> no metrics


def test_upload_track_with_server_network(client):
    r = client.post("/upload", files=_files(client.raw, "00000001", ("track",)), data={"network_trip_id": "00000001"})
    assert r.status_code == 200 and r.json()["n_nodes"] == 5


def test_upload_validation_errors(client):
    bad = [("files", ("x.track", b"1 2\n3 4\n", "text/plain"))]                    # 2 columns instead of 3
    assert client.post("/upload", files=bad).status_code == 422
    assert client.post("/upload", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))]).status_code == 422
    assert client.post("/upload", files=_files(client.raw, "00000001", ("track",))).status_code == 422          # no network
    far = "\n".join(f"{100 + i * 1e-4} 10 {i}" for i in range(5)).encode()                                      # other side of the world
    r = client.post("/upload", files=[("files", ("y.track", far, "text/plain"))] + _files(client.raw, "00000001", ("nodes", "arcs")))
    assert r.status_code == 422 and "road network" in r.json()["detail"]
    assert client.post("/match", json={"upload_id": "../../etc"}).status_code == 422
    assert client.post("/match", json={"upload_id": "unknown1"}).status_code == 404
    assert client.post("/match", json={}).status_code == 422

"""Tests for the evaluator-data loader and the adaptive feature heading (synthetic data only)."""
import copy
import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pathsnap import evaluator_data as ED
from pathsnap import inference as I
from pathsnap.config import load_config
from pathsnap.preprocessing import adaptive_bearing, preprocess_track


def _write_trip_zip(path: Path, trip="00000042", route=(0, 1, 2), extra_members=()):
    nodes = "\n".join(f"{4.0 + 0.001 * i:.6f}\t50.000000" for i in range(4))
    arcs = "0\t1\n1\t2\n2\t3\n1\t0\n"
    track = "\r\n".join(f"{4.0 + 0.00003 * k:.6f}\t50.000010\t{k + 1:.6f}" for k in range(80))
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"evaluator_test/{trip}/{trip}.nodes", nodes)
        zf.writestr(f"evaluator_test/{trip}/{trip}.arcs", arcs)
        zf.writestr(f"evaluator_test/{trip}/{trip}.track", track)
        zf.writestr(f"evaluator_test/{trip}/{trip}.route", "\n".join(map(str, route)))
        for name, data in extra_members:
            zf.writestr(name, data)


@pytest.fixture
def ev_cfg(tmp_path):
    cfg = copy.deepcopy(load_config())
    cfg["evaluator"] = {"dir": str(tmp_path / "ext"), "partition": "evaluator_test", "zip_glob": "*.zip"}
    cfg["data"]["raw_dir"] = str(tmp_path / "raw")          # empty development data
    (tmp_path / "ext").mkdir()
    return cfg


def test_extract_ignores_unsafe_members(ev_cfg, tmp_path):
    z = tmp_path / "ext" / "evaluator_test (1) (1).zip"
    _write_trip_zip(z, extra_members=[("../../evil.track", "x"), ("evaluator_test/00000042/notes.txt", "x"),
                                      ("evaluator_test/00000099/00000042.track", "x")])
    out = ED.extract(ev_cfg)
    files = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert files == ["00000042/00000042.arcs", "00000042/00000042.nodes", "00000042/00000042.route", "00000042/00000042.track"]
    assert not (tmp_path / "evil.track").exists()


def test_validate_confirms_zero_based_route(ev_cfg, tmp_path):
    _write_trip_zip(tmp_path / "ext" / "e.zip", route=(0, 1, 2))
    ED.extract(ev_cfg)
    man = ED.validate(ev_cfg)
    t = man["trips"][0]
    assert man["all_ok"] and t["route_in_range_0based"] == 1.0 and t["route_chain_0based"] == 1.0
    assert not t["id_overlap"] and not t["track_duplicate"]


def test_validate_flags_out_of_range_route(ev_cfg, tmp_path):
    _write_trip_zip(tmp_path / "ext" / "e.zip", route=(1, 2, 4))      # 4 does not exist (4 arcs: 0..3)
    ED.extract(ev_cfg)
    man = ED.validate(ev_cfg)
    assert not man["all_ok"] and "route ids outside 0..n_arcs-1" in man["trips"][0]["problems"]


def test_adaptive_bearing_needs_displacement():
    t = np.arange(0, 120, 1.0)
    x = 1.0 * t                                      # 1 m/s due east
    y = np.zeros_like(t)
    b, ok = adaptive_bearing(x, y, t, min_disp=20.0, max_hw=30.0)
    assert ok[20:100].all() and np.allclose(b[20:100], 90.0)
    b2, ok2 = adaptive_bearing(np.zeros_like(t), y, t, min_disp=20.0, max_hw=30.0)   # stationary
    assert not ok2.any() and np.isnan(b2).all()


def test_fixed_mode_feature_heading_equals_move_bearing():
    import pandas as pd
    rng = np.random.default_rng(0)
    t = np.arange(1, 200, 1.0)
    df = pd.DataFrame({"lon": 4.0 + 1e-5 * t + rng.normal(0, 2e-6, len(t)), "lat": 50.0 + rng.normal(0, 2e-6, len(t)), "t": t})
    obs, _, _ = preprocess_track(df, {})
    assert np.array_equal(obs.feat_bearing_valid.to_numpy(), obs.bearing_valid.to_numpy())
    assert np.allclose(obs.feat_bearing.to_numpy(), obs.move_bearing.to_numpy(), equal_nan=True)


def test_pipeline_cfg_applies_model_overrides_without_mutating():
    cfg = load_config()

    class Stub:
        pathsnap_preprocessing_overrides = {"feature_bearing_mode": "adaptive"}
    c2 = I.pipeline_cfg(cfg, Stub())
    assert c2["phase2"]["preprocessing"]["feature_bearing_mode"] == "adaptive"
    assert cfg["phase2"]["preprocessing"].get("feature_bearing_mode", "fixed") == "fixed"
    assert I.pipeline_cfg(cfg, object()) is cfg

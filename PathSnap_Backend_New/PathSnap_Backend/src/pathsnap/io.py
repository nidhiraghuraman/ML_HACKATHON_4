"""Raw-file loading for PathSnap trips.

Each trip lives in its own folder, e.g. data/raw/train/00000001/, containing:
  00000001.track  longitude latitude timestamp_s   (GPS observations, no header)
  00000001.nodes  longitude latitude               (row index = node_id)
  00000001.arcs   from_node_id to_node_id          (row index = arc_id)
  00000001.route  arc_id                           (reference route, chronological)

Every trip has its OWN road network: node 5 in trip A has nothing to do with node 5 in trip B.
Files are read without headers, and every row is checked; rows that cannot be parsed are
counted, not silently dropped, so the exploration report shows them.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

FILE_SPECS = {
    "track": ["lon", "lat", "t"],
    "nodes": ["lon", "lat"],
    "arcs": ["from_node", "to_node"],
    "route": ["arc_id"],
}


def count_nonempty_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def read_table(path: Path, columns: list[str]) -> tuple[pd.DataFrame, dict]:
    """Read a header-less whitespace-separated file and report parsing problems.

    Returns the parsed rows (all numeric, malformed rows removed) and a report with:
      lines            non-empty lines in the file
      wrong_col_count  rows whose number of fields != expected
      non_numeric      rows containing a value that is not a number (or is missing)
      kept             rows returned
    """
    lines = count_nonempty_lines(path)
    raw = pd.read_csv(path, sep=r"\s+", header=None, dtype=str, engine="c",
                      on_bad_lines="skip", skip_blank_lines=True)
    too_many = lines - len(raw)                      # rows skipped by the parser (extra fields)
    ncols = raw.shape[1]
    if ncols < len(columns):                         # whole file has too few columns
        for c in range(ncols, len(columns)):
            raw[c] = np.nan
    extra = raw.iloc[:, len(columns):].notna().any(axis=1) if ncols > len(columns) else pd.Series(False, index=raw.index)
    raw = raw.iloc[:, :len(columns)]
    raw.columns = columns
    num = raw.apply(pd.to_numeric, errors="coerce")
    missing_field = raw.isna().any(axis=1)
    non_numeric = num.isna().any(axis=1) & ~missing_field
    bad = missing_field | non_numeric | extra
    report = {
        "lines": int(lines),
        "wrong_col_count": int(too_many + missing_field.sum() + extra.sum()),
        "non_numeric": int(non_numeric.sum()),
        "kept": int((~bad).sum()),
    }
    # Note: dropping a bad row from .nodes/.arcs would shift row indices (= IDs). We therefore
    # keep the original row number in the index so IDs stay correct even if a row is removed.
    return num[~bad], report


@dataclass
class Trip:
    trip_id: str
    partition: str
    track: pd.DataFrame            # columns lon, lat, t  (original row order)
    nodes: np.ndarray              # (n_nodes, 2) lon, lat ; row = node_id
    arcs: np.ndarray               # (n_arcs, 2) int from_node, to_node ; row = arc_id
    route: np.ndarray | None       # (n_route,) int arc_id, or None if no .route file
    parse_report: dict = field(default_factory=dict)


def list_trips(raw_dir: Path, partition: str) -> list[Path]:
    pdir = Path(raw_dir) / partition
    if not pdir.exists():
        return []
    return sorted(p for p in pdir.iterdir() if p.is_dir())


def trip_file(trip_dir: Path, kind: str) -> Path:
    return trip_dir / f"{trip_dir.name}.{kind}"


def load_trip(trip_dir: Path, partition: str, load_route: bool = True) -> Trip:
    trip_dir = Path(trip_dir)
    report = {}
    tables = {}
    for kind, cols in FILE_SPECS.items():
        path = trip_file(trip_dir, kind)
        if kind == "route" and (not load_route or not path.exists()):
            tables[kind], report[kind] = None, {"present": path.exists()}
            continue
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
        df, rep = read_table(path, cols)
        rep["present"] = True
        rep["bytes"] = path.stat().st_size
        tables[kind], report[kind] = df, rep

    nodes_df, arcs_df = tables["nodes"], tables["arcs"]
    # IDs are row positions; if a malformed row was removed, refuse rather than shift IDs.
    for kind, df in (("nodes", nodes_df), ("arcs", arcs_df)):
        if report[kind]["kept"] != report[kind]["lines"]:
            raise ValueError(f"{trip_dir.name}.{kind}: {report[kind]['lines'] - report[kind]['kept']} "
                             "malformed rows; IDs would shift. Inspect the file.")
    route = None if tables["route"] is None else tables["route"]["arc_id"].to_numpy(np.int64)
    return Trip(
        trip_id=trip_dir.name,
        partition=partition,
        track=tables["track"].reset_index(drop=True),
        nodes=nodes_df.to_numpy(np.float64),
        arcs=arcs_df.to_numpy(np.int64),
        route=route,
        parse_report=report,
    )

"""PHASE 1 — Dataset exploration for PathSnap.

Run from the project folder (ML_Hackathon) with the venv active:
    python scripts\\explore_data.py

Reads every trip in the partitions listed in config.yaml, validates every file, computes
GPS / road-network / route statistics and writes everything to results/exploration/:
    trip_inventory.csv          one row per trip with all statistics
    partition_summary.json      totals per partition
    data_quality_issues.csv     every anomaly found, one row per (trip, issue)
    sample_rows.txt             first rows of each file type
    dataset_report.md           human-readable summary built from the numbers above
    *.png                       plots
    maps/trip_<id>.html         interactive Folium maps (GPS, road network, reference route)

Nothing here trains a model or changes the data. The reference route is used only to
DESCRIBE the dataset (how far GPS points are from it), never as a model input.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import shapely
import folium
from tqdm import tqdm

from pathsnap.config import load_config, resolve
from pathsnap.io import list_trips, load_trip, trip_file, Trip
from pathsnap.geo import LocalProjector, haversine_m


# ----------------------------------------------------------------------------- per-trip analysis
def analyse_trip(trip: Trip, cfg: dict) -> tuple[dict, dict]:
    """Return (stats row for the inventory, extras used for plots/maps)."""
    ex_cfg = cfg["exploration"]
    tr = trip.track
    lon, lat, t = tr["lon"].to_numpy(), tr["lat"].to_numpy(), tr["t"].to_numpy()
    nodes, arcs, route = trip.nodes, trip.arcs, trip.route
    n_nodes, n_arcs = len(nodes), len(arcs)
    rep = trip.parse_report

    row = {"partition": trip.partition, "trip_id": trip.trip_id}
    for kind in ("track", "nodes", "arcs", "route"):
        r = rep[kind]
        row[f"{kind}_present"] = r["present"]
        row[f"{kind}_bytes"] = r.get("bytes", 0)
        row[f"{kind}_malformed_rows"] = r.get("wrong_col_count", 0) + r.get("non_numeric", 0)

    # ---- GPS validity
    invalid = (np.abs(lon) > 180) | (np.abs(lat) > 90) | ((lon == 0) & (lat == 0))
    row["gps_points"] = len(tr)
    row["gps_invalid_coords"] = int(invalid.sum())
    row.update(lon_min=lon.min(), lon_max=lon.max(), lat_min=lat.min(), lat_max=lat.max(),
               lon_center=float(np.median(lon)), lat_center=float(np.median(lat)))

    # ---- timestamps
    dt = np.diff(t)
    row["t_first"], row["t_last"] = float(t[0]), float(t[-1])
    row["duration_s"] = float(t[-1] - t[0])
    row["t_non_increasing_steps"] = int((dt <= 0).sum())
    row["t_duplicate_steps"] = int((dt == 0).sum())
    row["t_integer_valued"] = bool(np.all(np.mod(t, 1) == 0))
    row["dt_frac_1s"] = float(np.mean(dt == 1)) if len(dt) else np.nan
    row["dt_median_s"] = float(np.median(dt)) if len(dt) else np.nan
    row["dt_max_s"] = float(dt.max()) if len(dt) else np.nan
    row[f"gaps_gt_{ex_cfg['large_gap_s']}s"] = int((dt > ex_cfg["large_gap_s"]).sum())
    row["gaps_gt_60s"] = int((dt > 60).sum())

    # ---- projection (metres) and movement
    proj = LocalProjector.for_points(lon, lat)
    x, y = proj.to_xy(lon, lat)
    step = np.hypot(np.diff(x), np.diff(y))
    hv = haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
    row["utm_epsg"] = proj.epsg
    row["proj_vs_haversine_max_rel_err"] = float(np.max(np.abs(step - hv) / np.maximum(hv, 1.0))) if len(hv) else 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(dt > 0, step / dt, np.nan)
    row["gps_path_length_m"] = float(step.sum())
    row["speed_median_mps"] = float(np.nanmedian(speed)) if len(speed) else np.nan
    row["speed_p95_mps"] = float(np.nanpercentile(speed, 95)) if len(speed) else np.nan
    row["implausible_speed_steps"] = int(np.nansum(speed > ex_cfg["implausible_speed_mps"]))
    row["stationary_steps"] = int((step < 0.5).sum())
    # Stationary detection: per-step speed is unreliable when parked (GPS jitters a few metres
    # every second), so we look at NET displacement over a 30 s window instead.
    j = np.searchsorted(t, t + 30.0)
    j = np.minimum(j, len(t) - 1)
    window_ok = (t[j] - t) >= 25.0
    disp = np.hypot(x[j] - x, y[j] - y)
    row["stationary_point_frac"] = float(np.mean(window_ok & (disp < 15.0)))
    row["bbox_width_km"] = float((x.max() - x.min()) / 1000)
    row["bbox_height_km"] = float((y.max() - y.min()) / 1000)

    # ---- road network
    nx_, ny_ = proj.to_xy(nodes[:, 0], nodes[:, 1])
    row["nodes"], row["arcs"] = n_nodes, n_arcs
    row["nodes_invalid_coords"] = int(((np.abs(nodes[:, 0]) > 180) | (np.abs(nodes[:, 1]) > 90)).sum())
    row["nodes_duplicate_coords"] = int(pd.DataFrame(nodes).duplicated().sum())
    a_from, a_to = arcs[:, 0], arcs[:, 1]
    bad_ref = (a_from < 0) | (a_to < 0) | (a_from >= n_nodes) | (a_to >= n_nodes)
    row["arcs_bad_node_ref"] = int(bad_ref.sum())
    row["arcs_self_loops"] = int((a_from == a_to).sum())
    key = a_from.astype(np.int64) * n_nodes + a_to
    rkey = a_to.astype(np.int64) * n_nodes + a_from
    row["arcs_duplicates"] = int(len(key) - len(np.unique(key)))
    row["arcs_with_reverse_twin"] = float(np.isin(key, rkey).mean())   # share of arcs whose reverse also exists
    deg = np.bincount(np.concatenate([a_from, a_to]), minlength=n_nodes)
    row["nodes_isolated"] = int((deg == 0).sum())
    g = coo_matrix((np.ones(n_arcs), (a_from, a_to)), shape=(n_nodes, n_nodes)).tocsr()
    n_wcc, wlab = connected_components(g, directed=True, connection="weak")
    n_scc, slab = connected_components(g, directed=True, connection="strong")
    used = deg > 0
    row["weak_components"] = int(n_wcc - (~used).sum())
    row["largest_weak_component_frac"] = float(np.bincount(wlab[used]).max() / used.sum())
    row["strong_components"] = int(n_scc - (~used).sum())
    row["largest_strong_component_frac"] = float(np.bincount(slab[used]).max() / used.sum())
    seg_len = np.hypot(nx_[a_to] - nx_[a_from], ny_[a_to] - ny_[a_from])
    row["arc_len_median_m"] = float(np.median(seg_len))
    row["arc_len_p95_m"] = float(np.percentile(seg_len, 95))
    row["arc_len_max_m"] = float(seg_len.max())
    row["network_bbox_km2"] = float((nx_.max() - nx_.min()) * (ny_.max() - ny_.min()) / 1e6)
    inside = (x >= nx_.min()) & (x <= nx_.max()) & (y >= ny_.min()) & (y <= ny_.max())
    row["gps_inside_network_bbox_frac"] = float(inside.mean())

    # ---- spatial index over all road segments (STRtree = R-tree from shapely)
    segs = shapely.linestrings(np.stack([np.c_[nx_[a_from], ny_[a_from]], np.c_[nx_[a_to], ny_[a_to]]], axis=1))
    tree = shapely.STRtree(segs)
    pts = shapely.points(x, y)
    _, d_any = tree.query_nearest(pts, return_distance=True, all_matches=False)
    # query_nearest returns one entry per point when all_matches=False
    row["gps_to_nearest_arc_median_m"] = float(np.median(d_any))
    row["gps_to_nearest_arc_p95_m"] = float(np.percentile(d_any, 95))
    row["gps_to_nearest_arc_max_m"] = float(d_any.max())

    extras = {"x": x, "y": y, "nx": nx_, "ny": ny_, "proj": proj, "dt": dt, "speed": speed,
              "d_any": d_any, "seg_len": seg_len}

    # ---- reference route (descriptive only)
    if route is not None:
        row["route_arcs"] = len(route)
        row["route_unique_arcs"] = int(len(np.unique(route)))
        row["route_bad_arc_ref"] = int(((route < 0) | (route >= n_arcs)).sum())
        r = route[(route >= 0) & (route < n_arcs)]
        if len(r) > 1:
            linked = arcs[r[:-1], 1] == arcs[r[1:], 0]
            row["route_breaks"] = int((~linked).sum())          # consecutive arcs that don't share a node
            row["route_consecutive_repeats"] = int((r[:-1] == r[1:]).sum())
        else:
            row["route_breaks"] = row["route_consecutive_repeats"] = 0
        row["route_length_m"] = float(seg_len[r].sum())
        row["route_len_over_gps_len"] = float(row["route_length_m"] / max(row["gps_path_length_m"], 1.0))
        row["gps_points_per_route_arc"] = float(len(tr) / max(len(r), 1))
        rtree = shapely.STRtree(segs[np.unique(r)])
        _, d_route = rtree.query_nearest(pts, return_distance=True, all_matches=False)
        row["gps_to_route_median_m"] = float(np.median(d_route))
        row["gps_to_route_p95_m"] = float(np.percentile(d_route, 95))
        row["gps_to_route_max_m"] = float(d_route.max())
        for thr in (10, 25, 50, 100):
            row[f"gps_within_{thr}m_of_route_frac"] = float(np.mean(d_route <= thr))
        # How often the geometrically nearest arc is SOME arc of the route (an upper-bound hint
        # for a nearest-road baseline; it ignores direction and position along the route).
        idx_any = tree.query_nearest(pts, all_matches=False)[1]
        row["nearest_arc_in_route_frac"] = float(np.isin(idx_any, r).mean())
        extras["d_route"] = d_route
        extras["route"] = r
    return row, extras


# ----------------------------------------------------------------------------- issues
def find_issues(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    gap_col = f"gaps_gt_{cfg['exploration']['large_gap_s']}s"
    rules = [
        ("malformed rows in a file", lambda r: r.track_malformed_rows + r.nodes_malformed_rows + r.arcs_malformed_rows + r.route_malformed_rows > 0),
        ("invalid GPS coordinates", lambda r: r.gps_invalid_coords > 0),
        ("timestamps not strictly increasing", lambda r: r.t_non_increasing_steps > 0),
        ("timestamps do not start at 1", lambda r: r.t_first != 1),
        ("under 90% of steps are 1 s (not 1 Hz)", lambda r: r.dt_frac_1s < 0.9),
        ("time gap > 60 s present", lambda r: r.gaps_gt_60s > 0),
        ("time gap > 1 hour present", lambda r: r.dt_max_s > 3600),
        ("implausible speed jumps (GPS spikes)", lambda r: r.implausible_speed_steps > 0),
        ("over 30% of points stationary (< 15 m moved in 30 s)", lambda r: r.stationary_point_frac > 0.3),
        ("isolated road nodes", lambda r: r.nodes_isolated > 0),
        ("duplicate node coordinates", lambda r: r.nodes_duplicate_coords > 0),
        ("arcs referencing missing nodes", lambda r: r.arcs_bad_node_ref > 0),
        ("self-loop arcs", lambda r: r.arcs_self_loops > 0),
        ("duplicate arcs", lambda r: r.arcs_duplicates > 0),
        ("road network split into several weak components", lambda r: r.weak_components > 1),
        ("route missing", lambda r: not r.route_present),
        ("route references missing arcs", lambda r: r.get("route_bad_arc_ref", 0) > 0),
        ("route has breaks (consecutive arcs not connected)", lambda r: r.get("route_breaks", 0) > 0),
        ("route revisits arcs (loops)", lambda r: r.get("route_unique_arcs", 0) < r.get("route_arcs", 0)),
        ("GPS points > 100 m from route", lambda r: r.get("gps_within_100m_of_route_frac", 1) < 1),
        ("GPS points outside road-network box", lambda r: r.gps_inside_network_bbox_frac < 1),
    ]
    out = []
    for _, r in df.iterrows():
        for name, fn in rules:
            try:
                if fn(r):
                    out.append({"partition": r.partition, "trip_id": r.trip_id, "issue": name})
            except Exception:
                pass
    issues = pd.DataFrame(out, columns=["partition", "trip_id", "issue"])
    # attach a short numeric detail for context
    detail_cols = {"time gap > 60 s present": "dt_max_s", "time gap > 1 hour present": "dt_max_s",
                   "under 90% of steps are 1 s (not 1 Hz)": "dt_frac_1s",
                   "implausible speed jumps (GPS spikes)": "implausible_speed_steps",
                   "isolated road nodes": "nodes_isolated", "route has breaks (consecutive arcs not connected)": "route_breaks",
                   "GPS points > 100 m from route": "gps_within_100m_of_route_frac",
                   "timestamps do not start at 1": "t_first",
                   "over 30% of points stationary (< 15 m moved in 30 s)": "stationary_point_frac", "duplicate arcs": "arcs_duplicates",
                   "duplicate node coordinates": "nodes_duplicate_coords", "self-loop arcs": "arcs_self_loops",
                   "road network split into several weak components": "weak_components",
                   "GPS points outside road-network box": "gps_inside_network_bbox_frac"}
    idx = df.set_index(["partition", "trip_id"])
    issues["detail"] = [
        f"{detail_cols[i]}={idx.loc[(p, t), detail_cols[i]]:.4g}" if i in detail_cols else ""
        for p, t, i in zip(issues.partition, issues.trip_id, issues.issue)
    ]
    return issues


# ----------------------------------------------------------------------------- plots
COLORS = {"train": "#2a6fdb", "student_test": "#e8833a"}


def save_plots(df: pd.DataFrame, pooled: dict, examples: list, out: Path) -> list[str]:
    files = []

    def finish(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
        plt.close(fig)
        files.append(name)

    # 1. where trips are
    fig, ax = plt.subplots(figsize=(10, 5))
    for p, g in df.groupby("partition"):
        ax.scatter(g.lon_center, g.lat_center, s=30, label=f"{p} ({len(g)})", c=COLORS.get(p, "grey"), edgecolor="k", lw=0.3)
    ax.set(xlim=(-180, 180), ylim=(-60, 75), xlabel="Longitude", ylabel="Latitude",
           title="Trip locations (median GPS position of each trip)")
    ax.grid(alpha=0.3); ax.legend()
    finish(fig, "trip_locations_world.png")

    # 2. GPS points per trip
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.logspace(np.log10(df.gps_points.min()), np.log10(df.gps_points.max()), 25)
    for p, g in df.groupby("partition"):
        ax.hist(g.gps_points, bins=bins, alpha=0.7, label=p, color=COLORS.get(p))
    ax.set(xscale="log", xlabel="GPS points per trip (log scale)", ylabel="Trips", title="Trajectory length")
    ax.legend(); finish(fig, "gps_points_per_trip.png")

    # 3. sampling intervals
    dt = pooled["dt"]
    fig, ax = plt.subplots(figsize=(8, 4))
    vals, counts = np.unique(np.clip(dt, 0, 31), return_counts=True)
    ax.bar(vals, counts, width=0.8, color="#2a6fdb")
    ax.set(yscale="log", xlabel="Seconds between consecutive GPS points (31 = 31 s or more)", ylabel="Count (log)",
           title=f"Sampling intervals, all trips ({np.mean(dt == 1):.1%} of steps are exactly 1 s)")
    finish(fig, "sampling_intervals.png")

    # 4. road network size
    fig, ax = plt.subplots(figsize=(7, 5))
    for p, g in df.groupby("partition"):
        ax.scatter(g.nodes, g.arcs, label=p, c=COLORS.get(p), edgecolor="k", lw=0.3)
    ax.set(xscale="log", yscale="log", xlabel="Nodes", ylabel="Arcs", title="Road-network size per trip")
    ax.grid(alpha=0.3, which="both"); ax.legend(); finish(fig, "network_size.png")

    # 5. GPS-to-route and GPS-to-nearest-road distances
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 100, 101)
    ax.hist(np.clip(pooled["d_any"], 0, 100), bins=bins, alpha=0.6, label="to nearest road segment", color="#888")
    if len(pooled["d_route"]):
        ax.hist(np.clip(pooled["d_route"], 0, 100), bins=bins, alpha=0.6, label="to nearest reference-route segment", color="#1a9e4a")
    ax.set(xlabel="Distance in metres (100 = 100 m or more)", ylabel="GPS points", title="How far GPS points lie from the roads (all trips)")
    ax.legend(); finish(fig, "gps_distance_to_roads.png")

    # 6. data-quality counts (missing / invalid / malformed) per partition
    q = df.groupby("partition")[["track_malformed_rows", "nodes_malformed_rows", "arcs_malformed_rows", "route_malformed_rows",
                                 "gps_invalid_coords", "t_non_increasing_steps", "implausible_speed_steps", "nodes_isolated"]].sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    q.T.plot.bar(ax=ax, color=[COLORS.get(c, "grey") for c in q.index])
    ax.set(ylabel="Count (summed over trips)", title="Missing / malformed / invalid values")
    for c in ax.containers:
        ax.bar_label(c, fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); finish(fig, "data_quality_counts.png")

    # 7. example trajectories with network and route, in metres
    if examples:
        n = len(examples)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
        axes = np.atleast_1d(axes)
        for ax, (row, ex, trip) in zip(axes, examples):
            x, y = ex["x"], ex["y"]
            pad = 300
            a = trip.arcs
            ax_nx, ax_ny = ex["nx"], ex["ny"]
            keep = ((ax_nx[a[:, 0]] > x.min() - pad) & (ax_nx[a[:, 0]] < x.max() + pad) &
                    (ax_ny[a[:, 0]] > y.min() - pad) & (ax_ny[a[:, 0]] < y.max() + pad))
            segs = np.stack([np.c_[ax_nx[a[keep, 0]], ax_ny[a[keep, 0]]], np.c_[ax_nx[a[keep, 1]], ax_ny[a[keep, 1]]]], axis=1)
            ax.add_collection(LineCollection(segs, colors="#cccccc", lw=0.5))
            if "route" in ex:
                r = ex["route"]
                rs = np.stack([np.c_[ax_nx[a[r, 0]], ax_ny[a[r, 0]]], np.c_[ax_nx[a[r, 1]], ax_ny[a[r, 1]]]], axis=1)
                ax.add_collection(LineCollection(rs, colors="#1a9e4a", lw=2))
            ax.scatter(x, y, s=2, c="#2a6fdb", zorder=3)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set(xlim=(x.min() - pad, x.max() + pad), ylim=(y.min() - pad, y.max() + pad), title=f"{trip.partition} {trip.trip_id}: {len(x)} pts", xlabel="x (m, UTM)", ylabel="y (m, UTM)")
        fig.suptitle("Grey = road network, green = reference route, blue = GPS points")
        finish(fig, "trajectory_examples.png")
    return files


# ----------------------------------------------------------------------------- folium
def folium_map(trip: Trip, ex: dict, cfg: dict, path: Path) -> None:
    ec = cfg["exploration"]
    lon, lat = trip.track["lon"].to_numpy(), trip.track["lat"].to_numpy()
    x, y, nx_, ny_, a = ex["x"], ex["y"], ex["nx"], ex["ny"], trip.arcs
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    # road segments near the trajectory (capped for file size), drawn as one multi-line layer
    pad = ec["network_buffer_m"]
    mx, my = (nx_[a[:, 0]] + nx_[a[:, 1]]) / 2, (ny_[a[:, 0]] + ny_[a[:, 1]]) / 2
    keep = np.where((mx > x.min() - pad) & (mx < x.max() + pad) & (my > y.min() - pad) & (my < y.max() + pad))[0]
    if len(keep) > ec["folium_max_network_arcs"]:
        cx, cy = np.median(x), np.median(y)
        keep = keep[np.argsort(np.hypot(mx[keep] - cx, my[keep] - cy))[: ec["folium_max_network_arcs"]]]
    nodes = trip.nodes
    net = [[[nodes[f, 1], nodes[f, 0]], [nodes[t, 1], nodes[t, 0]]] for f, t in a[keep]]
    fg_net = folium.FeatureGroup(name=f"Road network ({len(keep)} of {len(a)} arcs)")
    if net:
        folium.PolyLine(net, color="#7f7f7f", weight=1, opacity=0.6).add_to(fg_net)
    fg_net.add_to(m)

    if trip.route is not None and "route" in ex:
        r = ex["route"]
        fg_r = folium.FeatureGroup(name=f"Reference route ({len(r)} arcs)")
        folium.PolyLine([[[nodes[a[i, 0], 1], nodes[a[i, 0], 0]], [nodes[a[i, 1], 1], nodes[a[i, 1], 0]]] for i in r],
                        color="#1a9e4a", weight=5, opacity=0.8, tooltip="Reference route").add_to(fg_r)
        fg_r.add_to(m)

    fg_g = folium.FeatureGroup(name="Raw GPS points")
    step = max(1, int(np.ceil(len(lon) / ec["folium_max_gps_points"])))
    ts = trip.track["t"].to_numpy()
    for i in range(0, len(lon), step):
        folium.CircleMarker([lat[i], lon[i]], radius=2, color="#2a6fdb", fill=True, fill_opacity=0.8, weight=0,
                            tooltip=f"obs {i}, t={ts[i]:.0f}s").add_to(fg_g)
    fg_g.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = f"""<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;
      border:1px solid #999;border-radius:6px;font:13px sans-serif">
      <b>{trip.partition} trip {trip.trip_id}</b><br>
      <span style="color:#2a6fdb">&#9679;</span> Raw GPS ({len(lon)} pts{'' if step == 1 else f', every {step}th shown'})<br>
      <span style="color:#1a9e4a">&#9644;</span> Reference route<br>
      <span style="color:#7f7f7f">&#9644;</span> Road network</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    m.save(str(path))


# ----------------------------------------------------------------------------- report
def write_report(df: pd.DataFrame, issues: pd.DataFrame, summary: dict, out: Path, plots: list, maps: list, elapsed: float):
    tr = df[df.partition == "train"]
    L = ["# PathSnap — Dataset Exploration Report", "",
         "_Generated automatically by `scripts/explore_data.py` from the raw files. Every number below is computed, not typed in._", "",
         "## 1. Partitions", "", "| Partition | Trips | GPS points | Road nodes | Road arcs | Route arcs | Size on disk (MB) |",
         "|---|---|---|---|---|---|---|"]
    for p, s in summary["partitions"].items():
        L.append(f"| {p} | {s['trips']} | {s['gps_points']:,} | {s['nodes']:,} | {s['arcs']:,} | {s['route_arcs']:,} | {s['megabytes']:.1f} |")
    L += ["", "The evaluator-testing partition is **not** in the provided data; the pipeline must accept it later "
          "as `data/raw/evaluator_test/<trip_id>/` on Laptop 1.", "",
          "## 2. How the files relate", "",
          "- `.track` row *i* = GPS observation *i* (lon, lat, seconds). It is **not** linked to any node or arc row by index.",
          "- `.nodes` row *k* = node *k* (lon, lat). `.arcs` row *j* = directed road segment *j* from node `from` to node `to`.",
          "- `.route` = the reference arc IDs in travel order. In "
          f"{100 * (1 - (df.route_breaks > 0).mean()):.0f}% of trips EVERY consecutive pair of route arcs is connected "
          f"(`to` node of one = `from` node of the next); {int((df.route_breaks > 0).sum())} trips have at least one break.",
          f"- Each trip has its **own** road network (node/arc IDs restart per trip). Median GPS points per route arc "
          f"(train): {tr.gps_points_per_route_arc.median():.1f} — so many GPS points map to the same arc and labels must be built by alignment, not by row index.",
          "",
          "## 3. GPS trajectories (train partition)", "",
          f"- GPS points per trip: median {tr.gps_points.median():.0f}, min {tr.gps_points.min()}, max {tr.gps_points.max()}.",
          f"- Share of 1-second steps: median {tr.dt_frac_1s.median():.1%}; lowest {tr.dt_frac_1s.min():.1%}.",
          f"- Largest time gap in any train trip: {tr.dt_max_s.max():,.0f} s.",
          f"- Median speed per trip: {tr.speed_median_mps.median():.1f} m/s; trips with implausible jumps (> configured limit): {(tr.implausible_speed_steps > 0).sum()}.",
          f"- Trip locations: longitude {df.lon_min.min():.1f} to {df.lon_max.max():.1f}, latitude {df.lat_min.min():.1f} to {df.lat_max.max():.1f} (worldwide).",
          f"- Stationary points (moved < 15 m over the next 30 s): median {tr.stationary_point_frac.median():.1%} per trip; "
          f"max {tr.stationary_point_frac.max():.1%} (trip {tr.loc[tr.stationary_point_frac.idxmax(), 'trip_id']}). Long stationary periods "
          "(e.g. parked off-road) produce point clusters far from the route and must be handled in preprocessing.",
          f"- Projection check: max relative difference UTM vs haversine step length = {df.proj_vs_haversine_max_rel_err.max():.4%}.",
          "", "## 4. Road networks", "",
          f"- Nodes per trip: median {df.nodes.median():,.0f} (max {df.nodes.max():,}); arcs per trip: median {df.arcs.median():,.0f} (max {df.arcs.max():,}).",
          f"- Share of arcs whose reverse arc also exists (two-way roads stored as two arcs): median {df.arcs_with_reverse_twin.median():.0%}, min {df.arcs_with_reverse_twin.min():.0%}.",
          f"- Arc length: median of trip medians {df.arc_len_median_m.median():.0f} m; longest arc {df.arc_len_max_m.max():,.0f} m.",
          f"- Trips with isolated nodes: {(df.nodes_isolated > 0).sum()}; trips with >1 weak component: {(df.weak_components > 1).sum()}; "
          f"median share of nodes in the largest strongly-connected component: {df.largest_strong_component_frac.median():.1%}.",
          "", "## 5. GPS vs reference route (descriptive only — not a model input)", "",
          f"- Median distance GPS→route, per-trip medians: {df.gps_to_route_median_m.median():.1f} m; per-trip 95th percentile median {df.gps_to_route_p95_m.median():.1f} m; worst {df.gps_to_route_max_m.max():,.0f} m.",
          f"- Share of GPS points within 25 / 50 / 100 m of the route (all trips pooled by trip mean): "
          f"{df.gps_within_25m_of_route_frac.mean():.1%} / {df.gps_within_50m_of_route_frac.mean():.1%} / {df.gps_within_100m_of_route_frac.mean():.1%}.",
          f"- Share of GPS points whose nearest arc is *some* route arc (rough ceiling for a nearest-road method, ignoring direction): mean {df.nearest_arc_in_route_frac.mean():.1%}.",
          "", "## 6. Data-quality issues", "", "| Issue | Trips affected |", "|---|---|"]
    for name, g in issues.groupby("issue"):
        L.append(f"| {name} | {len(g)} |")
    if issues.empty:
        L.append("| none found | 0 |")
    L += ["", "Full list: `data_quality_issues.csv`.", "", "## 7. Outputs", ""]
    L += [f"- `{p}`" for p in plots] + [f"- `maps/{m}`" for m in maps]
    L += ["", f"Runtime: {elapsed:.0f} s.", ""]
    (out / "dataset_report.md").write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main():
    cfg = load_config()
    raw_dir = resolve(cfg["data"]["raw_dir"])
    out = resolve(cfg["exploration"]["output_dir"])
    (out / "maps").mkdir(parents=True, exist_ok=True)
    start = time.time()

    trip_dirs = [(p, d) for p in cfg["data"]["partitions"] for d in list_trips(raw_dir, p)]
    if not trip_dirs:
        sys.exit(f"No trips found under {raw_dir}. Expected folders like {raw_dir / 'train' / '00000001'}.")
    print(f"Found {len(trip_dirs)} trip folders under {raw_dir}")

    rows, pooled = [], {"dt": [], "d_any": [], "d_route": []}
    examples, map_jobs = [], []
    wanted_maps = set(cfg["exploration"]["folium_trips"])
    sample_written = False
    for part, d in tqdm(trip_dirs, desc="Exploring trips", unit="trip"):
        trip = load_trip(d, part)
        row, ex = analyse_trip(trip, cfg)
        rows.append(row)
        pooled["dt"].append(ex["dt"]); pooled["d_any"].append(ex["d_any"])
        if "d_route" in ex:
            pooled["d_route"].append(ex["d_route"])
        if not sample_written:
            with open(out / "sample_rows.txt", "w", encoding="utf-8") as f:
                for kind in ("track", "nodes", "arcs", "route"):
                    fp = trip_file(d, kind)
                    f.write(f"== {fp.name} (first 5 of {trip.parse_report[kind]['lines']} rows)\n")
                    with open(fp, encoding="utf-8") as src:
                        f.writelines([next(src) for _ in range(min(5, trip.parse_report[kind]["lines"]))])
                    f.write("\n")
            sample_written = True
        if trip.trip_id in wanted_maps:
            folium_map(trip, ex, cfg, out / "maps" / f"trip_{trip.trip_id}.html")
            map_jobs.append(f"trip_{trip.trip_id}.html")
            if len(examples) < 4:
                examples.append((row, ex, trip))

    df = pd.DataFrame(rows)
    df.to_csv(out / "trip_inventory.csv", index=False)
    issues = find_issues(df, cfg)
    issues.to_csv(out / "data_quality_issues.csv", index=False)
    pooled = {k: (np.concatenate(v) if v else np.array([])) for k, v in pooled.items()}

    summary = {"raw_dir": str(raw_dir), "partitions": {}}
    for p, g in df.groupby("partition"):
        summary["partitions"][p] = {
            "trips": int(len(g)), "gps_points": int(g.gps_points.sum()), "nodes": int(g.nodes.sum()),
            "arcs": int(g.arcs.sum()), "route_arcs": int(g.route_arcs.sum()) if "route_arcs" in g else 0,
            "megabytes": float(g[["track_bytes", "nodes_bytes", "arcs_bytes", "route_bytes"]].sum().sum() / 1e6),
            "trip_ids": g.trip_id.tolist(),
        }
    summary["evaluator_test_present"] = (raw_dir / "evaluator_test").exists()
    summary["issue_counts"] = issues.groupby("issue").size().to_dict()
    (out / "partition_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plots = save_plots(df, pooled, examples, out)
    elapsed = time.time() - start
    write_report(df, issues, summary, out, plots, map_jobs, elapsed)

    print("\n=== PARTITIONS ===")
    for p, s in summary["partitions"].items():
        print(f"  {p:<13} trips={s['trips']:>3}  gps_points={s['gps_points']:>8,}  arcs={s['arcs']:>10,}")
    print(f"\n=== DATA-QUALITY ISSUES (trips affected) ===")
    for k, v in summary["issue_counts"].items():
        print(f"  {v:>3}  {k}")
    print(f"\nWrote {len(plots)} plots, {len(map_jobs)} maps and the report to: {out}")
    print(f"Done in {elapsed:.0f} s. Open results\\exploration\\dataset_report.md first.")


if __name__ == "__main__":
    main()

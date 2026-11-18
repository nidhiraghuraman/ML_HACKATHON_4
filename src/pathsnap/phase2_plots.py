"""Phase 2 diagnostic plots and Folium maps. Each plot answers one question (see titles)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import folium

STATUS_COLORS = {"aligned": "#2a6fdb", "ambiguous": "#e8833a", "unalignable": "#555555"}
LABEL_COLORS = {1: "#1a9e4a", 0: "#d62728", -1: "#e8833a", -2: "#999999"}
LABEL_NAMES = {1: "positive", 0: "negative", -1: "ambiguous", -2: "unalignable obs"}


def _save(fig, out: Path, name: str, files: list):
    fig.tight_layout()
    fig.savefig(out / name, dpi=120)
    plt.close(fig)
    files.append(name)


def coverage_plots(cov: pd.DataFrame, actual: dict, out: Path, files: list):
    """Q: what radius and top-K are needed so the correct arc is among the candidates?"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for k in ["all", 32, 24, 16, 8, 4, 2]:
        s = cov[cov.k.astype(str) == str(k)]
        ax.plot(s.radius_m, s.coverage * 100, marker="o", ms=3, label=f"K = {k}")
    ax.set(xlabel="Search radius (m)", ylabel="Aligned arc among candidates (%)",
           title="Candidate coverage vs radius (train, alignable points)")
    ax.set_ylim(80, 100.5); ax.grid(alpha=0.3); ax.legend()
    ax = axes[1]
    rmax = cov.radius_m.max()
    s = cov[(cov.radius_m == rmax) & (cov.k.astype(str) != "all")]
    ax.plot(s.k.astype(int), s.coverage * 100, marker="o", color="#2a6fdb", label=f"fixed radius {rmax:.0f} m")
    ax.axhline(actual["coverage"] * 100, color="#1a9e4a", ls="--",
               label=f"actual pipeline (radii {actual['radii']}, K={actual['k']}): {actual['coverage']:.2%}")
    ax.set(xlabel="Top-K candidates kept", ylabel="Aligned arc among candidates (%)", title="Coverage vs K")
    ax.set_ylim(80, 100.5); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _save(fig, out, "coverage_vs_radius_and_k.png", files)


def candidate_count_plot(obs: pd.DataFrame, out: Path, files: list):
    """Q: how many candidates does a point get, and how often is the radius expanded / empty?"""
    fig, ax = plt.subplots(figsize=(8, 4))
    for part, g in obs.groupby("partition"):
        vc = g["n_candidates"].value_counts(normalize=True).sort_index()
        ax.bar(vc.index + (0.2 if part == "student_test" else -0.2), vc.values * 100, width=0.4, label=part)
    ax.set(xlabel="Candidates per GPS point", ylabel="% of points",
           title=f"Candidate count distribution (zero candidates: {(obs.n_candidates == 0).mean():.3%}, "
                 f"radius expanded: {obs.radius_expanded.mean():.2%})")
    ax.legend(); _save(fig, out, "candidate_count_distribution.png", files)


def label_distribution_plot(lq: pd.DataFrame, cand_counts: pd.DataFrame, out: Path, files: list):
    """Q: which trips have problematic labels (many unalignable / ambiguous points)?"""
    lq = lq.sort_values(["partition", "unalignable_frac"])
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={"height_ratios": [3, 1.2]})
    ax = axes[0]
    xs = np.arange(len(lq))
    b = np.zeros(len(lq))
    for col, name in [("aligned_frac", "aligned"), ("ambiguous_frac", "ambiguous"), ("unalignable_frac", "unalignable")]:
        ax.bar(xs, lq[col] * 100, bottom=b, color=STATUS_COLORS[name], label=name, width=0.85)
        b += lq[col].to_numpy() * 100
    ax.set_xticks(xs, [f"{p[0].upper()}{t[-3:]}" for p, t in zip(lq.partition, lq.trip_id)], rotation=90, fontsize=7)
    ax.set(ylabel="% of GPS points", title="Observation label status per trip (T = train, S = student_test; sorted by unalignable share)")
    ax.legend(loc="lower left")
    ax = axes[1]
    ax.bar([LABEL_NAMES[k] for k in cand_counts.index], cand_counts.values, color=[LABEL_COLORS[k] for k in cand_counts.index])
    for i, v in enumerate(cand_counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set(ylabel="Candidate rows", title="Candidate labels (train)", yscale="log")
    _save(fig, out, "label_distribution.png", files)


def alignment_distance_plot(obs: pd.DataFrame, out: Path, files: list, cfg: dict):
    """Q: how far are points from their aligned route arc, and where are the cut-offs?"""
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 100, 101)
    for st, g in obs.groupby("obs_status"):
        ax.hist(np.clip(g.align_dist_m.dropna(), 0, 100), bins=bins, alpha=0.6, color=STATUS_COLORS[st], label=f"{st} ({len(g):,})")
    ax.axvline(cfg["labels"]["unalign_dist_m"], color="k", ls="--", lw=1, label="unalignable cut-off")
    ax.axvline(cfg["labels"]["stationary_unalign_dist_m"], color="k", ls=":", lw=1, label="cut-off when stationary")
    ax.set(yscale="log", xlabel="Distance to aligned route arc (m; 100 = 100+)", ylabel="GPS points (log)",
           title="Alignment distance by status (train)")
    ax.legend(fontsize=8); _save(fig, out, "alignment_distance.png", files)


def feature_plots(cand: pd.DataFrame, auc: pd.DataFrame, out: Path, files: list, seed: int):
    """Q: which features separate positives from negatives, and which are redundant?"""
    d = cand[cand.label.isin([0, 1])]
    feats = ["dist_m", "dist_minus_nearest_m", "bearing_diff_deg", "dist_rank", "proj_frac", "connects_next"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, f in zip(axes.ravel(), feats):
        lo, hi = np.nanpercentile(d[f], [0, 99])
        bins = np.linspace(lo, hi if hi > lo else lo + 1, 41)
        for lab, name in [(0, "negative"), (1, "positive")]:
            v = d.loc[d.label == lab, f].dropna()
            ax.hist(np.clip(v, lo, hi), bins=bins, density=True, alpha=0.55, color=LABEL_COLORS[lab], label=name)
        a = auc.set_index("feature").loc[f, "auc"]
        ax.set(title=f"{f} (single-feature AUC {a:.3f})", yticks=[])
        ax.legend(fontsize=7)
    fig.suptitle("Feature distributions: positive vs negative candidates (train, density)")
    _save(fig, out, "feature_distributions.png", files)

    a = auc.dropna(subset=["auc_strength"])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(a.feature[::-1], a.auc_strength[::-1], color="#2a6fdb")
    ax.axvline(0.5, color="k", lw=0.8)
    ax.set(xlim=(0.45, 1.0), xlabel="max(AUC, 1 - AUC) using this feature alone",
           title="Single-feature ranking power (train, labels 1 vs 0)")
    _save(fig, out, "feature_single_auc.png", files)

    s = d.sample(min(len(d), 200_000), random_state=seed)
    from .features import FEATURE_COLUMNS
    corr = s[FEATURE_COLUMNS].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Spearman correlation between features (train sample)")
    _save(fig, out, "feature_correlation.png", files)
    return corr


def runtime_plot(rep: pd.DataFrame, out: Path, files: list):
    """Q: is the pipeline fast enough, and what drives the cost?"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(rep.net_n_arcs, rep.total_s, c="#2a6fdb", s=15)
    axes[0].set(xscale="log", xlabel="Road arcs in trip (log)", ylabel="Seconds", title="Total time per trip vs network size")
    cols = [c for c in ["load_s", "preprocess_s", "network_s", "candidates_s", "features_s", "labels_s"] if c in rep]
    axes[1].bar(cols, rep[cols].sum(), color="#e8833a")
    axes[1].set(ylabel="Seconds (all trips)", title="Where the time goes")
    plt.setp(axes[1].get_xticklabels(), rotation=30, ha="right")
    _save(fig, out, "runtime_per_trip.png", files)


# ----------------------------------------------------------------------------- example panels
def _draw_example(ax, obs, cand, net, centre_seq, half_window, title, pad=60):
    o = obs.iloc[max(0, centre_seq - half_window): centre_seq + half_window + 1]
    x, y = o.x.to_numpy(), o.y.to_numpy()
    x0, x1, y0, y1 = x.min() - pad, x.max() + pad, y.min() - pad, y.max() + pad
    mx, my = (net.x0 + net.x1) / 2, (net.y0 + net.y1) / 2
    near = np.flatnonzero(net.valid & (mx > x0 - 200) & (mx < x1 + 200) & (my > y0 - 200) & (my < y1 + 200))
    segs = np.stack([np.c_[net.x0[near], net.y0[near]], np.c_[net.x1[near], net.y1[near]]], axis=1)
    ax.add_collection(LineCollection(segs, colors="#dddddd", lw=1))
    c = cand[cand.seq == centre_seq]
    for lab in (0, -1, -2, 1):
        cc = c[c.label == lab]
        if len(cc):
            a = cc.arc_id.to_numpy()
            ss = np.stack([np.c_[net.x0[a], net.y0[a]], np.c_[net.x1[a], net.y1[a]]], axis=1)
            ax.add_collection(LineCollection(ss, colors=LABEL_COLORS[lab], lw=3 if lab == 1 else 2,
                                             label=f"{LABEL_NAMES[lab]} candidate ({len(cc)})"))
    for st, g in o.groupby("obs_status"):
        ax.scatter(g.x, g.y, s=10, c=STATUS_COLORS[st], label=f"GPS {st}", zorder=3)
    cp = obs.iloc[centre_seq]
    ax.scatter([cp.x], [cp.y], s=120, facecolors="none", edgecolors="k", lw=2, zorder=4, label="point whose candidates are shown")
    ax.set(xlim=(x0, x1), ylim=(y0, y1), title=title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=6, loc="best")


def example_panels(examples: list, out: Path, files: list):
    """Q: do labels look right in the hard cases (stationary, gap, ambiguous, normal)?"""
    if not examples:
        return
    fig, axes = plt.subplots(1, len(examples), figsize=(6 * len(examples), 6))
    for ax, ex in zip(np.atleast_1d(axes), examples):
        _draw_example(ax, *ex)
    _save(fig, out, "label_examples.png", files)


def pick_examples(obs: pd.DataFrame, cand: pd.DataFrame, net, trip_id: str) -> list:
    """Choose real example points from one trip (first match of each kind)."""
    ex = []
    o = obs.reset_index(drop=True)
    st = np.flatnonzero(o.is_stationary.to_numpy() & (o.obs_status.to_numpy() == "unalignable"))
    if len(st):
        ex.append(("stationary", (o, cand, net, int(st[len(st) // 2]), 40, f"{trip_id}: stationary, unalignable (seq {int(st[len(st) // 2])})")))
    gp = np.flatnonzero(o.gap_before.to_numpy() & (o.dt_prev_s.to_numpy() > 30) & (o.obs_status.to_numpy() != "unalignable")
                        & ~o.is_stationary.to_numpy())
    if len(gp):
        s = int(gp[0])
        ex.append(("gap", (o, cand, net, s, 6, f"{trip_id}: first point after a {o.dt_prev_s.iat[s]:.0f} s gap (seq {s})")))
    am = np.flatnonzero((o.obs_status.to_numpy() == "ambiguous") & ~o.is_stationary.to_numpy())
    if len(am):
        s = int(am[len(am) // 2])
        ex.append(("ambiguous", (o, cand, net, s, 8, f"{trip_id}: ambiguous (near a junction, seq {s})")))
    al = np.flatnonzero((o.obs_status.to_numpy() == "aligned") & ~o.is_stationary.to_numpy())
    if len(al):
        s = int(al[len(al) // 3])
        ex.append(("clear", (o, cand, net, s, 8, f"{trip_id}: clear aligned point (seq {s})")))
    return ex


# ----------------------------------------------------------------------------- folium
def folium_trip_map(obs: pd.DataFrame, cand: pd.DataFrame, ctx: dict, path: Path, max_obs_cand: int = 400,
                    buffer_m: float = 300, max_arcs: int = 15000):
    """Interactive map: road network, reference route, GPS by status, stationary points,
    gaps, and candidate arcs (coloured by label) for a sample of GPS points."""
    trip, net = ctx["trip"], ctx["net"]
    nodes = trip.nodes
    lon, lat = obs.lon.to_numpy(), obs.lat.to_numpy()
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    def arc_lines(ids):
        return [[[nodes[net.arc_from[a], 1], nodes[net.arc_from[a], 0]], [nodes[net.arc_to[a], 1], nodes[net.arc_to[a], 0]]] for a in ids]

    x, y = obs.x.to_numpy(), obs.y.to_numpy()
    mx, my = (net.x0 + net.x1) / 2, (net.y0 + net.y1) / 2
    near = np.flatnonzero(net.valid & (mx > x.min() - buffer_m) & (mx < x.max() + buffer_m) & (my > y.min() - buffer_m) & (my < y.max() + buffer_m))
    if len(near) > max_arcs:
        near = near[np.argsort(np.hypot(mx[near] - np.median(x), my[near] - np.median(y)))[:max_arcs]]
    fg = folium.FeatureGroup(name=f"Road network ({len(near):,} arcs)")
    if len(near):
        folium.PolyLine(arc_lines(near), color="#8c8c8c", weight=1, opacity=0.5).add_to(fg)
    fg.add_to(m)
    if trip.route is not None and len(trip.route):
        r = trip.route[(trip.route >= 0) & (trip.route < net.n_arcs)]
        fg = folium.FeatureGroup(name="Reference route")
        folium.PolyLine(arc_lines(r), color="#1a9e4a", weight=6, opacity=0.6).add_to(fg)
        fg.add_to(m)

    step = max(1, len(obs) // 3000)
    for st, color in STATUS_COLORS.items():
        g = obs[obs.obs_status == st].iloc[::step]
        fg = folium.FeatureGroup(name=f"GPS {st} ({(obs.obs_status == st).sum():,})")
        for _, p in g.iterrows():
            folium.CircleMarker([p.lat, p.lon], radius=2.5, color=color, fill=True, fill_opacity=0.9, weight=0,
                                tooltip=f"seq {p.seq}, t={p.t:.0f}s, {st}, d={p.align_dist_m:.1f} m").add_to(fg)
        fg.add_to(m)

    g = obs[obs.is_stationary].iloc[::max(1, step * 3)]
    fg = folium.FeatureGroup(name=f"Stationary points ({int(obs.is_stationary.sum()):,})", show=False)
    for _, p in g.iterrows():
        folium.CircleMarker([p.lat, p.lon], radius=4, color="#9467bd", fill=False, weight=1).add_to(fg)
    fg.add_to(m)

    g = obs[obs.gap_before]
    fg = folium.FeatureGroup(name=f"Time gaps ({len(g)})")
    for _, p in g.iterrows():
        folium.Marker([p.lat, p.lon], tooltip=f"gap of {p.dt_prev_s:.0f} s before seq {p.seq}",
                      icon=folium.Icon(color="red", icon="time")).add_to(fg)
    fg.add_to(m)

    # candidates for an evenly spread sample of GPS points (always include some ambiguous ones)
    seqs = obs.seq.to_numpy()
    pick = set(seqs[np.linspace(0, len(seqs) - 1, min(max_obs_cand, len(seqs))).astype(int)])
    amb = obs.seq[obs.obs_status == "ambiguous"].to_numpy()
    pick |= set(amb[np.linspace(0, len(amb) - 1, min(50, len(amb))).astype(int)]) if len(amb) else set()
    c = cand[cand.seq.isin(pick)]
    for lab in (0, -1, -2, 1):
        cc = c[c.label == lab]
        fg = folium.FeatureGroup(name=f"Candidates: {LABEL_NAMES[lab]} ({len(cc):,})", show=lab in (1, -1))
        if len(cc):
            folium.PolyLine(arc_lines(cc.arc_id.to_numpy()), color=LABEL_COLORS[lab],
                            weight=4 if lab == 1 else 2, opacity=0.8).add_to(fg)
        fg.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = f"""<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;
      border:1px solid #999;border-radius:6px;font:12px sans-serif;max-width:260px">
      <b>{trip.partition} trip {trip.trip_id}</b><br>
      GPS: <span style="color:{STATUS_COLORS['aligned']}">&#9679; aligned</span>
      <span style="color:{STATUS_COLORS['ambiguous']}">&#9679; ambiguous</span>
      <span style="color:{STATUS_COLORS['unalignable']}">&#9679; unalignable</span><br>
      <span style="color:#1a9e4a">&#9644;</span> reference route &nbsp;<span style="color:#8c8c8c">&#9644;</span> roads<br>
      Candidates (sample of {len(pick)} points): <span style="color:#1a9e4a">positive</span>,
      <span style="color:#d62728">negative</span>, <span style="color:#e8833a">ambiguous</span><br>
      Red pins = time gaps; purple rings = stationary</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    m.save(str(path))

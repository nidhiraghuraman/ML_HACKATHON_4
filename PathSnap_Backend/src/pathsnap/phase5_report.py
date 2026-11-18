"""Phase 5 maps, plots and report (all numbers read from the saved Phase 5 results)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium

ORDER = ["nearest", "classifier_only", "distance_hmm", "model_hmm"]
NAMES = {"nearest": "Nearest road", "classifier_only": "Classifier only", "distance_hmm": "Distance-only HMM",
         "model_hmm": "Classifier + HMM (PathSnap)"}
COLOR = {"nearest": "#2a78d6", "classifier_only": "#eb6834", "distance_hmm": "#1baf7a", "model_hmm": "#6250d6"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


# ----------------------------------------------------------------------------- maps
def trip_map(res: dict, ref_route, path: Path):
    """Raw GPS, reference route, classifier-only route and HMM route for one trip (layer control)."""
    trip, obs = res["trip"], res["obs"]
    nodes, arcs = trip.nodes, trip.arcs
    lat, lon = obs.lat.to_numpy(), obs.lon.to_numpy()
    m = folium.Map(location=[float(np.median(lat)), float(np.median(lon))], tiles="OpenStreetMap", control_scale=True)
    m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])

    def lines(route):
        return [[[nodes[arcs[a, 0], 1], nodes[arcs[a, 0], 0]], [nodes[arcs[a, 1], 1], nodes[arcs[a, 1], 0]]] for a in route]

    layers = []
    if ref_route is not None:
        layers.append(("Reference route", [int(a) for a in ref_route if 0 <= a < len(arcs)], "#1a9e4a", 8, 0.45, True))
    layers.append(("Classifier-only route", res["methods"]["classifier_only"]["route"], "#eb6834", 3, 0.9, False))
    layers.append(("Predicted route: classifier + HMM", res["methods"]["model_hmm"]["route"], "#6250d6", 4, 0.9, True))
    for name, route, color, w, op, show in layers:
        fg = folium.FeatureGroup(name=f"{name} ({len(route)} arcs)", show=show)
        if route:
            folium.PolyLine(lines(route), color=color, weight=w, opacity=op).add_to(fg)
        fg.add_to(m)
    fg = folium.FeatureGroup(name=f"Raw GPS ({len(obs):,} points)")
    step = max(1, len(obs) // 2500)
    for i in range(0, len(obs), step):
        folium.CircleMarker([lat[i], lon[i]], radius=2, color="#2a78d6", fill=True, fill_opacity=0.8, weight=0,
                            tooltip=f"seq {int(obs.seq.iat[i])}, t={obs.t.iat[i]:.0f}s").add_to(fg)
    fg.add_to(m)
    folium.Marker([lat[0], lon[0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([lat[-1], lon[-1]], tooltip="End", icon=folium.Icon(color="black", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    legend = f"""<div style="position:fixed;bottom:25px;left:25px;z-index:9999;background:white;padding:8px 10px;
      border:1px solid #999;border-radius:6px;font:12px sans-serif"><b>Trip {trip.trip_id}</b><br>
      <span style="color:#2a78d6">&#9679;</span> raw GPS<br><span style="color:#1a9e4a">&#9644;</span> reference route<br>
      <span style="color:#6250d6">&#9644;</span> predicted (classifier + HMM)<br><span style="color:#eb6834">&#9644;</span> classifier only (off by default)</div>"""
    m.get_root().html.add_child(folium.Element(legend))
    m.save(str(path))


# ----------------------------------------------------------------------------- plots
def _clean(ax, grid="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID); ax.set_axisbelow(True)


def plots(out: Path, summary: pd.DataFrame, per_trip: pd.DataFrame, grid: pd.DataFrame, part: str) -> list[str]:
    files = []
    # 1. method comparison
    metrics = [("point_acc", "Point accuracy (aligned GPS points, pooled)"), ("continuity_trip_mean", "Continuity (consecutive points connected)"),
               ("arc_f1_trip_mean", "Route arc F1 (trip mean)"), ("route_mismatch_trip_mean", "Route mismatch (lower is better)")]
    for pt in summary.partition.unique():
        s = summary[summary.partition == pt].set_index("method").reindex(ORDER)
        fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
        for ax, (m, title) in zip(axes, metrics):
            v = s[m].to_numpy()
            ax.barh(range(4)[::-1], v, color=[COLOR[k] for k in ORDER], height=0.6, edgecolor="white", linewidth=2)
            for yi, val in zip(range(4)[::-1], v):
                ax.text(val, yi, f" {val:.4f}", va="center", fontsize=8, color=INK)
            ax.set_yticks(range(4)[::-1], [NAMES[k] for k in ORDER], fontsize=8)
            ax.set_title(title, fontsize=9, loc="left"); ax.set_xlim(0, max(1.0, np.nanmax(v) * 1.15)); _clean(ax)
        fig.suptitle(f"Map-matching methods compared ({'student-test trips' if pt != 'train_oof' else 'training trips, out-of-fold probabilities (used for tuning)'})", fontsize=10)
        fig.tight_layout(); name = f"method_comparison_{pt}.png"; fig.savefig(out / name, dpi=130); plt.close(fig); files.append(name)
    # 2. per-trip point accuracy, classifier-only vs HMM
    p = per_trip[per_trip.partition == part].pivot(index="trip_id", columns="method", values="point_acc").reindex(columns=ORDER)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(p))
    for k in ORDER:
        ax.plot(x, p[k], "o", ms=8, color=COLOR[k], label=NAMES[k], markeredgecolor="white")
    for xi, (a, b) in enumerate(zip(p.classifier_only, p.model_hmm)):
        ax.plot([xi, xi], [a, b], color=MUTED, lw=1, zorder=0)
    ax.set_xticks(x, p.index, rotation=30, fontsize=8)
    ax.set(ylabel="Point accuracy (aligned points)", title=f"Per-trip point accuracy ({part}); grey line = classifier-only -> HMM")
    ax.legend(fontsize=8, frameon=False); _clean(ax, "y")
    fig.tight_layout(); fig.savefig(out / "per_trip_point_accuracy.png", dpi=130); plt.close(fig); files.append("per_trip_point_accuracy.png")
    # 3. tuning grid
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (kind, other) in zip(axes, [("model_hmm", "w_emission"), ("distance_hmm", "sigma_m")]):
        g = grid[grid.method == kind]
        for val, gg in g.groupby(other):
            ax.plot(gg.beta_m, gg.point_acc, "o-", lw=2, label=f"{other} = {val:g}")
        ax.set_xscale("log"); ax.set(xlabel="beta (m), log scale", ylabel="Point accuracy (training, out-of-fold)", title=f"Decoder tuning: {NAMES[kind]}")
        ax.legend(fontsize=8, frameon=False); _clean(ax, "both")
    fig.tight_layout(); fig.savefig(out / "decoder_tuning.png", dpi=130); plt.close(fig); files.append("decoder_tuning.png")
    return files


# ----------------------------------------------------------------------------- report
def write_report(out: Path, cfg: dict):
    c5 = cfg["phase5"]
    summary = pd.read_csv(out / "summary_metrics.csv")
    per_trip = pd.read_csv(out / "per_trip_metrics.csv", dtype={"trip_id": str})
    grid = pd.read_csv(out / "decoder_tuning_grid.csv")
    hp = json.loads((out / "hmm_params.json").read_text(encoding="utf-8"))
    run = json.loads((out / "phase5_run.json").read_text(encoding="utf-8"))
    part = run["partition"]
    timing = pd.read_csv(out / f"timing_{part}.csv", dtype={"trip_id": str})
    files = plots(out, summary, per_trip, grid, part)
    mh, dh = hp["model_hmm"], hp["distance_hmm"]

    L = ["# PathSnap — Phase 5 Report: HMM / Viterbi Route Reconstruction", "",
         f"_Generated by `scripts/run_phase5_hmm.py`; every number comes from this run ({run['elapsed_min']:.1f} min, Python {run['python']}, "
         f"scikit-learn {run['sklearn']}). Emission model: `{run['final_model']}`._", "",
         "## 1. How the decoder works", "",
         f"1. **Hidden states.** For every GPS point, its {int(mh['max_states'])} highest-probability candidate arcs (a beam). Each state is a "
         "directed arc and the position where the GPS point projects onto it. Arc IDs are the trip's own `.arcs` rows.",
         f"2. **Emission.** `log E = w * log p`, where p is the classifier's probability that the candidate is the correct road (tuned w = {mh['w_emission']:g}). "
         f"The distance-only baseline HMM uses `-0.5 (d / sigma)^2` instead (tuned sigma = {dh['sigma_m']:g} m).",
         f"3. **Transition.** `log T(a -> b) = -| route distance - GPS distance | / beta` (tuned beta = {mh['beta_m']:g} m): moving from arc a to arc b "
         "should take about as many metres along the DIRECTED road network as the straight-line GPS movement. One-way roads are respected because "
         "distances come from Dijkstra on the directed arcs.",
         f"4. **Impossible moves** score -inf: target unreachable within the search limit, or needing more than {mh['vmax_mps']:g} m/s "
         f"(+{mh['speed_slack_m']:g} m slack). A backward move of up to {mh['backtrack_tol_m']:g} m on the same arc is treated as standing still (GPS noise).",
         f"5. **Viterbi recurrence.** `delta_i(b) = log E_i(b) + max_a [ delta_(i-1)(a) + log T(a -> b) ]`, keeping the best predecessor of every "
         "state; backtracking from the best final state gives the most probable arc sequence.",
         f"6. **Fallbacks.** A new chain starts (decoding restarts from emissions alone) after a time gap longer than {mh['gap_break_s']:g} s, at a "
         "point with no candidates (it gets no arc), or when every transition is impossible (disconnected roads). Nothing is forced through an "
         "impossible connection.",
         "7. **Route reconstruction.** Consecutive matched arcs are joined by the shortest directed path between them, giving a connected arc "
         "sequence. A route break is counted where no path exists within the limits.", "",
         "The decoder never reads the reference route; it is used only afterwards, server-side, for evaluation.", "",
         "## 2. Decoder tuning (training trips only)", "",
         "Parameters were chosen on the 80 training trips using out-of-fold probabilities: the final model settings were refitted on each "
         "Phase 3 fold and scored that fold's held-out trips, so no trip's probabilities come from a model that saw it. Selection metric: "
         "pooled pointwise accuracy on aligned GPS points. The student-test trips played no part.", ""]
    for kind, pnames in [("model_hmm", ("beta_m", "w_emission")), ("distance_hmm", ("beta_m", "sigma_m"))]:
        g = grid[grid.method == kind].sort_values("point_acc", ascending=False)
        L.append(f"- {NAMES[kind]}: best {', '.join(f'{p} = {g.iloc[0][p]:g}' for p in pnames)} (point accuracy {g.iloc[0].point_acc:.4f}); "
                 f"worst setting in the grid {g.iloc[-1].point_acc:.4f}.")
    L += ["", "Full grid: `decoder_tuning_grid.csv`.", ""]

    def table(pt, title, note):
        s = summary[summary.partition == pt].set_index("method").reindex(ORDER)
        T = [f"## {title}", "", note, "",
             "| Method | Point accuracy | Ambiguous points (lenient) | Continuity | Route arc precision | Route arc recall | Route arc F1 | Length F1 | Route mismatch | Route breaks |",
             "|---|---|---|---|---|---|---|---|---|---|"]
        for m, r in s.iterrows():
            T.append(f"| {NAMES[m]} | {r.point_acc:.4f} | {r.get('amb_lenient_acc', np.nan):.4f} | {r.continuity_trip_mean:.4f} | {r.arc_precision_trip_mean:.4f} | "
                     f"{r.arc_recall_trip_mean:.4f} | {r.arc_f1_trip_mean:.4f} | {r.len_f1_trip_mean:.4f} | {r.route_mismatch_trip_mean:.4f} | {int(r.route_breaks_total)} |")
        return T + [""], s

    n_test = int(summary[summary.partition == part].trips.iloc[0])
    T, s_test = table(part, f"3. Results on the {part} trips (main result)",
                      f"{n_test} trips never used for training, model selection or decoder "
                      "tuning, processed through the same inference pipeline the API uses. Point accuracy is pooled over aligned points; the other "
                      "columns are means over trips.")
    L += T
    d_acc = s_test.loc["model_hmm", "point_acc"] - s_test.loc["classifier_only", "point_acc"]
    d_f1 = s_test.loc["model_hmm", "arc_f1_trip_mean"] - s_test.loc["classifier_only", "arc_f1_trip_mean"]
    d_vs_dist = s_test.loc["model_hmm", "point_acc"] - s_test.loc["distance_hmm", "point_acc"]
    pt = per_trip[per_trip.partition == part].pivot(index="trip_id", columns="method", values="point_acc")
    better = int((pt.model_hmm > pt.classifier_only).sum()); worse = int((pt.model_hmm < pt.classifier_only).sum())
    L += [f"**Did HMM decoding help?** On {part}, point accuracy changes by {100 * d_acc:+.2f} pp (classifier only → classifier + HMM) and route arc F1 "
          f"by {100 * d_f1:+.2f} pp; per trip the HMM is better on {better}, worse on {worse} of {len(pt)} trips. Compared with the distance-only HMM, "
          f"using the classifier as the emission changes point accuracy by {100 * d_vs_dist:+.2f} pp.", ""]
    worst = per_trip[(per_trip.partition == part) & (per_trip.method == "model_hmm")].sort_values("point_acc").head(3)
    L += ["Trips where the final system is weakest (point accuracy): " + "; ".join(f"{r.trip_id}: {r.point_acc:.4f} (route F1 {r.arc_f1:.4f}, "
          f"{int(r.route_breaks)} breaks)" for r in worst.itertuples()) + ".", ""]
    T, _ = table("train_oof", "4. Results on the training trips (out-of-fold probabilities; also the tuning set)",
                 "These trips were used to choose beta / w / sigma, so these numbers are slightly optimistic for the decoder; the model probabilities are out-of-fold.")
    L += T
    L += ["## 5. Runtime (inference pipeline, student-test trips)", "",
          f"Mean per trip: {timing.points.mean():.0f} GPS points; preprocessing + candidates + features {timing.preprocess_candidates_features_s.mean():.2f} s, "
          f"model {timing.model_s.mean():.3f} s, HMM decoding {timing.model_hmm_decode_s.mean():.2f} s. "
          f"Per 1,000 GPS points end to end: {1000 * timing.total_s.sum() / timing.points.sum():.2f} s (all methods were computed in these runs).", "",
          "## 6. Metric definitions", "",
          "- **Point accuracy**: share of aligned GPS points whose matched arc equals the arc assigned by the Phase 2 route alignment.",
          "- **Ambiguous points (lenient)**: for points near a junction, correct if the matched arc is the aligned arc or the equally close neighbouring route arc.",
          "- **Continuity**: share of consecutive point pairs (gap <= limit) whose matched arcs are the same or connected by a feasible directed path.",
          "- **Route arc precision / recall / F1**: predicted route arcs vs reference `.route` arcs (sets); length F1 weights arcs by length.",
          "- **Route mismatch** (Newson & Krumm 2009): (length wrongly added + length missed) / reference length; 0 is perfect.", "",
          "## 7. Limitations", "",
          "- Point accuracy is measured against the Phase 2 alignment, which treats the reference route as correct.",
          "- Route metrics compare arc sets: a route that visits the right arcs in a wrong order would not be penalised.",
          f"- The beam keeps {int(mh['max_states'])} states per point; a correct arc outside the beam cannot be chosen.",
          "- The evaluator-test trips are not available; their results will be produced by the API on the day.", "",
          "## 8. Files", "",
          f"`results/phase5/`: `hmm_params.json` (tuned decoder), `decoder_tuning_grid.csv`, `per_trip_metrics.csv`, `summary_metrics.csv`, "
          f"`predictions_{part}.csv` (matched arc per point and method), `timing_{part}.csv`, `maps/trip_*.html`, plots: "
          + ", ".join(f"`{f}`" for f in files) + ".", ""]
    (out / "phase5_report.md").write_text("\n".join(L), encoding="utf-8")

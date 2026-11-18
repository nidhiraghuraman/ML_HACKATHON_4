"""Phase 4 plots. Each figure answers one question (see its title)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
ERR_COLORS = {"reverse_direction": ORANGE, "adjacent_arc": AQUA, "other_road": MAGENTA}


def _clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID); ax.set_axisbelow(True)


def _save(fig, out: Path, name: str, files: list):
    fig.tight_layout(); fig.savefig(out / name, dpi=130, facecolor="white"); plt.close(fig); files.append(name)


def optimisation_history(trials: pd.DataFrame, out: Path, files: list):
    """Q: did the Bayesian search find better settings as trials went on?"""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    done = trials[trials.state == "COMPLETE"]
    pruned = trials[trials.state == "PRUNED"]
    ax.plot(done.trial, done.value_top1, "o", ms=8, color=BLUE, markeredgecolor="white", label="completed trial (mean Top-1 over search folds)")
    if len(pruned) and len(done):
        ax.plot(pruned.trial, np.full(len(pruned), done.value_top1.min()), "x", ms=7, color=MUTED, label="pruned trial (stopped early; drawn at bottom)")
    ax.step(done.trial, done.value_top1.cummax(), where="post", color=ORANGE, lw=2, label="best so far")
    if 0 in set(done.trial):
        ax.axhline(done.loc[done.trial == 0, "value_top1"].iat[0], color=INK, ls=":", lw=1, label="trial 0 = untuned Phase 3 settings")
    ax.set(xlabel="Trial number", ylabel="Top-1 (search folds)", title="Bayesian optimisation history (Optuna TPE)")
    ax.legend(fontsize=8, frameon=False, loc="lower right"); _clean(ax)
    _save(fig, out, "optimisation_history.png", files)


def param_effects(trials: pd.DataFrame, space: dict, out: Path, files: list):
    """Q: which hyperparameter values gave good Top-1 in the search?"""
    done = trials[trials.state == "COMPLETE"]
    params = [p for p in space if f"param_{p}" in done]
    if not params or done.empty:
        return
    n = len(params); cols = min(3, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.4 * rows), squeeze=False)
    best = done.value_top1.idxmax()
    for ax, p in zip(axes.ravel(), params):
        ax.scatter(done[f"param_{p}"], done.value_top1, s=40, color=BLUE, edgecolor="white")
        ax.scatter([done.loc[best, f"param_{p}"]], [done.loc[best, "value_top1"]], s=110, facecolor="none", edgecolor=ORANGE, lw=2, label="best")
        if space[p].get("log"):
            ax.set_xscale("log")
        ax.set(xlabel=p, ylabel="Top-1"); _clean(ax, "both")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle("Search trials: Top-1 against each hyperparameter (orange ring = best trial)", fontsize=10)
    _save(fig, out, "param_effects.png", files)


def tuned_vs_untuned(comp: pd.DataFrame, out: Path, files: list):
    """Q: on the same Phase 3 folds, is the tuned model better, and in every fold?"""
    metrics = [("top1", "Top-1 (aligned GPS points)"), ("macro_f1", "Macro F1 (candidates, threshold 0.5)"), ("log_loss", "Log loss (lower is better)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    u = comp[comp.model == "untuned"].set_index("fold"); t = comp[comp.model == "tuned"].set_index("fold")
    for ax, (m, title) in zip(axes, metrics):
        for f in u.index:
            ax.plot([0, 1], [u.loc[f, m], t.loc[f, m]], color=GRID, lw=1.5, zorder=1)
            ax.text(1.04, t.loc[f, m], f"fold {f}", fontsize=7, color=MUTED, va="center")
        ax.scatter(np.zeros(len(u)), u[m], s=60, color=BLUE, edgecolor="white", zorder=2, label="untuned")
        ax.scatter(np.ones(len(t)), t[m], s=60, color=ORANGE, edgecolor="white", zorder=2, label="tuned")
        ax.hlines([u[m].mean()], -0.15, 0.15, color=INK, lw=2); ax.hlines([t[m].mean()], 0.85, 1.15, color=INK, lw=2)
        ax.set_xticks([0, 1], [f"untuned\nmean {u[m].mean():.4f}", f"tuned\nmean {t[m].mean():.4f}"])
        ax.set_xlim(-0.4, 1.4); ax.set_title(title, fontsize=10); _clean(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Tuned vs untuned, paired by fold (same trips held out; lines join the same fold)", fontsize=10)
    _save(fig, out, "tuned_vs_untuned.png", files)


def reliability(calib: dict, out: Path, files: list):
    """Q: are the predicted probabilities honest (needed for the HMM in Phase 5)?"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    colors = {"untuned": BLUE, "tuned": ORANGE}
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1, label="perfect calibration")
    for tag, cs in calib.items():
        tb = cs["table"].dropna()
        ax.plot(tb.mean_pred, tb.observed, "o-", color=colors.get(tag, INK), lw=2, ms=6, label=f"{tag} (ECE {cs['ece']:.4f}, Brier {cs['brier']:.4f})")
    ax.set(xlabel="Mean predicted probability in bin", ylabel="Observed share of correct candidates", title="Reliability diagram (out-of-fold, candidates 0/1)")
    ax.legend(fontsize=8, frameon=False, loc="upper left"); _clean(ax, "both")
    ax = axes[1]
    tag = "tuned" if "tuned" in calib else "untuned"
    tb = calib[tag]["table"]
    ax.bar(tb.bin, tb.n, color=colors[tag], width=0.8)
    ax.set_yscale("log")
    ax.set_xticks(tb.bin, [f"{lo:.1f}-{hi:.1f}" for lo, hi in zip(tb.lo, tb.hi)], rotation=45, fontsize=8)
    ax.set(ylabel="Candidates (log scale)", title=f"Candidates per probability bin ({tag})"); _clean(ax)
    _save(fig, out, "reliability.png", files)


def error_plots(err: pd.DataFrame, slices: pd.DataFrame, out: Path, files: list):
    """Q: what kinds of mistakes remain, and where do they concentrate?"""
    wrong = err[err.error_type != "correct"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1, 2]})
    ax = axes[0]
    vc = wrong.error_type.value_counts().reindex(["reverse_direction", "adjacent_arc", "other_road"]).fillna(0)
    ax.bar(range(3), vc.values, color=[ERR_COLORS[k] for k in vc.index], width=0.6)
    for i, v in enumerate(vc.values):
        ax.text(i, v, f"{int(v):,}\n({v / max(len(wrong), 1):.0%})", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(3), ["wrong direction\n(same street)", "neighbouring arc\n(shares a node)", "different road"])
    ax.set(ylabel="GPS points with a wrong Top-1", title=f"Error types ({len(wrong):,} of {len(err):,} aligned points wrong)"); _clean(ax)
    ax = axes[1]
    s = slices[slices.slice != "all points"].copy()
    s["label"] = s.slice + ": " + s.group
    s = s.iloc[::-1]
    overall = slices[slices.slice == "all points"].top1_error_rate.iat[0]
    ax.barh(s.label, s.top1_error_rate * 100, color=BLUE, height=0.6)
    for yi, (v, n) in enumerate(zip(s.top1_error_rate, s.points)):
        ax.text(v * 100, yi, f"  {v:.1%}  (n={n:,})", va="center", fontsize=7)
    ax.axvline(overall * 100, color=ORANGE, lw=1.5, ls="--", label=f"all points: {overall:.1%}")
    ax.set(xlabel="Top-1 error rate (%)", title="Error rate by condition (out-of-fold)"); ax.tick_params(axis="y", labelsize=7)
    ax.legend(fontsize=8, frameon=False, loc="lower right"); _clean(ax, "x")
    _save(fig, out, "error_analysis.png", files)

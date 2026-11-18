"""Phase 3 plots. Each figure answers one question (see its title).

Colour = model identity, fixed order (never re-assigned by rank). Bars carry the model name on
the axis and the value as text, so identity never depends on colour alone.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

# validated categorical palette (light): blue, orange, aqua, yellow, magenta, green, violet
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
ORDER = ["baseline_nearest", "baseline_formula", "logistic_regression", "knn", "decision_tree",
         "random_forest", "hist_gradient_boosting"]
LABELS = {"baseline_nearest": "Nearest road (baseline)", "baseline_formula": "Distance+heading (baseline)",
          "logistic_regression": "Logistic Regression", "knn": "KNN (subset)", "decision_tree": "Decision Tree",
          "random_forest": "Random Forest", "hist_gradient_boosting": "Hist. Gradient Boosting"}
COLOR = dict(zip(ORDER, PALETTE))
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _save(fig, out: Path, name: str, files: list):
    fig.tight_layout()
    fig.savefig(out / name, dpi=130, facecolor="white")
    plt.close(fig)
    files.append(name)


def _hbar(ax, summary: pd.DataFrame, metric: str, title: str, xlabel: str, selected: str | None, xlim=None):
    s = summary[summary.model.isin(ORDER)].copy()
    s["o"] = s.model.map({m: i for i, m in enumerate(ORDER)})
    s = s.sort_values("o", ascending=False)
    y = np.arange(len(s))
    means = s[f"{metric}_mean"].to_numpy()
    stds = s[f"{metric}_std"].fillna(0).to_numpy()
    ok = np.isfinite(means)
    ax.barh(y[ok], means[ok], height=0.6, color=[COLOR[m] for m in s.model[ok]], edgecolor="white", linewidth=2)
    ax.errorbar(means[ok], y[ok], xerr=stds[ok], fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    for yi, m, sd, fin in zip(y, means, stds, ok):
        ax.text((m + sd) if fin else 0, yi, f"  {m:.4f}" if fin else "  n/a", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, [LABELS[m] + ("  ★" if m == selected else "") for m in s.model], fontsize=9)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if xlim:
        ax.set_xlim(*xlim)
    _style(ax)


def comparison_plots(summary: pd.DataFrame, selected: str, out: Path, files: list):
    """Q: which model ranks the right road first most often, and does it beat the baselines?"""
    lo = np.nanmin(summary["top1_mean"]) - 0.05
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    _hbar(axes[0, 0], summary, "top1", "Top-1 accuracy (GPS points, mean ± sd over 5 trip folds)", "share of aligned GPS points", selected, (max(lo, 0), 1.0))
    _hbar(axes[0, 1], summary, "top3", "Top-3 accuracy", "share of aligned GPS points", selected, (max(lo, 0), 1.0))
    _hbar(axes[1, 0], summary, "mrr", "Mean reciprocal rank", "MRR (1 = always ranked first)", selected, (max(lo, 0), 1.0))
    _hbar(axes[1, 1], summary, "f1", "Candidate-level F1 (positive class, threshold 0.5)", "F1", selected, (0, 1.0))
    fig.suptitle("Phase 3 model comparison (★ = selected); baselines have no threshold, F1 uses their top-1 pick", fontsize=11, color=INK)
    _save(fig, out, "model_comparison.png", files)


def curves_plot(oof: pd.DataFrame, models: list[str], out: Path, files: list):
    """Q: how well does each score separate positive from negative candidates at every threshold?"""
    d = oof[oof.label.isin([0, 1])]
    y = d.label.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for m in [x for x in ORDER if x in models]:
        s = d[m].to_numpy()
        p, r, _ = precision_recall_curve(y, s)
        axes[0].plot(r, p, color=COLOR[m], lw=2, label=LABELS[m])
        fpr, tpr, _ = roc_curve(y, s)
        axes[1].plot(fpr, tpr, color=COLOR[m], lw=2, label=LABELS[m])
    axes[0].axhline(y.mean(), color=MUTED, ls=":", lw=1, label=f"chance (positive rate {y.mean():.3f})")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (out-of-fold, all 80 trips)")
    axes[1].plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1, label="chance")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (out-of-fold, all 80 trips)")
    for ax in axes:
        ax.grid(color=GRID)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    _save(fig, out, "pr_roc_curves.png", files)


def fold_plot(folds: pd.DataFrame, out: Path, files: list):
    """Q: are the results stable across different sets of held-out trips?"""
    fig, ax = plt.subplots(figsize=(10, 5))
    models = [m for m in ORDER if m in set(folds.model)]
    for i, m in enumerate(models):
        v = folds[folds.model == m].sort_values("fold")
        x = np.full(len(v), i) + np.linspace(-0.2, 0.2, len(v))
        ax.plot(x, v.top1, "o", ms=8, color=COLOR[m], markeredgecolor="white", markeredgewidth=1.5)
        for xi, f, val in zip(x, v.fold, v.top1):
            ax.text(xi, val, f" {f}", fontsize=6, color=MUTED, va="center")
        ax.hlines(v.top1.mean(), i - 0.3, i + 0.3, color=INK, lw=1.5)
    ax.set_xticks(range(len(models)), [LABELS[m] for m in models], rotation=20, ha="right", fontsize=9)
    ax.set(ylabel="Top-1 accuracy on the fold's held-out trips", title="Top-1 per validation fold (dots, labelled with fold number; line = mean)")
    ax.grid(axis="y", color=GRID)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, out, "fold_variability.png", files)


def trip_plot(per_trip: pd.DataFrame, out: Path, files: list):
    """Q: is performance consistent across trips, or carried by a few large ones?"""
    models = [m for m in ORDER if m in per_trip.columns]
    fig, ax = plt.subplots(figsize=(11, 5))
    data = [per_trip[m].dropna().to_numpy() for m in models]
    bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.5, showfliers=True,
                    medianprops={"color": INK, "lw": 1.5}, flierprops={"marker": "o", "ms": 4, "markerfacecolor": MUTED, "markeredgecolor": "white"})
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(COLOR[m]); patch.set_alpha(0.85); patch.set_edgecolor("white")
    ax.set_xticks(range(1, len(models) + 1), [LABELS[m] for m in models], rotation=20, ha="right", fontsize=9)
    ax.set(ylabel="Top-1 accuracy within the trip", title="Top-1 per trip (80 trips, each scored when held out); box = middle 50%, dots = outlier trips")
    ax.grid(axis="y", color=GRID)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _save(fig, out, "trip_level_top1.png", files)


def confusion_plot(cm: dict, model: str, threshold: float, out: Path, files: list):
    """Q: what kinds of mistakes does the selected model make at the 0.5 threshold?"""
    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(m / m.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            frac = m[i, j] / m[i].sum()
            ax.text(j, i, f"{m[i, j]:,}\n({frac:.1%} of row)", ha="center", va="center",
                    color="white" if frac > 0.5 else INK, fontsize=10)
    ax.set_xticks([0, 1], ["predicted wrong road (0)", "predicted correct road (1)"], fontsize=8)
    ax.set_yticks([0, 1], ["truly wrong (0)", "truly correct (1)"], fontsize=8)
    ax.set_title(f"{LABELS[model]}: confusion matrix\n(out-of-fold, threshold {threshold})", fontsize=10)
    _save(fig, out, "confusion_matrix_selected.png", files)


def importance_plot(perm: pd.DataFrame, model: str, coef: pd.DataFrame | None, out: Path, files: list):
    """Q: which features does the selected model rely on (held-out permutation), and what does
    the linear model say for comparison?"""
    n = 2 if coef is not None else 1
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 7))
    axes = np.atleast_1d(axes)
    p = perm.sort_values("top1_drop_mean")
    axes[0].barh(p.feature, p.top1_drop_mean * 100, xerr=p.top1_drop_std * 100, color=COLOR[model], height=0.6,
                 error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
    axes[0].axvline(0, color=MUTED, lw=1)
    axes[0].set(xlabel="Drop in Top-1 when the feature is shuffled (percentage points)",
                title=f"{LABELS[model]}: permutation importance\n(held-out folds, mean ± sd over 5 folds)")
    _style(axes[0])
    if coef is not None:
        c = coef.reindex(coef.coef.abs().sort_values().index)
        axes[1].barh(c.feature, c.coef, color=[COLOR["logistic_regression"] if v > 0 else COLOR["baseline_formula"] for v in c.coef], height=0.6)
        axes[1].axvline(0, color=MUTED, lw=1)
        axes[1].set(xlabel="Coefficient on standardised feature (+ = more likely correct road)",
                    title="Logistic Regression coefficients\n(blue = raises, orange = lowers the score)")
        _style(axes[1])
    _save(fig, out, "feature_importance.png", files)


def runtime_plot(rt: pd.DataFrame, out: Path, files: list):
    """Q: what does each model cost to train and to run?"""
    r = rt[rt.model.isin(ORDER[2:])].copy()
    r["o"] = r.model.map({m: i for i, m in enumerate(ORDER)})
    r = r.sort_values("o", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, col, title in [(axes[0], "fit_s_mean", "Training time per fold (s)"),
                           (axes[1], "predict_s_per_1k_obs_mean", "Prediction time per 1,000 GPS points (s)")]:
        ax.barh(r.model.map(LABELS), r[col], color=[COLOR[m] for m in r.model], height=0.6, edgecolor="white", linewidth=2)
        for yi, v in enumerate(r[col]):
            ax.text(v, yi, f"  {v:.2f}", va="center", fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, loc="left"); _style(ax)
    fig.suptitle("Runtime (this machine; KNN trained on a subset)", fontsize=10)
    _save(fig, out, "runtime.png", files)


def student_plot(st: pd.DataFrame, selected: str, out: Path, files: list):
    """Q: (independent check, not used for selection) how do models do on the 10 student-test trips?"""
    s = st.rename(columns={c: f"{c}_mean" for c in ["top1", "macro_f1"]})
    for c in ["top1", "macro_f1"]:
        s[f"{c}_std"] = np.nan
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    lo = np.nanmin(s["top1_mean"]) - 0.05
    _hbar(axes[0], s, "top1", "Student-test Top-1 (10 trips, models fitted on all 80 train trips)", "share of aligned GPS points", selected, (max(lo, 0), 1.0))
    _hbar(axes[1], s, "macro_f1", "Student-test macro F1 (candidate level)", "macro F1", selected, (0, 1.0))
    fig.suptitle("Independent student-test check — computed AFTER the selection was saved; not used to choose", fontsize=10)
    _save(fig, out, "student_test_comparison.png", files)

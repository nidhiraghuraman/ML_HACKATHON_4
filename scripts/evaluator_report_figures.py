"""Figures for deliverables/EVALUATOR_EVALUATION_REPORT.md from results/evaluator/{baseline,final}/ (saved outputs)."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
R = Path(__file__).resolve().parents[1] / "results" / "evaluator"
out = R / "figures"; out.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e3e7ec", "axes.axisbelow": True})
b = pd.read_csv(R / "baseline/per_case.csv", dtype={"trip_id": str}).set_index("trip_id")
f = pd.read_csv(R / "final/per_case.csv", dtype={"trip_id": str}).set_index("trip_id")
fig, ax = plt.subplots(figsize=(10.5, 3.6)); x = np.arange(len(f)); w = 0.2
for j, (s, col, lab, c) in enumerate([(b, "classifier_only__point_acc", "baseline: classifier only", "#f2b48a"), (b, "model_hmm__point_acc", "baseline: classifier + HMM", "#b9a3d6"),
                                       (f, "classifier_only__point_acc", "final: classifier only", "#e07a2d"), (f, "model_hmm__point_acc", "final: classifier + HMM", "#6a3d9a")]):
    ax.bar(x + (j - 1.5) * w, s[col], w, color=c, label=lab)
ax.axhline(0.95, color="#c0392b", lw=0.8, ls="--"); ax.text(len(f) - 0.5, 0.951, "0.95", color="#c0392b", fontsize=8, ha="right", va="bottom")
ax.set_xticks(x); ax.set_xticklabels([f"{t}\n({n:,} pts)" for t, n in zip(f.index, f.aligned)], fontsize=7)
ax.set(ylim=(0.78, 1.005), ylabel="Point accuracy (aligned points)", title="Evaluator cases: exact arc-level point accuracy, baseline vs final pipeline")
ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2)); fig.tight_layout(); fig.savefig(out / "per_case_point_accuracy.png", dpi=170)
e = pd.concat([pd.read_csv(R / f"{t}/errors.csv").assign(run=t) for t in ("baseline", "final")])
cats = ["direction (reverse twin)", "adjacent arc (shares a node)", "other road", "beam miss (reference not in HMM top-8)", "candidate miss (reference arc not generated)"]
tab = e.groupby(["run", "method", "type"]).size().unstack(fill_value=0).reindex(columns=cats, fill_value=0)
fig, ax = plt.subplots(figsize=(10.5, 2.8)); left = np.zeros(4)
rows = [("baseline", "classifier_only"), ("baseline", "model_hmm"), ("final", "classifier_only"), ("final", "model_hmm")]
cols = ["#e07a2d", "#4c78a8", "#8794a4", "#59a14f", "#c0392b"]
for k, c in enumerate(cats):
    v = np.array([tab.loc[r, c] if r in tab.index else 0 for r in rows])
    ax.barh(range(4), v, left=left, color=cols[k], label=c, height=0.6); left += v
for i, tot in enumerate(left): ax.text(tot + 15, i, f"{int(tot)}", va="center", fontsize=8)
ax.set_yticks(range(4)); ax.set_yticklabels([f"{r}: {m.replace('_', ' ')}" for r, m in rows]); ax.invert_yaxis()
ax.set(xlabel="Wrong aligned points (of 15,542)", title="Error types on the evaluator cases"); ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.3))
fig.tight_layout(); fig.savefig(out / "error_types.png", dpi=170); print("ok")

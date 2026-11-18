import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BarChart3 } from "lucide-react";
import { api } from "../api/client";
import type { Evaluation as Ev } from "../api/types";
import { useApp } from "../state/AppContext";
import { Alert, Card, Empty, Spinner, Stat, MODEL_NAMES, num, pct } from "../components/ui";

const tip = { contentStyle: { background: "#0a0f1e", border: "1px solid rgba(255,255,255,.1)", borderRadius: 12, fontSize: 12 } };
const C = ["#22d3ee", "#a78bfa", "#f472b6", "#34d399"];

export default function Evaluation() {
  const { conn } = useApp();
  const [ev, setEv] = useState<Ev | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [session, setSession] = useState<{ evaluated_trips: number; mean: Record<string, number> } | null>(null);
  useEffect(() => {
    if (conn !== "online") return;
    api.evaluation().then(setEv).catch(e => setErr(e.message));
    api.metrics().then(setSession).catch(() => null);
  }, [conn]);
  if (conn !== "online") return <Card><Empty icon={<BarChart3 />} title="Backend not connected" /></Card>;
  if (err) return <Alert kind="error">{err}</Alert>;
  if (!ev) return <Spinner label="Loading evaluation results…" />;

  const models = (ev.models ?? []).map((m: any) => ({ ...m, name: MODEL_NAMES[m.model] ?? m.model }));
  const hmm = (ev.hmm?.methods ?? []).filter((m: any) => m.partition !== "train_oof").map((m: any) => ({ ...m, name: MODEL_NAMES[m.method] ?? m.method }));
  const perTrip: any[] = ev.hmm?.per_trip ?? [];
  const trips = [...new Set(perTrip.map(r => r.trip_id))];
  const tripRows = trips.map(t => {
    const row: any = { trip: t };
    perTrip.filter(r => r.trip_id === t).forEach(r => { row[r.method] = r.point_acc; });
    return row;
  });
  const sel = ev.selection ?? {};
  return (
    <div className="space-y-6">
      <div><h1 className="text-xl font-semibold text-white">Evaluation dashboard</h1>
        <p className="text-sm text-slate-400">All numbers come from the saved experiments (Phases 3-5): trip-grouped cross-validation on 80 training trips, and the 10 student-test trips.</p></div>

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Selected model (Phase 3)" value={<span className="text-lg">{MODEL_NAMES[sel.selected] ?? sel.selected}</span>} hint={`CV Top-1 ${pct(sel.selected_top1_mean)} ± ${pct(sel.selected_top1_std)}`} accent />
        <Stat label="After Bayesian tuning" value={pct(ev.tuning?.tuned_cv_top1)} hint={`CV Top-1; untuned ${pct(ev.tuning?.untuned_cv_top1)} · ${ev.tuning?.adopted ? "adopted" : "not adopted"}`} />
        <Stat label="HMM point accuracy (test)" value={pct(hmm.find((h: any) => h.method === "model_hmm")?.point_acc)} hint={`classifier only ${pct(hmm.find((h: any) => h.method === "classifier_only")?.point_acc)}`} />
        <Stat label="Calibration (ECE)" value={num(ev.calibration?.ece, 4)} hint={ev.calibration?.recalibration_recommended ? "recalibration recommended" : "probabilities usable as HMM emissions"} />
      </div>

      <Card title="Model comparison — 5-fold trip-grouped CV" subtitle="Top-1 = correct road ranked first; MRR = mean reciprocal rank; PR-AUC = ranking quality across thresholds">
        <div className="h-72"><ResponsiveContainer><BarChart data={models} margin={{ left: 0, right: 10 }}>
          <CartesianGrid stroke="rgba(255,255,255,.05)" /><XAxis dataKey="name" stroke="#64748b" fontSize={11} interval={0} angle={-12} textAnchor="end" height={50} />
          <YAxis domain={[0.4, 1]} stroke="#64748b" fontSize={11} /><Tooltip {...tip} formatter={(v: any) => num(v, 4)} /><Legend />
          <Bar dataKey="top1" name="Top-1" fill={C[0]} radius={[4, 4, 0, 0]} /><Bar dataKey="mrr" name="MRR" fill={C[1]} radius={[4, 4, 0, 0]} />
          <Bar dataKey="pr_auc" name="PR-AUC" fill={C[2]} radius={[4, 4, 0, 0]} />
        </BarChart></ResponsiveContainer></div>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="text-slate-400"><tr>{["Model", "Accuracy", "Precision", "Recall", "F1", "Macro F1", "ROC-AUC", "PR-AUC", "Log loss", "Top-1", "Top-3", "Top-5", "MRR", "Fit s/fold", "Predict s/1k pts"].map(h => <th key={h} className="px-2 py-2 font-medium">{h}</th>)}</tr></thead>
            <tbody className="font-mono">{models.map((m: any) => (
              <tr key={m.model} className={`border-t border-white/5 ${m.model === sel.selected ? "bg-cyan-400/5 text-cyan-100" : "text-slate-300"}`}>
                <td className="px-2 py-1.5 font-sans">{m.name}</td>{["accuracy", "precision", "recall", "f1", "macro_f1", "roc_auc", "pr_auc", "log_loss", "top1", "top3", "top5", "mrr"].map(k => <td key={k} className="px-2">{num(m[k], 4)}</td>)}
                <td className="px-2">{num(m.fit_s, 1)}</td><td className="px-2">{num(m.predict_s_per_1k_obs, 3)}</td>
              </tr>))}</tbody>
          </table>
          <p className="mt-2 text-[11px] text-slate-500">Baselines produce rankings, not probabilities: their F1 uses "top candidate per point" and log loss is not applicable.</p>
        </div>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Sequence decoding — student-test trips" subtitle="Does the HMM produce better, more coherent routes than independent per-point choices?">
          <div className="h-72"><ResponsiveContainer><BarChart data={hmm}>
            <CartesianGrid stroke="rgba(255,255,255,.05)" /><XAxis dataKey="name" stroke="#64748b" fontSize={11} /><YAxis domain={[0.4, 1]} stroke="#64748b" fontSize={11} />
            <Tooltip {...tip} formatter={(v: any) => num(v, 4)} /><Legend />
            <Bar dataKey="point_acc" name="Point accuracy" fill={C[0]} radius={[4, 4, 0, 0]} /><Bar dataKey="continuity_trip_mean" name="Continuity" fill={C[1]} radius={[4, 4, 0, 0]} />
            <Bar dataKey="arc_f1_trip_mean" name="Route F1" fill={C[2]} radius={[4, 4, 0, 0]} />
          </BarChart></ResponsiveContainer></div>
        </Card>
        <Card title="Per-trip point accuracy — student-test" subtitle="Classifier only vs classifier + HMM">
          <div className="h-72"><ResponsiveContainer><BarChart data={tripRows}>
            <CartesianGrid stroke="rgba(255,255,255,.05)" /><XAxis dataKey="trip" stroke="#64748b" fontSize={10} /><YAxis domain={[0.4, 1]} stroke="#64748b" fontSize={11} />
            <Tooltip {...tip} formatter={(v: any) => num(v, 4)} /><Legend />
            <Bar dataKey="classifier_only" name="Classifier only" fill="#f59e0b" radius={[4, 4, 0, 0]} /><Bar dataKey="model_hmm" name="Classifier + HMM" fill="#e879f9" radius={[4, 4, 0, 0]} />
          </BarChart></ResponsiveContainer></div>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card title="Bayesian hyperparameter optimisation" subtitle={`Optuna TPE · ${ev.tuning?.n_trials ?? "–"} trials (${ev.tuning?.pruned ?? 0} pruned) · ${num(ev.tuning?.search_minutes, 1)} min`}>
          <div className="grid grid-cols-2 gap-1.5 font-mono text-xs">
            {Object.entries(ev.tuning?.best_params ?? {}).map(([k, v]: any) => <div key={k} className="rounded-lg bg-ink-850 px-2.5 py-1.5"><span className="text-slate-500">{k}</span> <span className="text-cyan-200">{typeof v === "number" ? +v.toFixed(4) : String(v)}</span></div>)}
          </div>
          <table className="mt-4 w-full text-xs"><thead className="text-slate-400"><tr><th className="text-left">Fold</th><th>Untuned Top-1</th><th>Tuned Top-1</th></tr></thead>
            <tbody className="font-mono text-slate-300">{(ev.tuning?.folds ?? []).map((f: any) => <tr key={f.fold} className="border-t border-white/5 text-center"><td className="py-1 text-left">{f.fold}</td><td>{num(f.untuned_top1, 4)}</td><td>{num(f.tuned_top1, 4)}</td></tr>)}</tbody></table>
        </Card>
        <Card title="Live session" subtitle="Metrics computed by the server for trips matched since it started">
          {session && session.evaluated_trips ? (
            <div className="grid grid-cols-2 gap-3">
              <Stat label="Trips evaluated" value={session.evaluated_trips} />
              <Stat label="Mean point accuracy" value={pct(session.mean.point_acc)} />
              <Stat label="Mean route F1" value={pct(session.mean.arc_f1)} />
              <Stat label="Mean continuity" value={pct(session.mean.continuity)} />
            </div>
          ) : <Empty icon={<BarChart3 />} title="No trips evaluated in this session yet">Match trips from the Dashboard or Map page.</Empty>}
        </Card>
      </div>
    </div>
  );
}

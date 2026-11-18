import { useEffect, useMemo, useState } from "react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Brain } from "lucide-react";
import { api } from "../api/client";
import type { ModelInfo } from "../api/types";
import { useApp } from "../state/AppContext";
import { Card, Empty, Stat, num, pct , MODEL_NAMES } from "../components/ui";

const tip = { contentStyle: { background: "#0a0f1e", border: "1px solid rgba(255,255,255,.1)", borderRadius: 12, fontSize: 12 } };

export default function Analysis() {
  const { conn, result } = useApp();
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [seq, setSeq] = useState(0);
  useEffect(() => { if (conn === "online") api.model().then(setModel).catch(() => setModel(null)); }, [conn]);
  const pts = result?.points ?? [];

  const hist = useMemo(() => {
    const b = Array.from({ length: 10 }, (_, i) => ({ bin: `${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}`, points: 0 }));
    pts.forEach(p => { if (p.prob !== null) b[Math.min(9, Math.floor(p.prob * 10))].points++; });
    return b;
  }, [pts]);
  const series = useMemo(() => {
    const step = Math.max(1, Math.floor(pts.length / 600));
    return pts.filter((_, i) => i % step === 0).map(p => ({ seq: p.seq, prob: p.prob ?? 0, changed: p.arc_id !== p.classifier_arc_id ? 1 : 0 }));
  }, [pts]);
  const p = pts.find(x => x.seq === seq) ?? pts[0];

  return (
    <div className="grid gap-6 xl:grid-cols-3">
      <Card title="Selected model" subtitle={model?.source ?? "final model loaded by the backend"}>
        {model ? (
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <Stat label="Estimator" value={<span className="text-base break-words">{MODEL_NAMES[model.family] ?? model.estimator}</span>} hint={model.estimator} accent />
              <Stat label="Features" value={model.features.length} hint="inference-time only" />
            </div>
            <div>
              <div className="mb-2 text-xs uppercase tracking-widest text-slate-400">Hyperparameters</div>
              <div className="grid grid-cols-2 gap-1.5 font-mono text-xs">
                {Object.entries(model.params ?? {}).map(([k, v]) => <div key={k} className="rounded-lg bg-ink-850 px-2.5 py-1.5"><span className="text-slate-500">{k}</span> <span className="text-cyan-200">{typeof v === "number" ? +v.toFixed(4) : String(v)}</span></div>)}
              </div>
            </div>
            <div>
              <div className="mb-2 text-xs uppercase tracking-widest text-slate-400">HMM / Viterbi decoder</div>
              <div className="grid grid-cols-2 gap-1.5 font-mono text-xs">
                {Object.entries(model.decoder).map(([k, v]) => <div key={k} className="rounded-lg bg-ink-850 px-2.5 py-1.5"><span className="text-slate-500">{k}</span> <span className="text-fuchsia-200">{v}</span></div>)}
              </div>
            </div>
            <details className="text-xs text-slate-400"><summary className="cursor-pointer text-slate-300">Feature list ({model.features.length})</summary>
              <ul className="mt-2 space-y-1">{model.features.map(f => <li key={f.name}><span className="font-mono text-cyan-200">{f.name}</span> <span className="text-slate-500">[{f.unit}]</span> — {f.description}</li>)}</ul>
            </details>
          </div>
        ) : <Empty icon={<Brain />} title="Model information unavailable">Connect to the backend.</Empty>}
      </Card>

      <div className="space-y-6 xl:col-span-2">
        {!result || !pts.length ? <Card><Empty icon={<Brain />} title="No matched trajectory">Run a trip or upload to inspect candidate scores and sequence decoding.</Empty></Card> : (<>
          <div className="grid gap-4 sm:grid-cols-4">
            <Stat label="Mean matched probability" value={num(result.stats.mean_matched_prob, 3)} accent />
            <Stat label="Low-confidence points" value={pct(result.stats.low_confidence_share, 1)} hint="probability < 0.5" />
            <Stat label="Changed by HMM" value={pct(result.stats.hmm_changed_share, 1)} hint="vs classifier-only choice" />
            <Stat label="Candidates / point" value={num(result.stats.mean_candidates_per_point, 1)} />
          </div>
          <Card title="Confidence along the trajectory" subtitle="Model probability of the HMM-chosen arc per GPS point (sampled)">
            <div className="h-56"><ResponsiveContainer><AreaChart data={series}>
              <defs><linearGradient id="gp" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor="#22d3ee" stopOpacity={.5} /><stop offset="1" stopColor="#a855f7" stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid stroke="rgba(255,255,255,.05)" /><XAxis dataKey="seq" stroke="#64748b" fontSize={11} /><YAxis domain={[0, 1]} stroke="#64748b" fontSize={11} />
              <Tooltip {...tip} /><Area type="monotone" dataKey="prob" stroke="#22d3ee" fill="url(#gp)" strokeWidth={1.5} />
            </AreaChart></ResponsiveContainer></div>
          </Card>
          <div className="grid gap-6 lg:grid-cols-2">
            <Card title="Probability distribution" subtitle="Matched points per probability bin">
              <div className="h-56"><ResponsiveContainer><BarChart data={hist}>
                <CartesianGrid stroke="rgba(255,255,255,.05)" /><XAxis dataKey="bin" stroke="#64748b" fontSize={10} /><YAxis stroke="#64748b" fontSize={11} />
                <Tooltip {...tip} /><Bar dataKey="points" fill="#a78bfa" radius={[6, 6, 0, 0]} />
              </BarChart></ResponsiveContainer></div>
            </Card>
            <Card title="Candidate ranking inspector" subtitle="Top-3 candidates of one GPS point">
              <input type="range" min={0} max={pts.length - 1} value={Math.min(seq, pts.length - 1)} onChange={e => setSeq(+e.target.value)} className="w-full accent-cyan-400" />
              {p && (<div className="mt-3 space-y-2">
                <div className="text-xs text-slate-400">point <span className="font-mono text-slate-200">{p.seq}</span> · t = {num(p.t, 0)} s · chain {p.chain} · HMM chose <span className="font-mono text-fuchsia-200">{p.arc_id}</span>, classifier <span className="font-mono text-amber-200">{p.classifier_arc_id}</span></div>
                {p.top.map((c, i) => (
                  <div key={c.arc_id} className="rounded-lg bg-ink-850 px-3 py-2 text-xs">
                    <div className="flex justify-between"><span className="font-mono text-slate-200">#{i + 1} arc {c.arc_id}{c.arc_id === p.arc_id ? " · chosen" : ""}</span><span className="text-slate-400">{num(c.dist_m, 1)} m · p {num(c.prob, 3)}</span></div>
                    <div className="mt-1.5 h-1.5 rounded-full bg-white/5"><div className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-fuchsia-500" style={{ width: `${100 * c.prob}%` }} /></div>
                  </div>
                ))}
              </div>)}
            </Card>
          </div>
        </>)}
      </div>
    </div>
  );
}

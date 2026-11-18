import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowRight, Cpu, Database, Gauge, Network, Play, Radar } from "lucide-react";
import { api } from "../api/client";
import type { DatasetInfo, Evaluation, ModelInfo } from "../api/types";
import { useApp } from "../state/AppContext";
import { Alert, Button, Card, Empty, Spinner, Stat, pct , MODEL_NAMES } from "../components/ui";

export default function Dashboard() {
  const { conn, health, connError, trips, runMatch, running, runError, refresh } = useApp();
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [ds, setDs] = useState<DatasetInfo | null>(null);
  const [ev, setEv] = useState<Evaluation | null>(null);
  const nav = useNavigate();
  useEffect(() => {
    if (conn !== "online") return;
    api.model().then(setModel).catch(() => setModel(null));
    api.dataset().then(setDs).catch(() => setDs(null));
    api.evaluation().then(setEv).catch(() => setEv(null));
  }, [conn]);
  const hmmTest = ev?.hmm?.methods?.find((m: any) => m.partition !== "train_oof" && m.method === "model_hmm");
  const clfTest = ev?.hmm?.methods?.find((m: any) => m.partition !== "train_oof" && m.method === "classifier_only");

  const go = async (id: string) => { const r = await runMatch({ trip_id: id }); if (r) nav("/map"); };
  return (
    <div className="space-y-6">
      <section className="relative overflow-hidden rounded-3xl border border-white/[.07] bg-gradient-to-br from-ink-850 via-ink-900 to-ink-950 p-8">
        <div className="absolute -right-20 -top-20 h-72 w-72 rounded-full bg-fuchsia-500/20 blur-3xl" />
        <div className="absolute -bottom-24 left-1/3 h-72 w-72 rounded-full bg-cyan-500/10 blur-3xl" />
        <div className="relative">
          <div className="text-xs uppercase tracking-[.3em] text-cyan-300/80">Geospatial intelligence</div>
          <h1 className="mt-2 text-3xl font-semibold tracking-tight text-white md:text-4xl">Noisy GPS in. <span className="bg-gradient-to-r from-cyan-300 via-sky-300 to-fuchsia-400 bg-clip-text text-transparent">Coherent routes out.</span></h1>
          <p className="mt-3 max-w-2xl text-slate-400">PathSnap scores candidate roads with a trained gradient-boosting model and decodes the most probable connected sequence of directed arcs with an HMM and the Viterbi algorithm.</p>
          <div className="mt-6 flex flex-wrap gap-3">
            <Button onClick={() => nav("/upload")}>Upload a trajectory <ArrowRight className="h-4 w-4" /></Button>
            <Button variant="ghost" onClick={() => nav("/evaluation")}>View evaluation</Button>
          </div>
        </div>
      </section>

      {conn === "offline" && <Alert kind="error">{connError}{"\n"}<button className="mt-2 underline" onClick={refresh}>Retry</button> or open Connection (top right) to change the backend address.</Alert>}
      {conn === "checking" && <Spinner label="Connecting to the backend…" />}

      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <Stat label="Backend" value={conn === "online" ? "Online" : conn === "checking" ? "…" : "Offline"} hint={health ? `serving ${health.partition}` : "not connected"} accent={conn === "online"} />
        <Stat label="Model" value={<span className="text-lg">{model ? (MODEL_NAMES[model.family] ?? model.estimator) : "–"}</span>} hint={model?.source ?? (health ? "loaded" : "")} />
        <Stat label="Trips available" value={health?.n_trips ?? "–"} hint={ds ? ds.partitions.map(p => `${p.partition}: ${p.trips}`).join(" · ") : ""} />
        <Stat label="HMM decoder" value={health ? `β ${health.hmm.beta_m}` : "–"} hint={health ? `w ${health.hmm.w_emission} · ${health.hmm.max_states} states/point` : ""} />
      </div>

      <div className="grid gap-6 xl:grid-cols-3">
        <Card title="Measured results (student-test trips)" subtitle="From the saved Phase 3-5 experiments" className="xl:col-span-2">
          {ev ? (
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Stat label="Point accuracy · HMM" value={pct(hmmTest?.point_acc)} hint={`classifier only ${pct(clfTest?.point_acc)}`} accent />
              <Stat label="Route arc F1 · HMM" value={pct(hmmTest?.arc_f1_trip_mean)} hint={`classifier only ${pct(clfTest?.arc_f1_trip_mean)}`} />
              <Stat label="Continuity · HMM" value={pct(hmmTest?.continuity_trip_mean)} hint={`classifier only ${pct(clfTest?.continuity_trip_mean)}`} />
              <Stat label="CV Top-1 (tuned)" value={pct(ev.tuning?.tuned_cv_top1)} hint={`untuned ${pct(ev.tuning?.untuned_cv_top1)}`} />
            </div>
          ) : <Empty icon={<Gauge />} title="No saved evaluation">The backend has no evaluation summary yet.</Empty>}
          {ev?.dataset && (
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              {Object.entries(ev.dataset.partitions ?? {}).map(([k, v]: any) => (
                <div key={k} className="rounded-xl border border-white/[.06] bg-ink-850 p-3 text-xs text-slate-400">
                  <div className="mb-1 flex items-center gap-2 text-slate-200"><Database className="h-3.5 w-3.5 text-cyan-300" />{k}</div>
                  {v.trips} trips · {v.gps_points.toLocaleString()} GPS points · {v.arcs.toLocaleString()} road arcs
                </div>
              ))}
              {ev.dataset.coverage && <div className="rounded-xl border border-white/[.06] bg-ink-850 p-3 text-xs text-slate-400"><div className="mb-1 flex items-center gap-2 text-slate-200"><Network className="h-3.5 w-3.5 text-fuchsia-300" />Candidate coverage</div>train {pct(ev.dataset.coverage.train)} · student-test {pct(ev.dataset.coverage.student_test)}</div>}
            </div>
          )}
        </Card>

        <Card title="Server trips" subtitle="Run the full pipeline on a trip stored on the backend" right={<Cpu className="h-4 w-4 text-cyan-300" />}>
          {runError && <div className="mb-3"><Alert kind="error">{runError}</Alert></div>}
          {conn !== "online" ? <Empty icon={<Radar />} title="Not connected" /> : trips.length === 0 ? <Empty icon={<Radar />} title="No trips on the server" /> : (
            <div className="max-h-[360px] space-y-1.5 overflow-auto pr-1">
              {trips.map(t => (
                <div key={t.trip_id} className="flex items-center justify-between rounded-xl border border-white/[.05] bg-ink-850 px-3 py-2">
                  <div><div className="font-mono text-sm text-slate-100">{t.trip_id}</div><div className="text-xs text-slate-500">{t.n_points.toLocaleString()} points</div></div>
                  <Button variant="ghost" className="!px-3 !py-1.5" disabled={!!running} onClick={() => go(t.trip_id)}>
                    {running === t.trip_id ? <Spinner /> : <><Play className="h-3.5 w-3.5" />Match</>}
                  </Button>
                </div>
              ))}
            </div>
          )}
        </Card>
      </div>
    </div>
  );
}

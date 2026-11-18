import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FileUp, Play, UploadCloud, X } from "lucide-react";
import { api } from "../api/client";
import type { UploadResult } from "../api/types";
import { useApp } from "../state/AppContext";
import MapView from "../components/MapView";
import { Alert, Button, Card, Empty, Spinner, Stat, num } from "../components/ui";

const OK = /\.(zip|track|nodes|arcs|route)$/i;

export default function Upload() {
  const { conn, trips, runMatch, running, runError } = useApp();
  const [files, setFiles] = useState<File[]>([]);
  const [network, setNetwork] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [up, setUp] = useState<UploadResult | null>(null);
  const [drag, setDrag] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const nav = useNavigate();

  const add = (list: FileList | null) => {
    if (!list) return;
    const arr = Array.from(list);
    const bad = arr.filter(f => !OK.test(f.name));
    setErr(bad.length ? `Ignored unsupported files: ${bad.map(b => b.name).join(", ")} (use .zip, .track, .nodes, .arcs, .route)` : null);
    setFiles(prev => [...prev.filter(p => !arr.some(a => a.name === p.name)), ...arr.filter(f => OK.test(f.name))]);
    setUp(null);
  };
  const hasTrack = files.some(f => /\.(track|zip)$/i.test(f.name));
  const hasNet = files.some(f => /\.zip$/i.test(f.name)) || (files.some(f => /\.nodes$/i.test(f.name)) && files.some(f => /\.arcs$/i.test(f.name)));
  const canUpload = conn === "online" && hasTrack && (hasNet || !!network) && !busy;

  const doUpload = async () => {
    setBusy(true); setErr(null);
    try { setUp(await api.upload(files, hasNet ? undefined : network)); } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };
  const doMatch = async () => { if (!up) return; const r = await runMatch({ upload_id: up.upload_id }); if (r) nav("/map"); };

  return (
    <div className="grid gap-6 xl:grid-cols-5">
      <div className="space-y-6 xl:col-span-2">
        <Card title="Upload a GPS trajectory" subtitle="A trip folder as .zip, or the files .track (lon lat t) + .nodes + .arcs (optional .route for evaluation)">
          <div onDragOver={e => { e.preventDefault(); setDrag(true); }} onDragLeave={() => setDrag(false)}
            onDrop={e => { e.preventDefault(); setDrag(false); add(e.dataTransfer.files); }} onClick={() => input.current?.click()}
            className={`cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition ${drag ? "border-cyan-400 bg-cyan-400/5" : "border-white/10 hover:border-cyan-400/40"}`}>
            <UploadCloud className="mx-auto h-10 w-10 text-cyan-300" />
            <div className="mt-3 font-medium text-slate-100">Drop files here or click to browse</div>
            <div className="mt-1 text-xs text-slate-400">.zip · .track · .nodes · .arcs · .route — up to 60 MB</div>
            <input ref={input} type="file" multiple accept=".zip,.track,.nodes,.arcs,.route" className="hidden" onChange={e => add(e.target.files)} />
          </div>
          {files.length > 0 && (
            <ul className="mt-4 space-y-1.5">
              {files.map(f => (
                <li key={f.name} className="flex items-center justify-between rounded-lg bg-ink-850 px-3 py-2 text-sm">
                  <span className="flex items-center gap-2 font-mono text-slate-200"><FileUp className="h-3.5 w-3.5 text-cyan-300" />{f.name}</span>
                  <span className="flex items-center gap-3 text-xs text-slate-500">{(f.size / 1024).toFixed(1)} KB
                    <button onClick={() => { setFiles(files.filter(x => x !== f)); setUp(null); }}><X className="h-3.5 w-3.5" /></button></span>
                </li>
              ))}
            </ul>
          )}
          {hasTrack && !hasNet && (
            <div className="mt-4">
              <label className="text-xs uppercase tracking-widest text-slate-400">Road network (no .nodes/.arcs uploaded)</label>
              <select value={network} onChange={e => setNetwork(e.target.value)} className="mt-1.5 w-full rounded-xl border border-white/10 bg-ink-950 px-3 py-2 text-sm">
                <option value="">Choose a server trip's network…</option>
                {trips.map(t => <option key={t.trip_id} value={t.trip_id}>{t.trip_id}</option>)}
              </select>
            </div>
          )}
          {err && <div className="mt-4"><Alert kind="error">{err}</Alert></div>}
          {conn !== "online" && <div className="mt-4"><Alert kind="warn">Backend not connected.</Alert></div>}
          <div className="mt-5 flex gap-2">
            <Button onClick={doUpload} disabled={!canUpload}>{busy ? <Spinner label="Validating…" /> : "Upload & validate"}</Button>
            {up && <Button variant="ghost" onClick={doMatch} disabled={!!running}>{running ? <Spinner label="Matching…" /> : <><Play className="h-4 w-4" />Run map matching</>}</Button>}
          </div>
          {runError && <div className="mt-3"><Alert kind="error">{runError}</Alert></div>}
        </Card>
        {up && (
          <Card title="Validation" subtitle={`upload id ${up.upload_id}`}>
            <Alert kind="success">Files accepted: {up.files.join(" · ")}</Alert>
            <div className="mt-4 grid grid-cols-2 gap-3">
              <Stat label="GPS points" value={up.n_points.toLocaleString()} />
              <Stat label="Duration" value={`${num(up.duration_s / 60, 1)} min`} hint={`median step ${num(up.median_dt_s, 1)} s`} />
              <Stat label="Road network" value={(up.n_arcs ?? 0).toLocaleString()} hint={`arcs · ${(up.n_nodes ?? 0).toLocaleString()} nodes`} />
              <Stat label="Reference route" value={up.has_route ? "Yes" : "No"} hint={up.has_route ? "metrics will be computed" : "prediction only"} />
            </div>
            {up.warnings.length > 0 && <div className="mt-4 space-y-2">{up.warnings.map(w => <Alert key={w} kind="warn">{w}</Alert>)}</div>}
          </Card>
        )}
      </div>
      <div className="xl:col-span-3">
        <Card title="Trajectory preview" subtitle="Raw GPS as uploaded (before any processing)">
          {up ? <MapView gps={up.gps_geometry} height="68vh" /> : <Empty icon={<UploadCloud />} title="Nothing uploaded yet">The raw trajectory appears here after validation.</Empty>}
        </Card>
      </div>
    </div>
  );
}

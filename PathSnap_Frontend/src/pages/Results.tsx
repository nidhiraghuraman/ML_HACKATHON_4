import { useEffect, useMemo, useState } from "react";
import { ListTree, RefreshCw } from "lucide-react";
import { api } from "../api/client";
import type { SessionResult } from "../api/types";
import { useApp } from "../state/AppContext";
import { Alert, Button, Card, Empty, Stat, num, pct } from "../components/ui";

export default function Results() {
  const { result, conn, runMatch, running } = useApp();
  const [list, setList] = useState<SessionResult[]>([]);
  const [page, setPage] = useState(0);
  const load = () => api.results().then(r => setList(r.results)).catch(() => setList([]));
  useEffect(() => { if (conn === "online") load(); }, [conn, result]);
  const pts = result?.points ?? [];
  const rows = useMemo(() => pts.slice(page * 50, page * 50 + 50), [pts, page]);
  const pages = Math.ceil(pts.length / 50);
  return (
    <div className="space-y-6">
      <Card title="Session results" subtitle="Everything matched by the backend since it started" right={<Button variant="ghost" className="!px-3 !py-1.5" onClick={load}><RefreshCw className="h-3.5 w-3.5" />Refresh</Button>}>
        {list.length ? (
          <table className="w-full text-left text-xs"><thead className="text-slate-400"><tr>{["Time", "Source", "ID", "Points", "Route arcs", "Chains", "Seconds", "Point acc.", "Route F1", ""].map(h => <th key={h} className="px-2 py-2 font-medium">{h}</th>)}</tr></thead>
            <tbody className="font-mono text-slate-300">{list.map(r => (
              <tr key={r.source + r.id} className="border-t border-white/5"><td className="px-2 py-1.5">{r.at}</td><td className="px-2">{r.source}</td><td className="px-2 text-cyan-200">{r.id}</td><td className="px-2">{r.n_points}</td><td className="px-2">{r.route_arcs}</td><td className="px-2">{r.chains}</td><td className="px-2">{num(r.total_s, 2)}</td><td className="px-2">{r.evaluated ? pct(r.point_acc) : "–"}</td><td className="px-2">{r.evaluated ? pct(r.arc_f1) : "–"}</td>
                <td className="px-2"><button className="text-cyan-300 hover:underline disabled:opacity-40" disabled={!!running} onClick={() => runMatch(r.source === "trip" ? { trip_id: r.id } : { upload_id: r.id })}>open</button></td></tr>))}</tbody></table>
        ) : <Empty icon={<ListTree />} title="No results yet" />}
      </Card>
      {result ? (<>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
          <Stat label="Input" value={<span className="text-lg">{result.id}</span>} hint={result.source} accent />
          <Stat label="GPS points" value={result.n_points.toLocaleString()} hint={`${result.matched_points.toLocaleString()} matched`} />
          <Stat label="Route" value={`${result.route_arc_ids.length} arcs`} hint={`${num(result.route_length_m / 1000, 2)} km · ${result.route_breaks} breaks`} />
          <Stat label="Processing" value={`${num(result.timing_s.total_s, 2)} s`} hint={`features ${num(result.timing_s.preprocess_candidates_features_s, 2)} s · model ${num(result.timing_s.model_s, 3)} s · HMM ${num(result.timing_s.decode_s, 2)} s`} />
          <Stat label="Point accuracy" value={result.evaluated ? pct(result.metrics?.point_acc) : "n/a"} hint={result.evaluated ? `route F1 ${pct(result.metrics?.arc_f1)}` : "no reference route"} />
        </div>
        {result.warnings.map(w => <Alert key={w} kind="warn">{w}</Alert>)}
        <Card title="Reconstructed route" subtitle="Connected sequence of directed arc IDs (rows of the trip's .arcs file)">
          <div className="max-h-40 overflow-auto rounded-xl bg-ink-950 p-3 font-mono text-xs leading-relaxed text-cyan-100">{result.route_arc_ids.join(" → ")}</div>
        </Card>
        <Card title="Observations" subtitle="Matched arc, probability and top candidates per GPS point" right={<div className="flex items-center gap-2 text-xs text-slate-400"><Button variant="ghost" className="!px-2 !py-1" disabled={page === 0} onClick={() => setPage(page - 1)}>‹</Button>{page + 1} / {pages}<Button variant="ghost" className="!px-2 !py-1" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>›</Button></div>}>
          <table className="w-full text-left text-xs"><thead className="text-slate-400"><tr>{["Seq", "t (s)", "Lat", "Lon", "HMM arc", "Prob", "Classifier arc", "Chain", "Top candidates (arc:p)"].map(h => <th key={h} className="px-2 py-2 font-medium">{h}</th>)}</tr></thead>
            <tbody className="font-mono text-slate-300">{rows.map(p => (
              <tr key={p.seq} className={`border-t border-white/5 ${p.arc_id !== p.classifier_arc_id ? "bg-fuchsia-500/5" : ""}`}><td className="px-2 py-1">{p.seq}</td><td className="px-2">{num(p.t, 0)}</td><td className="px-2">{p.lat.toFixed(5)}</td><td className="px-2">{p.lon.toFixed(5)}</td>
                <td className="px-2 text-fuchsia-200">{p.arc_id}</td><td className="px-2">{num(p.prob, 3)}</td><td className="px-2 text-amber-200">{p.classifier_arc_id}</td><td className="px-2">{p.chain}</td>
                <td className="px-2 text-slate-400">{p.top.map(c => `${c.arc_id}:${c.prob.toFixed(2)}`).join("  ")}</td></tr>))}</tbody></table>
          <p className="mt-2 text-[11px] text-slate-500">Highlighted rows: the HMM chose a different arc than the classifier alone.</p>
        </Card>
      </>) : <Card><Empty icon={<ListTree />} title="No current result">Run a trip or an upload first.</Empty></Card>}
    </div>
  );
}

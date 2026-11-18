import { useState } from "react";
import { Map as MapIcon, Play } from "lucide-react";
import { useApp } from "../state/AppContext";
import MapView from "../components/MapView";
import { Alert, Button, Card, Empty, Spinner, Stat, num, pct } from "../components/ui";

export default function MapPage() {
  const { result, trips, runMatch, running, runError, conn } = useApp();
  const [sel, setSel] = useState("");
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <h1 className="text-xl font-semibold text-white">Interactive map</h1>
          <p className="text-sm text-slate-400">{result ? <>Showing <span className="font-mono text-cyan-200">{result.source} {result.id}</span> · {result.n_points.toLocaleString()} GPS points</> : "Run a trip or an upload to see results."}</p>
        </div>
        <div className="ml-auto flex items-center gap-2">
          <select value={sel} onChange={e => setSel(e.target.value)} className="rounded-xl border border-white/10 bg-ink-900 px-3 py-2 text-sm" disabled={conn !== "online"}>
            <option value="">Server trip…</option>{trips.map(t => <option key={t.trip_id}>{t.trip_id}</option>)}
          </select>
          <Button disabled={!sel || !!running} onClick={() => runMatch({ trip_id: sel })}>{running ? <Spinner label={`Matching ${running}…`} /> : <><Play className="h-4 w-4" />Run</>}</Button>
        </div>
      </div>
      {runError && <Alert kind="error">{runError}</Alert>}
      {result ? (
        <div className="grid gap-4 xl:grid-cols-4">
          <div className="xl:col-span-3"><MapView result={result} height="74vh" /></div>
          <div className="space-y-3">
            {result.evaluated && result.metrics ? (<>
              <Stat label="Point accuracy" value={pct(result.metrics.point_acc)} hint={`classifier only ${pct(result.metrics.classifier_only_point_acc)}`} accent />
              <Stat label="Route arc F1" value={pct(result.metrics.arc_f1)} hint={`P ${pct(result.metrics.arc_precision)} · R ${pct(result.metrics.arc_recall)}`} />
              <Stat label="Continuity" value={pct(result.metrics.continuity)} hint={`classifier only ${pct(result.metrics.classifier_only_continuity)}`} />
            </>) : <Alert kind="info">No reference route available for this input: prediction only, no accuracy metrics.</Alert>}
            <Stat label="Route" value={`${num(result.route_length_m / 1000, 2)} km`} hint={`${result.route_arc_ids.length} arcs · ${result.chains} chain(s) · ${result.route_breaks} breaks`} />
            <Stat label="Inference time" value={`${num(result.timing_s.total_s, 2)} s`} hint={`model ${num(result.timing_s.model_s, 3)} s · decode ${num(result.timing_s.decode_s, 2)} s`} />
            {result.warnings.map(w => <Alert key={w} kind="warn">{w}</Alert>)}
          </div>
        </div>
      ) : (
        <Card><Empty icon={<MapIcon />} title="No result yet">Choose a server trip above, or upload a trajectory on the Upload page.</Empty></Card>
      )}
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";
import { CircleMarker, MapContainer, Polyline, TileLayer, Tooltip, useMap } from "react-leaflet";
import L from "leaflet";
import type { LatLng, MatchResult } from "../api/types";

export type LayerKey = "gps" | "matched" | "route" | "classifier" | "candidates" | "markers";
export const LAYERS: { key: LayerKey; label: string; color: string }[] = [
  { key: "gps", label: "Noisy GPS", color: "#38bdf8" },
  { key: "route", label: "Reconstructed route (HMM)", color: "#e879f9" },
  { key: "matched", label: "Matched arcs", color: "#a78bfa" },
  { key: "classifier", label: "Classifier-only route", color: "#f59e0b" },
  { key: "candidates", label: "Candidate roads (sample)", color: "#94a3b8" },
  { key: "markers", label: "Start / end", color: "#34d399" },
];

function Fit({ points }: { points: LatLng[] }) {
  const map = useMap();
  useEffect(() => {
    if (points.length) { const b = L.latLngBounds(points as L.LatLngExpression[]); if (b.isValid()) map.fitBounds(b, { padding: [30, 30] }); }
  }, [points, map]);
  return null;
}

export default function MapView({ result, gps, height = "70vh", showControls = true }: { result?: MatchResult | null; gps?: LatLng[]; height?: string; showControls?: boolean }) {
  const [on, setOn] = useState<Record<LayerKey, boolean>>({ gps: true, route: true, matched: false, classifier: false, candidates: false, markers: true });
  const pts = useMemo(() => (result ? result.gps_geometry : gps) ?? [], [result, gps]);
  const renderer = useMemo(() => L.canvas({ padding: 0.5 }), []);
  return (
    <div className="relative overflow-hidden rounded-2xl border border-white/[.07]" style={{ height }}>
      <MapContainer center={[20, 0]} zoom={2} preferCanvas renderer={renderer} className="h-full w-full" zoomControl>
        <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" attribution='&copy; OpenStreetMap contributors &copy; CARTO' maxZoom={20} />
        <Fit points={pts} />
        {on.candidates && result?.candidate_geometry.map((c, i) => (
          <Polyline key={`c${i}`} positions={c.line} pathOptions={{ color: c.chosen ? "#22d3ee" : "#94a3b8", weight: c.chosen ? 3 : 2, opacity: 0.25 + 0.7 * c.prob, dashArray: c.chosen ? undefined : "4 4" }}>
            <Tooltip sticky>point {c.seq} · arc {c.arc_id} · p = {c.prob.toFixed(3)}{c.chosen ? " · chosen" : ""}</Tooltip>
          </Polyline>
        ))}
        {on.gps && pts.map((p, i) => <CircleMarker key={`g${i}`} center={p} radius={1.8} pathOptions={{ color: "#38bdf8", weight: 0, fillOpacity: 0.5 }} />)}
        {on.classifier && result && <Polyline positions={result.classifier_route_geometry} pathOptions={{ color: "#f59e0b", weight: 3, opacity: 0.8, dashArray: "6 6" }} />}
        {on.matched && result && <Polyline positions={result.matched_geometry} pathOptions={{ color: "#a78bfa", weight: 7, opacity: 0.5 }} />}
        {on.route && result && <Polyline positions={result.route_geometry} pathOptions={{ color: "#e879f9", weight: 4, opacity: 0.95 }} />}
        {on.markers && pts.length > 0 && (<>
          <CircleMarker center={pts[0]} radius={7} pathOptions={{ color: "#0a0f1e", weight: 2, fillColor: "#34d399", fillOpacity: 1 }}><Tooltip>Start</Tooltip></CircleMarker>
          <CircleMarker center={pts[pts.length - 1]} radius={7} pathOptions={{ color: "#0a0f1e", weight: 2, fillColor: "#f43f5e", fillOpacity: 1 }}><Tooltip>End</Tooltip></CircleMarker>
        </>)}
      </MapContainer>
      {showControls && (
        <div className="absolute right-3 top-3 z-[500] w-56 rounded-xl border border-white/10 bg-ink-950/85 p-3 text-xs backdrop-blur">
          <div className="mb-2 font-semibold uppercase tracking-widest text-slate-400">Layers</div>
          {LAYERS.filter(l => result || l.key === "gps" || l.key === "markers").map(l => (
            <label key={l.key} className="flex cursor-pointer items-center gap-2 py-1 text-slate-200">
              <input type="checkbox" checked={on[l.key]} onChange={() => setOn(s => ({ ...s, [l.key]: !s[l.key] }))} className="accent-cyan-400" />
              <span className="h-1 w-5 rounded-full" style={{ background: l.color }} />{l.label}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

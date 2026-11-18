// Central API client. The backend address comes from (in order):
//  1. a value saved in the app's Settings (localStorage), 2. VITE_API_BASE_URL in .env,
//  3. the same host that serves this page, port 8000.
import type { DatasetInfo, Evaluation, Health, MatchResult, ModelInfo, Preview, SessionResult, Trip, UploadResult } from "./types";

const KEY = "pathsnap.apiBase";
export function defaultBase(): string {
  const env = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim();
  if (env) return env.replace(/\/+$/, "");
  return `${location.protocol}//${location.hostname || "localhost"}:8000`;
}
export function getBase(): string {
  try { return (localStorage.getItem(KEY) || defaultBase()).replace(/\/+$/, ""); } catch { return defaultBase(); }
}
export function setBase(url: string | null) {
  try { url ? localStorage.setItem(KEY, url.trim().replace(/\/+$/, "")) : localStorage.removeItem(KEY); } catch { /* storage unavailable */ }
}

export class ApiError extends Error {
  constructor(message: string, public status?: number) { super(message); }
}

async function request<T>(path: string, init: RequestInit = {}, timeoutMs = 180000): Promise<T> {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  const base = getBase();
  try {
    const r = await fetch(base + path, { ...init, signal: ctl.signal });
    const text = await r.text();
    let body: any = null;
    try { body = text ? JSON.parse(text) : null; } catch { body = text; }
    if (!r.ok) {
      const d = body?.detail;
      const msg = typeof d === "string" ? d : Array.isArray(d) ? d.map((x: any) => x.msg).join("; ") : r.statusText;
      throw new ApiError(`${msg} (HTTP ${r.status})`, r.status);
    }
    return body as T;
  } catch (e: any) {
    if (e instanceof ApiError) throw e;
    if (e?.name === "AbortError") throw new ApiError(`The backend at ${base} did not answer within ${timeoutMs / 1000} s.`);
    throw new ApiError(`Cannot reach the backend at ${base}. Is it running with --host 0.0.0.0, on the same network, with port 8000 allowed by the firewall?`);
  } finally { clearTimeout(timer); }
}

const json = (body: unknown): RequestInit => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

export const api = {
  health: () => request<Health>("/health", {}, 8000),
  model: () => request<ModelInfo>("/model"),
  dataset: () => request<DatasetInfo>("/dataset"),
  evaluation: () => request<Evaluation>("/evaluation"),
  trips: () => request<{ partition: string; trips: Trip[] }>("/trips"),
  preview: (q: { trip_id?: string; upload_id?: string }) => request<Preview>(`/preview?${new URLSearchParams(q as Record<string, string>)}`),
  upload: (files: File[], networkTripId?: string) => {
    const fd = new FormData();
    files.forEach(f => fd.append("files", f, f.name));
    if (networkTripId) fd.append("network_trip_id", networkTripId);
    return request<UploadResult>("/upload", { method: "POST", body: fd });
  },
  match: (q: { trip_id?: string; upload_id?: string }) => request<MatchResult>("/match", json({ ...q, include_points: true, include_candidates: true })),
  results: () => request<{ results: SessionResult[] }>("/results"),
  metrics: () => request<{ evaluated_trips: number; mean: Record<string, number> }>("/metrics"),
};

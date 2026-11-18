import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from "react";
import { api, getBase } from "../api/client";
import type { Health, MatchResult, Trip } from "../api/types";

type Conn = "checking" | "online" | "offline";
interface AppState {
  base: string; conn: Conn; health: Health | null; connError: string | null;
  trips: Trip[]; result: MatchResult | null; running: string | null; runError: string | null;
  refresh: () => Promise<void>; runMatch: (q: { trip_id?: string; upload_id?: string }) => Promise<MatchResult | null>;
  setResult: (r: MatchResult | null) => void;
}
const Ctx = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [conn, setConn] = useState<Conn>("checking");
  const [health, setHealth] = useState<Health | null>(null);
  const [connError, setConnError] = useState<string | null>(null);
  const [trips, setTrips] = useState<Trip[]>([]);
  const [result, setResult] = useState<MatchResult | null>(null);
  const [running, setRunning] = useState<string | null>(null);
  const [runError, setRunError] = useState<string | null>(null);
  const [base, setBaseState] = useState(getBase());

  const refresh = useCallback(async () => {
    setBaseState(getBase()); setConn("checking"); setConnError(null);
    try {
      const h = await api.health();
      if (h.status !== "ok") throw new Error(h.detail || "Backend is running but not ready (model missing).");
      setHealth(h); setConn("online");
      const t = await api.trips(); setTrips(t.trips);
    } catch (e: any) { setConn("offline"); setHealth(null); setConnError(e.message); }
  }, []);

  const runMatch = useCallback(async (q: { trip_id?: string; upload_id?: string }) => {
    setRunning(q.trip_id || q.upload_id || "…"); setRunError(null);
    try { const r = await api.match(q); setResult(r); return r; }
    catch (e: any) { setRunError(e.message); return null; }
    finally { setRunning(null); }
  }, []);

  useEffect(() => { refresh(); const id = setInterval(() => api.health().then(() => setConn("online")).catch(() => setConn("offline")), 20000); return () => clearInterval(id); }, [refresh]);

  return <Ctx.Provider value={{ base, conn, health, connError, trips, result, running, runError, refresh, runMatch, setResult }}>{children}</Ctx.Provider>;
}
export function useApp() { const c = useContext(Ctx); if (!c) throw new Error("AppProvider missing"); return c; }

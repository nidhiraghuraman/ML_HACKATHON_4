import { useState, type ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { Activity, BarChart3, Brain, FileUp, LayoutDashboard, ListTree, Map as MapIcon, Settings2, X } from "lucide-react";
import { useApp } from "../state/AppContext";
import { defaultBase, getBase, setBase } from "../api/client";
import { Button } from "./ui";

const NAV = [
  { to: "/", label: "Dashboard", icon: LayoutDashboard },
  { to: "/upload", label: "Upload", icon: FileUp },
  { to: "/map", label: "Interactive map", icon: MapIcon },
  { to: "/analysis", label: "ML analysis", icon: Brain },
  { to: "/evaluation", label: "Evaluation", icon: BarChart3 },
  { to: "/results", label: "Results", icon: ListTree },
];

function StatusPill() {
  const { conn, health } = useApp();
  const color = conn === "online" ? "bg-emerald-400" : conn === "checking" ? "bg-amber-400" : "bg-rose-500";
  const text = conn === "online" ? `Backend online · ${health?.partition ?? ""}` : conn === "checking" ? "Connecting…" : "Backend offline";
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[.03] px-3 py-1 text-xs text-slate-300">
      <span className={`relative h-2 w-2 rounded-full ${color}`}>{conn === "online" && <span className="absolute inset-0 animate-ping rounded-full bg-emerald-400/60" />}</span>{text}
    </span>
  );
}

function Settings({ onClose }: { onClose: () => void }) {
  const { refresh } = useApp();
  const [v, setV] = useState(getBase());
  const save = async (val: string | null) => { setBase(val); await refresh(); onClose(); };
  return (
    <div className="fixed inset-0 z-[2000] flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm" onClick={onClose}>
      <div className="w-full max-w-lg rounded-2xl border border-white/10 bg-ink-900 p-6 shadow-2xl" onClick={e => e.stopPropagation()}>
        <div className="mb-4 flex items-center justify-between"><h2 className="font-semibold text-white">Backend connection</h2><button onClick={onClose}><X className="h-4 w-4 text-slate-400" /></button></div>
        <label className="text-xs uppercase tracking-widest text-slate-400">API base URL</label>
        <input value={v} onChange={e => setV(e.target.value)} spellCheck={false}
          className="mt-1.5 w-full rounded-xl border border-white/10 bg-ink-950 px-3 py-2 font-mono text-sm text-cyan-100 outline-none focus:border-cyan-400/60" />
        <p className="mt-2 text-xs text-slate-400">Use the backend laptop's LAN address, e.g. <span className="font-mono">http://192.168.1.10:8000</span> (find it there with <span className="font-mono">hostname -I</span>). <span className="font-mono">localhost</span> only works when the backend runs on this same computer. Default from .env: <span className="font-mono">{defaultBase()}</span></p>
        <div className="mt-5 flex justify-end gap-2"><Button variant="ghost" onClick={() => save(null)}>Reset to default</Button><Button onClick={() => save(v)}>Save & test</Button></div>
      </div>
    </div>
  );
}

export default function Layout({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const { base } = useApp();
  return (
    <div className="min-h-screen bg-[radial-gradient(1200px_600px_at_80%_-10%,rgba(168,85,247,.12),transparent),radial-gradient(900px_500px_at_-10%_10%,rgba(34,211,238,.10),transparent)]">
      <aside className="fixed inset-y-0 left-0 z-[1000] hidden w-64 flex-col border-r border-white/[.06] bg-ink-950/80 px-4 py-6 backdrop-blur lg:flex">
        <div className="mb-8 flex items-center gap-3 px-2">
          <img src="/favicon.svg" className="h-9 w-9" alt="" />
          <div><div className="text-lg font-semibold tracking-tight text-white">Path<span className="bg-gradient-to-r from-cyan-300 to-fuchsia-400 bg-clip-text text-transparent">Snap</span></div>
            <div className="text-[11px] uppercase tracking-[.2em] text-slate-500">map matching</div></div>
        </div>
        <nav className="flex flex-col gap-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) =>
              `flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition ${isActive ? "bg-gradient-to-r from-cyan-500/15 to-violet-500/10 text-white ring-1 ring-cyan-400/25" : "text-slate-400 hover:bg-white/[.04] hover:text-slate-100"}`}>
              <Icon className="h-4 w-4" />{label}
            </NavLink>
          ))}
        </nav>
        <div className="mt-auto rounded-xl border border-white/[.06] bg-white/[.02] p-3 text-xs text-slate-400">
          <div className="flex items-center gap-1.5 text-slate-300"><Activity className="h-3.5 w-3.5 text-cyan-300" />Pipeline</div>
          <div className="mt-1 leading-relaxed">GPS → candidates → features → ML scoring → HMM/Viterbi → route</div>
        </div>
      </aside>
      <div className="lg:pl-64">
        <header className="sticky top-0 z-[900] flex flex-wrap items-center gap-3 border-b border-white/[.06] bg-ink-950/70 px-6 py-3 backdrop-blur">
          <nav className="flex gap-1 overflow-x-auto lg:hidden">
            {NAV.map(({ to, label }) => <NavLink key={to} to={to} end={to === "/"} className={({ isActive }) => `whitespace-nowrap rounded-lg px-2.5 py-1 text-xs ${isActive ? "bg-cyan-500/15 text-white" : "text-slate-400"}`}>{label}</NavLink>)}
          </nav>
          <div className="ml-auto flex items-center gap-3">
            <span className="hidden font-mono text-xs text-slate-500 md:inline">{base}</span>
            <StatusPill />
            <Button variant="ghost" onClick={() => setOpen(true)} className="!px-3 !py-1.5"><Settings2 className="h-4 w-4" />Connection</Button>
          </div>
        </header>
        <main className="mx-auto max-w-[1500px] px-6 py-6">{children}</main>
      </div>
      {open && <Settings onClose={() => setOpen(false)} />}
    </div>
  );
}

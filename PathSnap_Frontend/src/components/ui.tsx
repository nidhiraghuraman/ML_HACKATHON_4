import type { ReactNode } from "react";
import { AlertTriangle, CheckCircle2, Info, Loader2 } from "lucide-react";

export function Card({ title, subtitle, right, children, className = "" }: { title?: ReactNode; subtitle?: ReactNode; right?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-2xl border border-white/[.07] bg-ink-900/80 backdrop-blur p-5 animate-fadein ${className}`}>
      {(title || right) && (
        <header className="mb-4 flex items-start justify-between gap-3">
          <div>{title && <h3 className="text-sm font-semibold tracking-wide text-slate-100">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-xs text-slate-400">{subtitle}</p>}</div>
          {right}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint, accent = false }: { label: string; value: ReactNode; hint?: ReactNode; accent?: boolean }) {
  return (
    <div className={`rounded-xl border p-4 ${accent ? "border-cyan-400/30 bg-gradient-to-br from-cyan-500/10 via-transparent to-fuchsia-500/10" : "border-white/[.07] bg-ink-850"}`}>
      <div className="text-[11px] uppercase tracking-[.14em] text-slate-400">{label}</div>
      <div className="mt-1.5 font-mono text-2xl font-semibold text-white tabular-nums">{value}</div>
      {hint && <div className="mt-1 text-xs text-slate-400">{hint}</div>}
    </div>
  );
}

export function Button({ children, onClick, disabled, variant = "primary", type = "button", className = "" }:
  { children: ReactNode; onClick?: () => void; disabled?: boolean; variant?: "primary" | "ghost"; type?: "button" | "submit"; className?: string }) {
  const base = "inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2 text-sm font-medium transition disabled:opacity-40 disabled:cursor-not-allowed";
  const v = variant === "primary"
    ? "bg-gradient-to-r from-cyan-500 to-violet-500 text-white shadow-glow hover:brightness-110"
    : "border border-white/10 bg-white/[.03] text-slate-200 hover:border-cyan-400/40 hover:bg-white/[.06]";
  return <button type={type} onClick={onClick} disabled={disabled} className={`${base} ${v} ${className}`}>{children}</button>;
}

export function Alert({ kind = "info", children }: { kind?: "info" | "error" | "warn" | "success"; children: ReactNode }) {
  const map = {
    info: ["border-cyan-400/25 bg-cyan-400/5 text-cyan-100", Info],
    error: ["border-rose-400/30 bg-rose-500/10 text-rose-100", AlertTriangle],
    warn: ["border-amber-400/30 bg-amber-500/10 text-amber-100", AlertTriangle],
    success: ["border-emerald-400/30 bg-emerald-500/10 text-emerald-100", CheckCircle2],
  } as const;
  const [cls, Icon] = map[kind];
  return <div className={`flex gap-2.5 rounded-xl border px-3.5 py-2.5 text-sm ${cls}`}><Icon className="mt-0.5 h-4 w-4 shrink-0" /><div className="min-w-0 whitespace-pre-wrap">{children}</div></div>;
}

export function Spinner({ label }: { label?: string }) {
  return <div className="flex items-center gap-2 text-sm text-slate-300"><Loader2 className="h-4 w-4 animate-spin text-cyan-300" />{label}</div>;
}

export function Empty({ icon, title, children }: { icon?: ReactNode; title: string; children?: ReactNode }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed border-white/10 px-6 py-14 text-center">
      <div className="mb-3 text-cyan-300/80">{icon}</div>
      <div className="font-medium text-slate-100">{title}</div>
      {children && <div className="mt-1 max-w-md text-sm text-slate-400">{children}</div>}
    </div>
  );
}

export const pct = (v?: number | null, d = 2) => (v === undefined || v === null || Number.isNaN(v) ? "–" : `${(100 * v).toFixed(d)}%`);
export const num = (v?: number | null, d = 2) => (v === undefined || v === null || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
export const MODEL_NAMES: Record<string, string> = {
  baseline_nearest: "Nearest road", baseline_formula: "Distance + heading", logistic_regression: "Logistic Regression", knn: "KNN",
  decision_tree: "Decision Tree", random_forest: "Random Forest", hist_gradient_boosting: "HistGradientBoosting",
  nearest: "Nearest road", classifier_only: "Classifier only", distance_hmm: "Distance-only HMM", model_hmm: "Classifier + HMM",
};

// Checks that the backend configured in .env (VITE_API_BASE_URL) is reachable from THIS laptop.
// Usage: npm run check:api      (or: node scripts/check-api.mjs http://192.168.1.10:8000)
import { readFileSync, existsSync } from "node:fs";
let url = process.argv[2];
if (!url) {
  for (const f of [".env.local", ".env"]) {
    if (existsSync(f)) {
      const m = readFileSync(f, "utf8").match(/^\s*VITE_API_BASE_URL\s*=\s*(\S+)/m);
      if (m) { url = m[1]; break; }
    }
  }
}
if (!url) { console.error("No VITE_API_BASE_URL found. Create .env from .env.example first."); process.exit(1); }
url = url.replace(/\/+$/, "");
try {
  const ctl = new AbortController(); setTimeout(() => ctl.abort(), 8000);
  const r = await fetch(url + "/health", { signal: ctl.signal });
  const j = await r.json();
  console.log(`Backend ${url} -> HTTP ${r.status}`, j);
  process.exit(j.status === "ok" ? 0 : 2);
} catch (e) {
  console.error(`Cannot reach ${url}/health: ${e.message}\nCheck: backend running on Laptop 1 with host 0.0.0.0, same network, firewall allows port 8000, correct IP.`);
  process.exit(1);
}

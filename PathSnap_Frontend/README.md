# PathSnap Frontend (Laptop 2)

React + TypeScript + Vite + Tailwind + Leaflet + Recharts evaluator interface for the PathSnap API.
Pages: Dashboard, Trajectory Upload, Interactive Map, ML Analysis, Evaluation, Results. All data comes from the backend; there is no mock data.

## Requirements
Node.js 18 or newer (`node -v`) and npm. Map tiles load from CARTO/OpenStreetMap, so the browser needs internet access for the basemap
(routes and points still draw without it).

## Setup
```bash
cd PathSnap_Frontend
npm ci                                   # installs exact versions from package-lock.json
cp .env.example .env
nano .env                                # set VITE_API_BASE_URL=http://<LAPTOP1_IP>:8000
npm run check:api                        # verifies the backend is reachable (calls /health)
```

## Run
Development server (hot reload), reachable from other devices too:
```bash
npm run dev            # http://localhost:5173
```
Production build + preview:
```bash
npm run build          # output in dist/
npm run preview        # http://localhost:4173
```

`VITE_API_BASE_URL` is read at build/dev-start time; after changing `.env`, restart `npm run dev` or rebuild.
The backend address can also be changed at runtime in the app (sidebar → **API settings**); it is stored in the browser.
If `VITE_API_BASE_URL` is empty, the app uses `http://<the host serving the page>:8000`.

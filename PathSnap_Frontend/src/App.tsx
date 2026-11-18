import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { AppProvider } from "./state/AppContext";
import Dashboard from "./pages/Dashboard";
import Upload from "./pages/Upload";
import MapPage from "./pages/MapPage";
import Analysis from "./pages/Analysis";
import Evaluation from "./pages/Evaluation";
import Results from "./pages/Results";

export default function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/map" element={<MapPage />} />
            <Route path="/analysis" element={<Analysis />} />
            <Route path="/evaluation" element={<Evaluation />} />
            <Route path="/results" element={<Results />} />
            <Route path="*" element={<Dashboard />} />
          </Routes>
        </Layout>
      </BrowserRouter>
    </AppProvider>
  );
}

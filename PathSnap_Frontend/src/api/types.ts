export type LatLng = [number, number];
export type Segment = [LatLng, LatLng];

export interface Health {
  status: string; model_loaded: boolean; model: string; partition: string; n_trips: number;
  evaluation_enabled: boolean; upload_enabled: boolean; hmm: Record<string, number>; detail?: string;
}
export interface Feature { name: string; unit: string; level: string; description: string }
export interface ModelInfo {
  model: string; family: string; estimator: string; params: Record<string, unknown> | null; source?: string;
  features: Feature[]; decoder: Record<string, number>; selection?: Record<string, unknown>; tuning?: Record<string, unknown>;
}
export interface DatasetInfo {
  raw_dir: string; partitions: { partition: string; trips: number; served: boolean }[];
  statistics?: Record<string, any> | null;
}
export interface Trip { trip_id: string; n_points: number }
export interface Preview {
  id: string; n_points: number; duration_s: number; median_dt_s: number; n_nodes?: number; n_arcs?: number;
  bbox: Record<string, number>; gps_geometry: LatLng[]; warnings: string[];
}
export interface UploadResult extends Omit<Preview, "id"> { upload_id: string; files: string[]; has_route: boolean }
export interface Candidate { arc_id: number; prob: number; dist_m: number }
export interface PointResult {
  seq: number; t: number; lon: number; lat: number; arc_id: number; prob: number | null;
  classifier_arc_id: number; chain: number; top: Candidate[];
}
export interface CandidateGeo { seq: number; arc_id: number; prob: number; chosen: boolean; line: Segment }
export interface MatchResult {
  source: "trip" | "upload"; id: string; n_points: number; matched_points: number; chains: number;
  route_arc_ids: number[]; route_breaks: number; route_length_m: number;
  route_geometry: Segment[]; matched_geometry: Segment[]; classifier_route_geometry: Segment[];
  candidate_geometry: CandidateGeo[]; gps_geometry: LatLng[]; points: PointResult[] | null;
  stats: Record<string, number>; timing_s: Record<string, number>; warnings: string[];
  metrics: Record<string, number> | null; evaluated: boolean; model: string;
}
export interface SessionResult {
  source: string; id: string; n_points: number; route_arcs: number; chains: number; total_s: number;
  evaluated: boolean; point_acc: number | null; arc_f1: number | null; at: string;
}
export type Evaluation = Record<string, any>;

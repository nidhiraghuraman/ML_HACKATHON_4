"""Phase 5: HMM map matching with Viterbi decoding (classifier probabilities as emissions).

Hidden states   for GPS point i: its K best candidate arcs (beam), each with the position where the
                GPS point projects onto the arc (proj_frac). Arc IDs are the trip's own .arcs rows.
Emission        log E_i(a) = w_emission * log p_i(a), p = model probability that arc a is correct.
                (Distance-only baseline HMM: log E = -0.5 * (dist / sigma)^2, Newson & Krumm 2009.)
Transition      log T(a -> b) = -| d_route(a, b) - d_gps(i-1, i) | / beta
                d_route = shortest DIRECTED road-network distance from the projected point on a to the
                projected point on b; d_gps = straight-line distance between the two GPS points (m).
                Impossible moves get -inf: b unreachable within the search limit, or d_route / dt above
                v_max (+ slack). Small backward moves on the same arc (<= backtrack_tol_m) count as
                standing still, because GPS noise makes a stopped car "move backwards" slightly.
Viterbi         delta_i(b) = log E_i(b) + max_a [ delta_{i-1}(a) + log T(a -> b) ], argmax kept for
                backtracking; the most probable state sequence is read back from the last point.
Breaks          a new chain starts (decoding restarts from emissions) when the time gap exceeds
                gap_break_s, when a point has no candidates, or when no transition is possible.
Route           consecutive matched arcs are joined with the shortest directed path between them, giving
                a connected arc sequence per chain.
Nothing here uses the reference route: decoding is pure inference.
"""
from __future__ import annotations
import heapq
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .road_network import RoadNetwork

DEFAULTS = {
    "max_states": 8,            # beam: candidates per GPS point kept as HMM states
    "beta_m": 10.0,             # transition scale (m)
    "w_emission": 1.0,          # weight of the classifier log-probability
    "sigma_m": 10.0,            # distance-only HMM emission scale (baseline)
    "vmax_mps": 50.0,           # 180 km/h
    "speed_slack_m": 30.0,
    "max_route_m": 6000.0,      # never search further than this between two points
    "gap_break_s": 60.0,
    "backtrack_tol_m": 15.0,
    "prob_floor": 1e-6,
}


# ============================================================================ shortest paths
class NetworkPaths:
    """Bounded Dijkstra on the trip's directed graph (edge weight = arc length), with caching."""

    def __init__(self, net: RoadNetwork):
        self.net = net
        ok = np.flatnonzero(net.valid)
        order = ok[np.argsort(net.arc_from[ok], kind="stable")]
        self.out_arcs = order
        self.ptr = np.searchsorted(net.arc_from[order], np.arange(net.n_nodes + 1))
        self._cache: dict[int, tuple[float, dict, dict]] = {}

    def from_node(self, u: int, limit: float) -> tuple[dict, dict]:
        """Distances (m) from node u to every node within `limit`, and the arc used to reach each node."""
        c = self._cache.get(u)
        if c is not None and c[0] >= limit:
            return c[1], c[2]
        dist, via = {u: 0.0}, {u: -1}
        heap = [(0.0, u)]
        L, A, T, P = self.net.length_m, self.out_arcs, self.net.arc_to, self.ptr
        while heap:
            d, v = heapq.heappop(heap)
            if d > dist.get(v, np.inf) or d > limit:
                continue
            for k in range(P[v], P[v + 1]):
                arc = A[k]
                w = T[arc]
                nd = d + L[arc]
                if nd <= limit and nd < dist.get(w, np.inf):
                    dist[w] = nd; via[w] = arc
                    heapq.heappush(heap, (nd, w))
        self._cache[u] = (limit, dist, via)
        return dist, via

    def path_arcs(self, u: int, v: int, limit: float) -> list[int] | None:
        """Arc IDs of the shortest path from node u to node v (empty if u == v), or None."""
        dist, via = self.from_node(u, limit)
        if dist.get(v, np.inf) > limit:          # the cache may hold results for a larger limit
            return None
        arcs = []
        while v != u:
            a = via[v]
            arcs.append(int(a))
            v = int(self.net.arc_from[a])
        return arcs[::-1]

    def route_distance(self, a: int, fa: float, b: int, fb: float, limit: float, backtrack_tol: float) -> float:
        L = self.net.length_m
        if a == b:
            delta = (fb - fa) * L[a]
            if delta >= -backtrack_tol:
                return max(delta, 0.0)
        head = (1.0 - fa) * L[a]
        dist, _ = self.from_node(int(self.net.arc_to[a]), max(limit - head, 0.0))
        d = dist.get(int(self.net.arc_from[b]))
        return np.inf if d is None else head + d + fb * L[b]


# ============================================================================ states and steps
@dataclass
class TripStates:
    """Per GPS point (in time order): candidate arcs kept as states, their projection fractions,
    emission log-scores, and the point's position/time."""
    arcs: list = field(default_factory=list)
    fracs: list = field(default_factory=list)
    emis: list = field(default_factory=list)
    x: np.ndarray = None
    y: np.ndarray = None
    t: np.ndarray = None
    kind: str = "model"


def build_states(obs: pd.DataFrame, cand: pd.DataFrame, score: np.ndarray, emission: str, cfg: dict) -> TripStates:
    """obs: one row per GPS point (seq 0..n-1, x, y, t). cand: candidate rows with seq, arc_id,
    proj_frac, dist_m, dist_rank. score: per-candidate model probability (emission='model') or
    ignored (emission='distance')."""
    c = {**DEFAULTS, **cfg}
    n = len(obs)
    df = pd.DataFrame({"seq": cand.seq.to_numpy(), "arc": cand.arc_id.to_numpy(), "frac": cand.proj_frac.to_numpy(),
                       "dist": cand.dist_m.to_numpy(), "rank": cand.dist_rank.to_numpy(), "score": np.asarray(score, float)})
    if emission == "model":
        df["e"] = np.log(np.clip(df.score, c["prob_floor"], 1.0))          # weighted by w_emission in viterbi()
        df = df.sort_values(["seq", "score", "rank"], ascending=[True, False, True], kind="mergesort")
    elif emission == "distance":
        df["e"] = -0.5 * df.dist ** 2                                      # divided by sigma^2 in viterbi()
        df = df.sort_values(["seq", "rank"], kind="mergesort")
    else:
        raise ValueError(emission)
    df = df.groupby("seq", sort=True).head(int(c["max_states"]))
    st = TripStates(x=obs.x.to_numpy(float), y=obs.y.to_numpy(float), t=obs.t.to_numpy(float), kind=emission)
    groups = dict(tuple(df.groupby("seq", sort=True)))
    for i in range(n):
        g = groups.get(i)
        if g is None:
            st.arcs.append(np.zeros(0, np.int64)); st.fracs.append(np.zeros(0)); st.emis.append(np.zeros(0))
        else:
            st.arcs.append(g.arc.to_numpy(np.int64)); st.fracs.append(g.frac.to_numpy(float)); st.emis.append(g.e.to_numpy(float))
    return st


def transition_inputs(st: TripStates, paths: NetworkPaths, cfg: dict) -> list:
    """For every consecutive pair (i-1, i): None if the chain must break there, else
    (d_route matrix K_{i-1} x K_i, d_gps). Independent of beta / w_emission, so it can be
    computed once and reused while tuning those parameters."""
    c = {**DEFAULTS, **cfg}
    steps = [None]
    for i in range(1, len(st.arcs)):
        A, B = st.arcs[i - 1], st.arcs[i]
        dt = st.t[i] - st.t[i - 1]
        if len(A) == 0 or len(B) == 0 or dt > c["gap_break_s"] or dt <= 0:
            steps.append(None); continue
        d_gps = float(np.hypot(st.x[i] - st.x[i - 1], st.y[i] - st.y[i - 1]))
        limit = min(c["vmax_mps"] * dt + c["speed_slack_m"], c["max_route_m"])
        D = np.full((len(A), len(B)), np.inf)
        for ia, (a, fa) in enumerate(zip(A, st.fracs[i - 1])):
            for ib, (b, fb) in enumerate(zip(B, st.fracs[i])):
                D[ia, ib] = paths.route_distance(int(a), float(fa), int(b), float(fb), limit, c["backtrack_tol_m"])
        D[D > limit] = np.inf
        steps.append((D, d_gps))
    return steps


def viterbi(st: TripStates, steps: list, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Most probable state per point. Returns (chosen arc per point or -1, chain id per point or -1)."""
    c = {**DEFAULTS, **cfg}
    n = len(st.arcs)
    best_state = np.full(n, -1, np.int64)
    chain = np.full(n, -1, np.int64)
    back = [None] * n
    delta = None
    chain_start, cid = None, -1

    def close_chain(end):
        if delta is None or chain_start is None:
            return
        k = int(np.argmax(delta))
        for j in range(end, chain_start - 1, -1):
            best_state[j] = k
            chain[j] = cid
            if j > chain_start:
                k = int(back[j][k])

    # emission weight: w_emission for model probabilities, 1/sigma^2 for the distance-only baseline
    w = float(c["w_emission"]) if st.kind == "model" else 1.0 / float(c["sigma_m"]) ** 2
    for i in range(n):
        E = w * st.emis[i]
        if len(E) == 0:
            close_chain(i - 1); delta, chain_start = None, None
            continue
        s = steps[i] if i > 0 else None
        if delta is None or s is None:
            close_chain(i - 1)
            delta, chain_start, cid = E.copy(), i, cid + 1
            continue
        D, d_gps = s
        logT = -np.abs(D - d_gps) / c["beta_m"]                    # -inf where D is inf
        tot = delta[:, None] + logT
        arg = np.argmax(tot, axis=0)
        best = tot[arg, np.arange(tot.shape[1])]
        if not np.isfinite(best).any():                            # no possible transition: new chain
            close_chain(i - 1)
            delta, chain_start, cid = E.copy(), i, cid + 1
            continue
        back[i] = arg
        delta = best + E
    close_chain(n - 1)
    arcs = np.array([st.arcs[i][k] if k >= 0 else -1 for i, k in enumerate(best_state)], np.int64)
    return arcs, chain


def classifier_only(st: TripStates) -> np.ndarray:
    """Top emission per point (no sequence information). With distance states this is the
    nearest-road baseline; with model states it is the classifier-only prediction."""
    return np.array([a[int(np.argmax(e))] if len(e) else -1 for a, e in zip(st.arcs, st.emis)], np.int64)


# ============================================================================ route reconstruction
def reconstruct_route(pred_arcs: np.ndarray, st: TripStates, paths: NetworkPaths, cfg: dict) -> tuple[list[int], int]:
    """Join consecutive matched arcs with shortest directed paths. Returns (arc list, number of
    breaks where no path within the limit existed or the time gap was too long)."""
    c = {**DEFAULTS, **cfg}
    route, breaks, prev, prev_t = [], 0, None, None
    for i, a in enumerate(pred_arcs):
        if a < 0:
            continue
        a = int(a)
        if prev is None:
            route.append(a)
        elif a != prev:
            dt = st.t[i] - prev_t
            limit = min(c["vmax_mps"] * max(dt, 1.0) + c["speed_slack_m"], c["max_route_m"])
            p = paths.path_arcs(int(paths.net.arc_to[prev]), int(paths.net.arc_from[a]), limit) if dt <= c["gap_break_s"] else None
            if p is None:
                breaks += 1
            else:
                route.extend(p)
            route.append(a)
        prev, prev_t = a, st.t[i]
    dedup = [r for k, r in enumerate(route) if k == 0 or r != route[k - 1]]
    return dedup, breaks


def continuity(pred_arcs: np.ndarray, st: TripStates, paths: NetworkPaths, cfg: dict) -> float:
    """Share of consecutive point pairs (both matched, time gap <= gap_break_s) whose matched arcs
    are the same or connected by a feasible directed path (same limit as the HMM)."""
    c = {**DEFAULTS, **cfg}
    ok = tot = 0
    for i in range(1, len(pred_arcs)):
        a, b = pred_arcs[i - 1], pred_arcs[i]
        dt = st.t[i] - st.t[i - 1]
        if a < 0 or b < 0 or dt > c["gap_break_s"] or dt <= 0:
            continue
        tot += 1
        if a == b:
            ok += 1; continue
        limit = min(c["vmax_mps"] * dt + c["speed_slack_m"], c["max_route_m"])
        dist, _ = paths.from_node(int(paths.net.arc_to[a]), limit)
        if dist.get(int(paths.net.arc_from[b]), np.inf) <= limit:
            ok += 1
    return ok / tot if tot else float("nan")


def route_metrics(pred_route: list[int], ref_route: np.ndarray, length_m: np.ndarray) -> dict:
    """Compare the reconstructed arc route with the reference route (arc sets, counted once).
    precision = share of predicted arcs that are on the reference route; recall = share of reference
    arcs that were predicted; length-weighted versions weight each arc by its length.
    route_mismatch (Newson & Krumm) = (length wrongly added + length missed) / reference length."""
    P = set(int(a) for a in pred_route); R = set(int(a) for a in np.asarray(ref_route))
    R = {a for a in R if 0 <= a < len(length_m)}
    if not R:
        return {k: float("nan") for k in ("arc_precision", "arc_recall", "arc_f1", "len_precision", "len_recall", "len_f1", "route_mismatch")}
    inter = P & R
    lp = sum(length_m[a] for a in P); lr = sum(length_m[a] for a in R); li = sum(length_m[a] for a in inter)
    prec = len(inter) / len(P) if P else 0.0; rec = len(inter) / len(R)
    lprec = li / lp if lp else 0.0; lrec = li / lr if lr else 0.0
    f = lambda p, r: 2 * p * r / (p + r) if (p + r) else 0.0
    return {"arc_precision": prec, "arc_recall": rec, "arc_f1": f(prec, rec), "len_precision": lprec, "len_recall": lrec,
            "len_f1": f(lprec, lrec), "route_mismatch": ((lp - li) + (lr - li)) / lr}


def point_accuracy(pred_arcs: np.ndarray, obs: pd.DataFrame, alt: dict | None = None) -> dict:
    """Pointwise accuracy against the aligned reference arc (aligned points), and lenient accuracy
    for ambiguous points (correct if the prediction is the aligned arc or a labelled alternative)."""
    st = obs.obs_status.to_numpy(); ra = obs.route_arc.to_numpy()
    al = st == "aligned"
    am = st == "ambiguous"
    out = {"aligned_points": int(al.sum()), "point_acc": float(np.mean(pred_arcs[al] == ra[al])) if al.any() else float("nan")}
    if am.any():
        hits = [(pred_arcs[i] == ra[i]) or (alt is not None and pred_arcs[i] in alt.get(int(obs.seq.iat[i]), ())) for i in np.flatnonzero(am)]
        out["ambiguous_points"] = int(am.sum()); out["amb_lenient_acc"] = float(np.mean(hits))
    out["matched_share"] = float(np.mean(pred_arcs >= 0))
    return out

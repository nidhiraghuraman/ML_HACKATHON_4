"""Road geometry for ONE trip, built from its own .nodes and .arcs.

Every arc keeps its original arc_id (= row in .arcs), its from/to node IDs and its direction.
Geometry is a straight segment from the from-node to the to-node in the trip's UTM metres.
Arcs that reference a missing node or have zero length are reported (not silently dropped)
and are excluded from the spatial index so they can never become candidates.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import shapely
from scipy.sparse import csr_matrix

from .geo import LocalProjector, bearing_deg


@dataclass
class RoadNetwork:
    n_nodes: int
    arc_from: np.ndarray          # (n_arcs,) int64 from-node id
    arc_to: np.ndarray            # (n_arcs,) int64 to-node id
    x0: np.ndarray                # start x (m)
    y0: np.ndarray
    x1: np.ndarray                # end x (m)
    y1: np.ndarray
    length_m: np.ndarray
    bearing: np.ndarray           # heading from-node -> to-node, degrees (NaN if zero length)
    valid: np.ndarray             # bool: usable as a candidate
    has_twin: np.ndarray          # bool: the reverse arc (to -> from) also exists
    node_in_deg: np.ndarray
    node_out_deg: np.ndarray
    valid_ids: np.ndarray         # arc ids that are in the spatial index (tree index -> arc id)
    tree: shapely.STRtree
    successors: csr_matrix        # node x node adjacency (1 where an arc from i to j exists)
    report: dict = field(default_factory=dict)

    @property
    def n_arcs(self) -> int:
        return len(self.arc_from)


def build_road_network(nodes_lonlat: np.ndarray, arcs: np.ndarray, projector: LocalProjector) -> RoadNetwork:
    n_nodes = len(nodes_lonlat)
    a_from = arcs[:, 0].astype(np.int64)
    a_to = arcs[:, 1].astype(np.int64)
    n_arcs = len(arcs)

    nx_, ny_ = projector.to_xy(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
    bad_ref = (a_from < 0) | (a_to < 0) | (a_from >= n_nodes) | (a_to >= n_nodes)
    f = np.where(bad_ref, 0, a_from)
    t = np.where(bad_ref, 0, a_to)
    x0, y0, x1, y1 = nx_[f], ny_[f], nx_[t], ny_[t]
    x0[bad_ref] = y0[bad_ref] = x1[bad_ref] = y1[bad_ref] = np.nan
    length = np.hypot(x1 - x0, y1 - y0)
    zero_len = ~bad_ref & (length == 0)
    valid = ~bad_ref & ~zero_len & np.isfinite(length)
    brg = np.where(valid, bearing_deg(x1 - x0, y1 - y0), np.nan)

    key = a_from * max(n_nodes, 1) + a_to
    rkey = a_to * max(n_nodes, 1) + a_from
    has_twin = np.isin(key, rkey) & ~bad_ref & (a_from != a_to)

    ok_f, ok_t = a_from[~bad_ref], a_to[~bad_ref]
    out_deg = np.bincount(ok_f, minlength=n_nodes)
    in_deg = np.bincount(ok_t, minlength=n_nodes)
    succ = csr_matrix((np.ones(len(ok_f), dtype=np.int8), (ok_f, ok_t)), shape=(n_nodes, n_nodes))

    valid_ids = np.flatnonzero(valid)
    segs = shapely.linestrings(np.stack([np.c_[x0[valid_ids], y0[valid_ids]],
                                         np.c_[x1[valid_ids], y1[valid_ids]]], axis=1))
    tree = shapely.STRtree(segs)

    dup = int(len(key[~bad_ref]) - len(np.unique(key[~bad_ref])))
    report = {
        "n_nodes": int(n_nodes), "n_arcs": int(n_arcs),
        "invalid_node_ref": int(bad_ref.sum()), "zero_length": int(zero_len.sum()),
        "self_loops": int((a_from == a_to).sum()), "duplicate_arcs": dup,
        "indexed_arcs": int(len(valid_ids)), "twin_frac": float(has_twin.mean()) if n_arcs else 0.0,
    }
    return RoadNetwork(n_nodes, a_from, a_to, x0, y0, x1, y1, length, brg, valid, has_twin,
                       in_deg, out_deg, valid_ids, tree, succ, report)

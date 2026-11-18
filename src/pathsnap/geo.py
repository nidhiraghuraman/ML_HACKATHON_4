"""Geographic helpers.

Longitude/latitude are angles, not metres. One degree of longitude is ~111 km at the
equator but ~55 km at 60 degrees north, and this dataset spans 34S to 61N. So every trip is
projected into its own UTM zone (a flat metre-based grid chosen from the trip's centre),
where straight-line distances are accurate to well under 1% across a trip-sized area.
"""
from __future__ import annotations
import numpy as np
from pyproj import Transformer

EARTH_RADIUS_M = 6_371_008.8


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing (lon, lat). 326xx = north, 327xx = south."""
    zone = int((lon + 180) // 6) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


class LocalProjector:
    """Converts lon/lat <-> metres (x, y) in the UTM zone of a reference point."""

    def __init__(self, ref_lon: float, ref_lat: float):
        self.epsg = utm_epsg(ref_lon, ref_lat)
        self._fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)
        self._inv = Transformer.from_crs(self.epsg, 4326, always_xy=True)

    @classmethod
    def for_points(cls, lon: np.ndarray, lat: np.ndarray) -> "LocalProjector":
        return cls(float(np.median(lon)), float(np.median(lat)))

    def to_xy(self, lon, lat):
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def to_lonlat(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x, float), np.asarray(y, float))
        return np.asarray(lon), np.asarray(lat)


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (used to cross-check the projection)."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def bearing_deg(dx, dy):
    """Compass-style heading of a movement (dx east, dy north) in degrees: 0 = north, 90 = east.

    Computed in the trip's UTM grid. Grid north differs from true north by a few degrees away
    from the zone's central meridian, but GPS and road headings are both measured in the same
    grid, so their DIFFERENCE (what the features use) is unaffected.
    """
    return np.mod(np.degrees(np.arctan2(np.asarray(dx, float), np.asarray(dy, float))), 360.0)


def angle_diff_deg(a, b):
    """Smallest absolute difference between two headings, in [0, 180] degrees (NaN-safe)."""
    d = np.abs(np.mod(np.asarray(a, float) - np.asarray(b, float), 360.0))
    return np.minimum(d, 360.0 - d)


def point_segment_distance(px, py, x0, y0, x1, y1):
    """Vectorised distance from points (px, py) to segments (x0,y0)->(x1,y1), all in metres.

    Returns (distance_m, frac) where frac in [0, 1] is how far along the segment the closest
    point lies (0 = start node, 1 = end node). Zero-length segments get frac = 0.
    """
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(L2 > 0, ((px - x0) * dx + (py - y0) * dy) / L2, 0.0)
    frac = np.clip(frac, 0.0, 1.0)
    cx, cy = x0 + frac * dx, y0 + frac * dy
    return np.hypot(px - cx, py - cy), frac

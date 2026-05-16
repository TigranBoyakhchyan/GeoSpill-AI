"""
Geographic utilities for SAR oil spill detection.

Provides:
    - Pixel-to-coordinate conversion for oil spill masks
    - Spill area calculation in km²
    - WGS84 bounding box extraction from GeoTIFFs
    - Linear-to-decibel SAR data conversion
    - Nearby coastline / country proximity analysis
"""

import math
import numpy as np


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

DB_CLIP_MIN = -50.0
DB_CLIP_MAX =  0.0

_EARTH_RADIUS_KM = 6_371.0
_EARTH_RADIUS_M  = 6_371_000.0


# ─── SAR DATA CONVERSION ─────────────────────────────────────────────────────

def is_linear_scale(img: np.ndarray) -> bool:
    """
    Heuristic: detect whether SAR data is in linear power scale vs decibels.
    Linear values are typically in [0, ~5]; dB values are negative.
    """
    band = img[0] if img.ndim == 3 else img
    valid = band[np.isfinite(band)]
    if len(valid) == 0:
        return False
    return float(valid.mean()) > -5.0 and float(valid.min()) >= 0.0


def linear_to_db(img: np.ndarray, floor_db: float = DB_CLIP_MIN) -> np.ndarray:
    """Convert SAR backscatter from linear power to decibels."""
    return np.where(
        img > 0,
        10.0 * np.log10(img + 1e-10),
        floor_db,
    ).astype(np.float32)


# ─── COORDINATE CONVERSION ───────────────────────────────────────────────────

def mask_to_latlon(mask: np.ndarray, transform, crs) -> list:
    """Convert oil spill mask pixels to (lat, lon) WGS-84 coordinates."""
    rows, cols = np.where(mask > 0.5)
    if len(rows) == 0:
        return []

    xs = transform.c + cols * transform.a
    ys = transform.f + rows * transform.e

    if crs and str(crs) != "EPSG:4326":
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            xs, ys = transformer.transform(xs, ys)
        except Exception:
            pass

    return list(zip(ys.tolist(), xs.tolist()))


def subsample(coords: list, max_points: int = 3000) -> list:
    """Subsample a coordinate list for map rendering performance."""
    if len(coords) <= max_points:
        return coords
    step = max(1, len(coords) // max_points)
    return coords[::step]


def get_wgs84_bounds(transform, crs, H: int, W: int):
    """Return (south, west, north, east) in WGS-84 degrees, or None."""
    if crs is None:
        return None
    try:
        from rasterio.warp import transform_bounds
        left   = transform.c
        top    = transform.f
        right  = left + W * transform.a
        bottom = top  + H * transform.e
        west, south, east, north = transform_bounds(
            crs, "EPSG:4326", left, bottom, right, top
        )
        return (south, west, north, east)
    except Exception:
        return None


# ─── AREA CALCULATION ─────────────────────────────────────────────────────────

def _pixel_area_projected(transform) -> float:
    """Pixel area in m² for a projected CRS."""
    return abs(transform.a * transform.e)


def _pixel_area_geographic(transform, mask: np.ndarray) -> np.ndarray:
    """Per-pixel area in m² for a geographic CRS (EPSG:4326)."""
    H, W = mask.shape
    d_lon = abs(transform.a)
    d_lat = abs(transform.e)

    row_indices = np.arange(H)
    lats = transform.f + (row_indices + 0.5) * transform.e

    rad = np.deg2rad(np.abs(lats))
    area_per_row = d_lon * d_lat * (math.pi / 180.0) ** 2 * _EARTH_RADIUS_M ** 2 * np.cos(rad)
    return np.broadcast_to(area_per_row[:, None], (H, W))


def calculate_spill_area(mask: np.ndarray, transform, crs) -> dict:
    """
    Calculate the area of detected oil spill in km².

    Returns dict with: area_km2, area_m2, oil_pixels, pixel_area_m2, crs_type
    """
    oil_pixels = int(np.sum(mask > 0.5))

    if oil_pixels == 0:
        return {"area_km2": 0.0, "area_m2": 0.0, "oil_pixels": 0,
                "pixel_area_m2": 0.0, "crs_type": "unknown"}

    if crs is None:
        return {"area_km2": None, "area_m2": None, "oil_pixels": oil_pixels,
                "pixel_area_m2": None, "crs_type": "no_crs"}

    is_geographic = crs.is_geographic if hasattr(crs, "is_geographic") else False

    if is_geographic:
        pixel_areas = _pixel_area_geographic(transform, mask)
        total_m2 = float(np.sum(pixel_areas * (mask > 0.5)))
        avg_pixel_area = float(np.mean(pixel_areas))
        crs_type = "geographic"
    else:
        pixel_m2 = _pixel_area_projected(transform)
        total_m2 = oil_pixels * pixel_m2
        avg_pixel_area = pixel_m2
        crs_type = "projected"

    return {"area_km2": total_m2 / 1e6, "area_m2": total_m2,
            "oil_pixels": oil_pixels, "pixel_area_m2": avg_pixel_area,
            "crs_type": crs_type}


# ─── PROXIMITY ANALYSIS ──────────────────────────────────────────────────────

def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two WGS-84 points."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return _EARTH_RADIUS_KM * 2 * math.asin(math.sqrt(a))


def _bearing(lat1, lon1, lat2, lon2) -> float:
    """Initial bearing in degrees from point 1 to point 2."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def _bearing_to_compass(bearing: float) -> str:
    """Convert bearing degrees to 8-point compass direction."""
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = round(bearing / 45) % 8
    return dirs[idx]


def spill_centroid(coords: list) -> tuple:
    """Compute the centroid (mean lat, mean lon) of oil pixel coordinates."""
    if not coords:
        return (0.0, 0.0)
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    return (float(np.mean(lats)), float(np.mean(lons)))


def find_nearby_coasts(center_lat: float,
                       center_lon: float,
                       radius_km:  float = 200.0,
                       max_results: int  = 8) -> list:
    """
    Find countries whose coastlines are within radius_km of a point.

    Uses the Natural Earth low-resolution country dataset bundled with
    geopandas (no downloads, no API calls, works offline).

    Args:
        center_lat:  spill centroid latitude (WGS-84)
        center_lon:  spill centroid longitude (WGS-84)
        radius_km:   search radius in km
        max_results: maximum number of countries to return

    Returns:
        List of dicts, sorted by distance, each with:
            name:         country name
            distance_km:  distance from spill centroid to nearest coast point
            bearing_deg:  bearing from spill to nearest coast point
            direction:    compass direction (e.g. "NE")
            nearest_lat:  latitude of nearest coast point
            nearest_lon:  longitude of nearest coast point
        Empty list if geopandas is not installed or no coasts found.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point
        from shapely.ops import nearest_points
    except ImportError:
        return []

    world = _load_countries(gpd)
    if world is None or world.empty:
        return []

    # Detect the correct column name for country name and ISO code.
    # Different versions and sources use different column names:
    #   naturalearth_lowres (old gpd): "name", "iso_a3"
    #   ne_110m_admin_0_countries:     "ADMIN" or "NAME", "ISO_A3"
    #   geodatasets:                   "name", "iso_a3"
    name_col = _find_column(world, ["name", "NAME", "ADMIN", "NAME_EN",
                                     "admin", "NAME_LONG", "SOVEREIGNT"])
    iso_col  = _find_column(world, ["iso_a3", "ISO_A3", "ISO_A3_EH",
                                     "SOV_A3", "ADM0_A3"])

    spill_point = Point(center_lon, center_lat)

    # Rough degree-based pre-filter (1° latitude ≈ 111 km)
    deg_buffer = radius_km / 111.0 * 1.5
    candidates = world.cx[
        center_lon - deg_buffer : center_lon + deg_buffer,
        center_lat - deg_buffer : center_lat + deg_buffer
    ]

    results = []
    for _, row in candidates.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Find nearest point on country boundary to spill
        nearest_on_coast, _ = nearest_points(geom.boundary, spill_point)
        coast_lat = nearest_on_coast.y
        coast_lon = nearest_on_coast.x

        dist = _haversine(center_lat, center_lon, coast_lat, coast_lon)
        if dist > radius_km:
            continue

        brg = _bearing(center_lat, center_lon, coast_lat, coast_lon)

        country_name = str(row[name_col]) if name_col else "Unknown"
        iso_code     = str(row[iso_col])  if iso_col  else "???"

        results.append({
            "name":        country_name,
            "iso_a3":      iso_code,
            "distance_km": round(dist, 1),
            "bearing_deg": round(brg, 1),
            "direction":   _bearing_to_compass(brg),
            "nearest_lat": round(coast_lat, 4),
            "nearest_lon": round(coast_lon, 4),
        })

    results.sort(key=lambda x: x["distance_km"])
    return results[:max_results]


def _load_countries(gpd):
    """
    Load world country polygons, trying multiple sources in order:
        1. geodatasets package (geopandas 1.0+ replacement)
        2. Legacy gpd.datasets (geopandas < 1.0)
        3. Direct URL download from Natural Earth
    """
    # 1. geodatasets (geopandas >= 1.0)
    try:
        import geodatasets
        path = geodatasets.get_path("naturalearth.land")
        # geodatasets has land but not countries — try the admin one
    except Exception:
        pass

    try:
        import geodatasets
        return gpd.read_file(
            geodatasets.get_url("naturalearth.admin_0_countries")
        )
    except Exception:
        pass

    # 2. Legacy gpd.datasets (geopandas < 1.0)
    try:
        return gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    except Exception:
        pass

    # 3. Direct URL download
    urls = [
        "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip",
        "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson",
    ]
    for url in urls:
        try:
            return gpd.read_file(url)
        except Exception:
            continue

    return None


def _find_column(df, candidates: list):
    """Find the first matching column name from a list of candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None
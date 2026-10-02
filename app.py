"""
AeroRoute // Airspace Deconfliction Control
Aerospace Mission Control Center & Tactical Airspace Management System
Featuring Discrete Mathematics Graph Operations, Warshall Reachability Engine,
and Interactive Flight Corridor Deconfliction with Custom Airspace Datasets.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, shape
import shapely.wkt
import json
import io
import networkx as nx

# ─────────────────────────────────────────────
# 1. PAGE CONFIGURATION & THEME
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AeroRoute // Airspace Deconfliction Control",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# MISSION CONTROL CSS — High-Tech Dark Aerospace HUD
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  .stApp {
    background: radial-gradient(circle at 50% 0%, #0d1627 0%, #070a14 55%, #04060b 100%);
    color: #e2e8f0;
  }

  /* Mission Control Top Header Bar */
  .mission-header {
    background: linear-gradient(135deg, rgba(13, 22, 39, 0.95) 0%, rgba(10, 16, 29, 0.92) 100%);
    border: 1px solid rgba(56, 189, 248, 0.22);
    border-radius: 14px;
    padding: 16px 24px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    backdrop-filter: blur(14px);
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.05);
  }

  .mission-title-box {
    display: flex;
    flex-direction: column;
  }

  .mission-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    color: #38bdf8;
    letter-spacing: 1.6px;
    text-transform: uppercase;
    margin-bottom: 2px;
  }

  .mission-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.65rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #ffffff 0%, #38bdf8 60%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.4px;
  }

  .mission-subtitle {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 3px;
  }

  /* Live System Status Badges */
  .mission-status-group {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }

  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 999px;
    padding: 6px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
  }

  .status-pill.online {
    border-color: rgba(16, 185, 129, 0.4);
    background: rgba(16, 185, 129, 0.08);
    color: #34d399;
  }

  .status-pill.standby {
    border-color: rgba(245, 158, 11, 0.4);
    background: rgba(245, 158, 11, 0.08);
    color: #fbbf24;
  }

  .status-pill.cyan {
    border-color: rgba(56, 189, 248, 0.35);
    background: rgba(56, 189, 248, 0.08);
    color: #38bdf8;
  }

  .status-pill.purple {
    border-color: rgba(167, 139, 250, 0.35);
    background: rgba(167, 139, 250, 0.08);
    color: #c084fc;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
  }

  .dot-green {
    background: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: radarPulse 1.8s infinite;
  }

  .dot-amber {
    background: #f59e0b;
    box-shadow: 0 0 8px #f59e0b;
  }

  .dot-cyan {
    background: #0ea5e9;
    box-shadow: 0 0 8px #0ea5e9;
  }

  .dot-purple {
    background: #a855f7;
    box-shadow: 0 0 8px #a855f7;
  }

  @keyframes radarPulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
    70% { transform: scale(1.15); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
  }

  /* Metric Containers */
  [data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(13, 20, 36, 0.8) 0%, rgba(17, 24, 39, 0.75) 100%) !important;
    border: 1px solid rgba(56, 189, 248, 0.18) !important;
    border-radius: 10px !important;
    padding: 10px 16px !important;
    backdrop-filter: blur(8px) !important;
    transition: transform 0.2s ease, border-color 0.2s ease;
  }
  [data-testid="stMetric"]:hover {
    border-color: rgba(56, 189, 248, 0.4) !important;
    transform: translateY(-1px);
  }
  [data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    color: #64748b !important;
  }
  [data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #38bdf8 !important;
    font-weight: 700 !important;
    font-size: 1.55rem !important;
  }

  /* Sidebar Tactical Styling */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090e18 0%, #060910 100%) !important;
    border-right: 1px solid rgba(56, 189, 248, 0.15) !important;
  }

  .sidebar-hud-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    border-bottom: 1px solid rgba(56, 189, 248, 0.12);
    padding-bottom: 4px;
    margin: 18px 0 10px 0;
  }

  [data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.3px !important;
    padding: 9px 0 !important;
    box-shadow: 0 4px 14px rgba(2, 132, 199, 0.28) !important;
    transition: all 0.2s ease !important;
  }
  [data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(2, 132, 199, 0.45) !important;
  }
  [data-testid="stSidebar"] .stButton > button:disabled {
    background: rgba(30, 41, 59, 0.4) !important;
    color: #475569 !important;
    border-color: rgba(51, 65, 85, 0.4) !important;
    box-shadow: none !important;
    cursor: not-allowed !important;
  }

  /* Telemetry Inspector Tabs */
  [data-baseweb="tab-list"] {
    background: rgba(13, 20, 36, 0.75) !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
  }
  [data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    border-radius: 6px !important;
    padding: 8px 14px !important;
  }
  [data-baseweb="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.28) 0%, rgba(99, 102, 241, 0.28) 100%) !important;
    color: #38bdf8 !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
  }

  /* Radar map container border */
  .radar-frame {
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
  }

  .flight-row {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(56, 189, 248, 0.1);
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 6px;
    transition: all 0.2s ease;
  }
  .flight-row:hover {
    background: rgba(15, 23, 42, 0.85);
    border-color: rgba(56, 189, 248, 0.3);
  }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATHEMATICAL & COLOR UTILITIES
# ─────────────────────────────────────────────
def interpolate_point(p1, p2, t):
    return [p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t]

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(max(0.0, a)))

def corridor_color(layer, status):
    if status == "conflict": return "#ef4444"
    if status == "priority": return "#f59e0b"
    return {"low": "#34d399", "mid": "#38bdf8", "high": "#a78bfa"}.get(layer, "#94a3b8")

def node_color(t):
    return {"hub":"#f59e0b","terminal":"#38bdf8","relay":"#a78bfa","depot":"#34d399",
            "priority":"#ef4444","nfz":"#dc2626","landing":"#fb923c"}.get(t, "#38bdf8")

def node_icon(t):
    return {"hub":"home","terminal":"plane","relay":"signal","depot":"archive",
            "priority":"plus","nfz":"ban","landing":"map-marker"}.get(t, "circle")


# ─────────────────────────────────────────────
# AIRSPACE DATASET PARSER (CSV / GeoJSON)
# ─────────────────────────────────────────────
def parse_airspace_dataset(uploaded_file) -> tuple[dict, list, list, list]:
    """
    Parse an uploaded CSV or GeoJSON dataset into:
    - waypoints: dict[node_id -> {name, lat, lon, type, alt}]
    - corridors: list[(src_id, dst_id, layer, speed, status)]
    - restricted_zones: list[{name, geometry, raw_geometry, type, coordinates, radius}]
    - drones: list[{id, from, to, progress, speed, status}]
    """
    if uploaded_file is None:
        return {}, [], [], []

    try:
        uploaded_file.seek(0)
    except Exception:
        pass

    filename = uploaded_file.name.lower()
    waypoints = {}
    corridors = []
    restricted_zones = []
    drones = []

    try:
        # ── 1. GeoJSON Parsing via GeoPandas ──
        if filename.endswith(".geojson") or filename.endswith(".json"):
            try:
                gdf = gpd.read_file(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                data = json.load(uploaded_file)
                if "features" in data:
                    gdf = gpd.GeoDataFrame.from_features(data["features"])
                else:
                    gdf = gpd.GeoDataFrame(geometry=[shape(data.get("geometry", data))])

            if gdf.crs is not None and str(gdf.crs).lower() not in ("epsg:4326", "wgs 84", "4326"):
                try:
                    gdf = gdf.to_crs(epsg=4326)
                except Exception:
                    pass

            for idx, row in gdf.iterrows():
                geom = row.geometry
                if geom is None or geom.is_empty:
                    continue

                name = str(row.get("name") or row.get("label") or row.get("id") or f"Zone_{idx+1}")
                ptype = str(row.get("type") or "").lower()

                if (geom.geom_type in ("Polygon", "MultiPolygon") or
                    ptype in ("nfz", "restricted", "no_fly_zone") or
                    "nfz" in name.lower() or "restricted" in name.lower()):

                    coords = []
                    if geom.geom_type == "Polygon":
                        coords = [[lat, lon] for lon, lat in geom.exterior.coords]
                    elif geom.geom_type == "MultiPolygon":
                        coords = [[[lat, lon] for lon, lat in p.exterior.coords] for p in geom.geoms]
                    elif geom.geom_type == "Point":
                        coords = [geom.y, geom.x]

                    radius_m = float(row.get("radius") or row.get("radius_m") or 150.0)
                    eff_geom = geom
                    if geom.geom_type == "Point":
                        eff_geom = geom.buffer(radius_m / 111320.0)

                    restricted_zones.append({
                        "name": name,
                        "geometry": eff_geom,
                        "raw_geometry": geom,
                        "type": geom.geom_type,
                        "coordinates": coords,
                        "radius": radius_m
                    })

                elif geom.geom_type == "Point":
                    node_id = str(row.get("id") or row.get("node_id") or f"WP_{idx+1:03d}")
                    waypoints[node_id] = {
                        "name": name,
                        "lat": float(geom.y),
                        "lon": float(geom.x),
                        "type": ptype if ptype in ("hub", "terminal", "relay", "depot", "landing") else "terminal",
                        "alt": float(row.get("alt") or row.get("altitude") or 80.0),
                        "osm_id": node_id,
                        "is_restricted": False
                    }

                elif geom.geom_type == "LineString":
                    src = str(row.get("source") or row.get("from") or "")
                    dst = str(row.get("target") or row.get("to") or "")
                    layer = str(row.get("layer") or "mid").lower()
                    speed = int(row.get("speed") or 65)
                    corridors.append((src, dst, layer, speed, "active", geom))

            if waypoints:
                for c_idx, c in enumerate(corridors):
                    if len(c) == 6:
                        src, dst, layer, speed, status, l_geom = c
                        if (not src or src not in waypoints) and l_geom:
                            start_pt = l_geom.coords[0]
                            src = min(waypoints.keys(), key=lambda k: (waypoints[k]["lon"] - start_pt[0])**2 + (waypoints[k]["lat"] - start_pt[1])**2)
                        if (not dst or dst not in waypoints) and l_geom:
                            end_pt = l_geom.coords[-1]
                            dst = min(waypoints.keys(), key=lambda k: (waypoints[k]["lon"] - end_pt[0])**2 + (waypoints[k]["lat"] - end_pt[1])**2 if k != src else 999)
                        corridors[c_idx] = (src, dst, layer, speed, status)

        # ── 2. CSV Parsing via Pandas ──
        elif filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            col_map = {str(c).lower().strip(): c for c in df.columns}

            lat_col = next((col_map[c] for c in ["latitude", "lat", "y", "lat_deg"] if c in col_map), None)
            lon_col = next((col_map[c] for c in ["longitude", "lon", "lng", "x", "lon_deg"] if c in col_map), None)
            id_col = next((col_map[c] for c in ["id", "node_id", "waypoint_id", "code"] if c in col_map), None)
            name_col = next((col_map[c] for c in ["name", "label", "title"] if c in col_map), None)
            type_col = next((col_map[c] for c in ["type", "entity", "classification", "kind"] if c in col_map), None)
            alt_col = next((col_map[c] for c in ["altitude", "alt", "height"] if c in col_map), None)
            radius_col = next((col_map[c] for c in ["radius", "radius_m", "buffer"] if c in col_map), None)
            conn_col = next((col_map[c] for c in ["connected_to", "connections", "edges", "target", "to"] if c in col_map), None)
            wkt_col = next((col_map[c] for c in ["geometry", "wkt", "geom", "polygon"] if c in col_map), None)

            for idx, row in df.iterrows():
                row_type = str(row[type_col]).lower().strip() if type_col and pd.notna(row[type_col]) else ""
                name = str(row[name_col]) if name_col and pd.notna(row[name_col]) else f"Item_{idx+1}"

                if wkt_col and pd.notna(row[wkt_col]):
                    try:
                        g_val = shapely.wkt.loads(str(row[wkt_col]).strip())
                        if g_val.geom_type in ("Polygon", "MultiPolygon") or row_type in ("nfz", "restricted"):
                            coords = [[lat, lon] for lon, lat in g_val.exterior.coords] if g_val.geom_type == "Polygon" else []
                            rad = float(row[radius_col]) if radius_col and pd.notna(row[radius_col]) else 150.0
                            restricted_zones.append({
                                "name": name,
                                "geometry": g_val,
                                "raw_geometry": g_val,
                                "type": g_val.geom_type,
                                "coordinates": coords,
                                "radius": rad
                            })
                            continue
                    except Exception:
                        pass

                if lat_col and lon_col and pd.notna(row[lat_col]) and pd.notna(row[lon_col]):
                    lat = float(row[lat_col])
                    lon = float(row[lon_col])

                    if row_type in ("nfz", "restricted", "no_fly_zone") or "nfz" in name.lower() or "restricted" in name.lower():
                        rad = float(row[radius_col]) if radius_col and pd.notna(row[radius_col]) else 150.0
                        pt = Point(lon, lat)
                        restricted_zones.append({
                            "name": name,
                            "geometry": pt.buffer(rad / 111320.0),
                            "raw_geometry": pt,
                            "type": "Point",
                            "coordinates": [lat, lon],
                            "radius": rad
                        })
                    else:
                        node_id = str(row[id_col]) if id_col and pd.notna(row[id_col]) else f"WP_{idx+1:03d}"
                        waypoints[node_id] = {
                            "name": name,
                            "lat": lat,
                            "lon": lon,
                            "type": row_type if row_type in ("hub", "terminal", "relay", "depot", "landing") else "terminal",
                            "alt": float(row[alt_col]) if alt_col and pd.notna(row[alt_col]) else 80.0,
                            "osm_id": node_id,
                            "is_restricted": False
                        }
                        if conn_col and pd.notna(row[conn_col]):
                            targets = [t.strip() for t in str(row[conn_col]).replace(";", ",").split(",") if t.strip()]
                            for tgt in targets:
                                corridors.append((node_id, tgt, "mid", 65, "active"))

        # Build graph corridors between nearest nodes if none were specified
        if waypoints and not corridors:
            wp_keys = list(waypoints.keys())
            layers = ["low", "mid", "high"]
            speeds = [50, 65, 80]
            for i in range(len(wp_keys)):
                distances = []
                for j in range(len(wp_keys)):
                    if i != j:
                        d = haversine_km(waypoints[wp_keys[i]]["lat"], waypoints[wp_keys[i]]["lon"],
                                         waypoints[wp_keys[j]]["lat"], waypoints[wp_keys[j]]["lon"])
                        distances.append((d, wp_keys[j]))
                distances.sort()
                for d, neighbor in distances[:2]:
                    layer = layers[len(corridors) % 3]
                    speed = speeds[len(corridors) % 3]
                    corridors.append((wp_keys[i], neighbor, layer, speed, "active"))

    except Exception as e:
        st.sidebar.error(f"Error parsing airspace dataset: {e}")
        return {}, [], [], []

    return waypoints, corridors, restricted_zones, drones


# ─────────────────────────────────────────────
# DISCRETE MATHEMATICS — GRAPH OPERATIONS MODULE
# ─────────────────────────────────────────────
def build_adjacency_matrix(nodes: list[str], edges: list[tuple[str, str]]) -> np.ndarray:
    """Build a boolean Adjacency Matrix from a list of node IDs and directed edges."""
    n = len(nodes)
    index = {node: i for i, node in enumerate(nodes)}
    A = np.zeros((n, n), dtype=bool)

    for edge in edges:
        src, dst = edge[0], edge[1]
        if src in index and dst in index:
            A[index[src]][index[dst]] = True

    return A


def warshall_closure(A: np.ndarray) -> np.ndarray:
    """
    Compute the Transitive Closure of a directed graph using Warshall's Algorithm.
    Recurrence: R[i][j] = R[i][j] or (R[i][k] and R[k][j])
    Complexity: O(n³)
    """
    n = A.shape[0]
    R = A.copy()

    for k in range(n):
        for i in range(n):
            if R[i, k]:
                for j in range(n):
                    R[i, j] = R[i, j] or R[k, j]

    return R


def matrices_as_dataframes(
    nodes: list[str],
    A: np.ndarray,
    R: np.ndarray,
    label_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wrap adjacency and reachability matrices as labelled DataFrames."""
    labels = [label_map.get(n, n) if label_map else n for n in nodes]
    df_adj   = pd.DataFrame(A.astype(int), index=labels, columns=labels)
    df_reach = pd.DataFrame(R.astype(int), index=labels, columns=labels)
    return df_adj, df_reach


def find_path_bfs(start: str, end: str, adj_list: dict[str, list[str]]) -> list[str] | None:
    """Find shortest-hop directed path using BFS."""
    if start == end:
        return [start]
    queue: list[list[str]] = [[start]]
    visited: set[str] = {start}

    while queue:
        path = queue.pop(0)
        current = path[-1]
        for neighbour in adj_list.get(current, []):
            if neighbour in visited:
                continue
            new_path = path + [neighbour]
            if neighbour == end:
                return new_path
            visited.add(neighbour)
            queue.append(new_path)
    return None


def validate_flight_plan(
    start: str,
    end: str,
    graph_nodes: list[str],
    reach_matrix: np.ndarray,
    adj_list: dict[str, list[str]],
    nfz_nodes: set[str],
    waypoints: dict,
    corridors: list | None = None,
    restricted_zones: list | None = None,
) -> dict:
    """Validate flight corridor using Warshall Reachability Matrix & NFZ scan with safe bypass fallback."""
    corridors = corridors or []
    restricted_zones = restricted_zones or []

    if start == end:
        return {"status": "same_node", "path": [start], "conflicts": [], "distance": 0.0, "has_fallback": False}

    if start not in graph_nodes or end not in graph_nodes:
        return {"status": "unreachable", "path": [], "conflicts": [], "distance": 0.0, "has_fallback": False}

    start_wp = waypoints.get(start, {})
    end_wp = waypoints.get(end, {})
    start_restricted = start_wp.get("is_restricted", False) or (start in nfz_nodes) or start_wp.get("type") == "nfz"
    end_restricted = end_wp.get("is_restricted", False) or (end in nfz_nodes) or end_wp.get("type") == "nfz"

    if start_restricted or end_restricted:
        conflicts = []
        if start_restricted:
            conflicts.append(f"{start} (Origin in NFZ)")
        if end_restricted:
            conflicts.append(f"{end} (Destination in NFZ)")
        return {
            "status": "nfz_conflict",
            "has_fallback": False,
            "path": [start, end],
            "conflicts": conflicts,
            "distance": 0.0,
            "fallback_path": None,
            "fallback_distance": 0.0
        }

    # 1. Primary path extraction via BFS / Warshall
    idx = {n: i for i, n in enumerate(graph_nodes)}
    reach_ok = bool(reach_matrix[idx[start]][idx[end]]) if (reach_matrix.size > 0 and start in idx and end in idx) else False
    primary_path = find_path_bfs(start, end, adj_list)

    if not reach_ok and not primary_path:
        # Check if safe graph can find a bypass path even if standard directed reachability had a gap
        safe_fb, safe_dist = find_safe_path_networkx(start, end, graph_nodes, corridors, waypoints, restricted_zones, nfz_nodes)
        if safe_fb:
            return {
                "status": "safe",
                "path": safe_fb,
                "conflicts": [],
                "distance": safe_dist,
                "has_fallback": False
            }
        return {"status": "unreachable", "path": [], "conflicts": [], "distance": 0.0, "has_fallback": False}

    if primary_path is None:
        primary_path = [start, end]

    # 2. Check primary path for NFZ conflicts
    conflicts = [n for n in primary_path if waypoints.get(n, {}).get("is_restricted", False) or n in nfz_nodes or waypoints.get(n, {}).get("type") == "nfz"]

    for i in range(len(primary_path) - 1):
        u, v = primary_path[i], primary_path[i+1]
        if u in waypoints and v in waypoints:
            u_pt = Point(waypoints[u]["lon"], waypoints[u]["lat"])
            v_pt = Point(waypoints[v]["lon"], waypoints[v]["lat"])
            corridor_line = LineString([u_pt, v_pt])
            for uz in restricted_zones:
                if uz["geometry"].intersects(corridor_line):
                    tag = f"{uz['name']} (Intersects {u}→{v})"
                    if tag not in conflicts:
                        conflicts.append(tag)

    primary_dist = sum(
        haversine_km(
            waypoints[primary_path[i]]["lat"], waypoints[primary_path[i]]["lon"],
            waypoints[primary_path[i+1]]["lat"], waypoints[primary_path[i+1]]["lon"]
        )
        for i in range(len(primary_path) - 1)
    ) if len(primary_path) >= 2 else 0.0

    # 3. Route decision
    if not conflicts:
        return {
            "status": "safe",
            "path": primary_path,
            "conflicts": [],
            "distance": primary_dist,
            "has_fallback": False,
            "fallback_path": None,
            "fallback_distance": 0.0
        }

    # Primary route rejected due to NFZ conflict! Compute automated safe-path routing fallback via NetworkX
    safe_path, safe_dist = find_safe_path_networkx(
        start, end, graph_nodes, corridors, waypoints, restricted_zones, nfz_nodes
    )

    if safe_path and len(safe_path) >= 2:
        return {
            "status": "nfz_conflict",
            "has_fallback": True,
            "path": primary_path,
            "conflicts": conflicts,
            "distance": primary_dist,
            "fallback_path": safe_path,
            "fallback_distance": safe_dist,
            "fallback_hops": len(safe_path) - 1
        }
    else:
        return {
            "status": "nfz_conflict",
            "has_fallback": False,
            "path": primary_path,
            "conflicts": conflicts,
            "distance": primary_dist,
            "fallback_path": None,
            "fallback_distance": 0.0
        }


def find_safe_path_networkx(
    start: str,
    end: str,
    graph_nodes: list[str],
    corridors: list,
    waypoints: dict,
    restricted_zones: list,
    nfz_nodes: set[str]
) -> tuple[list[str] | None, float]:
    """
    Compute automated safe detour corridor using NetworkX Dijkstra / BFS pathfinding.
    All nodes with is_restricted == True (and their immediate safety radius buffers)
    as well as corridor segments intersecting restricted airspace are strictly excluded.
    """
    if start == end:
        return [start], 0.0

    # 1. Filter nodes: Exclude any node marked is_restricted or in NFZ buffer
    G_safe = nx.DiGraph()
    for n in graph_nodes:
        wp = waypoints.get(n)
        if not wp:
            continue
        if wp.get("is_restricted", False) or n in nfz_nodes or wp.get("type") == "nfz":
            continue
        pt = Point(wp["lon"], wp["lat"])
        # Check immediate safety radius buffer from restricted zones
        in_buffer = any(
            uz["geometry"].contains(pt) or uz["geometry"].distance(pt) < 0.0005
            for uz in restricted_zones
        )
        if in_buffer:
            continue
        G_safe.add_node(n)

    if start not in G_safe or end not in G_safe:
        return None, 0.0

    # 2. Add safe edges with geographic distance weights
    for c in corridors:
        u, v = c[0], c[1]
        status = c[4] if len(c) > 4 else "active"
        if status == "conflict":
            continue
        if u in G_safe and v in G_safe:
            u_pt = Point(waypoints[u]["lon"], waypoints[u]["lat"])
            v_pt = Point(waypoints[v]["lon"], waypoints[v]["lat"])
            corridor_line = LineString([u_pt, v_pt])
            if any(uz["geometry"].intersects(corridor_line) for uz in restricted_zones):
                continue
            dist_w = haversine_km(
                waypoints[u]["lat"], waypoints[u]["lon"],
                waypoints[v]["lat"], waypoints[v]["lon"]
            )
            G_safe.add_edge(u, v, weight=dist_w)

    # 3. Pathfinding via Dijkstra (nx.shortest_path with weight)
    try:
        path = nx.shortest_path(G_safe, source=start, target=end, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        # Fallback to undirected traversal if directed path is blocked but safe segments allow bidirectional travel
        try:
            path = nx.shortest_path(G_safe.to_undirected(), source=start, target=end, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None, 0.0

    total_dist = sum(
        haversine_km(
            waypoints[path[i]]["lat"], waypoints[path[i]]["lon"],
            waypoints[path[i+1]]["lat"], waypoints[path[i+1]]["lon"]
        )
        for i in range(len(path) - 1)
    )
    return path, total_dist



# ─────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────
if "airspace_data" not in st.session_state:
    st.session_state["airspace_data"] = {
        "waypoints": {},
        "corridors": [],
        "restricted_zones": [],
        "drones": []
    }
if "dataset_filename" not in st.session_state:
    st.session_state["dataset_filename"] = ""
if "warshall_executed" not in st.session_state:
    st.session_state["warshall_executed"] = False
if "fp_result" not in st.session_state:
    st.session_state["fp_result"] = None
if "fp_start" not in st.session_state:
    st.session_state["fp_start"] = ""
if "fp_end" not in st.session_state:
    st.session_state["fp_end"] = ""
if "show_conflicts" not in st.session_state:
    st.session_state.show_conflicts = True
if "show_priority" not in st.session_state:
    st.session_state.show_priority = True
if "show_low" not in st.session_state:
    st.session_state.show_low = True
if "show_mid" not in st.session_state:
    st.session_state.show_mid = True
if "show_high" not in st.session_state:
    st.session_state.show_high = True
if "altitude_filter" not in st.session_state:
    st.session_state.altitude_filter = (0, 250)
if "conflict_radius" not in st.session_state:
    st.session_state.conflict_radius = 120

# Extract data strictly from session state
WAYPOINTS = dict(st.session_state["airspace_data"].get("waypoints", {}))
CORRIDORS = list(st.session_state["airspace_data"].get("corridors", []))
RESTRICTED_ZONES = list(st.session_state["airspace_data"].get("restricted_zones", []))
DRONES = list(st.session_state["airspace_data"].get("drones", []))

has_valid_data = bool(WAYPOINTS or RESTRICTED_ZONES)

# ─────────────────────────────────────────────
# DECONFLICTION: UPDATE WAYPOINTS & CORRIDORS AGAINST RESTRICTED ZONES
# ─────────────────────────────────────────────
if has_valid_data:
    for wp_id, wp in WAYPOINTS.items():
        wp.setdefault("is_restricted", wp.get("type") == "nfz")

if has_valid_data and RESTRICTED_ZONES:
    for wp_id, wp in WAYPOINTS.items():
        wpt = Point(wp["lon"], wp["lat"])
        for uz in RESTRICTED_ZONES:
            if uz["geometry"].contains(wpt) or uz["geometry"].distance(wpt) < 0.0005:
                wp["type"] = "nfz"
                wp["is_restricted"] = True
                if not wp["name"].startswith("⛔"):
                    wp["name"] = f"⛔ NFZ ({uz['name']}): {wp['name']}"
                break

    for idx, c in enumerate(CORRIDORS):
        u, v, layer, speed, status = c[0], c[1], c[2], c[3], c[4]
        if u in WAYPOINTS and v in WAYPOINTS:
            u_pt = Point(WAYPOINTS[u]["lon"], WAYPOINTS[u]["lat"])
            v_pt = Point(WAYPOINTS[v]["lon"], WAYPOINTS[v]["lat"])
            corridor_line = LineString([u_pt, v_pt])

            is_blocked = (
                WAYPOINTS[u].get("is_restricted", False) or
                WAYPOINTS[v].get("is_restricted", False) or
                WAYPOINTS[u]["type"] == "nfz" or
                WAYPOINTS[v]["type"] == "nfz" or
                any(uz["geometry"].intersects(corridor_line) or uz["geometry"].contains(u_pt) or uz["geometry"].contains(v_pt) for uz in RESTRICTED_ZONES)
            )
            if is_blocked:
                CORRIDORS[idx] = (u, v, layer, speed, "conflict")

# ─────────────────────────────────────────────
# PRECOMPUTE MATRICES (ONLY WHEN VALID DATA EXISTS)
# ─────────────────────────────────────────────
_GRAPH_NODES: list[str] = list(WAYPOINTS.keys())
_GRAPH_EDGES: list[tuple[str, str]] = [(c[0], c[1]) for c in CORRIDORS if c[0] in WAYPOINTS and c[1] in WAYPOINTS]
_SHORT_LABEL: dict[str, str] = {nid: nid[-4:] for nid in _GRAPH_NODES}

if has_valid_data and _GRAPH_NODES:
    _ADJ_MATRIX = build_adjacency_matrix(_GRAPH_NODES, _GRAPH_EDGES)
    _REACH_MATRIX = warshall_closure(_ADJ_MATRIX)
    _DF_ADJ, _DF_REACH = matrices_as_dataframes(_GRAPH_NODES, _ADJ_MATRIX, _REACH_MATRIX, label_map=_SHORT_LABEL)
else:
    _ADJ_MATRIX = np.zeros((0, 0), dtype=bool)
    _REACH_MATRIX = np.zeros((0, 0), dtype=bool)
    _DF_ADJ = pd.DataFrame()
    _DF_REACH = pd.DataFrame()

_NFZ_NODES: set[str] = {wp_id for wp_id, wp in WAYPOINTS.items() if wp["type"] == "nfz"}

_ADJ_LIST: dict[str, list[str]] = {n: [] for n in _GRAPH_NODES}
for _src, _dst in _GRAPH_EDGES:
    _ADJ_LIST[_src].append(_dst)


# ─────────────────────────────────────────────
# SIDEBAR — MISSION CONTROL DISPATCH & UPLOADER
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:6px 0 14px;">
      <div style="font-family:'Space Grotesk',sans-serif;font-size:1.45rem;font-weight:800;
                  background:linear-gradient(90deg,#38bdf8,#818cf8);-webkit-background-clip:text;
                  -webkit-text-fill-color:transparent;letter-spacing:-0.4px;">
        🛸 AEROROUTE // HUD
      </div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#64748b;letter-spacing:1.2px;text-transform:uppercase;margin-top:2px;">
        Mission Control Dispatch
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Dataset Upload Widget ────────────────────────────────────────────────
    st.markdown('<div class="sidebar-hud-title">📡 Airspace Telemetry Ingest</div>', unsafe_allow_html=True)

    uploaded_zones_file = st.file_uploader(
        "Upload Restricted Zones (CSV/GeoJSON)",
        type=["csv", "geojson", "json"],
        help="Upload CSV or GeoJSON dataset containing nodes, corridors, and restricted no-fly zones."
    )

    process_btn = st.button("Process Uploaded Data", use_container_width=True)

    if process_btn:
        if uploaded_zones_file is not None:
            wps, cors, rzs, drs = parse_airspace_dataset(uploaded_zones_file)
            if wps or rzs:
                st.session_state["airspace_data"] = {
                    "waypoints": wps,
                    "corridors": cors,
                    "restricted_zones": rzs,
                    "drones": drs
                }
                st.session_state["dataset_filename"] = uploaded_zones_file.name
                st.session_state["warshall_executed"] = True
                st.session_state["fp_result"] = None
                st.sidebar.success(f"✅ Ingested: {len(wps)} nodes, {len(cors)} corridors, {len(rzs)} NFZs")
                st.rerun()
            else:
                st.sidebar.warning("⚠️ No valid nodes or restricted zones found in file.")
        else:
            st.sidebar.warning("⚠️ Please select a CSV or GeoJSON file to upload first.")

    if has_valid_data:
        st.markdown(
            f"""<div style="font-size:11px;color:#38bdf8;background:rgba(56,189,248,0.1);
                        border:1px solid rgba(56,189,248,0.25);border-radius:6px;padding:7px 10px;margin:8px 0;">
              🛰️ <b>Active Sector:</b> {st.session_state.get('dataset_filename', 'Uploaded file')}<br>
              <span style="font-size:10px;color:#94a3b8;">
                {len(WAYPOINTS)} waypoints &bull; {len(CORRIDORS)} corridors &bull; {len(RESTRICTED_ZONES)} NFZs
              </span>
            </div>""",
            unsafe_allow_html=True
        )
        if st.button("🗑️ Purge Airspace Sector", use_container_width=True):
            st.session_state["airspace_data"] = {"waypoints": {}, "corridors": [], "restricted_zones": [], "drones": []}
            st.session_state["dataset_filename"] = ""
            st.session_state["warshall_executed"] = False
            st.session_state["fp_result"] = None
            st.rerun()

    # ── Altitude & Corridor Rules ────────────────────────────────────────────
    st.markdown('<div class="sidebar-hud-title">🏔 Altitude & Speed Bands</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    st.session_state.show_low  = c1.checkbox("Low\n60–100m",   value=st.session_state.show_low)
    st.session_state.show_mid  = c2.checkbox("Mid\n100–150m",  value=st.session_state.show_mid)
    st.session_state.show_high = c3.checkbox("High\n150–200m", value=st.session_state.show_high)
    st.session_state.altitude_filter = st.slider("Ceiling Filter (m)", 0, 250, st.session_state.altitude_filter, 5)
    st.session_state.conflict_radius = st.slider("NFZ Safety Bubble (m)", 50, 300, st.session_state.conflict_radius, 10)

    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;text-align:center;font-size:0.65rem;color:#475569;margin-top:20px;padding-top:10px;border-top:1px solid rgba(56,189,248,0.1);">
      AEROROUTE // OPS v2.8.0<br>Discrete Math Flight Deconfliction
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 2. TOP HEADER BAR WITH LIVE STATUS BADGES
# ─────────────────────────────────────────────
n_conflict = sum(1 for c in CORRIDORS if c[4] == "conflict")
status_label = "Online" if has_valid_data else "Standby"
status_class = "online" if has_valid_data else "standby"
dot_class = "dot-green" if has_valid_data else "dot-amber"
corridor_checks_count = len(CORRIDORS)
security_status = "Deconflicted" if (has_valid_data and n_conflict == 0) else ("Intervention Req." if n_conflict > 0 else "Active")

st.markdown(f"""
<div class="mission-header">
  <div class="mission-title-box">
    <div class="mission-tag">AEROSPACE MISSION CONTROL // DISPATCH RADAR</div>
    <h1 class="mission-title">AeroRoute // Airspace Deconfliction Control</h1>
    <div class="mission-subtitle">Autonomous UAV Corridor Coordination &bull; Warshall Reachability Matrix &bull; Real-time Safety Barrier</div>
  </div>
  <div class="mission-status-group">
    <div class="status-pill {status_class}">
      <span class="status-dot {dot_class}"></span>
      <span>System Status: <b>{status_label}</b></span>
    </div>
    <div class="status-pill cyan">
      <span class="status-dot dot-cyan"></span>
      <span>Active Corridor Checks: <b>{corridor_checks_count}</b></span>
    </div>
    <div class="status-pill purple">
      <span class="status-dot dot-purple"></span>
      <span>Security Protocols: <b>{security_status}</b></span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 4. CUSTOM METRIC CONTAINERS (Tracking Nodes, Hops, Conflicts)
# ─────────────────────────────────────────────
total_km = sum(haversine_km(WAYPOINTS[c[0]]["lat"], WAYPOINTS[c[0]]["lon"],
                            WAYPOINTS[c[1]]["lat"], WAYPOINTS[c[1]]["lon"])
               for c in CORRIDORS if c[0] in WAYPOINTS and c[1] in WAYPOINTS) if CORRIDORS else 0

total_pairs = len(_GRAPH_NODES) * len(_GRAPH_NODES) if _GRAPH_NODES else 0
n_reach = int(_REACH_MATRIX.sum()) if _REACH_MATRIX.size > 0 else 0
coverage_pct = f"{(n_reach / total_pairs * 100):.1f}%" if total_pairs > 0 else "0.0%"

m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
with m_col1:
    st.metric(
        label="Sector Nodes |V|",
        value=len(WAYPOINTS),
        delta="Active Waypoints" if has_valid_data else "No Data"
    )
with m_col2:
    st.metric(
        label="Flight Corridors |E|",
        value=len(CORRIDORS),
        delta=f"{total_km:.1f} km span" if total_km > 0 else "0 km"
    )
with m_col3:
    st.metric(
        label="Restricted NFZs",
        value=len(RESTRICTED_ZONES),
        delta="No-Fly Zones" if RESTRICTED_ZONES else "Zero Barriers",
        delta_color="inverse" if RESTRICTED_ZONES else "normal"
    )
with m_col4:
    st.metric(
        label="Collision Flags",
        value=n_conflict,
        delta="⚠ NFZ Intersect" if n_conflict > 0 else "Airspace Clear",
        delta_color="inverse" if n_conflict > 0 else "normal"
    )
with m_col5:
    st.metric(
        label="Reachability Index",
        value=f"{n_reach}/{total_pairs}" if total_pairs > 0 else "0/0",
        delta=f"{coverage_pct} Transitive"
    )

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 2. MAIN BODY SPLIT: st.columns([2, 1])
# Left: Interactive Folium Map
# Right: Interactive Telemetry & Math Inspector Panel
# ─────────────────────────────────────────────
col_left_map, col_right_telemetry = st.columns([2, 1])


# ─────────────────────────────────────────────
# LEFT COLUMN: INTERACTIVE FOLIUM MAP / STANDBY HUD
# ─────────────────────────────────────────────
with col_left_map:
    if not has_valid_data:
        # ── STANDBY EMPTY STATE RADAR ─────────────────────────────────────────
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(13, 20, 36, 0.85) 0%, rgba(9, 14, 26, 0.88) 100%);
                    border: 2px dashed rgba(56, 189, 248, 0.3); border-radius: 14px; padding: 48px 32px;
                    text-align: center; backdrop-filter: blur(12px); box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);">
          <div style="font-size: 56px; margin-bottom: 14px; filter: drop-shadow(0 0 16px rgba(56, 189, 248, 0.45));">🛰️</div>
          <h2 style="font-family: 'Space Grotesk', sans-serif; font-size: 1.55rem; font-weight: 700; color: #38bdf8; margin: 0 0 10px 0;">
            AeroRoute Airspace Empty
          </h2>
          <p style="color: #94a3b8; font-size: 1.0rem; max-width: 580px; margin: 0 auto 24px auto; line-height: 1.6;">
            Please upload a valid CSV or GeoJSON dataset containing nodes and restricted zones to initialize the radar and corridor engine.
          </p>
          <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(56, 189, 248, 0.2); border-radius: 8px; padding: 12px 16px; text-align: left; min-width: 170px;">
              <div style="font-weight: 700; color: #38bdf8; font-size: 12px;">📍 1. Waypoint Nodes</div>
              <div style="color: #64748b; font-size: 11px;">Coordinates, altitudes, and corridor links.</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 8px; padding: 12px 16px; text-align: left; min-width: 170px;">
              <div style="font-weight: 700; color: #ef4444; font-size: 12px;">⛔ 2. Restricted NFZs</div>
              <div style="color: #64748b; font-size: 11px;">No-Fly barriers & hazard zones.</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(167, 139, 250, 0.25); border-radius: 8px; padding: 12px 16px; text-align: left; min-width: 170px;">
              <div style="font-weight: 700; color: #a78bfa; font-size: 12px;">⚡ 3. Warshall Closure</div>
              <div style="color: #64748b; font-size: 11px;">Transitive matrix path planning.</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        sample_csv_text = """id,name,latitude,longitude,type,altitude,connections
WP_01,Central Skyport,19.0760,72.8777,hub,120,WP_02;WP_03
WP_02,Terminal North,19.0830,72.8820,terminal,90,WP_01;WP_04
WP_03,Cargo Logistics Depot,19.0720,72.8890,depot,80,WP_01;WP_04
WP_04,East Skyway Relay,19.0810,72.8950,relay,100,WP_02;WP_03
NFZ_01,Downtown Restricted Airspace,19.0775,72.8835,nfz,0,
"""
        sample_geojson_text = json.dumps({
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"id": "WP_01", "name": "Central Skyport", "type": "hub", "alt": 120}, "geometry": {"type": "Point", "coordinates": [72.8777, 19.0760]}},
                {"type": "Feature", "properties": {"id": "WP_02", "name": "Terminal North", "type": "terminal", "alt": 90}, "geometry": {"type": "Point", "coordinates": [72.8820, 19.0830]}},
                {"type": "Feature", "properties": {"id": "WP_03", "name": "Cargo Logistics Depot", "type": "depot", "alt": 85}, "geometry": {"type": "Point", "coordinates": [72.8890, 19.0720]}},
                {"type": "Feature", "properties": {"source": "WP_01", "target": "WP_02", "speed": 65, "layer": "mid"}, "geometry": {"type": "LineString", "coordinates": [[72.8777, 19.0760], [72.8820, 19.0830]]}},
                {"type": "Feature", "properties": {"source": "WP_02", "target": "WP_03", "speed": 75, "layer": "high"}, "geometry": {"type": "LineString", "coordinates": [[72.8820, 19.0830], [72.8890, 19.0720]]}},
                {"type": "Feature", "properties": {"name": "Restricted Airspace Zone", "type": "nfz"}, "geometry": {"type": "Polygon", "coordinates": [[[72.880, 19.076], [72.887, 19.076], [72.887, 19.080], [72.880, 19.080], [72.880, 19.076]]]}}
            ]
        }, indent=2)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                label="📥 Download Template CSV Dataset",
                data=sample_csv_text,
                file_name="sample_airspace.csv",
                mime="text/csv",
                use_container_width=True
            )
        with dl2:
            st.download_button(
                label="📥 Download Template GeoJSON Dataset",
                data=sample_geojson_text,
                file_name="sample_airspace.geojson",
                mime="application/json",
                use_container_width=True
            )
        map_data = None

    else:
        # ── ACTIVE RADAR FOLIUM MAP ──────────────────────────────────────────
        all_lats, all_lons = [], []
        for wp in WAYPOINTS.values():
            all_lats.append(wp["lat"])
            all_lons.append(wp["lon"])
        for rz in RESTRICTED_ZONES:
            if rz["type"] == "Point":
                all_lats.append(rz["coordinates"][0])
                all_lons.append(rz["coordinates"][1])
            elif rz["type"] == "Polygon":
                for pt in rz["coordinates"]:
                    all_lats.append(pt[0])
                    all_lons.append(pt[1])

        if all_lats and all_lons:
            center_lat = float(np.mean(all_lats))
            center_lon = float(np.mean(all_lons))
            map_bounds = [[min(all_lats), min(all_lons)], [max(all_lats), max(all_lons)]]
        else:
            center_lat, center_lon = 19.0760, 72.8777
            map_bounds = None

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=14,
            tiles="OpenStreetMap",
            prefer_canvas=True
        )

        fg_corridors = folium.FeatureGroup(name="🛸 Drone Flight Corridors", show=True)
        fg_nodes     = folium.FeatureGroup(name="📍 Waypoint Nodes", show=True)
        fg_uploaded  = folium.FeatureGroup(name="⛔ Restricted Airspace (NFZs)", show=True)
        fg_plan      = folium.FeatureGroup(name="✅ Approved / Bypass Flight Path", show=True)

        # ── Highlight Restricted Zones in RED on Folium Map Before Warshall ──
        for u_zone in RESTRICTED_ZONES:
            z_geom = u_zone["geometry"]
            z_type = u_zone["type"]
            z_name = u_zone["name"]
            z_rad  = u_zone.get("radius", 150.0)

            if z_type == "Polygon":
                coords = u_zone["coordinates"]
                folium.Polygon(
                    locations=coords,
                    color="#dc2626",
                    weight=3,
                    fill=True,
                    fill_color="#ef4444",
                    fill_opacity=0.35,
                    dash_array="6 4",
                    popup=folium.Popup(f"<b style='color:#ef4444'>🚫 Restricted Airspace</b><br>{z_name}<br>Strict No-Fly Zone", max_width=220),
                    tooltip=f"⛔ Uploaded NFZ: {z_name}"
                ).add_to(fg_uploaded)

                c_lat = float(z_geom.centroid.y)
                c_lon = float(z_geom.centroid.x)
                folium.Marker(
                    location=[c_lat, c_lon],
                    icon=folium.DivIcon(
                        html=f"""<div style="font-family:Inter,sans-serif;font-size:10px;font-weight:700;
                                   color:#ffffff;background:rgba(220,38,38,0.9);border:1px solid #ef4444;
                                   border-radius:4px;padding:2px 7px;white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,0.4);">
                                   ⛔ {z_name}</div>""",
                        icon_size=(80, 22), icon_anchor=(40, 11)
                    )
                ).add_to(fg_uploaded)

            elif z_type == "MultiPolygon":
                for poly_coords in u_zone["coordinates"]:
                    folium.Polygon(
                        locations=poly_coords,
                        color="#dc2626",
                        weight=3,
                        fill=True,
                        fill_color="#ef4444",
                        fill_opacity=0.35,
                        dash_array="6 4",
                        popup=folium.Popup(f"<b style='color:#ef4444'>🚫 Restricted Airspace</b><br>{z_name}<br>Strict No-Fly Zone", max_width=220),
                        tooltip=f"⛔ Uploaded NFZ: {z_name}"
                    ).add_to(fg_uploaded)

            elif z_type == "Point":
                raw_pt = u_zone["raw_geometry"]
                folium.Circle(
                    location=[raw_pt.y, raw_pt.x],
                    radius=z_rad,
                    color="#dc2626",
                    weight=3,
                    fill=True,
                    fill_color="#ef4444",
                    fill_opacity=0.35,
                    dash_array="6 4",
                    popup=folium.Popup(f"<b style='color:#ef4444'>🚫 Restricted Airspace</b><br>{z_name}<br>Safety bubble: {z_rad:.0f}m", max_width=220),
                    tooltip=f"⛔ Uploaded NFZ: {z_name} ({z_rad:.0f}m)"
                ).add_to(fg_uploaded)

                folium.Marker(
                    location=[raw_pt.y, raw_pt.x],
                    icon=folium.DivIcon(
                        html=f"""<div style="font-family:Inter,sans-serif;font-size:10px;font-weight:700;
                                   color:#ffffff;background:rgba(220,38,38,0.9);border:1px solid #ef4444;
                                   border-radius:4px;padding:2px 6px;white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,0.4);">
                                   ⛔ {z_name}</div>""",
                        icon_size=(75, 20), icon_anchor=(37, 10)
                    )
                ).add_to(fg_uploaded)

        # ── Render Flight Corridors ──────────────────────────────────────────
        alt_min, alt_max = st.session_state.altitude_filter
        active_layers = {l for l, f in [("low", st.session_state.show_low),
                                          ("mid", st.session_state.show_mid),
                                          ("high", st.session_state.show_high)] if f}

        for corridor in CORRIDORS:
            from_id, to_id, layer, speed, status = corridor[0], corridor[1], corridor[2], corridor[3], corridor[4]
            if from_id not in WAYPOINTS or to_id not in WAYPOINTS:
                continue
            if layer not in active_layers:
                continue

            wf, wt = WAYPOINTS[from_id], WAYPOINTS[to_id]
            avg_alt = (wf["alt"] + wt["alt"]) / 2
            if not (alt_min <= avg_alt <= alt_max):
                continue

            color  = corridor_color(layer, status)
            weight = 4 if status in ("conflict", "priority") else 3
            dash   = "10 5" if status == "conflict" else None

            dist_m = haversine_km(wf["lat"], wf["lon"], wt["lat"], wt["lon"]) * 1000
            popup_html = f"""
              <div style="font-family:Inter,sans-serif;font-size:13px;min-width:200px;">
                <b style="color:{color}">{wf['name']} → {wt['name']}</b>
                <hr style="border-color:#333;margin:6px 0">
                <b>Layer:</b> {layer.capitalize()}<br>
                <b>Speed:</b> {speed} km/h &nbsp;|&nbsp; <b>Avg Alt:</b> {avg_alt:.0f} m<br>
                <b>Distance:</b> {dist_m:.0f} m<br>
                <b>Status:</b> <span style="color:{color}">{status.upper()}</span>
              </div>"""

            folium.PolyLine(
                locations=[[wf["lat"], wf["lon"]], [wt["lat"], wt["lon"]]],
                color=color, weight=weight, opacity=0.85,
                dash_array=dash,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"{wf['name']} → {wt['name']} | {layer.upper()} | {speed} km/h | {status.upper()}"
            ).add_to(fg_corridors)

            ap = interpolate_point([wf["lat"], wf["lon"]], [wt["lat"], wt["lon"]], 0.70)
            angle = math.degrees(math.atan2(wt["lon"] - wf["lon"], wt["lat"] - wf["lat"]))
            folium.Marker(
                location=ap,
                icon=folium.DivIcon(html=f"""
                  <div style="transform:rotate({angle}deg);width:0;height:0;
                              border-left:6px solid transparent;border-right:6px solid transparent;
                              border-bottom:12px solid {color};opacity:0.9;margin:-6px 0 0 -6px;"></div>""",
                  icon_size=(12, 12), icon_anchor=(6, 6))
            ).add_to(fg_corridors)

        # ── Render Waypoint Nodes ────────────────────────────────────────────
        for wp_id, wp in WAYPOINTS.items():
            if wp["type"] == "nfz":
                continue
            color = node_color(wp["type"])

            popup_html = f"""
              <div style="font-family:Inter,sans-serif;font-size:13px;min-width:180px;">
                <b style="color:{color}">{wp['name']}</b>
                <hr style="border-color:#333;margin:6px 0">
                <b>ID:</b> {wp_id}<br><b>Classification:</b> {wp['type'].capitalize()}<br>
                <b>Altitude:</b> {wp['alt']} m<br>
                <b>Coordinates:</b> {wp['lat']:.5f}, {wp['lon']:.5f}
              </div>"""

            folium.Marker(
                location=[wp["lat"], wp["lon"]],
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=f"📍 {wp['name']} | {wp['type'].capitalize()} | {wp['alt']}m",
                icon=folium.Icon(color="white", icon_color=color, icon=node_icon(wp["type"]), prefix="fa")
            ).add_to(fg_nodes)

        # ── Render Approved Flight Plan Overlay / Automated Safe Bypass ──────
        fp_res = st.session_state.get("fp_result")
        if fp_res:
            render_path = None
            is_bypass = False

            if fp_res.get("status") == "safe" and len(fp_res.get("path", [])) >= 2:
                render_path = fp_res["path"]
                is_bypass = False
            elif fp_res.get("status") == "nfz_conflict" and fp_res.get("has_fallback") and fp_res.get("fallback_path"):
                # Clear the red violating trajectory and render newly computed safe alternative path in bright green
                render_path = fp_res["fallback_path"]
                is_bypass = True

            if render_path and len(render_path) >= 2:
                SAFE_COLOR = "#10b981"
                prefix_label = "🛡️ Bypass Vector" if is_bypass else "✅ Approved Corridor"

                for i in range(len(render_path) - 1):
                    n_from, n_to = render_path[i], render_path[i + 1]
                    if n_from in WAYPOINTS and n_to in WAYPOINTS:
                        seg_from = WAYPOINTS[n_from]
                        seg_to   = WAYPOINTS[n_to]
                        seg_dist = haversine_km(seg_from["lat"], seg_from["lon"], seg_to["lat"], seg_to["lon"]) * 1000

                        folium.PolyLine(
                            locations=[[seg_from["lat"], seg_from["lon"]], [seg_to["lat"], seg_to["lon"]]],
                            color=SAFE_COLOR,
                            weight=6 if not is_bypass else 5,
                            opacity=0.95,
                            dash_array="8 6" if is_bypass else "10 5",
                            tooltip=f"{prefix_label} Hop {i+1}: {seg_from['name']} → {seg_to['name']} ({seg_dist:.0f} m)",
                        ).add_to(fg_plan)

                # START Marker
                start_node = render_path[0]
                if start_node in WAYPOINTS:
                    wps = WAYPOINTS[start_node]
                    folium.Marker(
                        location=[wps["lat"], wps["lon"]],
                        tooltip=f"🟢 START: {wps['name']}",
                        icon=folium.DivIcon(
                            html=f"""
                              <div style="font-size:22px;text-align:center;line-height:1;">&#128994;</div>
                              <div style="font-size:9px;font-weight:700;color:{SAFE_COLOR};background:rgba(0,0,0,0.85);
                                          border-radius:3px;padding:1px 4px;white-space:nowrap;">START</div>""",
                            icon_size=(40, 48), icon_anchor=(20, 24)
                        )
                    ).add_to(fg_plan)

                # GOAL Marker
                end_node = render_path[-1]
                if end_node in WAYPOINTS:
                    wpe = WAYPOINTS[end_node]
                    folium.Marker(
                        location=[wpe["lat"], wpe["lon"]],
                        tooltip=f"🏁 DESTINATION: {wpe['name']}",
                        icon=folium.DivIcon(
                            html=f"""
                              <div style="font-size:22px;text-align:center;line-height:1;">&#127937;</div>
                              <div style="font-size:9px;font-weight:700;color:#f59e0b;background:rgba(0,0,0,0.85);
                                          border-radius:3px;padding:1px 4px;white-space:nowrap;">GOAL</div>""",
                            icon_size=(40, 48), icon_anchor=(20, 24)
                        )
                    ).add_to(fg_plan)

                # Render intermediate detour waypoint badges if this is a bypass
                if is_bypass:
                    for mid_node in render_path[1:-1]:
                        if mid_node in WAYPOINTS:
                            wp_mid = WAYPOINTS[mid_node]
                            folium.CircleMarker(
                                location=[wp_mid["lat"], wp_mid["lon"]],
                                radius=7,
                                color="#10b981",
                                weight=2,
                                fill=True,
                                fill_color="#34d399",
                                fill_opacity=0.9,
                                tooltip=f"🛡️ Safe Detour Waypoint: {wp_mid['name']} ({mid_node})"
                            ).add_to(fg_plan)

        fg_corridors.add_to(m)
        fg_nodes.add_to(m)
        fg_uploaded.add_to(m)
        fg_plan.add_to(m)

        if map_bounds:
            m.fit_bounds(map_bounds, padding=(30, 30))

        folium.LayerControl(position="topright", collapsed=False).add_to(m)

        st.markdown(f"<div style='font-family:Space Grotesk,sans-serif;font-size:0.9rem;font-weight:700;color:#38bdf8;margin-bottom:6px;'>🗺️ LIVE RADAR VIEW // {st.session_state.get('dataset_filename', 'AIRSPACE SECTOR')}</div>", unsafe_allow_html=True)
        with st.container():
            map_data = st_folium(m, width="100%", height=620, returned_objects=["last_object_clicked"])


# ─────────────────────────────────────────────
# 3. RIGHT COLUMN: TELEMETRY & MATH INSPECTOR TABS
# ─────────────────────────────────────────────
with col_right_telemetry:
    st.markdown("<div style='font-family:Space Grotesk,sans-serif;font-size:0.9rem;font-weight:700;color:#818cf8;margin-bottom:6px;'>📊 TELEMETRY &amp; MATH INSPECTOR</div>", unsafe_allow_html=True)

    tab_validator, tab_discrete_math, tab_corridors = st.tabs([
        "✈️ Flight Plan Validator",
        "⚡ Discrete Math Engine",
        "📋 Sector Telemetry"
    ])

    # ── TAB 1: FLIGHT PLAN VALIDATOR ─────────────────────────────────────────
    with tab_validator:
        st.markdown("<div style='font-size:12px;color:#94a3b8;margin-bottom:12px;'>Dispatch route validator against Warshall reachability matrix and designated No-Fly Zones.</div>", unsafe_allow_html=True)

        _selectable = [n for n in _GRAPH_NODES if WAYPOINTS.get(n, {}).get("type") != "nfz"]
        _sel_labels = {n: f"{WAYPOINTS[n]['name']} ({n})" for n in _selectable}

        if _selectable and len(_selectable) >= 2:
            fp_start_idx = _selectable.index(st.session_state.fp_start) if st.session_state.fp_start in _selectable else 0
            fp_end_idx   = _selectable.index(st.session_state.fp_end)   if st.session_state.fp_end   in _selectable else 1

            t_val_start = st.selectbox(
                "Departure Waypoint",
                _selectable,
                index=fp_start_idx,
                format_func=lambda n: _sel_labels.get(n, n),
                key="tab_val_start"
            )
            t_val_end = st.selectbox(
                "Destination Waypoint",
                _selectable,
                index=fp_end_idx,
                format_func=lambda n: _sel_labels.get(n, n),
                key="tab_val_end"
            )
        else:
            st.info("🔒 Upload an airspace dataset with at least 2 non-restricted waypoints to enable flight plan planning.")
            t_val_start, t_val_end = None, None

        # Route calculation button disabled until valid dataset uploaded
        plan_route_btn = st.button(
            "🚀 Validate & Plan Flight Corridor",
            disabled=not (has_valid_data and t_val_start and t_val_end and len(_selectable) >= 2),
            use_container_width=True,
            key="btn_val_route",
            help="Evaluate route clearance against Warshall reachability matrix and No-Fly Zones."
        )

        if plan_route_btn and t_val_start and t_val_end:
            st.session_state.fp_start = t_val_start
            st.session_state.fp_end = t_val_end
            st.session_state.fp_result = validate_flight_plan(
                t_val_start,
                t_val_end,
                _GRAPH_NODES,
                _REACH_MATRIX,
                _ADJ_LIST,
                _NFZ_NODES,
                WAYPOINTS,
                CORRIDORS,
                RESTRICTED_ZONES
            )
            st.rerun()

        # Display Validation Results
        fp_res = st.session_state.get("fp_result")
        if fp_res:
            st.markdown("<hr style='margin:14px 0 10px 0;'>", unsafe_allow_html=True)
            _s  = fp_res.get("status")
            _p  = fp_res.get("path", [])
            _cf = fp_res.get("conflicts", [])
            _d  = fp_res.get("distance", 0.0)
            _src_name = WAYPOINTS.get(st.session_state.fp_start, {}).get("name", st.session_state.fp_start)
            _dst_name = WAYPOINTS.get(st.session_state.fp_end,   {}).get("name", st.session_state.fp_end)

            if _s == "nfz_conflict":
                # ── STEP 1: VIOLATION NOTICE ──
                conflict_tags = " ".join(
                    f"""<span style="display:inline-block;background:rgba(239,68,68,0.2);
                        color:#ef4444;border:1px solid rgba(239,68,68,0.5);border-radius:4px;
                        padding:2px 8px;font-size:11px;font-weight:700;margin:2px 3px 2px 0;">{nid}</span>"""
                    for nid in _cf
                )
                path_str = " → ".join(WAYPOINTS.get(n, {}).get("name", n) for n in _p) if _p else f"{_src_name} → {_dst_name}"
                st.markdown(f"""
                <div style="background:rgba(239,68,68,0.1);border:1px solid #ef4444;border-radius:10px;padding:14px 16px;margin-bottom:12px;">
                  <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-size:20px;">🚫</span>
                    <b style="color:#ef4444;font-size:13px;letter-spacing:0.5px;">REJECTED // NFZ CONFLICT</b>
                  </div>
                  <div style="font-size:11.5px;color:#cbd5e1;margin-top:6px;">
                    Primary trajectory intersects <b>{len(_cf)} restricted zone(s)</b>. Corridor blocked for safety.
                  </div>
                  <div style="margin-top:8px;">{conflict_tags}</div>
                  <div style="font-size:10.5px;color:#94a3b8;margin-top:8px;font-style:italic;">
                    Attempted Primary: {path_str}
                  </div>
                </div>""", unsafe_allow_html=True)

                # ── STEP 2: ALTERNATIVE SAFE ROUTE BANNER ──
                if fp_res.get("has_fallback") and fp_res.get("fallback_path"):
                    _fb = fp_res["fallback_path"]
                    _fbd = fp_res.get("fallback_distance", 0.0)
                    fb_hops = [
                        f"""<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid rgba(16,185,129,0.15);">
                              <span style="color:#10b981;font-size:11px;font-weight:700;">{'%02d'%(i+1)}</span>
                              <span style="font-size:14px;">{'🛫' if i==0 else ('🛬' if i==len(_fb)-1 else '🛡️')}</span>
                              <div style="flex:1;">
                                <div style="font-size:11.5px;color:#e2e8f0;font-weight:600;">{WAYPOINTS.get(_fb[i], {}).get('name', _fb[i])}</div>
                                <div style="font-size:10px;color:#64748b;">{_fb[i]} &bull; {WAYPOINTS.get(_fb[i], {}).get('alt', 80)}m</div>
                              </div>
                            </div>"""
                        for i in range(len(_fb))
                    ]
                    st.markdown(f"""
                    <div style="background:rgba(16,185,129,0.09);border:1px solid #10b981;border-radius:10px;padding:14px 16px;">
                      <div style="display:flex;align-items:center;gap:8px;">
                        <span style="font-size:20px;">🛡️</span>
                        <b style="color:#10b981;font-size:13px;letter-spacing:0.5px;">SAFE ROUTE FALLBACK ACTIVATED</b>
                      </div>
                      <div style="font-size:12px;color:#34d399;font-weight:600;margin-top:6px;line-height:1.4;">
                        Primary corridor blocked by safety protocol. Alternative safe route computed via bypass vector.
                      </div>
                      <div style="font-size:11px;color:#94a3b8;margin:8px 0 10px 0;">
                        Detour: <b>{len(_fb)-1} hops</b> &bull; <b>{_fbd:.2f} km</b> &bull; Computed via Dijkstra / Safe Filter
                      </div>
                      <div style="max-height: 180px; overflow-y: auto;">
                        {''.join(fb_hops)}
                      </div>
                      <div style="margin-top:10px;font-size:10.5px;color:#6ee7b7;font-weight:500;">
                        ⚡ Safe bypass path rendered in bright emerald green on radar. Violating corridor cleared.
                      </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:rgba(239,68,68,0.06);border:1px dashed #ef4444;border-radius:8px;padding:10px 14px;">
                      <span style="color:#f87171;font-size:11px;font-weight:600;">
                        ⚠️ No viable bypass vector found around restricted zones. All alternative corridors severed.
                      </span>
                    </div>""", unsafe_allow_html=True)

            elif _s == "unreachable":
                st.markdown(f"""
                <div style="background:rgba(239,68,68,0.1);border:1px solid #ef4444;border-radius:10px;padding:14px 16px;">
                  <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-size:20px;">🚫</span>
                    <b style="color:#ef4444;font-size:13px;">UNREACHABLE // NO PATH</b>
                  </div>
                  <div style="font-size:11.5px;color:#cbd5e1;margin-top:6px;">
                    Warshall reachability confirmed no directed transitive path connects <b>{_src_name}</b> to <b>{_dst_name}</b>.
                  </div>
                </div>""", unsafe_allow_html=True)

            elif _s == "same_node":
                st.warning("⚠️ Origin and Destination waypoints are identical.")

            else:   # Safe approved route
                hops = [
                    f"""<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid rgba(16,185,129,0.15);">
                          <span style="color:#10b981;font-size:11px;font-weight:700;">{'%02d'%(i+1)}</span>
                          <span style="font-size:14px;">{'🛫' if i==0 else ('🛬' if i==len(_p)-1 else '📌')}</span>
                          <div style="flex:1;">
                            <div style="font-size:11.5px;color:#e2e8f0;font-weight:600;">{WAYPOINTS.get(_p[i], {}).get('name', _p[i])}</div>
                            <div style="font-size:10px;color:#64748b;">{_p[i]} &bull; {WAYPOINTS.get(_p[i], {}).get('alt', 80)}m</div>
                          </div>
                        </div>"""
                    for i in range(len(_p))
                ]
                st.markdown(f"""
                <div style="background:rgba(16,185,129,0.08);border:1px solid #10b981;border-radius:10px;padding:14px 16px;">
                  <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-size:20px;">✅</span>
                    <b style="color:#10b981;font-size:13px;">APPROVED // AIRSPACE DECONFLICTED</b>
                  </div>
                  <div style="font-size:11.5px;color:#94a3b8;margin:6px 0 10px 0;">
                    {_src_name} → {_dst_name} &bull; <b>{len(_p)-1} hops</b> &bull; <b>{_d:.2f} km</b>
                  </div>
                  <div style="max-height: 200px; overflow-y: auto;">
                    {''.join(hops)}
                  </div>
                  <div style="margin-top:10px;font-size:10.5px;color:#34d399;">
                    Emerald trajectory rendered live on the mission radar map.
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:11px;color:#64748b;margin-top:14px;'>Select waypoints and click Validate to run collision detection.</div>", unsafe_allow_html=True)

    # ── TAB 2: DISCRETE MATH ENGINE ──────────────────────────────────────────
    with tab_discrete_math:
        st.markdown("<div style='font-size:12px;color:#94a3b8;margin-bottom:12px;'>Transitive Closure Reachability computation using Warshall's Algorithm.</div>", unsafe_allow_html=True)

        warshall_calc_btn = st.button(
            "⚡ Run Warshall's Algorithm",
            disabled=not (has_valid_data and len(_GRAPH_NODES) > 0),
            use_container_width=True,
            key="tab_btn_warshall",
            help="Compute Warshall reachability matrix on current sector graph."
        )

        if warshall_calc_btn:
            st.session_state["warshall_executed"] = True
            st.success("✅ Warshall's Transitive Closure calculated!")

        if has_valid_data and len(_GRAPH_NODES) > 0:
            n_nodes = len(_GRAPH_NODES)
            n_edges = len(_GRAPH_EDGES)
            n_reach = int(_REACH_MATRIX.sum())
            total_pairs = n_nodes * n_nodes
            density = (n_edges / (n_nodes * (n_nodes - 1))) * 100 if n_nodes > 1 else 0

            st.markdown(f"""
            <div style="background:rgba(15,23,42,0.6);border-left:3px solid #818cf8;border-radius:4px;padding:8px 12px;margin:10px 0;font-size:11px;color:#94a3b8;">
              <b style="color:#818cf8;">Warshall Recurrence:</b> <code>R[i, j] |= R[i, k] &amp; R[k, j]</code><br>
              Complexity <b style="color:#e2e8f0;">O(n³)</b> &bull; {n_nodes} nodes &bull; {n_nodes**3:,} iterations<br>
              Graph Density: <b style="color:#38bdf8;">{density:.1f}%</b> &bull; Transitive Reach: <b style="color:#a78bfa;">{n_reach/total_pairs*100:.1f}%</b>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<b style='font-size:12px;color:#38bdf8;'>🔗 Adjacency Matrix (A)</b>", unsafe_allow_html=True)
            def _style_adj(val):
                return "background-color:rgba(56,189,248,0.2);color:#38bdf8;font-weight:700;" if val == 1 else "color:#334155;"

            st.dataframe(
                _DF_ADJ.style.map(_style_adj),
                use_container_width=True,
                height=min(32 * n_nodes + 40, 240)
            )

            st.markdown("<b style='font-size:12px;color:#a78bfa;margin-top:10px;'>🌐 Warshall Reachability Matrix (R)</b>", unsafe_allow_html=True)
            def _style_reach(val):
                return "background-color:rgba(167,139,250,0.2);color:#a78bfa;font-weight:700;" if val == 1 else "color:#334155;"

            st.dataframe(
                _DF_REACH.style.map(_style_reach),
                use_container_width=True,
                height=min(32 * n_nodes + 40, 240)
            )
        else:
            st.info("Upload and process an airspace dataset to compute the Adjacency and Warshall matrices.")

    # ── TAB 3: SECTOR TELEMETRY & NODES ──────────────────────────────────────
    with tab_corridors:
        st.markdown("<div style='font-size:12px;color:#94a3b8;margin-bottom:12px;'>Airspace corridor channels and clicked waypoint telemetry.</div>", unsafe_allow_html=True)

        # Selected Map Feature Inspector
        if map_data and map_data.get("last_object_clicked"):
            clicked = map_data["last_object_clicked"]
            lat, lon = clicked.get("lat", 0), clicked.get("lng", 0)
            if WAYPOINTS:
                nearest_id, nearest_wp = min(WAYPOINTS.items(), key=lambda kv: haversine_km(lat, lon, kv[1]["lat"], kv[1]["lon"]))
                dist_m = haversine_km(lat, lon, nearest_wp["lat"], nearest_wp["lon"]) * 1000
                color  = node_color(nearest_wp["type"])
                outgoing = "".join(f'<span style="color:#38bdf8;"> → {WAYPOINTS[c[1]]["name"][:12]}</span>' for c in CORRIDORS if c[0] == nearest_id and c[1] in WAYPOINTS)
                st.markdown(f"""
                <div class="info-panel" style="margin-bottom:12px;">
                  <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:700;color:{color};margin-bottom:8px;">
                    📍 {nearest_wp['name']}
                  </div>
                  <table style="width:100%;font-size:11.5px;border-collapse:collapse;">
                    <tr><td style="color:#64748b;padding:2px 0;">ID</td><td style="color:#e2e8f0;text-align:right;">{nearest_id}</td></tr>
                    <tr><td style="color:#64748b;padding:2px 0;">Class</td><td style="color:{color};text-align:right;font-weight:600;">{nearest_wp['type'].capitalize()}</td></tr>
                    <tr><td style="color:#64748b;padding:2px 0;">Altitude</td><td style="color:#e2e8f0;text-align:right;">{nearest_wp['alt']} m</td></tr>
                    <tr><td style="color:#64748b;padding:2px 0;">Coordinates</td><td style="color:#e2e8f0;text-align:right;">{nearest_wp['lat']:.4f}, {nearest_wp['lon']:.4f}</td></tr>
                    <tr><td style="color:#64748b;padding:2px 0;">Distance</td><td style="color:#94a3b8;text-align:right;">{dist_m:.0f} m</td></tr>
                  </table>
                  <div style="margin-top:6px;font-size:10.5px;color:#475569;">Channels: {outgoing or 'None'}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<b style='font-size:12px;color:#e2e8f0;'>Active Corridor Channels</b>", unsafe_allow_html=True)
        if CORRIDORS:
            for c in CORRIDORS[:8]:
                from_id, to_id, layer, speed, status = c[0], c[1], c[2], c[3], c[4]
                if from_id not in WAYPOINTS or to_id not in WAYPOINTS:
                    continue
                lc = corridor_color(layer, status)
                icon = "🔴" if status == "conflict" else ("🟡" if status == "priority" else "🟢")
                st.markdown(f"""
                <div class="flight-row">
                  <div style="display:flex;align-items:center;gap:8px;">
                    <span style="font-size:12px;">{icon}</span>
                    <div style="flex:1;">
                      <div style="font-size:11px;font-weight:600;color:#e2e8f0;">
                        {WAYPOINTS[from_id]['name'][:13]} → {WAYPOINTS[to_id]['name'][:13]}
                      </div>
                      <div style="font-size:9.5px;color:#64748b;">
                        <span style="color:{lc};">{layer.upper()}</span> &bull; {speed} km/h &bull; <span style="color:{lc};">{status.upper()}</span>
                      </div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.caption("No corridors active in current sector.")

# ─────────────────────────────────────────────
# MISSION CONTROL FOOTER
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="border-top:1px solid rgba(56,189,248,0.12);padding:14px 0;display:flex;justify-content:space-between;font-size:0.72rem;color:#475569;font-family:'JetBrains Mono',monospace;">
  <span>🛸 AEROROUTE // TACTICAL MISSION DISPATCH &bull; URBAN AIR MOBILITY LAB &bull; v2.8.0</span>
  <span>OPENSTREETMAP ODbL &bull; DEFCON 1 SAFETY PROTOCOL ACTIVE</span>
</div>
""", unsafe_allow_html=True)

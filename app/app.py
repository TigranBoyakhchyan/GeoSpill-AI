"""
Streamlit web app for SAR oil spill detection and visualization.

Upload a Sentinel-1 GeoTIFF → model predicts oil spill mask → map visualization.

Note: If no trained model is found, runs in DEMO MODE using a simple
threshold-based detector so the UI is fully functional for testing.

Run:
    streamlit run app/app.py
"""

import os
import sys
import json
import tempfile
import warnings
import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
from rasterio.warp import transform_bounds
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAR Oil Spill Detector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --navy:   #0a1628;
    --blue:   #1a3a5c;
    --accent: #00d4ff;
    --oil:    #ff4444;
    --white:  #e8f4f8;
    --muted:  #7a9ab0;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--navy);
    color: var(--white);
}

.stApp { background-color: var(--navy); }

h1, h2, h3 {
    font-family: 'Space Mono', monospace;
    letter-spacing: -0.02em;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #0a1628 0%, #1a3a5c 50%, #0d2540 100%);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,212,255,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent);
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.03em;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin: 0;
    font-weight: 300;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #0f2035, #162d4a);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
}
.badge-demo  { background: rgba(255,165,0,0.15); color: #ffa500; border: 1px solid rgba(255,165,0,0.3); }
.badge-model { background: rgba(0,212,255,0.12); color: var(--accent); border: 1px solid rgba(0,212,255,0.25); }
.badge-oil   { background: rgba(255,68,68,0.12);  color: var(--oil);    border: 1px solid rgba(255,68,68,0.25); }

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--muted);
    border-bottom: 1px solid rgba(0,212,255,0.15);
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

/* Pixel table */
.pixel-table {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    width: 100%;
    border-collapse: collapse;
}
.pixel-table th {
    color: var(--accent);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 0.5rem 0.75rem;
    border-bottom: 1px solid rgba(0,212,255,0.2);
    text-align: left;
}
.pixel-table td {
    padding: 0.4rem 0.75rem;
    color: var(--white);
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.pixel-table tr:hover td { background: rgba(0,212,255,0.04); }

/* Upload area styling override */
[data-testid="stFileUploader"] {
    background: rgba(26, 58, 92, 0.3);
    border: 1px dashed rgba(0,212,255,0.3);
    border-radius: 10px;
    padding: 1rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1f33 0%, #0a1628 100%);
    border-right: 1px solid rgba(0,212,255,0.1);
}
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    """
    Load the trained U-Net model if available.
    Falls back to demo mode (threshold detector) if no checkpoint found.
    """
    checkpoint_path = "checkpoints/best_model.pth"
    stats_path      = "data/train_stats.json"

    try:
        import torch
        from src.model import get_model

        if not os.path.exists(checkpoint_path):
            return None, None, "demo"

        model = get_model(model_type="smp", in_channels=2)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        stats = {"mean": [-32.0, -32.0], "std": [7.0, 7.0]}
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)

        return model, stats, "model"

    except Exception:
        return None, None, "demo"


def load_tif(path: str):
    """Load a GeoTIFF and return image array, transform, and CRS."""
    with rasterio.open(path) as src:
        img       = src.read().astype(np.float32)   # (bands, H, W)
        transform = src.transform
        crs       = src.crs
        profile   = src.profile
    return img, transform, crs, profile


def predict_demo(img: np.ndarray) -> np.ndarray:
    """
    Demo mode: simple threshold detector.
    SAR oil spills appear dark — pixels below a threshold are flagged.
    Uses band 0 (VV polarization).
    """
    vv = img[0] if img.ndim == 3 else img
    vv_clipped = np.clip(vv, -50, 0)

    # Normalize to [0, 1] for thresholding
    vv_norm = (vv_clipped - vv_clipped.min()) / (vv_clipped.max() - vv_clipped.min() + 1e-6)

    # Dark pixels (low backscatter) = potential oil
    # Threshold: bottom 8% of backscatter values flagged as oil
    threshold = np.percentile(vv_norm, 8)
    mask = (vv_norm < threshold).astype(np.float32)
    return mask


def predict_model(img: np.ndarray, model, stats: dict) -> np.ndarray:
    """Run full U-Net inference with sliding window on a 2-band image."""
    import torch

    PATCH_SIZE = 256
    STRIDE     = 128

    mean = np.array(stats["mean"], dtype=np.float32)
    std  = np.array(stats["std"],  dtype=np.float32)

    # Clip and normalize
    img = np.clip(img[:2], -50, 0)
    img = (img - mean[:, None, None]) / (std[:, None, None] + 1e-6)

    _, H, W = img.shape
    pred_sum   = np.zeros((H, W), dtype=np.float32)
    pred_count = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, H - PATCH_SIZE + 1, STRIDE):
            for x in range(0, W - PATCH_SIZE + 1, STRIDE):
                patch = img[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                t = torch.from_numpy(patch).unsqueeze(0)
                pred = torch.sigmoid(model(t)).squeeze().numpy()
                pred_sum  [y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pred
                pred_count[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

    avg = pred_sum / (pred_count + 1e-6)
    return (avg > 0.5).astype(np.float32)


def mask_to_latlon(mask: np.ndarray, transform, crs) -> list:
    """Convert oil spill mask pixels to (lat, lon) coordinates."""
    rows, cols = np.where(mask == 1)

    if len(rows) == 0:
        return []

    # Convert pixel coords to projected coords
    xs = transform.c + cols * transform.a
    ys = transform.f + rows * transform.e

    # Reproject to WGS84 if needed
    if crs and str(crs) != "EPSG:4326":
        from pyproj import Transformer
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        xs, ys = transformer.transform(xs, ys)

    # (lat, lon) pairs — note xs=lon, ys=lat after WGS84
    return list(zip(ys.tolist(), xs.tolist()))


def subsample(coords: list, max_points: int = 3000) -> list:
    """Subsample coordinate list for map performance."""
    if len(coords) <= max_points:
        return coords
    step = len(coords) // max_points
    return coords[::step]


def build_map(coords: list, image_bounds=None) -> folium.Map:
    """Build a Folium map with oil spill markers."""
    if coords:
        center_lat = np.mean([c[0] for c in coords])
        center_lon = np.mean([c[1] for c in coords])
    else:
        center_lat, center_lon = 0, 0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="CartoDB dark_matter",
    )

    if not coords:
        return m

    # Add oil spill markers as a cluster of small circles
    display = subsample(coords, 3000)
    for lat, lon in display:
        folium.CircleMarker(
            location=[lat, lon],
            radius=2,
            color="#ff4444",
            fill=True,
            fill_color="#ff4444",
            fill_opacity=0.6,
            weight=0,
        ).add_to(m)

    # Add a legend
    legend_html = """
    <div style="position:fixed; bottom:20px; left:20px; z-index:1000;
                background:#0a1628; border:1px solid rgba(0,212,255,0.3);
                border-radius:8px; padding:12px 16px; font-family:monospace;
                font-size:12px; color:#e8f4f8;">
        <div style="color:#00d4ff; font-weight:bold; margin-bottom:8px;">LEGEND</div>
        <div><span style="color:#ff4444;">●</span> Oil spill pixels</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="section-header">⚙ Configuration</div>', unsafe_allow_html=True)

    threshold_pct = st.slider(
        "Oil detection sensitivity (demo mode)",
        min_value=2, max_value=20, value=8, step=1,
        help="Percentile threshold for demo mode. Lower = more pixels flagged as oil."
    )

    st.markdown("---")
    st.markdown('<div class="section-header">ℹ About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#7a9ab0; line-height:1.6;">
    Upload a Sentinel-1 SAR GeoTIFF to detect oil spills.<br><br>
    <b style="color:#e8f4f8;">Demo mode</b> uses backscatter thresholding.<br>
    <b style="color:#00d4ff;">Model mode</b> uses the trained U-Net.<br><br>
    Oil spills appear dark in SAR imagery due to dampened capillary waves.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">🔗 Model Status</div>', unsafe_allow_html=True)
    model, stats, mode = load_model()
    if mode == "model":
        st.markdown('<span class="badge badge-model">✓ MODEL LOADED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-demo">⚡ DEMO MODE</span>', unsafe_allow_html=True)
        st.caption("Train the model first to enable U-Net inference.")


# ─── MAIN LAYOUT ──────────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <p class="hero-title">🛰️ SAR Oil Spill Detector</p>
    <p class="hero-sub">Sentinel-1 GeoTIFF → Deep Learning Segmentation → Geographic Visualization</p>
</div>
""", unsafe_allow_html=True)

# Upload
st.markdown('<div class="section-header">01 — Upload Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a Sentinel-1 GeoTIFF here",
    type=["tif", "tiff"],
    help="2-band SAR image (VV + VH polarization), 2048×2048 pixels"
)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center; padding:3rem; color:#7a9ab0;">
        <div style="font-size:3rem; margin-bottom:1rem;">📡</div>
        <div style="font-family:'Space Mono',monospace; font-size:0.9rem;">
            Awaiting satellite imagery...
        </div>
        <div style="font-size:0.8rem; margin-top:0.5rem;">
            Upload a .tif file to begin detection
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─── PROCESS ──────────────────────────────────────────────────────────────────

with st.spinner("Reading GeoTIFF..."):
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        img, transform, crs, profile = load_tif(tmp_path)
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        st.stop()
    finally:
        os.unlink(tmp_path)

n_bands, H, W = img.shape if img.ndim == 3 else (1, *img.shape)

# ─── IMAGE STATS ──────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">02 — Image Properties</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{W}×{H}</div>
        <div class="metric-label">Resolution (px)</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{n_bands}</div>
        <div class="metric-label">Bands</div>
    </div>""", unsafe_allow_html=True)
with col3:
    vmin = float(img.min())
    vmax = float(img.max())
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{vmin:.1f}</div>
        <div class="metric-label">Min dB</div>
    </div>""", unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{vmax:.1f}</div>
        <div class="metric-label">Max dB</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── PREDICTION ───────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">03 — Detection</div>', unsafe_allow_html=True)

with st.spinner("Running oil spill detection..."):
    if mode == "model" and model is not None:
        mask = predict_model(img, model, stats)
        mode_label = "U-Net Model"
    else:
        # Use sidebar threshold setting
        vv = img[0] if img.ndim == 3 else img
        vv_clipped = np.clip(vv, -50, 0)
        vv_norm = (vv_clipped - vv_clipped.min()) / (vv_clipped.max() - vv_clipped.min() + 1e-6)
        threshold_val = np.percentile(vv_norm, threshold_pct)
        mask = (vv_norm < threshold_val).astype(np.float32)
        mode_label = f"Threshold Demo (p={threshold_pct})"

    oil_pixels  = int(mask.sum())
    total_pixels = H * W
    oil_pct     = 100.0 * oil_pixels / total_pixels

# Detection summary
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{oil_pixels:,}</div>
        <div class="metric-label">Oil Pixels Detected</div>
    </div>""", unsafe_allow_html=True)
with col_b:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{oil_pct:.2f}%</div>
        <div class="metric-label">Coverage</div>
    </div>""", unsafe_allow_html=True)
with col_c:
    badge_class = "badge-model" if mode == "model" else "badge-demo"
    st.markdown(f"""
    <div class="metric-card">
        <div style="margin-top:0.4rem;">
            <span class="badge {badge_class}">{mode_label}</span>
        </div>
        <div class="metric-label" style="margin-top:0.6rem;">Detection Method</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── MAP + PIXEL TABLE ────────────────────────────────────────────────────────

map_col, table_col = st.columns([3, 2])

with map_col:
    st.markdown('<div class="section-header">04 — Geographic Map</div>', unsafe_allow_html=True)

    with st.spinner("Building map..."):
        coords = mask_to_latlon(mask, transform, crs)

    if not coords:
        st.warning("No oil pixels detected or image has no geospatial metadata.")
    else:
        folium_map = build_map(coords)
        st_folium(folium_map, width=700, height=500, returned_objects=[])

        if crs is None or str(crs) == "None":
            st.caption("⚠️ No CRS found in file — map may not be geographically accurate.")

with table_col:
    st.markdown('<div class="section-header">05 — Oil Pixel Coordinates</div>', unsafe_allow_html=True)

    if not coords:
        st.info("No coordinates to display.")
    else:
        # Show first N coordinates in a table
        DISPLAY_LIMIT = 200
        display_coords = coords[:DISPLAY_LIMIT]

        # Build HTML table
        rows = ""
        for i, (lat, lon) in enumerate(display_coords):
            rows += f"""
            <tr>
                <td>{i+1}</td>
                <td>{lat:.6f}</td>
                <td>{lon:.6f}</td>
            </tr>"""

        table_html = f"""
        <div style="max-height:420px; overflow-y:auto; border:1px solid rgba(0,212,255,0.1); border-radius:8px;">
        <table class="pixel-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th>Latitude</th>
                    <th>Longitude</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
        <div style="font-size:0.75rem; color:#7a9ab0; margin-top:0.5rem;">
            Showing {min(DISPLAY_LIMIT, len(coords)):,} of {len(coords):,} oil pixels
            (subsampled for display)
        </div>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        # Download full coordinates as CSV
        st.markdown("<br>", unsafe_allow_html=True)
        csv_data = "latitude,longitude\n" + "\n".join(
            f"{lat:.8f},{lon:.8f}" for lat, lon in coords
        )
        st.download_button(
            label="⬇ Download All Coordinates (CSV)",
            data=csv_data,
            file_name="oil_spill_coordinates.csv",
            mime="text/csv",
        )
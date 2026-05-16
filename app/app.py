"""
Streamlit web app for SAR oil spill detection.

Upload a Sentinel-1 GeoTIFF → U-Net predicts oil spill mask → geographic visualization.

Logic lives in src/ modules:
    src/inference.py — model loading, sliding-window prediction, demo mode
    src/geo.py       — coordinates, area, dB conversion, proximity analysis

Run:
    streamlit run app.py
"""

import os
import io
import sys
import tempfile
import warnings

import numpy as np
import streamlit as st
import folium
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from streamlit_folium import st_folium
from PIL import Image

import rasterio
from rasterio.errors import NotGeoreferencedWarning

sys.path.insert(0, os.path.join(os.path.dirname(__file__),"../src"))

from inference import load_checkpoint, predict_sliding, predict_demo
from geo import (
    is_linear_scale, linear_to_db, mask_to_latlon, subsample,
    get_wgs84_bounds, calculate_spill_area, find_nearby_coasts, spill_centroid,
)

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GeoSpill AI — SAR Oil Spill Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── STYLES ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600;700&display=swap');

:root {
    --bg-deep:    #060d18;
    --bg-card:    #0b1726;
    --bg-raised:  #0f1f35;
    --border:     rgba(56, 189, 248, 0.08);
    --border-glow:rgba(56, 189, 248, 0.20);
    --cyan:       #38bdf8;
    --cyan-dim:   #1e6fa0;
    --teal:       #2dd4bf;
    --oil-red:    #f87171;
    --oil-glow:   rgba(248, 113, 113, 0.15);
    --amber:      #fbbf24;
    --text:       #e2e8f0;
    --text-dim:   #64748b;
    --text-muted: #475569;
}

.stApp {
    background: var(--bg-deep);
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 40%, rgba(56,189,248,0.03), transparent),
        radial-gradient(ellipse 60% 40% at 80% 60%, rgba(45,212,191,0.02), transparent);
}
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: var(--text);
}

/* ── Hero ────────────────────────────────────────────────────── */
.hero {
    position: relative; overflow: hidden;
    background: linear-gradient(160deg, #060d18 0%, #0f2847 40%, #0a1e3a 70%, #060d18 100%);
    border: 1px solid var(--border-glow);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
}
.hero::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; bottom: 0;
    background:
        repeating-linear-gradient(
            90deg, transparent, transparent 60px,
            rgba(56,189,248,0.02) 60px, rgba(56,189,248,0.02) 61px
        ),
        repeating-linear-gradient(
            0deg, transparent, transparent 60px,
            rgba(56,189,248,0.02) 60px, rgba(56,189,248,0.02) 61px
        );
    pointer-events: none;
}
.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 500;
    color: var(--teal); letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem; font-weight: 700;
    color: #f8fafc;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.03em;
}
.hero-title span { color: var(--cyan); }
.hero-sub {
    color: var(--text-dim);
    font-size: 1rem; font-weight: 300;
    margin: 0; max-width: 600px; line-height: 1.6;
}

/* ── Section Headers ─────────────────────────────────────────── */
.sec {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.18em;
    color: var(--text-muted);
    padding-bottom: 0.6rem; margin-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 0.6rem;
}
.sec-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 6px;
    background: rgba(56,189,248,0.08);
    border: 1px solid rgba(56,189,248,0.15);
    font-size: 0.6rem; color: var(--cyan);
}

/* ── Cards ────────────────────────────────────────────────────── */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: border-color 0.2s;
}
.card:hover { border-color: var(--border-glow); }
.card-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem; font-weight: 700;
    color: var(--cyan); line-height: 1;
}
.card-value.red { color: var(--oil-red); }
.card-value.teal { color: var(--teal); }
.card-value.amber { color: var(--amber); }
.card-label {
    font-size: 0.72rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-top: 0.5rem; font-weight: 500;
}

/* ── Badge ────────────────────────────────────────────────────── */
.badge {
    display: inline-block; padding: 0.3rem 0.8rem;
    border-radius: 20px; font-size: 0.7rem;
    font-family: 'JetBrains Mono', monospace; font-weight: 600;
}
.badge-ok {
    background: rgba(56,189,248,0.1); color: var(--cyan);
    border: 1px solid rgba(56,189,248,0.2);
}
.badge-demo {
    background: rgba(251,191,36,0.1); color: var(--amber);
    border: 1px solid rgba(251,191,36,0.2);
}

/* ── Proximity Cards ──────────────────────────────────────────── */
.coast-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    display: flex; align-items: center; gap: 1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.coast-card:hover { border-color: var(--border-glow); }
.coast-dir {
    width: 42px; height: 42px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 10px;
    background: rgba(56,189,248,0.06);
    border: 1px solid rgba(56,189,248,0.12);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem; font-weight: 700; color: var(--cyan);
    flex-shrink: 0;
}
.coast-info { flex: 1; min-width: 0; }
.coast-name {
    font-weight: 600; font-size: 0.95rem;
    color: #f1f5f9; margin-bottom: 0.2rem;
}
.coast-dist {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; color: var(--text-dim);
}
.coast-bar-bg {
    height: 4px; border-radius: 2px;
    background: rgba(56,189,248,0.08);
    margin-top: 0.4rem; overflow: hidden;
}
.coast-bar {
    height: 100%; border-radius: 2px;
    background: linear-gradient(90deg, var(--cyan), var(--teal));
}

/* ── Coordinate Table ─────────────────────────────────────────── */
.coord-table {
    font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;
    width: 100%; border-collapse: collapse;
}
.coord-table th {
    color: var(--cyan); font-size: 0.65rem; text-transform: uppercase;
    letter-spacing: 0.1em; padding: 0.6rem 0.8rem;
    border-bottom: 1px solid var(--border-glow); text-align: left;
    font-weight: 500;
}
.coord-table td {
    padding: 0.45rem 0.8rem; color: var(--text-dim);
    border-bottom: 1px solid var(--border);
}
.coord-table tr:hover td { background: rgba(56,189,248,0.03); color: var(--text); }
</style>
""", unsafe_allow_html=True)


# ─── CONFIG ───────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "models/best_model_20260416_145208.pt"
DEFAULT_STATS_PATH = "data/train_stats.json"
MAP_MAX_MARKERS    = 3000


# ─── CACHED MODEL LOADING ────────────────────────────────────────────────────

@st.cache_resource
def _load_model(cp, sp):
    return load_checkpoint(cp, sp)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def sec(num, label):
    return f'<div class="sec"><span class="sec-num">{num}</span>{label}</div>'

def card(value, label, color=""):
    cls = f" {color}" if color else ""
    return f"""<div class="card">
        <div class="card-value{cls}">{value}</div>
        <div class="card-label">{label}</div>
    </div>"""


def build_dot_map(coords):
    if coords:
        center = [np.mean([c[0] for c in coords]), np.mean([c[1] for c in coords])]
    else:
        center = [0, 0]

    m = folium.Map(location=center, zoom_start=11, tiles="CartoDB dark_matter")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False, control=True,
    ).add_to(m)

    if coords:
        for lat, lon in subsample(coords, MAP_MAX_MARKERS):
            folium.CircleMarker(
                location=[lat, lon], radius=2,
                color="#f87171", fill=True,
                fill_color="#f87171", fill_opacity=0.7, weight=0,
            ).add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def render_preview(img, mask):
    from geo import DB_CLIP_MIN, DB_CLIP_MAX
    vv = np.clip(img[0], DB_CLIP_MIN, DB_CLIP_MAX)
    oil_cmap = ListedColormap([(0, 0, 0, 0), (1.0, 0.47, 0.47, 1.0)])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#060d18")
    for ax in axes:
        ax.set_facecolor("#060d18")
        ax.axis("off")

    axes[0].imshow(vv, cmap="gray")
    axes[0].set_title("VV Band (dB)", color="#94a3b8", fontfamily="monospace", fontsize=10)

    axes[1].imshow(mask, cmap=oil_cmap, vmin=0, vmax=1)
    axes[1].set_title("Predicted Mask", color="#94a3b8", fontfamily="monospace", fontsize=10)

    axes[2].imshow(vv, cmap="gray")
    axes[2].imshow(mask, cmap=oil_cmap, vmin=0, vmax=1, alpha=0.55)
    axes[2].set_title("Overlay", color="#94a3b8", fontfamily="monospace", fontsize=10)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor="#060d18", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0 0.5rem;">
        <div style="font-family:'Outfit',sans-serif; font-size:1.2rem; font-weight:700; color:#f8fafc;">
            🛰️ GeoSpill AI
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#475569;
                    letter-spacing:0.15em; text-transform:uppercase; margin-top:0.3rem;">
            Configuration
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    model_path = st.text_input("Model checkpoint", value=DEFAULT_MODEL_PATH)
    stats_path = st.text_input("Stats file", value=DEFAULT_STATS_PATH)

    model, stats, err = _load_model(model_path, stats_path)
    has_model = model is not None

    st.markdown("---")

    if has_model:
        dev = "GPU" if next(model.parameters()).is_cuda else "CPU"
        st.markdown(f'<span class="badge badge-ok">✓ Model loaded ({dev})</span>',
                    unsafe_allow_html=True)
        threshold  = st.slider("Detection threshold", 0.1, 0.9, 0.5, 0.05)
        patch_size = st.selectbox("Patch size", [256, 512], index=0)
        stride     = st.selectbox("Stride", [patch_size // 2, patch_size], index=0)
    else:
        st.markdown('<span class="badge badge-demo">⚡ Demo mode</span>',
                    unsafe_allow_html=True)
        st.caption(err)
        threshold_pct = st.slider("Oil sensitivity (%ile)", 2, 20, 8, 1)

    st.markdown("---")
    proximity_radius = st.slider("Coast search radius (km)", 50, 500, 200, 25)


# ─── HERO ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <div class="hero-tag">Sentinel-1 SAR Analysis</div>
    <div class="hero-title">Geo<span>Spill</span> AI</div>
    <div class="hero-sub">
        Deep learning oil spill detection from synthetic aperture radar imagery.
        Upload a GeoTIFF to detect spills, measure area, and identify nearby coastlines.
    </div>
</div>
""", unsafe_allow_html=True)


# ── 01 Upload ─────────────────────────────────────────────────────────────────

st.markdown(sec("01", "Upload Image"), unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a Sentinel-1 GeoTIFF here", type=["tif", "tiff"],
    help="2-band SAR image (VV + VH polarization)",
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center; padding:4rem 2rem; border:1px dashed rgba(56,189,248,0.15);
                border-radius:12px; background:rgba(56,189,248,0.02);">
        <div style="font-size:2.5rem; margin-bottom:0.8rem; opacity:0.5;">📡</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:#64748b;">
            Awaiting satellite imagery...
        </div>
        <div style="font-size:0.78rem; color:#475569; margin-top:0.4rem;">
            Upload a Sentinel-1 .tif file to begin detection
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Read TIF ──────────────────────────────────────────────────────────────────

with st.spinner("Reading GeoTIFF..."):
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name
    try:
        with rasterio.open(tmp_path) as src:
            img       = src.read().astype(np.float32)
            transform = src.transform
            crs       = src.crs
    except Exception as e:
        st.error(f"Failed to read GeoTIFF: {e}")
        st.stop()
    finally:
        os.unlink(tmp_path)

if img.ndim == 2:
    img = img[None, ...]

if is_linear_scale(img):
    st.info("📐 Detected linear-scale data — auto-converting to decibels.")
    img = linear_to_db(img)

n_bands, H, W = img.shape


# ── 02 Image Properties ──────────────────────────────────────────────────────

st.markdown(sec("02", "Image Properties"), unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
c1.markdown(card(f"{W}×{H}", "Resolution"), unsafe_allow_html=True)
c2.markdown(card(f"{n_bands}", "Bands"), unsafe_allow_html=True)
c3.markdown(card(f"{float(img[0].min()):.1f} dB", "Min (VV)"), unsafe_allow_html=True)
c4.markdown(card(f"{float(img[0].max()):.1f} dB", "Max (VV)"), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ── 03 Detection ──────────────────────────────────────────────────────────────

st.markdown(sec("03", "Detection Results"), unsafe_allow_html=True)

if has_model:
    mean = np.array(stats["mean"], dtype=np.float32)
    std  = np.array(stats["std"],  dtype=np.float32)
    if img.shape[0] < 2:
        st.error(f"Image has {img.shape[0]} band(s); model needs 2 (VV + VH).")
        st.stop()

    pb = st.progress(0.0, text="Running inference...")
    def _cb(d, t):
        pb.progress(d / t, text=f"U-Net inference · tile {d}/{t}")
    mask = predict_sliding(img, model, mean, std,
                           patch_size=patch_size, stride=stride,
                           threshold=threshold, progress_cb=_cb)
    pb.empty()
    mode_label = "U-Net Model"
else:
    mask = predict_demo(img, threshold_pct)
    mode_label = f"Demo (p{threshold_pct})"

area_info = calculate_spill_area(mask, transform, crs)
oil_pixels = area_info["oil_pixels"]
oil_pct    = 100.0 * oil_pixels / (H * W)

c1, c2, c3, c4 = st.columns(4)
c1.markdown(card(f"{oil_pixels:,}", "Oil Pixels", "red"), unsafe_allow_html=True)
c2.markdown(card(f"{oil_pct:.2f}%", "Coverage", "red"), unsafe_allow_html=True)

if area_info["area_km2"] is not None and area_info["area_km2"] > 0:
    akm = area_info["area_km2"]
    c3.markdown(card(f"{akm:.2f}" if akm >= 1 else f"{akm*1000:.0f} m²×10³",
                     "Spill Area (km²)", "amber"), unsafe_allow_html=True)
else:
    c3.markdown(card("—", "Spill Area"), unsafe_allow_html=True)

badge_cls = "badge-ok" if has_model else "badge-demo"
c4.markdown(f"""<div class="card" style="display:flex; flex-direction:column;
    align-items:center; justify-content:center; min-height:80px;">
    <span class="badge {badge_cls}">{mode_label}</span>
    <div class="card-label" style="margin-top:0.5rem;">Method</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ── 04 Prediction Preview ────────────────────────────────────────────────────

st.markdown(sec("04", "Prediction Preview"), unsafe_allow_html=True)
st.image(render_preview(img, mask), use_container_width=True)
st.markdown("<br>", unsafe_allow_html=True)

dc1, dc2 = st.columns(2)

with dc1:
    if crs is not None:
        buf = io.BytesIO()
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(driver="GTiff", height=H, width=W, count=1,
                              dtype="uint8", crs=crs, transform=transform,
                              compress="deflate") as dst:
                dst.write(mask.astype(np.uint8), 1)
            buf.write(memfile.read())
        buf.seek(0)
        st.download_button("⬇ Download mask (GeoTIFF)", data=buf.getvalue(),
                           file_name="oil_spill_mask.tif", mime="image/tiff")
    else:
        st.caption("GeoTIFF unavailable (no CRS).")

with dc2:
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("⬇ Download mask (PNG)", data=buf.getvalue(),
                       file_name="oil_spill_mask.png", mime="image/png")

st.markdown("<br>", unsafe_allow_html=True)

# ── 05 Map + 06 Coordinates ──────────────────────────────────────────────────

map_col, table_col = st.columns([3, 2])

with map_col:
    st.markdown(sec("05", "Geographic Map"), unsafe_allow_html=True)
    with st.spinner("Mapping oil pixels..."):
        coords = mask_to_latlon(mask, transform, crs)

    if not coords:
        st.markdown("""
        <div style="text-align:center; padding:3rem; border:1px solid rgba(248,113,113,0.15);
                    border-radius:12px; background:rgba(248,113,113,0.03);">
            <div style="font-size:1.5rem; margin-bottom:0.5rem;">🔍</div>
            <div style="color:#94a3b8; font-size:0.9rem;">No oil pixels detected</div>
        </div>""", unsafe_allow_html=True)
    else:
        disp = subsample(coords, MAP_MAX_MARKERS)
        st.caption(f"Rendering {len(disp):,} of {len(coords):,} pixels")
        st_folium(build_dot_map(disp), width=None, height=480, returned_objects=[])

with table_col:
    st.markdown(sec("06", "Coordinates"), unsafe_allow_html=True)

    if not coords:
        st.caption("No data.")
    else:
        LIMIT = 150
        rows = ""
        for i, (lat, lon) in enumerate(coords[:LIMIT]):
            rows += f"<tr><td>{i+1}</td><td>{lat:.6f}</td><td>{lon:.6f}</td></tr>"

        st.markdown(f"""
        <div style="max-height:400px; overflow-y:auto;
                    border:1px solid var(--border); border-radius:10px;">
        <table class="coord-table">
            <thead><tr><th>#</th><th>Lat</th><th>Lon</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>
        </div>
        <div style="font-size:0.7rem; color:#475569; margin-top:0.5rem;">
            {min(LIMIT, len(coords)):,} of {len(coords):,} shown
        </div>
        """, unsafe_allow_html=True)

        csv_data = "latitude,longitude\n" + "\n".join(
            f"{lat:.8f},{lon:.8f}" for lat, lon in coords
        )
        st.download_button("⬇ Download coordinates (CSV)",
                           data=csv_data, file_name="oil_spill_coords.csv",
                           mime="text/csv")

st.markdown("<br>", unsafe_allow_html=True)


# ── 07 Proximity Analysis ────────────────────────────────────────────────────

st.markdown(sec("07", "Proximity Analysis — Nearby Coastlines"), unsafe_allow_html=True)

if not coords:
    st.caption("No spill detected — proximity analysis requires oil pixel coordinates.")
else:
    centroid = spill_centroid(coords)
    st.caption(f"Spill centroid: {centroid[0]:.4f}°N, {centroid[1]:.4f}°E · "
               f"Search radius: {proximity_radius} km")

    with st.spinner("Searching nearby coastlines..."):
        coasts = find_nearby_coasts(centroid[0], centroid[1],
                                    radius_km=proximity_radius)

    if not coasts:
        st.markdown(f"""
        <div style="text-align:center; padding:2rem; border:1px solid var(--border);
                    border-radius:12px; background:var(--bg-card);">
            <div style="color:#64748b; font-size:0.9rem;">
                No coastlines found within {proximity_radius} km.
                Try increasing the search radius in the sidebar.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        max_dist = max(c["distance_km"] for c in coasts) if coasts else 1

        for c in coasts:
            bar_pct = max(5, 100 - (c["distance_km"] / max(proximity_radius, 1)) * 100)
            direction_word = {
                "N": "north", "NE": "northeast", "E": "east", "SE": "southeast",
                "S": "south", "SW": "southwest", "W": "west", "NW": "northwest",
            }.get(c["direction"], c["direction"])
            st.markdown(f"""
            <div class="coast-card">
                <div class="coast-dir">{c['direction']}</div>
                <div class="coast-info">
                    <div class="coast-name">{c['name']}</div>
                    <div class="coast-dist">
                        {c['distance_km']:.1f} km {direction_word} ({c['bearing_deg']:.0f}°)
                        · nearest point {c['nearest_lat']:.3f}°, {c['nearest_lon']:.3f}°
                    </div>
                    <div class="coast-bar-bg">
                        <div class="coast-bar" style="width:{bar_pct:.0f}%;"></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:0.7rem; color:#475569; margin-top:0.8rem;
                    font-family:'JetBrains Mono',monospace;">
            ℹ Distance-based proximity only — does not account for
            currents, wind, or drift. Not a spill trajectory forecast.
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
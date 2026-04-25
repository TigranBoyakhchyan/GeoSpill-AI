"""
app.py — Streamlit web app for SAR oil spill detection.

Upload a Sentinel-1 GeoTIFF → model predicts oil spill mask → map visualization.

Modes:
    MODEL mode: uses the trained U-Net for inference
    DEMO  mode: uses a simple VV-band threshold when no model is available

Run:
    streamlit run app.py

Expected files (for MODEL mode):
    models/best_model_YYYYMMDD_HHMMSS.pt   — trained checkpoint from training.py
    data/train_stats.json                   — normalization stats from preprocess.py
"""

import os
import io
import json
import tempfile
import warnings

import numpy as np
import streamlit as st
import folium
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from streamlit_folium import st_folium

import rasterio
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SAR Oil Spill Detector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── STYLES ───────────────────────────────────────────────────────────────────

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

.stApp { background-color: var(--navy); }
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--white); }
h1, h2, h3 { font-family: 'Space Mono', monospace; letter-spacing: -0.02em; }

.hero {
    background: linear-gradient(135deg, #0a1628 0%, #1a3a5c 50%, #0d2540 100%);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem; font-weight: 700;
    color: var(--accent); margin: 0 0 0.4rem 0;
}
.hero-sub { color: var(--muted); font-size: 0.95rem; margin: 0; font-weight: 300; }

.metric-card {
    background: linear-gradient(135deg, #0f2035, #162d4a);
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 10px;
    padding: 1.2rem 1.5rem; text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem; font-weight: 700;
    color: var(--accent); line-height: 1;
}
.metric-label {
    font-size: 0.78rem; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em; margin-top: 0.4rem;
}

.badge {
    display: inline-block; padding: 0.25rem 0.75rem;
    border-radius: 20px; font-size: 0.75rem;
    font-family: 'Space Mono', monospace; font-weight: 700;
}
.badge-ok   { background: rgba(0,212,255,0.12); color: var(--accent); border: 1px solid rgba(0,212,255,0.25); }
.badge-demo { background: rgba(255,165,0,0.15); color: #ffa500; border: 1px solid rgba(255,165,0,0.3); }
.badge-err  { background: rgba(255,68,68,0.15); color: var(--oil); border: 1px solid rgba(255,68,68,0.3); }

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.15em; color: var(--muted);
    border-bottom: 1px solid rgba(0,212,255,0.15);
    padding-bottom: 0.5rem; margin-bottom: 1rem;
}

.pixel-table {
    font-family: 'Space Mono', monospace; font-size: 0.8rem;
    width: 100%; border-collapse: collapse;
}
.pixel-table th {
    color: var(--accent); font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.08em; padding: 0.5rem 0.75rem;
    border-bottom: 1px solid rgba(0,212,255,0.2); text-align: left;
}
.pixel-table td {
    padding: 0.4rem 0.75rem; color: var(--white);
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.pixel-table tr:hover td { background: rgba(0,212,255,0.04); }
</style>
""", unsafe_allow_html=True)


# ─── CONFIG ───────────────────────────────────────────────────────────────────

DEFAULT_MODEL_PATH = "models/best_model_20260416_145208.pt"
DEFAULT_STATS_PATH = "data/train_stats.json"

DB_CLIP_MIN = -50.0
DB_CLIP_MAX = 0.0

MAP_MAX_MARKERS = 3000    # folium performance limit


# ─── MODEL LOADING ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(checkpoint_path: str, stats_path: str):
    """
    Load the trained U-Net and normalization stats.
    Returns (model, stats, error_message). On success error_message is None.
    """
    if not os.path.exists(checkpoint_path):
        return None, None, f"Checkpoint not found: {checkpoint_path}"
    if not os.path.exists(stats_path):
        return None, None, f"Stats file not found: {stats_path}"

    try:
        import torch
        import segmentation_models_pytorch as smp

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = smp.Unet(
            encoder_name    = "resnet34",
            encoder_weights = None,
            in_channels     = 2,
            classes         = 1,
        )

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # training.py saves under key "model_state"
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt

        model.load_state_dict(state)
        model.to(device).eval()

        with open(stats_path) as f:
            stats = json.load(f)

        return model, stats, None

    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


# ─── INFERENCE ────────────────────────────────────────────────────────────────

def tile_positions(length: int, patch: int, stride: int) -> list:
    """Generate sliding-window start positions that fully cover [0, length)."""
    if length <= patch:
        return [0]
    positions = list(range(0, length - patch + 1, stride))
    if positions[-1] + patch < length:
        positions.append(length - patch)
    return positions


def predict_model(img: np.ndarray,
                  model,
                  mean: np.ndarray,
                  std:  np.ndarray,
                  patch_size: int = 256,
                  stride:     int = 128,
                  threshold:  float = 0.5,
                  progress_cb=None) -> np.ndarray:
    """
    Run U-Net sliding-window inference on a 2-band SAR image.
    Returns binary mask of shape (H, W), values 0.0 / 1.0.
    """
    import torch
    device = next(model.parameters()).device

    img_norm = np.clip(img[:2], DB_CLIP_MIN, DB_CLIP_MAX).astype(np.float32)
    img_norm = (img_norm - mean[:, None, None]) / (std[:, None, None] + 1e-6)

    _, H, W = img_norm.shape
    pred_sum   = np.zeros((H, W), dtype=np.float32)
    pred_count = np.zeros((H, W), dtype=np.float32)

    ys = tile_positions(H, patch_size, stride)
    xs = tile_positions(W, patch_size, stride)
    total_tiles = len(ys) * len(xs)
    done = 0

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = img_norm[:, y:y + patch_size, x:x + patch_size]
                t = torch.from_numpy(patch).unsqueeze(0).to(device)
                logits = model(t)
                probs  = torch.sigmoid(logits).squeeze().cpu().numpy()
                pred_sum  [y:y + patch_size, x:x + patch_size] += probs
                pred_count[y:y + patch_size, x:x + patch_size] += 1.0
                done += 1
                if progress_cb:
                    progress_cb(done, total_tiles)

    avg_prob = pred_sum / np.maximum(pred_count, 1e-6)
    return (avg_prob > threshold).astype(np.float32)


def predict_demo(img: np.ndarray, threshold_pct: int = 8) -> np.ndarray:
    """
    Demo mode: simple threshold detector for testing the UI.
    SAR oil spills appear dark — bottom N% of VV backscatter is flagged.
    """
    vv = img[0] if img.ndim == 3 else img
    vv_clipped = np.clip(vv, DB_CLIP_MIN, DB_CLIP_MAX)
    vv_norm = (vv_clipped - vv_clipped.min()) / (vv_clipped.max() - vv_clipped.min() + 1e-6)
    threshold_val = np.percentile(vv_norm, threshold_pct)
    return (vv_norm < threshold_val).astype(np.float32)


# ─── GEO HELPERS ──────────────────────────────────────────────────────────────

def mask_to_latlon(mask: np.ndarray, transform, crs) -> list:
    """Convert oil spill mask pixels to (lat, lon) coordinates."""
    rows, cols = np.where(mask > 0.5)
    if len(rows) == 0:
        return []

    # Pixel coords → projected coords
    xs = transform.c + cols * transform.a
    ys = transform.f + rows * transform.e

    # Reproject to WGS84 if needed
    if crs and str(crs) != "EPSG:4326":
        try:
            from pyproj import Transformer
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            xs, ys = transformer.transform(xs, ys)
        except Exception:
            pass

    # (lat, lon) — after WGS84 transform, xs=lon, ys=lat
    return list(zip(ys.tolist(), xs.tolist()))


def subsample(coords: list, max_points: int = MAP_MAX_MARKERS) -> list:
    """Subsample coordinate list for map performance."""
    if len(coords) <= max_points:
        return coords
    step = max(1, len(coords) // max_points)
    return coords[::step]


def build_dot_map(coords: list) -> folium.Map:
    """Build a folium map with oil spill pixels as red dot markers."""
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

    # Satellite layer toggle
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    if not coords:
        return m

    # Draw subsampled oil pixels as small red circles
    display = subsample(coords, MAP_MAX_MARKERS)
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

    folium.LayerControl(collapsed=False).add_to(m)

    # Legend
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


# ─── VISUALIZATION ────────────────────────────────────────────────────────────

def render_side_by_side(img: np.ndarray, mask: np.ndarray) -> bytes:
    """Render VV band, predicted mask, and overlay as one PNG."""
    vv = img[0]
    vv_show = np.clip(vv, DB_CLIP_MIN, DB_CLIP_MAX)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0a1628")

    axes[0].imshow(vv_show, cmap="gray")
    axes[0].set_title("VV Band (dB)", color="#e8f4f8")
    axes[0].axis("off")

    oil_cmap = ListedColormap([(0, 0, 0, 0), (1.0, 0.27, 0.27, 1.0)])
    axes[1].imshow(mask, cmap=oil_cmap, vmin=0, vmax=1)
    axes[1].set_facecolor("#0a1628")
    axes[1].set_title("Predicted Mask", color="#e8f4f8")
    axes[1].axis("off")

    axes[2].imshow(vv_show, cmap="gray")
    axes[2].imshow(mask, cmap=oil_cmap, vmin=0, vmax=1, alpha=0.55)
    axes[2].set_title("Overlay", color="#e8f4f8")
    axes[2].axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor="#0a1628", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div class="section-header">⚙ Configuration</div>', unsafe_allow_html=True)

    model_path = st.text_input("Model checkpoint (.pt)", value=DEFAULT_MODEL_PATH)
    stats_path = st.text_input("Stats file (train_stats.json)", value=DEFAULT_STATS_PATH)

    st.markdown("---")

    # Try loading model
    model, stats, err = load_model(model_path, stats_path)
    has_model = model is not None

    if has_model:
        device_str = "GPU" if next(model.parameters()).is_cuda else "CPU"
        st.markdown(f'<span class="badge badge-ok">✓ MODEL LOADED ({device_str})</span>',
                    unsafe_allow_html=True)

        threshold = st.slider(
            "Detection threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05,
            help="Probability threshold for binarizing the U-Net output."
        )
        patch_size = st.selectbox("Patch size", [256, 512], index=0)
        stride     = st.selectbox("Stride", [patch_size // 2, patch_size], index=0)
    else:
        st.markdown('<span class="badge badge-demo">⚡ DEMO MODE</span>',
                    unsafe_allow_html=True)
        st.caption(f"Reason: {err}")
        st.caption("Using threshold-based detection for UI testing.")

        threshold_pct = st.slider(
            "Oil sensitivity (percentile)",
            min_value=2, max_value=20, value=8, step=1,
            help="Lower = more pixels flagged as oil. Bottom N% of backscatter."
        )

    st.markdown("---")
    st.markdown('<div class="section-header">ℹ About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.82rem; color:#7a9ab0; line-height:1.6;">
    Upload a Sentinel-1 SAR GeoTIFF to detect oil spills.<br><br>
    <b style="color:#00d4ff;">Model mode</b> uses the trained U-Net.<br>
    <b style="color:#ffa500;">Demo mode</b> uses backscatter thresholding.<br><br>
    Oil spills appear dark in SAR imagery due to dampened capillary waves.
    </div>
    """, unsafe_allow_html=True)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <p class="hero-title">🛰️ SAR Oil Spill Detector</p>
    <p class="hero-sub">Sentinel-1 GeoTIFF → Deep Learning Segmentation → Geographic Visualization</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-header">01 — Upload Image</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop a Sentinel-1 GeoTIFF here",
    type=["tif", "tiff"],
    help="2-band SAR image (VV + VH polarization)",
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

# ── Read the TIF ──────────────────────────────────────────────────────────────

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

n_bands, H, W = img.shape

# After reading the image, check if values look like linear power
# (mostly positive, range 0-5ish) vs decibels (mostly negative, range -50 to 0)
if img[0].mean() > 0 and img[0].max() < 10:
    # Likely linear power — convert to dB
    st.info("Detected linear-scale data — converting to decibels (dB = 10·log₁₀).")
    img = np.where(img > 0, 10.0 * np.log10(img + 1e-10), -50.0).astype(np.float32)
# ── Image Properties ──────────────────────────────────────────────────────────

st.markdown('<div class="section-header">02 — Image Properties</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
for col, value, label in [
    (c1, f"{W}×{H}", "Resolution (px)"),
    (c2, f"{n_bands}", "Bands"),
    (c3, f"{float(img[0].min()):.1f}", "Min dB (VV)"),
    (c4, f"{float(img[0].max()):.1f}", "Max dB (VV)"),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Prediction ────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">03 — Detection</div>', unsafe_allow_html=True)

if has_model:
    # MODEL MODE
    mean = np.array(stats["mean"], dtype=np.float32)
    std  = np.array(stats["std"],  dtype=np.float32)

    if img.shape[0] < 2:
        st.error(f"Image has only {img.shape[0]} band(s); model requires 2 (VV + VH).")
        st.stop()

    progress_bar = st.progress(0.0, text="Running U-Net sliding-window inference...")
    def _cb(done, total):
        progress_bar.progress(done / total, text=f"Running U-Net... ({done}/{total} tiles)")

    mask = predict_model(
        img, model, mean, std,
        patch_size=patch_size, stride=stride, threshold=threshold,
        progress_cb=_cb,
    )
    progress_bar.empty()
    mode_label = "U-Net Model"
else:
    # DEMO MODE
    mask = predict_demo(img, threshold_pct)
    mode_label = f"Threshold Demo (p={threshold_pct})"

oil_pixels   = int(mask.sum())
total_pixels = H * W
oil_pct      = 100.0 * oil_pixels / total_pixels

# Detection summary
c1, c2, c3 = st.columns(3)
c1.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{oil_pixels:,}</div>
    <div class="metric-label">Oil Pixels Detected</div>
</div>""", unsafe_allow_html=True)
c2.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{oil_pct:.2f}%</div>
    <div class="metric-label">Coverage</div>
</div>""", unsafe_allow_html=True)
badge_class = "badge-ok" if has_model else "badge-demo"
c3.markdown(f"""
<div class="metric-card">
    <div style="margin-top:0.4rem;">
        <span class="badge {badge_class}">{mode_label}</span>
    </div>
    <div class="metric-label" style="margin-top:0.6rem;">Detection Method</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Side-by-side preview ──────────────────────────────────────────────────────

st.markdown('<div class="section-header">04 — Prediction Preview</div>', unsafe_allow_html=True)
preview_png = render_side_by_side(img, mask)
st.image(preview_png, use_container_width=True)

# ── Map + Coordinates ─────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)

map_col, table_col = st.columns([3, 2])

with map_col:
    st.markdown('<div class="section-header">05 — Geographic Map</div>', unsafe_allow_html=True)

    with st.spinner("Converting pixels to coordinates..."):
        coords = mask_to_latlon(mask, transform, crs)

    if not coords:
        st.warning("No oil pixels detected — nothing to display on the map.")
    else:
        display_coords = subsample(coords, MAP_MAX_MARKERS)
        st.caption(f"Showing {len(display_coords):,} of {len(coords):,} oil pixels "
                   f"(subsampled to {MAP_MAX_MARKERS:,} max for performance)")

        folium_map = build_dot_map(display_coords)
        st_folium(folium_map, width=700, height=500, returned_objects=[])

        if crs is None or str(crs) == "None":
            st.caption("⚠️ No CRS found — map coordinates may not be geographically accurate.")

with table_col:
    st.markdown('<div class="section-header">06 — Oil Pixel Coordinates</div>', unsafe_allow_html=True)

    if not coords:
        st.info("No coordinates to display.")
    else:
        # Show first N in a scrollable table
        DISPLAY_LIMIT = 200
        table_coords = coords[:DISPLAY_LIMIT]

        rows_html = ""
        for i, (lat, lon) in enumerate(table_coords):
            rows_html += f"<tr><td>{i+1}</td><td>{lat:.6f}</td><td>{lon:.6f}</td></tr>"

        st.markdown(f"""
        <div style="max-height:420px; overflow-y:auto;
                    border:1px solid rgba(0,212,255,0.1); border-radius:8px;">
        <table class="pixel-table">
            <thead><tr>
                <th>#</th><th>Latitude</th><th>Longitude</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        <div style="font-size:0.75rem; color:#7a9ab0; margin-top:0.5rem;">
            Showing {min(DISPLAY_LIMIT, len(coords)):,} of {len(coords):,} oil pixels
        </div>
        """, unsafe_allow_html=True)

        # CSV download
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

# ── Downloads ─────────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">07 — Download Mask</div>', unsafe_allow_html=True)

dc1, dc2 = st.columns(2)

with dc1:
    # Mask as GeoTIFF (georeferenced)
    if crs is not None:
        buf = io.BytesIO()
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(
                driver="GTiff",
                height=H, width=W, count=1,
                dtype="uint8",
                crs=crs, transform=transform,
                compress="deflate",
            ) as dst:
                dst.write(mask.astype(np.uint8), 1)
            buf.write(memfile.read())
        buf.seek(0)
        st.download_button(
            "⬇ Download mask (GeoTIFF)",
            data=buf.getvalue(),
            file_name="oil_spill_mask.tif",
            mime="image/tiff",
        )
    else:
        st.caption("GeoTIFF unavailable (source had no CRS).")

with dc2:
    # Mask as PNG
    from PIL import Image as PILImage
    mask_img = PILImage.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        "⬇ Download mask (PNG)",
        data=buf.getvalue(),
        file_name="oil_spill_mask.png",
        mime="image/png",
    )
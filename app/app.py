"""
app.py — GeoSpill AI — SAR Oil Spill Detection

Multi-image support with button-triggered detection.
Inference only runs when the user clicks "Run Detection."
Sidebar changes, coast radius adjustments, and file uploads are instant.
"""

import os, io, sys, tempfile, warnings
import numpy as np
import streamlit as st
import folium
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from streamlit_folium import st_folium
from PIL import Image
import rasterio
from rasterio.errors import NotGeoreferencedWarning

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from inference import load_checkpoint, predict_sliding, predict_demo, filter_small_regions
from geo import (is_linear_scale, linear_to_db, mask_to_latlon, subsample,
    calculate_spill_area, find_nearby_coasts, spill_centroid, DB_CLIP_MIN, DB_CLIP_MAX)

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

SPILL_COLORS = [("#f87171","Red"),("#fb923c","Orange"),("#fbbf24","Amber"),
    ("#a78bfa","Purple"),("#34d399","Emerald"),("#f472b6","Pink"),
    ("#38bdf8","Sky"),("#e879f9","Fuchsia")]
MAP_MAX_MARKERS = 3000
DEFAULT_MODEL_PATH = "models/best_model_20260604_091953.pt"
DEFAULT_STATS_PATH = "data/train_stats.json"

st.set_page_config(page_title="GeoSpill AI", page_icon="🛰️",
                   layout="wide", initial_sidebar_state="expanded")

# ─── STYLES ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Outfit:wght@300;400;500;600;700&display=swap');
:root{--bg:#060d18;--bg2:#0b1726;--br:rgba(56,189,248,.08);--brg:rgba(56,189,248,.2);
--c:#38bdf8;--t:#2dd4bf;--r:#f87171;--a:#fbbf24;--tx:#e2e8f0;--td:#64748b;--tm:#475569}
.stApp{background:var(--bg);background-image:radial-gradient(ellipse 80% 50% at 20% 40%,rgba(56,189,248,.03),transparent),radial-gradient(ellipse 60% 40% at 80% 60%,rgba(45,212,191,.02),transparent)}
html,body,[class*="css"]{font-family:'Outfit',sans-serif;color:var(--tx)}
.hero{position:relative;overflow:hidden;background:linear-gradient(160deg,#060d18,#0f2847 40%,#0a1e3a 70%,#060d18);border:1px solid var(--brg);border-radius:16px;padding:2.5rem 3rem;margin-bottom:2rem}
.hero::before{content:'';position:absolute;inset:0;background:repeating-linear-gradient(90deg,transparent,transparent 60px,rgba(56,189,248,.02) 60px,rgba(56,189,248,.02) 61px),repeating-linear-gradient(0deg,transparent,transparent 60px,rgba(56,189,248,.02) 60px,rgba(56,189,248,.02) 61px);pointer-events:none}
.hero-tag{font-family:'JetBrains Mono',monospace;font-size:.65rem;font-weight:500;color:var(--t);letter-spacing:.2em;text-transform:uppercase;margin-bottom:.6rem}
.hero-title{font-family:'Outfit';font-size:2.2rem;font-weight:700;color:#f8fafc;margin:0 0 .5rem;letter-spacing:-.03em}
.hero-title span{color:var(--c)}
.hero-sub{color:var(--td);font-size:1rem;font-weight:300;margin:0;max-width:620px;line-height:1.6}
.sec{font-family:'JetBrains Mono',monospace;font-size:.65rem;font-weight:500;text-transform:uppercase;letter-spacing:.18em;color:var(--tm);padding-bottom:.6rem;margin-bottom:1.2rem;border-bottom:1px solid var(--br);display:flex;align-items:center;gap:.6rem}
.sec-n{display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:6px;background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.15);font-size:.6rem;color:var(--c)}
.card{background:var(--bg2);border:1px solid var(--br);border-radius:12px;padding:1.2rem 1.4rem;text-align:center}
.cv{font-family:'JetBrains Mono',monospace;font-size:1.35rem;font-weight:700;color:var(--c);line-height:1}
.cv.r{color:var(--r)}.cv.a{color:var(--a)}.cv.t{color:var(--t)}
.cl{font-size:.7rem;color:var(--tm);text-transform:uppercase;letter-spacing:.1em;margin-top:.4rem;font-weight:500}
.badge{display:inline-block;padding:.3rem .8rem;border-radius:20px;font-size:.7rem;font-family:'JetBrains Mono',monospace;font-weight:600}
.b-ok{background:rgba(56,189,248,.1);color:var(--c);border:1px solid rgba(56,189,248,.2)}
.b-dm{background:rgba(251,191,36,.1);color:var(--a);border:1px solid rgba(251,191,36,.2)}
.cc{background:var(--bg2);border:1px solid var(--br);border-radius:12px;padding:1rem 1.2rem;display:flex;align-items:center;gap:1rem;margin-bottom:.6rem}
.cd{width:42px;height:42px;display:flex;align-items:center;justify-content:center;border-radius:10px;background:rgba(56,189,248,.06);border:1px solid rgba(56,189,248,.12);font-family:'JetBrains Mono',monospace;font-size:.75rem;font-weight:700;color:var(--c);flex-shrink:0}
.ci{flex:1}.cn{font-weight:600;font-size:.95rem;color:#f1f5f9;margin-bottom:.2rem}
.cdt{font-family:'JetBrains Mono',monospace;font-size:.78rem;color:var(--td)}
.cb{height:4px;border-radius:2px;background:rgba(56,189,248,.08);margin-top:.4rem;overflow:hidden}
.cbf{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--c),var(--t))}
.ct{font-family:'JetBrains Mono',monospace;font-size:.78rem;width:100%;border-collapse:collapse}
.ct th{color:var(--c);font-size:.65rem;text-transform:uppercase;letter-spacing:.1em;padding:.6rem .8rem;border-bottom:1px solid var(--brg);text-align:left;font-weight:500}
.ct td{padding:.45rem .8rem;color:var(--td);border-bottom:1px solid var(--br)}
.ct tr:hover td{background:rgba(56,189,248,.03);color:var(--tx)}
.run-btn{margin:1rem 0}

/* Button styled to match the "Sentinel-1 SAR Analysis" tag */
.stButton > button {
    background-color: #2bb597 !important; /* same as .hero-tag */
    border-color: #2bb597 !important;
    color: #060d18 !important;
    border-radius: 10px !important;
    padding: .6rem 1rem !important;
    font-weight: 700 !important;
    box-shadow: 0 8px 18px rgba(45,212,191,0.12) !important;
    transition: transform .12s ease, box-shadow .12s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 30px rgba(45,212,191,0.18) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

</style>""", unsafe_allow_html=True)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def sec(n, label):
    return f'<div class="sec"><span class="sec-n">{n}</span>{label}</div>'
def crd(v, l, c=""):
    cls = f" {c}" if c else ""
    return f'<div class="card"><div class="cv{cls}">{v}</div><div class="cl">{l}</div></div>'

def render_preview(img, mask):
    vv = np.clip(img[0], DB_CLIP_MIN, DB_CLIP_MAX)
    oil_cmap = ListedColormap([(0,0,0,0),(1,.47,.47,1)])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="#060d18")
    for a in ax: a.set_facecolor("#060d18"); a.axis("off")
    ax[0].imshow(vv, cmap="gray")
    ax[0].set_title("VV Band", color="#94a3b8", fontfamily="monospace", fontsize=9)
    ax[1].imshow(mask, cmap=oil_cmap, vmin=0, vmax=1)
    ax[1].set_title("Mask", color="#94a3b8", fontfamily="monospace", fontsize=9)
    ax[2].imshow(vv, cmap="gray")
    ax[2].imshow(mask, cmap=oil_cmap, vmin=0, vmax=1, alpha=.55)
    ax[2].set_title("Overlay", color="#94a3b8", fontfamily="monospace", fontsize=9)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="#060d18", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

def process_single_file(file_bytes, filename, model, stats, has_model,
                        threshold, patch_size, stride, threshold_pct, min_area,
                        progress_container):
    """Process one TIF file — read, infer, filter, compute everything."""
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp.write(file_bytes); tmp_path = tmp.name
    try:
        with rasterio.open(tmp_path) as src:
            img = src.read().astype(np.float32)
            tf6 = tuple(src.transform)[:6]
            crs_wkt = src.crs.to_wkt() if src.crs else None
    finally:
        os.unlink(tmp_path)

    if img.ndim == 2: img = img[None, ...]
    converted = False
    if is_linear_scale(img): img = linear_to_db(img); converted = True
    n_bands, H, W = img.shape

    from rasterio.transform import Affine; from rasterio.crs import CRS
    transform = Affine(*tf6)
    crs = CRS.from_wkt(crs_wkt) if crs_wkt else None

    # Inference
    if has_model and n_bands >= 2:
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype=np.float32)
        pb = progress_container.progress(0.0, text=f"Processing {filename}...")
        def _cb(d, t): pb.progress(d/t, text=f"{filename}: tile {d}/{t}")
        mask, prob_map = predict_sliding(img, model, mean, std, patch_size=patch_size,
                               stride=stride, threshold=threshold, progress_cb=_cb)
        pb.empty()
        mode = "U-Net"
    else:
        mask, prob_map = predict_demo(img, threshold_pct)
        mode = "Demo"

    # Post-processing
    mask = filter_small_regions(mask, min_pixels=min_area)

    # Compute results
    area_info = calculate_spill_area(mask, transform, crs)
    coords = mask_to_latlon(mask, transform, crs)

    # Render preview
    preview_png = render_preview(img, mask)

    # Render confidence heatmap
    from matplotlib.colors import LinearSegmentedColormap
    heatmap_colors = [
        (0.0,  "#060d18"),  # background — no confidence
        (0.15, "#0c2341"),  # very low — deep blue
        (0.35, "#164e6e"),  # low — teal hint
        (0.50, "#2dd4bf"),  # threshold zone — teal
        (0.65, "#fbbf24"),  # above threshold — amber
        (0.80, "#f87171"),  # confident — red
        (1.0,  "#ff2222"),  # very confident — bright red
    ]
    positions = [c[0] for c in heatmap_colors]
    colors_hex = [c[1] for c in heatmap_colors]
    # Convert hex to RGB tuples
    colors_rgb = []
    for h in colors_hex:
        h = h.lstrip("#")
        colors_rgb.append(tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)))
    oil_heatmap_cmap = LinearSegmentedColormap.from_list("oil_conf", list(zip(positions, colors_rgb)))

    fig_h, ax_h = plt.subplots(1, 1, figsize=(15, 5), facecolor="#060d18")
    ax_h.set_facecolor("#060d18")
    ax_h.axis("off")
    im = ax_h.imshow(prob_map, cmap=oil_heatmap_cmap, vmin=0, vmax=1)
    # Threshold line in colorbar
    cbar = fig_h.colorbar(im, ax=ax_h, fraction=0.025, pad=0.02)
    cbar.set_label("Oil Probability", color="#94a3b8", fontfamily="monospace", fontsize=9)
    cbar.ax.tick_params(colors="#64748b", labelsize=8)
    cbar.ax.axhline(y=threshold, color="#ffffff", linewidth=1.5, linestyle="--")
    cbar.ax.text(1.5, threshold, f" threshold ({threshold})", color="#ffffff",
                 fontsize=7, fontfamily="monospace", va="center",
                 transform=cbar.ax.get_yaxis_transform())
    plt.tight_layout()
    buf_h = io.BytesIO()
    fig_h.savefig(buf_h, format="png", dpi=110, facecolor="#060d18", bbox_inches="tight")
    plt.close(fig_h)
    heatmap_png = buf_h.getvalue()
    mi = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    buf = io.BytesIO(); mi.save(buf, format="PNG"); mask_png = buf.getvalue()

    geotiff_bytes = None
    if crs is not None:
        buf = io.BytesIO()
        with rasterio.io.MemoryFile() as mf:
            with mf.open(driver="GTiff", height=H, width=W, count=1, dtype="uint8",
                         crs=crs, transform=transform, compress="deflate") as dst:
                dst.write(mask.astype(np.uint8), 1)
            buf.write(mf.read())
        geotiff_bytes = buf.getvalue()

    return {
        "filename": filename, "n_bands": n_bands, "H": H, "W": W,
        "min_db": float(img[0].min()), "max_db": float(img[0].max()),
        "converted": converted, "mode": mode,
        "oil_pixels": area_info["oil_pixels"],
        "oil_pct": 100.0 * area_info["oil_pixels"] / (H * W),
        "area_info": area_info, "coords": coords,
        "preview_png": preview_png, "heatmap_png": heatmap_png,
        "mask_png": mask_png, "geotiff_bytes": geotiff_bytes,
    }


# ─── MODEL (CACHED) ──────────────────────────────────────────────────────────

@st.cache_resource
def _load_model(cp, sp):
    return load_checkpoint(cp, sp)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown('<div style="text-align:center;padding:1rem 0 .5rem">'
        '<div style="font-family:Outfit;font-size:1.2rem;font-weight:700;color:#f8fafc">'
        '🛰️ GeoSpill AI</div>'
        '<div style="font-family:JetBrains Mono,monospace;font-size:.6rem;color:#475569;'
        'letter-spacing:.15em;text-transform:uppercase;margin-top:.3rem">'
        'Configuration</div></div>', unsafe_allow_html=True)
    st.markdown("---")

    model_path = st.text_input("Model checkpoint", value=DEFAULT_MODEL_PATH)
    stats_path = st.text_input("Stats file", value=DEFAULT_STATS_PATH)
    model, stats, err = _load_model(model_path, stats_path)
    has_model = model is not None
    st.markdown("---")

    if has_model:
        dev = "GPU" if next(model.parameters()).is_cuda else "CPU"
        st.markdown(f'<span class="badge b-ok">✓ Model ({dev})</span>', unsafe_allow_html=True)

        threshold = st.slider("Detection threshold", 0.1, 0.9, 0.75, 0.05)
        threshold_pct = 8

        with st.expander("Advanced settings", expanded=False):
            patch_size = st.selectbox("Patch size", [256, 512], index=0)
            stride = st.selectbox("Stride", [patch_size // 2, patch_size], index=0)
            min_area = st.slider("Min spill size (pixels)", 100, 2000, 500, 100,
                                 help="Connected regions smaller than this are removed as noise.",
                                 key="min_area_model")
    else:
        st.markdown('<span class="badge b-dm">⚡ Demo</span>', unsafe_allow_html=True)
        st.caption(err)
        threshold_pct = st.slider("Oil sensitivity (%ile)", 2, 20, 8, 1)
        threshold = 0.5; patch_size = 256; stride = 128
        min_area = 500
    st.markdown("---")
    proximity_radius = st.slider("Coast search radius (km)", 50, 500, 200, 25)


# ─── HERO ─────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero"><div class="hero-tag">Sentinel-1 SAR Analysis</div>'
    '<div class="hero-title">Geo<span>Spill</span> AI</div>'
    '<div class="hero-sub">Deep learning oil spill detection from SAR imagery. '
    'Upload one or more GeoTIFFs, adjust settings, then click Run Detection.</div>'
    '</div>', unsafe_allow_html=True)


# ─── 01 UPLOAD ────────────────────────────────────────────────────────────────

st.markdown(sec("01", "Upload Images"), unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "upload", type=["tif", "tiff"],
    accept_multiple_files=True, label_visibility="collapsed",
)

if not uploaded_files:
    st.markdown('<div style="text-align:center;padding:4rem 2rem;border:1px dashed '
        'rgba(56,189,248,.15);border-radius:12px;background:rgba(56,189,248,.02)">'
        '<div style="font-size:2.5rem;margin-bottom:.8rem;opacity:.5">📡</div>'
        '<div style="font-family:JetBrains Mono,monospace;font-size:.85rem;color:#64748b">'
        'Awaiting satellite imagery...</div></div>', unsafe_allow_html=True)
    st.stop()

# Show uploaded file summary
for i, f in enumerate(uploaded_files):
    color, _ = SPILL_COLORS[i % len(SPILL_COLORS)]
    st.markdown(f'<div style="display:flex;align-items:center;gap:.5rem;margin:.3rem 0;'
        f'font-size:.85rem"><span style="color:{color}">●</span>'
        f'<span style="color:#94a3b8">{f.name}</span>'
        f'<span style="color:#475569;font-size:.75rem">'
        f'({f.size / 1e6:.1f} MB)</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─── 02 RUN DETECTION ────────────────────────────────────────────────────────

st.markdown(sec("02", "Detection"), unsafe_allow_html=True)

# Build current params fingerprint (excludes coast radius — that's display-only)
current_params = {
    "files": [f.name for f in uploaded_files],
    "file_sizes": [f.size for f in uploaded_files],
    "has_model": has_model,
    "threshold": threshold,
    "patch_size": patch_size,
    "stride": stride,
    "threshold_pct": threshold_pct,
    "min_area": min_area,
}

# Check if we need to (re)run
has_results = "detection_results" in st.session_state
params_match = has_results and st.session_state.get("detection_params") == current_params

if has_results and not params_match:
    st.warning("⚠️ Settings or files changed since last detection. "
               "Click **Run Detection** to update results.")

# The button
_, btn_col, _ = st.columns([2, 1, 2])
with btn_col:
    run_clicked = st.button(
        f"🔍 Run Detection",
        type="primary",
        width="stretch",
    )

if run_clicked:
    results = []
    progress_area = st.container()
    overall = progress_area.progress(0.0, text="Starting...")

    for i, uploaded in enumerate(uploaded_files):
        overall.progress(i / len(uploaded_files),
                         text=f"Image {i+1}/{len(uploaded_files)}: {uploaded.name}")
        file_bytes = uploaded.getvalue()
        r = process_single_file(
            file_bytes, uploaded.name, model, stats, has_model,
            threshold, patch_size, stride, threshold_pct, min_area,
            progress_area,
        )
        results.append(r)

    overall.progress(1.0, text="Detection complete!")
    overall.empty()

    # Store in session state
    st.session_state.detection_results = results
    st.session_state.detection_params = current_params
    has_results = True
    params_match = True
    st.rerun()

if not has_results:
    st.info("Upload images and click **Run Detection** to begin.")
    st.stop()


# ─── DISPLAY RESULTS (from session_state) ─────────────────────────────────────

results = st.session_state.detection_results

# ── 03 Per-image results ──────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(sec("03", f"Results — {len(results)} Image(s)"), unsafe_allow_html=True)

all_map_entries = []

for idx, r in enumerate(results):
    color, _ = SPILL_COLORS[idx % len(SPILL_COLORS)]
    short = r["filename"][:35] + ("..." if len(r["filename"]) > 35 else "")
    label = f"Image {idx+1}" if len(results) > 1 else "Image"

    with st.expander(f"🖼 {label} — {short}", expanded=(idx == 0)):
        if r["converted"]:
            st.info("📐 Linear → dB conversion applied")

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(crd(f"{r['W']}×{r['H']}", "Resolution"), unsafe_allow_html=True)
        c2.markdown(crd(f"{r['n_bands']}", "Bands"), unsafe_allow_html=True)
        c3.markdown(crd(f"{r['min_db']:.1f}", "Min dB"), unsafe_allow_html=True)
        c4.markdown(crd(f"{r['max_db']:.1f}", "Max dB"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        r1.markdown(crd(f"{r['oil_pixels']:,}", "Oil Pixels", "r"), unsafe_allow_html=True)
        r2.markdown(crd(f"{r['oil_pct']:.2f}%", "Coverage", "r"), unsafe_allow_html=True)
        ai = r["area_info"]
        if ai.get("area_km2") and ai["area_km2"] > 0:
            akm = ai["area_km2"]
            r3.markdown(crd(f"{akm:.2f}" if akm >= 1 else f"{akm*1000:.0f} m²×10³",
                            "Area (km²)", "a"), unsafe_allow_html=True)
        else:
            r3.markdown(crd("—", "Area"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Preview + download buttons underneath
        st.image(r["preview_png"], width="stretch")

        # Confidence heatmap
        show_heatmap = st.checkbox("Show confidence heatmap", value=False,
                                    key=f"hm_{idx}")
        if show_heatmap:
            st.image(r["heatmap_png"], width="stretch")
            st.markdown(
                '<div style="font-family:JetBrains Mono,monospace;font-size:.7rem;'
                'color:#475569;margin-top:.3rem">'
                'Brighter = higher oil probability. '
                'White dashed line on colorbar = detection threshold. '
                'Pixels above the threshold become the binary mask.</div>',
                unsafe_allow_html=True)
        
        d1, d2, d3 = st.columns(3)
        with d1:
            if r["geotiff_bytes"]:
                st.download_button("⬇ Mask GeoTIFF", data=r["geotiff_bytes"],
                    file_name=f"mask_{idx+1}.tif", key=f"gt_{idx}",
                    width="stretch")
        with d2:
            st.download_button("⬇ Mask PNG", data=r["mask_png"],
                file_name=f"mask_{idx+1}.png", key=f"pn_{idx}",
                width="stretch")
        with d3:
            if r["coords"]:
                csv = "latitude,longitude\n" + "\n".join(
                    f"{la:.8f},{lo:.8f}" for la, lo in r["coords"])
                st.download_button("⬇ Coords CSV", data=csv,
                    file_name=f"coords_{idx+1}.csv", key=f"cv_{idx}",
                    width="stretch")

        # Per-image proximity analysis
        if r["coords"]:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f'<div style="font-family:JetBrains Mono,monospace;font-size:.65rem;'
                f'font-weight:500;text-transform:uppercase;letter-spacing:.18em;color:#475569;'
                f'padding-bottom:.4rem;margin-bottom:.8rem;border-bottom:1px solid rgba(56,189,248,.08)">'
                f'Nearby Coastlines</div>', unsafe_allow_html=True)

            img_centroid = spill_centroid(r["coords"])
            st.caption(f"Centroid: {img_centroid[0]:.4f}°N, {img_centroid[1]:.4f}°E")

            try:
                img_coasts = find_nearby_coasts(img_centroid[0], img_centroid[1],
                                                radius_km=proximity_radius)
            except Exception as ex:
                img_coasts = []
                st.warning(f"Proximity error: {ex}")

            if not img_coasts:
                st.caption(f"No coastlines within {proximity_radius} km.")
            else:
                CW = {"N":"north","NE":"northeast","E":"east","SE":"southeast",
                      "S":"south","SW":"southwest","W":"west","NW":"northwest"}
                for c in img_coasts:
                    bp = max(5, 100 - (c["distance_km"] / max(proximity_radius, 1)) * 100)
                    dw = CW.get(c["direction"], c["direction"])
                    d = c["distance_km"]
                    # Color by urgency: red < 50km, amber 50-150km, cyan > 150km
                    if d < 50:
                        dist_color = "#f87171"; urgency = "HIGH RISK"
                    elif d < 150:
                        dist_color = "#fbbf24"; urgency = "MODERATE"
                    else:
                        dist_color = "#38bdf8"; urgency = "LOW RISK"

                    st.markdown(f"""
                    <div class="cc" style="padding:1.2rem 1.4rem">
                        <div class="cd" style="width:52px;height:52px;font-size:.85rem;
                            border-color:{dist_color}40;color:{dist_color}">{c['direction']}</div>
                        <div class="ci">
                            <div style="display:flex;align-items:baseline;gap:.6rem;margin-bottom:.3rem">
                                <span style="font-family:'JetBrains Mono',monospace;font-size:1.5rem;
                                    font-weight:700;color:{dist_color}">{d:.1f}</span>
                                <span style="font-size:.85rem;color:{dist_color};font-weight:500">km {dw}</span>
                                <span style="font-family:'JetBrains Mono',monospace;font-size:.6rem;
                                    padding:.2rem .5rem;border-radius:10px;
                                    background:{dist_color}15;color:{dist_color};
                                    border:1px solid {dist_color}30">{urgency}</span>
                            </div>
                            <div class="cn" style="font-size:1rem">{c['name']}</div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:.7rem;
                                color:#475569;margin-top:.2rem">
                                bearing {c['bearing_deg']:.0f}° · nearest point
                                {c['nearest_lat']:.3f}°, {c['nearest_lon']:.3f}°</div>
                            <div class="cb" style="margin-top:.5rem">
                                <div class="cbf" style="width:{bp:.0f}%;
                                    background:linear-gradient(90deg,{dist_color},{dist_color}88)"></div>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

    if r["coords"]:
        all_map_entries.append({
            "coords": r["coords"], "color": color,
            "label": f"{label}: {short}",
        })




# ── 04 Combined map ───────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(sec("04", "Combined Map"), unsafe_allow_html=True)

if not all_map_entries:
    st.info("No oil pixels detected in any image.")
else:
    total_oil = sum(len(e["coords"]) for e in all_map_entries)
    st.caption(f"{total_oil:,} oil pixels across {len(all_map_entries)} image(s)")

    all_pts = []
    for e in all_map_entries: all_pts.extend(e["coords"])
    cen = [np.mean([p[0] for p in all_pts]), np.mean([p[1] for p in all_pts])]
    m = folium.Map(location=cen, zoom_start=8, tiles="CartoDB dark_matter")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite", overlay=False, control=True).add_to(m)
    pl = max(100, MAP_MAX_MARKERS // max(len(all_map_entries), 1))
    for e in all_map_entries:
        for lat, lon in subsample(e["coords"], pl):
            folium.CircleMarker(location=[lat, lon], radius=2, color=e["color"],
                fill=True, fill_color=e["color"], fill_opacity=.7, weight=0,
                tooltip=e["label"]).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    leg = "".join(f'<div style="margin:3px 0"><span style="color:{e["color"]}">●</span> '
                  f'{e["label"]}</div>' for e in all_map_entries)
    m.get_root().html.add_child(folium.Element(
        f'<div style="position:fixed;bottom:20px;left:20px;z-index:1000;background:#0b1726;'
        f'border:1px solid rgba(56,189,248,.2);border-radius:10px;padding:12px 16px;'
        f'font-family:monospace;font-size:11px;color:#e2e8f0;max-width:240px">'
        f'<div style="color:#38bdf8;font-weight:bold;margin-bottom:6px;font-size:10px;'
        f'letter-spacing:.1em">DETECTED SPILLS</div>{leg}</div>'))

    st_folium(m, width=None, height=520, returned_objects=[])
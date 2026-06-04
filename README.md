# 🛰️ GeoSpill AI — SAR Oil Spill Detection

Deep learning-powered oil spill detection from Sentinel-1 Synthetic Aperture Radar (SAR) satellite imagery. Upload a GeoTIFF, detect oil spills, measure their area, and identify nearby coastlines at risk.

**[▶ Try the Live Demo on Hugging Face](https://huggingface.co/spaces/TigranBoyakhchyan/GeoSpill-AI)**

---

## Overview

GeoSpill AI uses a U-Net segmentation model with a pretrained ResNet34 encoder to detect oil spills in SAR satellite imagery. Oil spills dampen ocean surface waves, creating characteristically dark regions in radar backscatter that the model learns to distinguish from natural lookalikes (calm water, wind shadows, biogenic slicks).

The project includes a complete pipeline from raw GeoTIFF preprocessing through model training to an interactive web application for detection and analysis.

### Key Features

- **Oil Spill Detection** — U-Net with ResNet34 encoder trained on Sentinel-1 VV+VH dual-polarization SAR data
- **Area Calculation** — Computes spill area in km² using the image's geographic reference system, handling both projected and geographic coordinate systems
- **Proximity Analysis** — Identifies nearby coastlines and countries within a configurable radius, with distance and compass direction from the spill centroid
- **Confidence Heatmap** — Visualizes the model's probability output, showing where it's confident vs. uncertain about oil presence
- **Automatic Format Detection** — Detects whether uploaded SAR data is in linear power scale or decibels and converts automatically
- **Noise Filtering** — Removes small isolated detections using connected component analysis to reduce false positives
- **Geographic Visualization** — Plots detected oil pixels on an interactive map with satellite imagery overlay
- **Export Options** — Download detection masks as GeoTIFF (georeferenced) or PNG, and oil pixel coordinates as CSV

---

## Architecture

| Component | Details |
|---|---|
| Model | U-Net (segmentation_models_pytorch) |
| Encoder | ResNet34, pretrained on ImageNet |
| Input | 2 channels (Sentinel-1 VV + VH bands) |
| Output | 1 channel (binary oil/water mask) |
| Loss | Focal Loss (α=0.5, γ=2.0) + Dice Loss |
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | Cosine annealing with linear warmup (5 epochs) |
| Inference | Sliding window with overlap averaging (256×256 patches, stride 128) |
| Post-processing | Connected component filtering (min 500 pixels) |

---

## Live Demo

The easiest way to try GeoSpill AI is through the hosted demo:

**[https://huggingface.co/spaces/TigranBoyakhchyan/GeoSpill-AI](https://huggingface.co/spaces/TigranBoyakhchyan/GeoSpill-AI)**

### How to use the demo

1. Open the link above
2. Upload a Sentinel-1 GeoTIFF file (2-band SAR image with VV and VH polarization)
3. Adjust the **Detection threshold** in the sidebar if needed (default: 0.75; higher = fewer false positives, lower = more sensitive)
4. Click **"Run Detection"**
5. View results: prediction preview, interactive map, area calculation, and nearby coastlines
6. Download the mask (GeoTIFF or PNG) or coordinates (CSV) using the buttons below the prediction

### Notes on the demo

- The demo runs on CPU, so inference takes 1-3 minutes per image depending on size
- If your SAR data is in linear power scale (values 0-5 instead of -50 to 0 dB), the app detects this automatically and converts to decibels
- The model was trained on SAR imagery in decibel scale with values in the range [-50, 0] dB; images outside this range may produce unreliable results

---

## Local Installation

### Prerequisites

- Python 3.9 or higher
- Git
- A trained model checkpoint (`.pt` file)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/TigranBoyakhchyan/GeoSpill-AI.git
cd GeoSpill-AI
```

2. **Create a virtual environment**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Place your model and stats files**

```
GeoSpill-AI/
├── models/
│   └── best_model_20260416_145208.pt    ← your trained checkpoint
└── data/
    └── train_stats.json                  ← normalization statistics
```

5. **Run the app**

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

### Local usage with GPU

If you have a CUDA-capable GPU, the app will automatically use it for inference. This reduces detection time from 1-3 minutes (CPU) to a few seconds (GPU). No additional configuration is needed — PyTorch detects the GPU automatically.

---

## Project Structure

```
GeoSpill-AI/
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
├── src/
│   ├── preprocess.py         # GeoTIFF → compressed .npz conversion
│   ├── dataset.py            # PyTorch Dataset for training
│   ├── training.py           # Training loop with early stopping
│   ├── inference.py          # Model loading + sliding window prediction
│   └── geo.py                # Geographic utilities (area, proximity, coordinates)
├── models/                   # Trained model checkpoints
├── data/                     # Normalization statistics (train_stats.json)
└── notebooks/                # Kaggle/Colab training notebooks
```

### Module descriptions

**`src/preprocess.py`** — Converts raw Sentinel-1 GeoTIFF images and binary masks into compressed `.npz` format. Computes per-band normalization statistics (mean/std) from the training split only to avoid data leakage. Supports chunked processing for large datasets and resumable execution.

**`src/dataset.py`** — PyTorch Dataset that loads preprocessed `.npz` files and returns `(image, mask)` tensor pairs. Supports random cropping to a configurable patch size and geometric augmentations (flips, 90° rotations) that are safe for SAR data (no intensity augmentations that would corrupt radar backscatter values).

**`src/training.py`** — Full training loop with Focal+Dice loss, AdamW optimizer, cosine annealing with linear warmup, gradient clipping, and per-epoch validation. Includes checkpoint resumption (`--resume` flag) and early stopping (configurable patience). Saves the best model by validation IoU and logs all metrics to CSV.

**`src/inference.py`** — Model loading from checkpoints, sliding window inference with configurable overlap, connected component noise filtering, and a demo mode (backscatter threshold) for testing without a trained model.

**`src/geo.py`** — Geographic utilities including pixel-to-coordinate conversion, spill area calculation (handles both projected and geographic CRS), nearby coastline detection using Natural Earth data, and automatic linear-to-decibel SAR data conversion.

---

## Training

### Data format

The training pipeline expects:
- **Images**: 2-band Sentinel-1 GeoTIFF files (VV + VH polarization) in decibel scale
- **Masks**: Single-band binary GeoTIFF files (0 = water, 1 or 255 = oil spill)
- Image and mask files must have matching filenames

### Preprocessing

```bash
python src/preprocess.py \
    --images_dir path/to/images \
    --masks_dir  path/to/masks \
    --output_dir data/npz_cache \
    --stats_file data/train_stats.json
```

This produces:
- Compressed `.npz` files (float16 images, uint8 masks) in `output_dir`
- `train_stats.json` with normalization constants and train/val/test split paths (70/15/15)

### Training

```bash
python src/training.py \
    --stats_file data/train_stats.json \
    --output_dir results \
    --epochs     50 \
    --batch_size 8 \
    --patch_size 256 \
    --lr         1e-4
```

To resume from a checkpoint:

```bash
python src/training.py \
    --stats_file data/train_stats.json \
    --output_dir results \
    --resume     results/best_model_YYYYMMDD_HHMMSS.pt \
    --epochs     50
```

Training on a single T4 GPU takes approximately 30-50 minutes for 50 epochs with 840 training samples.

### Training on Kaggle

For users without a local GPU, training can be done on Kaggle with free T4 GPU access (30 hours/week):

1. Upload raw `.tif` dataset as a Kaggle Dataset
2. Run preprocessing in a Kaggle notebook (CPU, ~40 minutes)
3. Save preprocessed output as a new Kaggle Dataset
4. Create a GPU notebook, attach the preprocessed dataset, and run training

---

## Configuration

### Detection threshold (default: 0.75)

Controls the probability cutoff for classifying a pixel as oil. Higher values produce fewer false positives but may miss faint spills. Lower values are more sensitive but flag more natural dark patches.

| Threshold | Behavior |
|---|---|
| 0.5 | Sensitive — detects faint spills but more false positives |
| 0.75 | Balanced — recommended default |
| 0.85+ | Conservative — only high-confidence detections |

### Min spill size (default: 500 pixels)

Connected regions smaller than this are removed as noise. Real oil spills form large coherent regions; isolated small dark patches are typically calm water, wind shadows, or sensor artifacts.

### Coast search radius (default: 200 km)

The radius for the proximity analysis that identifies nearby coastlines. Increase for open-ocean spills far from shore.

---

## Limitations

- **False positives on calm water**: The model may flag calm-water regions, wind shadows behind landmasses, and biogenic slicks as oil. Raising the detection threshold and min spill size helps mitigate this.
- **Training data bias**: Model performance depends on the diversity and quality of the training dataset. The model may underperform on SAR imagery from regions or conditions not represented in training.
- **Proximity analysis is static**: The nearby coastlines feature calculates straight-line distance only. It does not model ocean currents, wind-driven drift, or oil weathering. It is not a spill trajectory forecast.
- **Linear-scale data**: The automatic linear-to-dB conversion is heuristic-based. For best results, provide SAR data already calibrated to decibel scale.
- **Single-image processing**: The web application processes one image at a time for stability.

---

## Dependencies

```
streamlit
streamlit-folium
folium
torch
torchvision
segmentation-models-pytorch
rasterio
pyproj
geopandas
shapely
matplotlib
numpy
pillow
scipy
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgments

- [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch) for the U-Net implementation
- [Natural Earth](https://www.naturalearthdata.com/) for the coastline and country boundary data
- [Copernicus / ESA](https://www.copernicus.eu/) for Sentinel-1 SAR satellite data

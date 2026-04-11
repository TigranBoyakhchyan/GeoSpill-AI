"""
src/preprocess.py
-----------------
Computes global normalization statistics (mean/std) from the training set
and saves them to data/train_stats.json.

Run this ONCE before training:
    python src/preprocess.py

Output:
    data/train_stats.json   <- mean and std per band, used by dataset.py
    data/splits.json        <- train/val/test image path lists
"""

import os
import json
import numpy as np
import rasterio
import warnings
import logging

from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
logging.getLogger("rasterio").setLevel(logging.ERROR)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
IMAGES_DIR = "LocalTrainingDataset/images"   # change to FullDataset/images for full run
MASKS_DIR  = "LocalTrainingDataset/masks"
STATS_FILE = "data/train_stats.json"
SPLITS_FILE = "data/splits.json"

DB_CLIP_MIN = -50.0
DB_CLIP_MAX =  0.0


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def find_pairs(images_dir: str, masks_dir: str) -> list:
    """Find all matching (image_path, mask_path) pairs by filename."""
    pairs = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith((".tif", ".tiff")):
            continue
        img_path  = os.path.join(images_dir, fname)
        mask_path = os.path.join(masks_dir, fname)
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    return pairs


def load_image(path: str) -> np.ndarray:
    """Load a SAR GeoTIFF, return float32 array of shape (2, H, W)."""
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
    return img


def compute_mean_std(image_paths: list) -> tuple:
    """
    Compute per-band mean and std across all training images.

    Uses online accumulation so we never load all images into memory.
    Stats are computed AFTER clipping to remove extreme outliers.

    Args:
        image_paths: training image paths ONLY — never val/test (data leakage)

    Returns:
        mean:    np.array (2,)
        std:     np.array (2,)
        skipped: number of corrupted files skipped
    """
    band_sums    = np.zeros(2, dtype=np.float64)
    band_sq_sums = np.zeros(2, dtype=np.float64)
    pixel_count  = 0
    skipped      = 0

    for i, path in enumerate(image_paths):
        print(f"  [{i+1}/{len(image_paths)}] {os.path.basename(path)}", end="\r")
        try:
            img = load_image(path)
            img = np.clip(img, DB_CLIP_MIN, DB_CLIP_MAX)

            n = img.shape[1] * img.shape[2]
            band_sums    += img.reshape(2, -1).sum(axis=1)
            band_sq_sums += (img.reshape(2, -1) ** 2).sum(axis=1)
            pixel_count  += n

        except Exception as e:
            print(f"\n  SKIPPING {os.path.basename(path)}: {e}")
            skipped += 1

    print()  # newline after progress
    if pixel_count == 0:
        raise RuntimeError("No valid images found.")

    mean = band_sums / pixel_count
    std  = np.sqrt(np.maximum(band_sq_sums / pixel_count - mean ** 2, 0))
    return mean, std, skipped


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run_preprocessing():
    print("=" * 60)
    print("SAR Oil Spill — Preprocessing (stats only, no patches)")
    print("=" * 60)

    # ── Find pairs ────────────────────────────────────────────
    pairs = find_pairs(IMAGES_DIR, MASKS_DIR)
    print(f"\nFound {len(pairs)} image-mask pairs in {IMAGES_DIR}")
    if len(pairs) == 0:
        print("ERROR: No pairs found. Check IMAGES_DIR and MASKS_DIR paths.")
        return

    # ── Split at image level (70 / 15 / 15) ──────────────────
    # We save the split to disk so dataset.py uses the exact same split
    # every run — reproducibility is critical
    np.random.seed(42)
    indices = np.random.permutation(len(pairs))
    n_train = int(0.70 * len(pairs))
    n_val   = int(0.15 * len(pairs))

    train_pairs = [pairs[i] for i in indices[:n_train]]
    val_pairs   = [pairs[i] for i in indices[n_train:n_train + n_val]]
    test_pairs  = [pairs[i] for i in indices[n_train + n_val:]]

    print(f"Split: {len(train_pairs)} train | {len(val_pairs)} val | {len(test_pairs)} test")

    # Save splits so dataset.py loads the same images every run
    os.makedirs("data", exist_ok=True)
    splits = {
        "train": [p[0] for p in train_pairs],
        "val":   [p[0] for p in val_pairs],
        "test":  [p[0] for p in test_pairs],
        "masks": {p[0]: p[1] for p in train_pairs + val_pairs + test_pairs},
    }
    with open(SPLITS_FILE, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Splits saved to : {SPLITS_FILE}")

    # ── Compute stats on training images only ─────────────────
    print(f"\nComputing mean/std from {len(train_pairs)} training images...")
    mean, std, skipped = compute_mean_std([p[0] for p in train_pairs])

    print(f"  Band 0 (VV): mean={mean[0]:.4f}, std={std[0]:.4f}")
    print(f"  Band 1 (VH): mean={mean[1]:.4f}, std={std[1]:.4f}")
    if skipped:
        print(f"  Skipped {skipped} corrupted files")

    # ── Save stats ────────────────────────────────────────────
    stats = {
        "mean":        mean.tolist(),
        "std":         std.tolist(),
        "db_clip_min": DB_CLIP_MIN,
        "db_clip_max": DB_CLIP_MAX,
        "n_train":     len(train_pairs),
        "n_val":       len(val_pairs),
        "n_test":      len(test_pairs),
    }
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Stats saved to  : {STATS_FILE}")
    print("\n" + "=" * 60)
    print("Done! No patches generated — dataset.py crops on the fly.")
    print("Next step: python src/train.py")
    print("=" * 60)


if __name__ == "__main__":
    run_preprocessing()
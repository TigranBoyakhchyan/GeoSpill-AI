"""
src/preprocess.py
-----------------
Converts SAR GeoTIFF images and masks to compressed .npz format,
and computes normalization statistics (mean/std) from the training set.

Usage (command line):
    python src/preprocess.py \
        --images_dir  LocalTrainingDataset/images \
        --masks_dir   LocalTrainingDataset/masks \
        --output_dir  data/npz_cache \
        --stats_file  data/train_stats.json

Usage (from Colab notebook):
    import src.preprocess as pre
    pre.run(
        images_dir = '/content/drive/MyDrive/Geo_Spill_Data/images',
        masks_dir  = '/content/drive/MyDrive/Geo_Spill_Data/masks',
        output_dir = '/content/data/npz_cache',
        stats_file = '/content/data/train_stats.json',
    )

Output:
    output_dir/
        images/  <- one .npz per image, key='data', shape (2, H, W), normalized
        masks/   <- one .npz per mask,  key='data', shape (H, W), values 0.0/1.0
    stats_file   <- JSON with mean, std, and train/val/test split paths
"""

import os
import gc
import json
import argparse
import numpy as np
import rasterio
import warnings
import logging
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
logging.getLogger("rasterio").setLevel(logging.ERROR)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
DB_CLIP_MIN = -50.0
DB_CLIP_MAX =  0.0
TRAIN_RATIO =  0.70
VAL_RATIO   =  0.15
# TEST_RATIO  =  0.15  (remainder)

# How often to force garbage collection (every N files)
GC_EVERY = 50


# ─── FILE DISCOVERY ───────────────────────────────────────────────────────────

def find_pairs(images_dir: str, masks_dir: str) -> list:
    """
    Find all matched (image_path, mask_path) pairs by filename.

    Both directories must contain files with matching names.
    Example: images/00001.tif is matched with masks/00001.tif

    Returns:
        list of (image_path, mask_path) tuples, sorted by filename
    """
    pairs = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith((".tif", ".tiff")):
            continue
        img_path  = os.path.join(images_dir, fname)
        mask_path = os.path.join(masks_dir, fname)
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
        else:
            print(f"  WARNING: no mask found for {fname} — skipping")
    return pairs


# ─── LOADING ──────────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """
    Load a 2-band SAR GeoTIFF.

    Returns:
        np.ndarray of shape (2, H, W), dtype float32
    """
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def load_mask(path: str) -> np.ndarray:
    """
    Load a binary mask GeoTIFF.

    Normalizes pixel values to 0.0 (water) or 1.0 (oil spill).
    Handles both 0/255 and 0/1 masks.

    Returns:
        np.ndarray of shape (H, W), dtype float32
    """
    with rasterio.open(path) as src:
        mask = src.read(1).astype(np.float32)
    if mask.max() > 1.0:
        mask = (mask > 127).astype(np.float32)
    return mask


# ─── NORMALIZATION STATS ──────────────────────────────────────────────────────

def compute_mean_std(image_paths: list) -> tuple:
    """
    Compute per-band mean and std across all given images.

    Uses online accumulation so we never load all images into memory at once.
    Stats are computed AFTER clipping to remove extreme outlier values.

    Memory-optimized: uses in-place clip and computes squared sums
    one band at a time to avoid allocating a full copy of the image.

    IMPORTANT: only pass training image paths — never val/test.
    Including val/test would be data leakage.

    Returns:
        mean: np.ndarray of shape (2,), dtype float32
        std:  np.ndarray of shape (2,), dtype float32
    """
    band_sums    = np.zeros(2, dtype=np.float64)
    band_sq_sums = np.zeros(2, dtype=np.float64)
    pixel_count  = 0
    skipped      = 0
    total        = len(image_paths)

    for i, path in enumerate(image_paths):
        print(f"  [{i+1:4d}/{total}] {os.path.basename(path)}", end="\r")
        try:
            img = load_image(path)                 # (2, H, W) float32
            np.clip(img, DB_CLIP_MIN, DB_CLIP_MAX, out=img)  # in-place

            n = img.shape[1] * img.shape[2]
            for b in range(2):
                band_flat = img[b].ravel()         # view, no copy
                band_sums[b]    += band_flat.sum()
                band_sq_sums[b] += np.dot(band_flat, band_flat)  # sum of squares without allocating img**2
            pixel_count += n

            del img, band_flat                     # free immediately
        except Exception as e:
            print(f"\n  SKIPPING {os.path.basename(path)}: {e}")
            skipped += 1

        # Periodic garbage collection to return memory to OS
        if (i + 1) % GC_EVERY == 0:
            gc.collect()

    print()

    if pixel_count == 0:
        raise RuntimeError("No valid images found — cannot compute stats.")

    mean = (band_sums / pixel_count).astype(np.float32)
    std  = np.sqrt(
        np.maximum(band_sq_sums / pixel_count - (band_sums / pixel_count) ** 2, 0)
    ).astype(np.float32)

    if skipped:
        print(f"  Skipped {skipped} files during stats computation")

    return mean, std


# ─── CONVERSION ───────────────────────────────────────────────────────────────

def convert_image(src_path: str,
                  dst_stem: str,
                  mean: np.ndarray,
                  std: np.ndarray) -> bool:
    """
    Load a SAR image, clip, normalize, and save as .npz.

    Memory-optimized:
        • clip and normalize are done in-place (no extra copies)
        • uses np.savez (ZIP_STORED) instead of np.savez_compressed
          (ZIP_DEFLATED) to avoid zlib buffering the entire array in RAM
        • file size is ~2× larger on disk, but that is fine for local SSD

    The file stores one array under key 'data'.
    Load with: np.load('file.npz')['data']

    Returns:
        True on success, False on failure
    """
    try:
        img = load_image(src_path)                                   # (2, H, W)
        np.clip(img, DB_CLIP_MIN, DB_CLIP_MAX, out=img)             # in-place
        img -= mean[:, None, None]                                   # in-place
        img /= (std[:, None, None] + 1e-6)                          # in-place
        np.savez(dst_stem, data=img)                                 # no compression
        del img
        return True
    except Exception as e:
        print(f"\n  SKIPPING {os.path.basename(src_path)}: {e}")
        return False


def convert_mask(src_path: str, dst_stem: str) -> bool:
    """
    Load a binary mask and save as .npz.

    The file stores one array under key 'data'.
    Load with: np.load('file.npz')['data']

    Returns:
        True on success, False on failure
    """
    try:
        mask = load_mask(src_path)
        np.savez(dst_stem, data=mask)                                # no compression
        del mask
        return True
    except Exception as e:
        print(f"\n  SKIPPING {os.path.basename(src_path)}: {e}")
        return False


def _get_ram_mb() -> str:
    """Return current process RSS in MB (Linux only, silent fail)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return f"{int(line.split()[1]) / 1024:.0f} MB"
    except Exception:
        pass
    return "N/A"


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run(images_dir: str,
        masks_dir:  str,
        output_dir: str,
        stats_file: str = "data/train_stats.json") -> dict:
    """
    Full preprocessing pipeline.

    Steps:
        1. Find all matched image-mask pairs
        2. Split into train/val/test at the image level (70/15/15)
        3. Compute mean/std from training images only
        4. Convert all images and masks from .tif to .npz
        5. Save stats and split paths to stats_file

    Args:
        images_dir: directory containing .tif SAR images
        masks_dir:  directory containing .tif binary masks
        output_dir: where .npz files are saved
                    (creates output_dir/images/ and output_dir/masks/)
        stats_file: path where train_stats.json is saved

    Returns:
        dict with stats and split paths (same as what is written to stats_file)
    """
    print("=" * 60)
    print("SAR Oil Spill — Preprocessing")
    print("=" * 60)
    print(f"\nImages dir : {images_dir}")
    print(f"Masks dir  : {masks_dir}")
    print(f"Output dir : {output_dir}")
    print(f"Stats file : {stats_file}")

    # ── 1. Find pairs ─────────────────────────────────────────────
    print("\nScanning dataset...")
    pairs = find_pairs(images_dir, masks_dir)
    if len(pairs) == 0:
        raise RuntimeError(
            "No image-mask pairs found. "
            "Check that images_dir and masks_dir contain matching filenames."
        )
    print(f"Found {len(pairs)} image-mask pairs")

    # ── 2. Split at image level ───────────────────────────────────
    np.random.seed(42)
    indices = np.random.permutation(len(pairs))
    n_train = int(TRAIN_RATIO * len(pairs))
    n_val   = int(VAL_RATIO   * len(pairs))

    train_pairs = [pairs[i] for i in indices[:n_train]]
    val_pairs   = [pairs[i] for i in indices[n_train: n_train + n_val]]
    test_pairs  = [pairs[i] for i in indices[n_train + n_val:]]

    print(f"Split      : {len(train_pairs)} train | "
          f"{len(val_pairs)} val | {len(test_pairs)} test")

    # ── 3. Compute stats from training images only ────────────────
    print(f"\nComputing mean/std from {len(train_pairs)} training images...")
    mean, std = compute_mean_std([p[0] for p in train_pairs])
    print(f"  Band 0 (VV): mean={mean[0]:.4f}, std={std[0]:.4f}")
    print(f"  Band 1 (VH): mean={mean[1]:.4f}, std={std[1]:.4f}")

    # Free any memory held by the stats pass before conversion begins
    gc.collect()
    print(f"  RAM after stats: {_get_ram_mb()}")

    # ── 4. Create output directories ──────────────────────────────
    img_out_dir  = os.path.join(output_dir, "images")
    mask_out_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_out_dir,  exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)
    stats_dir = os.path.dirname(os.path.abspath(stats_file))
    if stats_dir:
        os.makedirs(stats_dir, exist_ok=True)

    # ── 5. Convert all pairs to .npz ─────────────────────────────
    all_pairs = train_pairs + val_pairs + test_pairs
    total     = len(all_pairs)
    ok = skip = already = 0

    print(f"\nConverting {total} image-mask pairs to .npz...")

    for i, (img_path, mask_path) in enumerate(all_pairs):
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        img_stem = os.path.join(img_out_dir,  stem)   # without .npz
        msk_stem = os.path.join(mask_out_dir, stem)   # without .npz

        # Skip if already converted — allows safe resume after interruption
        if (os.path.exists(img_stem  + ".npz") and
                os.path.exists(msk_stem + ".npz")):
            already += 1
            continue

        img_ok  = convert_image(img_path,  img_stem,  mean, std)
        mask_ok = convert_mask(mask_path, msk_stem)

        if img_ok and mask_ok:
            ok += 1
        else:
            skip += 1

        # Periodic GC + progress with RAM usage
        if (i + 1) % GC_EVERY == 0:
            gc.collect()
            print(f"  [{i+1:4d}/{total}] {stem}  (RAM: {_get_ram_mb()})")
        else:
            print(f"  [{i+1:4d}/{total}] {stem}", end="\r")

    print(f"\nConversion : {ok} new, {already} already existed, {skip} failed")
    print(f"  Final RAM: {_get_ram_mb()}")

    # ── 6. Build split path lists ─────────────────────────────────
    def npz_path(tif_path: str, subdir: str) -> str:
        stem = os.path.splitext(os.path.basename(tif_path))[0]
        return os.path.join(output_dir, subdir, stem + ".npz")

    splits = {
        "train": [npz_path(p[0], "images") for p in train_pairs],
        "val":   [npz_path(p[0], "images") for p in val_pairs],
        "test":  [npz_path(p[0], "images") for p in test_pairs],
        "masks": {
            npz_path(p[0], "images"): npz_path(p[1], "masks")
            for p in all_pairs
        },
    }

    # ── 7. Save stats + splits ────────────────────────────────────
    result = {
        "mean":        mean.tolist(),
        "std":         std.tolist(),
        "db_clip_min": DB_CLIP_MIN,
        "db_clip_max": DB_CLIP_MAX,
        "n_train":     len(train_pairs),
        "n_val":       len(val_pairs),
        "n_test":      len(test_pairs),
        "splits":      splits,
    }

    with open(stats_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nStats saved : {stats_file}")
    print("=" * 60)
    print("Done!")
    print(f"  NPZ cache  : {output_dir}/")
    print(f"  Stats file : {stats_file}")
    print("=" * 60)

    return result


# ─── COMMAND LINE INTERFACE ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SAR GeoTIFF dataset to normalized .npz files"
    )
    parser.add_argument(
        "--images_dir", required=True,
        help="Directory containing .tif SAR images"
    )
    parser.add_argument(
        "--masks_dir", required=True,
        help="Directory containing .tif binary masks"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for .npz files"
    )
    parser.add_argument(
        "--stats_file", default="data/train_stats.json",
        help="Path to save train_stats.json (default: data/train_stats.json)"
    )
    args = parser.parse_args()

    run(
        images_dir = args.images_dir,
        masks_dir  = args.masks_dir,
        output_dir = args.output_dir,
        stats_file = args.stats_file,
    )

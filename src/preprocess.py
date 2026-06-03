"""
Converts SAR GeoTIFF images and masks to compressed .npz format,
and computes normalization statistics (mean/std) from the training set.

Storage format:
    images/<stem>.npz  — key 'data', shape (2, H, W), dtype float32, normalized,
                         ZIP_DEFLATED compression
    masks/<stem>.npz   — key 'data', shape (H, W),    dtype uint8,   0 or 1,
                         ZIP_DEFLATED compression

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

Chunked usage (process only a slice of files):
    pre.run(..., start_idx=0,   end_idx=400)
    pre.run(..., start_idx=400, end_idx=800)
    pre.run(..., start_idx=800, end_idx=None)
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

GC_EVERY = 50


# ─── FILE DISCOVERY ───────────────────────────────────────────────────────────

def find_pairs(images_dir: str, masks_dir: str) -> list:
    """Find all matched (image_path, mask_path) pairs by filename."""
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
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def load_mask(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        mask = src.read(1).astype(np.float32)
    if mask.max() > 1.0:
        mask = (mask > 127).astype(np.float32)
    return mask


# ─── NORMALIZATION STATS ──────────────────────────────────────────────────────

def compute_mean_std(image_paths: list) -> tuple:
    """
    Compute per-band mean and std across all given images.

    Memory-optimized: in-place clip, sum-of-squares via np.dot (no img**2 copy),
    and periodic gc.collect() to keep RAM flat across long runs.

    IMPORTANT: only pass training image paths — never val/test.
    """
    band_sums    = np.zeros(2, dtype=np.float64)
    band_sq_sums = np.zeros(2, dtype=np.float64)
    pixel_count  = 0
    skipped      = 0
    total        = len(image_paths)

    for i, path in enumerate(image_paths):
        print(f"  [{i+1:4d}/{total}] {os.path.basename(path)}", end="\r")
        try:
            img = load_image(path)
            np.clip(img, DB_CLIP_MIN, DB_CLIP_MAX, out=img)

            n = img.shape[1] * img.shape[2]
            for b in range(2):
                band_flat = img[b].ravel()
                band_sums[b]    += band_flat.sum()
                band_sq_sums[b] += np.dot(band_flat, band_flat)
            pixel_count += n

            del img, band_flat
        except Exception as e:
            print(f"\n  SKIPPING {os.path.basename(path)}: {e}")
            skipped += 1

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
    Load, clip, normalize, save as compressed .npz (float32, ZIP_DEFLATED).
    """
    try:
        img = load_image(src_path)
        np.clip(img, DB_CLIP_MIN, DB_CLIP_MAX, out=img)
        img -= mean[:, None, None]
        img /= (std[:, None, None] + 1e-6)
        np.savez_compressed(dst_stem, data=img.astype(np.float16))
        del img
        return True
    except Exception as e:
        print(f"\n  SKIPPING {os.path.basename(src_path)}: {e}")
        return False


def convert_mask(src_path: str, dst_stem: str) -> bool:
    """
    Load binary mask, cast to uint8, save as compressed .npz.

    Binary masks compress very well (long runs of identical values).
    """
    try:
        mask = load_mask(src_path)         # float32, values 0.0 / 1.0
        mask = mask.astype(np.uint8)       # 4x smaller before compression
        np.savez_compressed(dst_stem, data=mask)
        del mask
        return True
    except Exception as e:
        print(f"\n  SKIPPING {os.path.basename(src_path)}: {e}")
        return False


def _get_ram_mb() -> str:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return f"{int(line.split()[1]) / 1024:.0f} MB"
    except Exception:
        pass
    return "N/A"


# ─── STATS FILE HELPERS ───────────────────────────────────────────────────────

def _build_splits_and_result(train_pairs, val_pairs, test_pairs,
                             output_dir, mean, std):
    """Build the splits dict and final stats dict — same for all chunks."""
    all_pairs = train_pairs + val_pairs + test_pairs

    def npz_path(tif_path, subdir):
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

    return {
        "mean":        mean.tolist(),
        "std":         std.tolist(),
        "db_clip_min": DB_CLIP_MIN,
        "db_clip_max": DB_CLIP_MAX,
        "n_train":     len(train_pairs),
        "n_val":       len(val_pairs),
        "n_test":      len(test_pairs),
        "splits":      splits,
    }


# ─── MAIN PIPELINE ────────────────────────────────────────────────────────────

def run(images_dir: str,
        masks_dir:  str,
        output_dir: str,
        stats_file: str = "data/train_stats.json",
        start_idx:  int = 0,
        end_idx:    int = None) -> dict:
    """
    Full preprocessing pipeline — optionally restricted to a slice of files.

    File order is deterministic (np.random.seed(42)), so chunked calls
    produce the same output as a single full run.

    Stats are computed on the first call, cached to stats_file, and
    reused by subsequent chunks.
    """
    print("=" * 60)
    print("SAR Oil Spill — Preprocessing")
    if start_idx != 0 or end_idx is not None:
        print(f"  CHUNK MODE: processing indices [{start_idx}, "
              f"{'end' if end_idx is None else end_idx})")
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

    # ── 3. Stats ──────────────────────────────────────────────────
    if os.path.exists(stats_file):
        with open(stats_file) as f:
            cached = json.load(f)
        mean = np.array(cached["mean"], dtype=np.float32)
        std  = np.array(cached["std"],  dtype=np.float32)
        print(f"\nReusing cached stats from {stats_file}")
        print(f"  Band 0 (VV): mean={mean[0]:.4f}, std={std[0]:.4f}")
        print(f"  Band 1 (VH): mean={mean[1]:.4f}, std={std[1]:.4f}")
    else:
        print(f"\nComputing mean/std from {len(train_pairs)} training images...")
        mean, std = compute_mean_std([p[0] for p in train_pairs])
        print(f"  Band 0 (VV): mean={mean[0]:.4f}, std={std[0]:.4f}")
        print(f"  Band 1 (VH): mean={mean[1]:.4f}, std={std[1]:.4f}")

        stats_dir = os.path.dirname(os.path.abspath(stats_file))
        if stats_dir:
            os.makedirs(stats_dir, exist_ok=True)
        result = _build_splits_and_result(train_pairs, val_pairs, test_pairs,
                                          output_dir, mean, std)
        with open(stats_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Stats + splits saved to {stats_file}")

    gc.collect()
    print(f"  RAM after stats: {_get_ram_mb()}")

    # ── 4. Create output directories ──────────────────────────────
    img_out_dir  = os.path.join(output_dir, "images")
    mask_out_dir = os.path.join(output_dir, "masks")
    os.makedirs(img_out_dir,  exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    # ── 5. Slice the conversion range ─────────────────────────────
    all_pairs = train_pairs + val_pairs + test_pairs
    total_all = len(all_pairs)

    if end_idx is None:
        end_idx = total_all
    start_idx = max(0, start_idx)
    end_idx   = min(total_all, end_idx)

    chunk = all_pairs[start_idx:end_idx]
    total = len(chunk)

    if total == 0:
        print(f"\nNo files in range [{start_idx}, {end_idx}) — nothing to do.")
        return _build_splits_and_result(train_pairs, val_pairs, test_pairs,
                                        output_dir, mean, std)

    ok = skip = already = 0
    print(f"\nConverting {total} pairs in range [{start_idx}, {end_idx}) "
          f"out of {total_all} total...")

    for i, (img_path, mask_path) in enumerate(chunk):
        stem     = os.path.splitext(os.path.basename(img_path))[0]
        img_stem = os.path.join(img_out_dir,  stem)
        msk_stem = os.path.join(mask_out_dir, stem)

        global_idx = start_idx + i + 1

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

        if (i + 1) % GC_EVERY == 0:
            gc.collect()
            print(f"  [{global_idx:4d}/{total_all}] {stem}  "
                  f"(RAM: {_get_ram_mb()})")
        else:
            print(f"  [{global_idx:4d}/{total_all}] {stem}", end="\r")

    print(f"\nChunk done : {ok} new, {already} already existed, {skip} failed")
    print(f"  Final RAM: {_get_ram_mb()}")

    # ── 6. Re-save stats file (idempotent) ────────────────────────
    result = _build_splits_and_result(train_pairs, val_pairs, test_pairs,
                                      output_dir, mean, std)
    with open(stats_file, "w") as f:
        json.dump(result, f, indent=2)

    # ── 7. Report overall progress ────────────────────────────────
    done_imgs = len([f for f in os.listdir(img_out_dir)  if f.endswith(".npz")])
    done_msks = len([f for f in os.listdir(mask_out_dir) if f.endswith(".npz")])
    print(f"\nOverall   : {done_imgs}/{total_all} images, "
          f"{done_msks}/{total_all} masks converted")
    print("=" * 60)
    if done_imgs >= total_all and done_msks >= total_all:
        print("All chunks complete! Ready to train.")
    else:
        remaining = total_all - done_imgs
        print(f"~{remaining} files remaining. "
              f"Run run(..., start_idx={end_idx}) next.")
    print("=" * 60)

    return result


# ─── COMMAND LINE INTERFACE ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert SAR GeoTIFF dataset to normalized .npz files"
    )
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--masks_dir",  required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stats_file", default="data/train_stats.json")
    parser.add_argument("--start_idx",  type=int, default=0,
                        help="First index (inclusive) to convert")
    parser.add_argument("--end_idx",    type=int, default=None,
                        help="Last index (exclusive) to convert. Omit for end.")
    args = parser.parse_args()

    run(
        images_dir = args.images_dir,
        masks_dir  = args.masks_dir,
        output_dir = args.output_dir,
        stats_file = args.stats_file,
        start_idx  = args.start_idx,
        end_idx    = args.end_idx,
    )

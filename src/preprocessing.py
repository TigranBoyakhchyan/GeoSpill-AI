"""
src/preprocess.py
-----------------
Handles all preprocessing for the SAR oil spill dataset:
    1. Loading SAR images and masks from GeoTIFF files
    2. Normalizing SAR backscatter values (dB scale, 2 bands)
    3. Computing global mean/std from the training set
    4. Extracting 256x256 patches from full 2048x2048 images
    5. Saving patches to disk for fast training

Run this script ONCE before training:
    python src/preprocess.py
"""

import os
import json
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Adjust these paths if needed
IMAGES_DIR   = "LocalTrainingDataset/Image"   # local subset for now
MASKS_DIR    = "LocalTrainingDataset/Mask"
OUTPUT_DIR   = "data/patches"                  # where patches will be saved
STATS_FILE   = "data/train_stats.json"         # mean/std saved here

PATCH_SIZE   = 256    # width and height of each patch in pixels
STRIDE       = 256    # how many pixels to move between patches
                      # stride = patch_size means NO overlap (good for local training)
                      # stride = 128 means 50% overlap (use on Colab for more data)

DB_CLIP_MIN  = -50.0  # clip dB values below this (removes extreme outliers like -101)
DB_CLIP_MAX  =  0.0   # clip dB values above this (values above 0 dB are noise)

# ─── STEP 1: LOADING ──────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """
    Load a SAR GeoTIFF image and return it as a float32 numpy array.

    Your images have 2 bands (VV and VH polarization).
    rasterio.read() returns shape (bands, height, width) = (2, 2048, 2048).
    We keep this channel-first format because PyTorch also uses (C, H, W).

    Args:
        path: path to the .tif image file

    Returns:
        img: numpy array of shape (2, H, W), dtype float32
    """
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)   # reads ALL bands → (2, 2048, 2048)
    return img


def load_mask(path: str) -> np.ndarray:
    """
    Load a binary mask GeoTIFF and return it as a float32 numpy array.

    Masks are single-band images where:
        white pixels (255) = oil spill
        black pixels (0)   = water / background

    We normalize to binary (0.0 or 1.0) because:
        - PyTorch loss functions expect float values
        - Some masks may use 255 instead of 1 for white

    Args:
        path: path to the .tif mask file

    Returns:
        mask: numpy array of shape (H, W), dtype float32, values 0.0 or 1.0
    """
    with rasterio.open(path) as src:
        mask = src.read(1).astype(np.float32)  # read only band 1 → (2048, 2048)

    # Normalize: if mask uses 255 for white, convert to 1.0
    # The threshold 127 handles any value above "half-white" as oil
    if mask.max() > 1.0:
        mask = (mask > 127).astype(np.float32)

    return mask


# ─── STEP 2: NORMALIZATION ────────────────────────────────────────────────────

def clip_db(img: np.ndarray) -> np.ndarray:
    """
    Clip dB values to a physically meaningful range.

    Your EDA showed values ranging from -101 to +5.5 dB.
    In reality, Sentinel-1 SAR rarely goes below -50 dB over ocean.
    Values like -101 are sensor artifacts or areas with no valid return.

    Clipping to [-50, 0]:
        - Removes extreme outliers that would skew normalization
        - Keeps the full dynamic range of real backscatter
        - Oil spills typically appear around -20 to -30 dB
        - Open water typically appears around -15 to -20 dB

    Args:
        img: array of shape (2, H, W) with raw dB values

    Returns:
        img: clipped array, same shape
    """
    return np.clip(img, DB_CLIP_MIN, DB_CLIP_MAX)


def compute_mean_std(image_paths: list) -> tuple:
    """
    Compute the global mean and std of the TRAINING SET only.

    Why training set only?
        If you include validation/test images in the stats, information
        from those images leaks into your normalization — this is called
        data leakage and will give you falsely optimistic results.

    Why global (not per-image)?
        Per-image normalization makes every image look the same regardless
        of actual backscatter intensity. Global stats preserve the relative
        differences between images (e.g., calm sea vs. rough sea).

    We compute stats per band because VV and VH have different
    backscatter characteristics and different value distributions.

    Args:
        image_paths: list of file paths to TRAINING images only

    Returns:
        mean: numpy array of shape (2,) — one mean per band
        std:  numpy array of shape (2,) — one std per band
    """
    print(f"  Computing mean/std from {len(image_paths)} training images...")

    # Accumulators — we use Welford's online algorithm to avoid
    # loading all images into memory at once
    band_sums    = np.zeros(2, dtype=np.float64)
    band_sq_sums = np.zeros(2, dtype=np.float64)
    pixel_count  = 0

    for path in image_paths:
        img = load_image(path)          # (2, 2048, 2048)
        img = clip_db(img)              # clip outliers first

        # img.shape[1] * img.shape[2] = number of pixels per band
        n = img.shape[1] * img.shape[2]
        band_sums    += img.reshape(2, -1).sum(axis=1)
        band_sq_sums += (img.reshape(2, -1) ** 2).sum(axis=1)
        pixel_count  += n

    mean = band_sums / pixel_count
    # var = E[x²] - E[x]²
    std  = np.sqrt(band_sq_sums / pixel_count - mean ** 2)

    print(f"  Band 0 (VV): mean={mean[0]:.4f}, std={std[0]:.4f}")
    print(f"  Band 1 (VH): mean={mean[1]:.4f}, std={std[1]:.4f}")

    return mean, std


def normalize(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization to a 2-band SAR image.

    Z-score normalization: x_norm = (x - mean) / std

    After this, each band will have approximately mean=0 and std=1.
    This is important because:
        - Neural networks train much faster with zero-centered inputs
        - Prevents one band from dominating due to scale differences
        - Works well with batch normalization layers inside the model

    mean and std are shape (2,) so we expand them to (2, 1, 1)
    to broadcast correctly over the (2, H, W) image.

    Args:
        img:  array of shape (2, H, W), already clipped
        mean: array of shape (2,)
        std:  array of shape (2,)

    Returns:
        normalized array of shape (2, H, W)
    """
    mean = mean[:, None, None]   # (2,) → (2, 1, 1) for broadcasting
    std  = std[:, None, None]
    return (img - mean) / (std + 1e-6)   # 1e-6 prevents division by zero


# ─── STEP 3: PATCH EXTRACTION ─────────────────────────────────────────────────

def extract_patches(img: np.ndarray, mask: np.ndarray,
                    patch_size: int = 256, stride: int = 256) -> tuple:
    """
    Slice a full 2048x2048 image into smaller patches for training.

    Why patches?
        A 2048x2048 image with 2 bands takes ~64MB of GPU memory.
        With a batch size of 8, that's 512MB just for the images,
        before the model even runs. Patches let you control memory usage
        precisely and increase the effective dataset size.

    How it works:
        We slide a 256x256 window across the image with a given stride.
        - stride = 256 (no overlap): each pixel appears in exactly 1 patch
          → faster, fewer patches, good for local training
        - stride = 128 (50% overlap): each pixel appears in ~4 patches
          → more patches, better coverage, use on Colab

    Patch filtering (handling class imbalance):
        With only ~4% oil coverage, most patches will be pure water.
        Training on too many empty patches teaches the model to always
        predict water. We handle this by:
        - Always keeping patches that contain ANY oil pixel
        - Randomly keeping only 20% of pure-water patches

    Args:
        img:        normalized image, shape (2, H, W)
        mask:       binary mask, shape (H, W)
        patch_size: size of each square patch
        stride:     step between patches

    Returns:
        patches_img:  list of arrays, each shape (2, patch_size, patch_size)
        patches_mask: list of arrays, each shape (patch_size, patch_size)
    """
    _, H, W = img.shape
    patches_img  = []
    patches_mask = []

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            p_img  = img[:, y:y+patch_size, x:x+patch_size]   # (2, 256, 256)
            p_mask = mask[y:y+patch_size, x:x+patch_size]     # (256, 256)

            has_oil = p_mask.sum() > 0

            if has_oil:
                # Always keep patches with oil
                patches_img.append(p_img)
                patches_mask.append(p_mask)
            elif np.random.rand() < 0.2:
                # Keep only 20% of pure-water patches
                # This reduces imbalance without throwing away all context
                patches_img.append(p_img)
                patches_mask.append(p_mask)

    return patches_img, patches_mask


# ─── STEP 4: SAVE PATCHES TO DISK ─────────────────────────────────────────────

def save_patches(patches_img: list, patches_mask: list,
                 output_dir: str, split: str, start_idx: int = 0) -> int:
    """
    Save extracted patches to disk as .npy files.

    Why .npy instead of .tif?
        - Much faster to load during training (no rasterio overhead)
        - Already normalized — no processing needed at load time
        - Smaller file size for float32 arrays

    Naming convention: {split}_{index:05d}_img.npy / _mask.npy
    Example: train_00042_img.npy, train_00042_mask.npy

    Args:
        patches_img:  list of image patches
        patches_mask: list of mask patches
        output_dir:   root folder to save into
        split:        "train", "val", or "test"
        start_idx:    offset for patch index (for multi-image processing)

    Returns:
        number of patches saved
    """
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    for i, (p_img, p_mask) in enumerate(zip(patches_img, patches_mask)):
        idx = start_idx + i
        np.save(os.path.join(split_dir, f"{idx:05d}_img.npy"),  p_img)
        np.save(os.path.join(split_dir, f"{idx:05d}_mask.npy"), p_mask)

    return len(patches_img)


# ─── STEP 5: MAIN PIPELINE ────────────────────────────────────────────────────

def find_pairs(images_dir: str, masks_dir: str) -> list:
    """
    Find all matching (image, mask) file path pairs.

    Matches by filename — image '00001.tif' is paired with mask '00001.tif'.
    """
    pairs = []
    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith((".tif", ".tiff")):
            continue
        img_path  = os.path.join(images_dir, fname)
        mask_path = os.path.join(masks_dir, fname)
        if os.path.exists(mask_path):
            pairs.append((img_path, mask_path))
    return pairs


def run_preprocessing():
    """
    Full preprocessing pipeline — run this once before training.

    Steps:
        1. Find all image-mask pairs
        2. Split into train / val / test at the IMAGE level
        3. Compute mean/std from training images only
        4. Save stats to disk (needed by dataset.py and inference.py)
        5. Extract and save patches for all splits
    """
    print("=" * 60)
    print("SAR Oil Spill — Preprocessing Pipeline")
    print("=" * 60)

    # ── 1. Find pairs ──────────────────────────────────────────
    pairs = find_pairs(IMAGES_DIR, MASKS_DIR)
    print(f"\nFound {len(pairs)} image-mask pairs")

    if len(pairs) == 0:
        print("ERROR: No pairs found. Check IMAGES_DIR and MASKS_DIR paths.")
        return

    # ── 2. Split at IMAGE level ────────────────────────────────
    # CRITICAL: always split by image, never by patch.
    # If patches from the same image end up in both train and val,
    # the model will memorize those images instead of generalizing.
    # This is called data leakage.
    np.random.seed(42)   # fixed seed for reproducibility
    indices = np.random.permutation(len(pairs))

    # 70% train, 15% val, 15% test
    n_train = int(0.70 * len(pairs))
    n_val   = int(0.15 * len(pairs))

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs   = [pairs[i] for i in val_idx]
    test_pairs  = [pairs[i] for i in test_idx]

    print(f"\nSplit: {len(train_pairs)} train | {len(val_pairs)} val | {len(test_pairs)} test")

    # ── 3. Compute and save stats from training set only ───────
    print("\nComputing normalization statistics...")
    train_image_paths = [p[0] for p in train_pairs]
    mean, std = compute_mean_std(train_image_paths)

    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    stats = {"mean": mean.tolist(), "std": std.tolist()}
    with open(STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to: {STATS_FILE}")
    print("  (Use these SAME stats on Colab — do not recompute on full dataset)")

    # ── 4. Extract and save patches for each split ─────────────
    print(f"\nExtracting patches (size={PATCH_SIZE}, stride={STRIDE})...")

    for split_name, split_pairs in [("train", train_pairs),
                                     ("val",   val_pairs),
                                     ("test",  test_pairs)]:
        total_patches = 0
        total_oil     = 0

        for img_path, mask_path in split_pairs:
            img  = load_image(img_path)          # (2, 2048, 2048)
            img  = clip_db(img)                  # clip outliers
            img  = normalize(img, mean, std)     # z-score normalize
            mask = load_mask(mask_path)          # (2048, 2048)

            patches_img, patches_mask = extract_patches(
                img, mask, PATCH_SIZE, STRIDE
            )

            # Count oil patches for reporting
            oil_count = sum(m.sum() > 0 for m in patches_mask)
            total_oil += oil_count

            saved = save_patches(patches_img, patches_mask,
                                 OUTPUT_DIR, split_name, total_patches)
            total_patches += saved

        print(f"  {split_name:<6}: {total_patches} patches "
              f"({total_oil} with oil, "
              f"{total_patches - total_oil} water-only)")

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Patches saved to: {OUTPUT_DIR}/")
    print("Next step: write src/dataset.py to load these patches")
    print("=" * 60)


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_preprocessing()
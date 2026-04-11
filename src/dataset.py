"""
src/dataset.py
--------------
PyTorch Dataset that loads raw SAR GeoTIFF files and crops patches
on-the-fly during training. No pre-extracted patches needed.

Each call to __getitem__:
    1. Loads the full 2048x2048 SAR image and mask from disk
    2. Clips and normalizes the image
    3. Applies a random 256x256 crop + augmentations
    4. Returns a (image, mask) tensor pair

This uses more CPU per batch than loading pre-saved patches, but:
    - Zero extra disk space (raw files stay as-is)
    - Each epoch sees different random crops -> better generalization
    - Works directly from Google Drive on Colab

Usage:
    from src.dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import rasterio
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
import logging

from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
logging.getLogger("rasterio").setLevel(logging.ERROR)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
STATS_FILE  = "data/train_stats.json"
SPLITS_FILE = "data/splits.json"

PATCH_SIZE  = 256
BATCH_SIZE  = 8
NUM_WORKERS = 0      # keep 0 on Windows; set 4 on Colab (Linux)
PIN_MEMORY  = False  # set True when training on GPU


# ─── DATASET CLASS ────────────────────────────────────────────────────────────

class OilSpillDataset(Dataset):
    """
    Loads full SAR images from disk and crops random patches on-the-fly.

    Why on-the-fly instead of pre-saved patches?
        Pre-saving 1200 images worth of patches at stride=128 produces
        ~135GB of data. On-the-fly cropping uses zero extra disk space
        because we just read the original files and crop in memory.

    Each image is used multiple times per epoch via random crops.
    The number of crops per image is controlled by crops_per_image.

    Args:
        image_paths:     list of full paths to SAR .tif images
        mask_paths:      corresponding list of mask .tif paths
        mean:            per-band mean for normalization, shape (2,)
        std:             per-band std for normalization, shape (2,)
        transform:       albumentations pipeline (includes RandomCrop)
        crops_per_image: how many random crops to take per image per epoch
                         (higher = more samples but slower epoch)
    """

    def __init__(self,
                 image_paths:     list,
                 mask_paths:      list,
                 mean:            np.ndarray,
                 std:             np.ndarray,
                 transform=None,
                 crops_per_image: int = 5):

        self.image_paths     = image_paths
        self.mask_paths      = mask_paths
        self.mean            = mean[:, None, None].astype(np.float32)  # (2,1,1)
        self.std             = std[:, None, None].astype(np.float32)
        self.transform       = transform
        self.crops_per_image = crops_per_image

        # Each image appears crops_per_image times in the index
        # index 0..crops_per_image-1 all map to image 0
        # index crops_per_image..2*crops_per_image-1 map to image 1, etc.
        self.total = len(image_paths) * crops_per_image

        print(f"  Dataset: {len(image_paths)} images × "
              f"{crops_per_image} crops = {self.total} samples per epoch")

    def __len__(self) -> int:
        return self.total

    def _load_and_normalize(self, img_path: str) -> np.ndarray:
        """Load a SAR image, clip dB values, and z-score normalize."""
        with rasterio.open(img_path) as src:
            img = src.read().astype(np.float32)   # (2, H, W)
        img = np.clip(img, -50.0, 0.0)
        img = (img - self.mean) / (self.std + 1e-6)
        return img

    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load binary mask, normalize to 0.0/1.0."""
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)  # (H, W)
        if mask.max() > 1.0:
            mask = (mask > 127).astype(np.float32)
        return mask

    def __getitem__(self, idx: int) -> tuple:
        """
        Load one (image, mask) crop.

        Maps flat index → image index via integer division,
        then applies a random crop (different every call due to
        albumentations' internal randomness).
        """
        # Map flat idx to image idx
        img_idx = idx % len(self.image_paths)

        try:
            img  = self._load_and_normalize(self.image_paths[img_idx])
            mask = self._load_mask(self.mask_paths[img_idx])
        except Exception as e:
            # If a file is corrupted, return a zero patch and continue
            img  = np.zeros((2, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
            print(f"\n  Warning: skipping corrupted file "
                  f"{os.path.basename(self.image_paths[img_idx])}: {e}")
            return torch.from_numpy(img), torch.zeros(1, PATCH_SIZE, PATCH_SIZE)

        if self.transform:
            # Albumentations expects (H, W, C) — transpose from (C, H, W)
            img_hwc   = img.transpose(1, 2, 0).astype(np.float32)
            augmented = self.transform(image=img_hwc, mask=mask)
            img  = augmented["image"]        # tensor (2, H, W) via ToTensorV2
            mask = augmented["mask"]         # tensor (H, W)
        else:
            img  = torch.from_numpy(img)
            mask = torch.from_numpy(mask)

        # Add channel dim to mask: (H, W) -> (1, H, W)
        mask = mask.unsqueeze(0)
        return img, mask


# ─── AUGMENTATION PIPELINES ───────────────────────────────────────────────────

def get_train_transform(patch_size: int = PATCH_SIZE) -> A.Compose:
    """
    Training augmentation with random crop.

    RandomCrop is the key addition vs the old patch-based approach.
    It picks a random 256x256 region from the full 2048x2048 image,
    so each call to __getitem__ sees a different region.
    """
    return A.Compose([
        # Random crop — this replaces pre-extracted patches
        A.RandomCrop(patch_size, patch_size),

        # Geometric augmentations — valid for SAR (no preferred orientation)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # Noise — simulates SAR speckle variation
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GaussNoise(noise_scale_factor=0.02, p=0.3),

        ToTensorV2(),
    ])


def get_val_transform(patch_size: int = PATCH_SIZE) -> A.Compose:
    """
    Validation/test: fixed center crop, no augmentation.

    CenterCrop instead of RandomCrop for reproducible evaluation —
    the same region is evaluated every epoch so val metrics are comparable.
    """
    return A.Compose([
        A.CenterCrop(patch_size, patch_size),
        ToTensorV2(),
    ])


# ─── DATALOADER FACTORY ───────────────────────────────────────────────────────

def get_dataloaders(
    stats_file:      str  = STATS_FILE,
    splits_file:     str  = SPLITS_FILE,
    batch_size:      int  = BATCH_SIZE,
    num_workers:     int  = NUM_WORKERS,
    pin_memory:      bool = PIN_MEMORY,
    crops_per_image: int  = 5,
) -> tuple:
    """
    Build train, val, test DataLoaders from splits.json and train_stats.json.

    crops_per_image controls how many random crops per image per epoch:
        - 5  → 46 images × 5 = 230 samples locally (fast, good for testing)
        - 10 → 1200 images × 10 = 12,000 samples on Colab (good for training)

    Args:
        stats_file:      path to train_stats.json
        splits_file:     path to splits.json
        batch_size:      samples per batch
        num_workers:     parallel loading workers (0 on Windows)
        pin_memory:      faster GPU transfer (True when using CUDA)
        crops_per_image: random crops per image per epoch

    Returns:
        train_loader, val_loader, test_loader
    """
    # Load normalization stats
    if not os.path.exists(stats_file):
        raise FileNotFoundError(
            f"Stats file not found: {stats_file}\n"
            "Run src/preprocess.py first."
        )
    with open(stats_file) as f:
        stats = json.load(f)
    mean = np.array(stats["mean"], dtype=np.float32)
    std  = np.array(stats["std"],  dtype=np.float32)

    # Load splits
    if not os.path.exists(splits_file):
        raise FileNotFoundError(
            f"Splits file not found: {splits_file}\n"
            "Run src/preprocess.py first."
        )
    with open(splits_file) as f:
        splits = json.load(f)

    mask_map = splits["masks"]   # image_path -> mask_path

    print("\nInitializing datasets...")
    loaders = {}
    for split in ["train", "val", "test"]:
        img_paths  = splits[split]
        mask_paths = [mask_map[p] for p in img_paths]

        transform = (get_train_transform() if split == "train"
                     else get_val_transform())

        # Val/test: 1 crop per image is enough for evaluation
        n_crops = crops_per_image if split == "train" else 1

        dataset = OilSpillDataset(
            image_paths     = img_paths,
            mask_paths      = mask_paths,
            mean            = mean,
            std             = std,
            transform       = transform,
            crops_per_image = n_crops,
        )

        loaders[split] = DataLoader(
            dataset,
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = (split == "train"),
        )

    print(f"\n  Batch size  : {batch_size}")
    print(f"  num_workers : {num_workers}")
    print(f"  Train batches : {len(loaders['train'])}")
    print(f"  Val batches   : {len(loaders['val'])}")
    print(f"  Test batches  : {len(loaders['test'])}")

    return loaders["train"], loaders["val"], loaders["test"]


# ─── SANITY CHECK ─────────────────────────────────────────────────────────────

def run_sanity_check():
    """
    Verify the dataset loads correctly and tensors have the right shape.
    Run with: python src/dataset.py
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Dataset Sanity Check")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)

    images, masks = next(iter(train_loader))

    print(f"\nBatch shapes:")
    print(f"  images : {images.shape}  (batch, bands, H, W)")
    print(f"  masks  : {masks.shape}   (batch, 1, H, W)")
    print(f"\nImage stats (after normalization):")
    print(f"  min  : {images.min():.4f}")
    print(f"  max  : {images.max():.4f}")
    print(f"  mean : {images.mean():.4f}  (should be close to 0)")
    print(f"  std  : {images.std():.4f}   (should be close to 1)")
    print(f"\nMask unique values: {masks.unique().tolist()}  (should be [0.0, 1.0])")
    print(f"Oil coverage: {100 * masks.mean().item():.2f}%")

    # Save a quick visual
    os.makedirs("eda_output", exist_ok=True)
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle("Dataset Sanity Check — On-the-fly Crops", fontsize=13)
    for i in range(min(4, images.shape[0])):
        axes[i][0].imshow(images[i][0].numpy(), cmap="gray")
        axes[i][0].set_title(f"VV band (sample {i+1})", fontsize=8)
        axes[i][0].axis("off")
        axes[i][1].imshow(images[i][1].numpy(), cmap="gray")
        axes[i][1].set_title(f"VH band (sample {i+1})", fontsize=8)
        axes[i][1].axis("off")
        axes[i][2].imshow(masks[i][0].numpy(), cmap="gray", vmin=0, vmax=1)
        oil_pct = 100 * masks[i].mean().item()
        axes[i][2].set_title(f"Mask ({oil_pct:.1f}% oil)", fontsize=8)
        axes[i][2].axis("off")

    plt.tight_layout()
    out = "eda_output/dataset_sanity_check.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSanity check plot saved: {out}")
    print("\nDataset working correctly.")


if __name__ == "__main__":
    run_sanity_check()
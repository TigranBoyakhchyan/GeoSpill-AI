"""
src/dataset.py
--------------
PyTorch Dataset and DataLoader for the SAR oil spill segmentation task.

This file:
    1. Defines OilSpillDataset — loads pre-processed .npy patches from disk
    2. Applies data augmentation during training
    3. Provides a get_dataloaders() helper that returns ready-to-use loaders

Assumes preprocess.py has already been run and patches exist at:
    data/patches/train/
    data/patches/val/
    data/patches/test/

Usage:
    from src.dataset import get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders()
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─── CONFIG ───────────────────────────────────────────────────────────────────
PATCHES_DIR = "data/patches"     # root folder containing train/ val/ test/
STATS_FILE  = "data/train_stats.json"

PATCH_SIZE  = 256
BATCH_SIZE  = 8     # reduce to 4 if you run out of memory locally
NUM_WORKERS = 0     # IMPORTANT: keep 0 on Windows to avoid DataLoader errors
                    # change to 4 on Linux / Google Colab for faster loading
PIN_MEMORY  = False # set True only when training on GPU


# ─── STEP 1: DATASET CLASS ────────────────────────────────────────────────────

class OilSpillDataset(Dataset):
    """
    PyTorch Dataset that loads pre-processed SAR patches from disk.

    Each sample is a pair of:
        image: float32 tensor of shape (2, 256, 256)  — 2 bands (VV, VH)
        mask:  float32 tensor of shape (1, 256, 256)  — binary oil mask

    The patches were already normalized in preprocess.py so we only
    need to load them and optionally apply augmentation here.

    Args:
        split:     one of "train", "val", "test"
        transform: albumentations transform pipeline (None for val/test)
    """

    def __init__(self, split: str, transform=None):
        self.split     = split
        self.transform = transform
        self.patch_dir = os.path.join(PATCHES_DIR, split)

        if not os.path.exists(self.patch_dir):
            raise FileNotFoundError(
                f"Patch directory not found: {self.patch_dir}\n"
                f"Did you run src/preprocess.py first?"
            )

        # Collect all image patch indices by scanning for *_img.npy files.
        # We store just the indices (e.g. 0, 1, 2 ...) and build paths on the fly.
        # This avoids loading everything into RAM at once.
        self.indices = sorted([
            int(f.replace("_img.npy", ""))
            for f in os.listdir(self.patch_dir)
            if f.endswith("_img.npy")
        ])

        if len(self.indices) == 0:
            raise RuntimeError(f"No patches found in {self.patch_dir}")

        print(f"  [{split}] Loaded {len(self.indices)} patches from {self.patch_dir}")

    def __len__(self) -> int:
        # tells PyTorch how many samples are in this dataset
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple:
        """
        Load one (image, mask) pair by index.

        __getitem__ is called by the DataLoader for each sample in a batch.
        It must return tensors, not numpy arrays.

        Flow:
            1. Load .npy files from disk
            2. Optionally apply augmentation
            3. Convert to PyTorch tensors
            4. Add channel dim to mask: (H, W) → (1, H, W)

        Args:
            idx: position in self.indices list (not the patch filename number)

        Returns:
            image: tensor (2, 256, 256)
            mask:  tensor (1, 256, 256)
        """
        patch_idx = self.indices[idx]

        # Load pre-processed numpy arrays from disk
        img  = np.load(os.path.join(self.patch_dir, f"{patch_idx:05d}_img.npy"))   # (2, H, W)
        mask = np.load(os.path.join(self.patch_dir, f"{patch_idx:05d}_mask.npy"))  # (H, W)

        if self.transform:
            # Albumentations expects images in (H, W, C) format — opposite of PyTorch.
            # We temporarily transpose from (2, H, W) → (H, W, 2) for augmentation,
            # then ToTensorV2 converts back to (2, H, W) automatically.
            img_hwc = img.transpose(1, 2, 0).astype(np.float32)   # (H, W, 2), ensure float32

            augmented = self.transform(image=img_hwc, mask=mask)

            img  = augmented["image"]   # ToTensorV2 → tensor (2, H, W)
            mask = augmented["mask"]    # still (H, W) — we add channel dim below
        else:
            # No augmentation — just convert to tensors manually
            img  = torch.from_numpy(img)           # (2, H, W)
            mask = torch.from_numpy(mask)          # (H, W)

        # Add channel dimension to mask: (H, W) → (1, H, W)
        # This matches the model output shape and loss function expectations
        mask = mask.unsqueeze(0)

        return img, mask


# ─── STEP 2: AUGMENTATION PIPELINES ──────────────────────────────────────────

def get_train_transform() -> A.Compose:
    """
    Augmentation pipeline for TRAINING only.

    Goal: artificially increase dataset diversity so the model generalizes
    better and doesn't memorize the 46 training images.

    Rules for SAR augmentation:
        ✅ Geometric transforms (flip, rotate) — physically valid for SAR
        ✅ Slight blur — simulates speckle variation
        ✅ Gaussian noise — simulates sensor noise
        ❌ Color jitter / hue / saturation — SAR has no color, meaningless
        ❌ CLAHE / brightness — changes physical backscatter meaning

    Every transform is applied with a probability (p=...).
    The same transform is applied to BOTH image and mask automatically
    by albumentations — so the mask stays aligned with the image.
    """
    return A.Compose([
        # Random horizontal flip — oil spills have no preferred orientation
        A.HorizontalFlip(p=0.5),

        # Random vertical flip — same reasoning
        A.VerticalFlip(p=0.5),

        # Random 90° rotation — SAR geometry is rotation-invariant
        A.RandomRotate90(p=0.5),

        # Slight Gaussian blur — simulates speckle smoothing variation
        # blur_limit controls kernel size range (must be odd numbers)
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),

        # Gaussian noise — simulates SAR thermal noise variation
        A.GaussNoise(noise_scale_factor=0.02, p=0.3),

        # Convert numpy (H, W, C) → PyTorch tensor (C, H, W)
        # Also converts dtype to float32 if not already
        ToTensorV2(),
    ])


def get_val_transform() -> A.Compose:
    """
    Transform pipeline for VALIDATION and TEST sets.

    No augmentation — we want to evaluate on clean, unmodified patches
    so metrics reflect true model performance.

    We still need ToTensorV2 to convert numpy → tensor.
    """
    return A.Compose([
        ToTensorV2(),
    ])


# ─── STEP 3: DATALOADER FACTORY ───────────────────────────────────────────────

def get_dataloaders(
    batch_size:  int  = BATCH_SIZE,
    num_workers: int  = NUM_WORKERS,
    pin_memory:  bool = PIN_MEMORY,
) -> tuple:
    """
    Create and return train, val, and test DataLoaders.

    The DataLoader wraps the Dataset and handles:
        - Batching: groups samples into batches of size batch_size
        - Shuffling: randomizes order each epoch (train only)
        - Parallel loading: uses multiple CPU workers (num_workers)
        - GPU transfer: pin_memory speeds up CPU→GPU data transfer

    Args:
        batch_size:  number of patches per batch
        num_workers: parallel loading workers (0 = main process, safe on Windows)
        pin_memory:  faster GPU transfer (True only when using CUDA)

    Returns:
        train_loader, val_loader, test_loader
    """
    print("\nInitializing datasets...")

    train_dataset = OilSpillDataset(split="train", transform=get_train_transform())
    val_dataset   = OilSpillDataset(split="val",   transform=get_val_transform())
    test_dataset  = OilSpillDataset(split="test",  transform=get_val_transform())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # shuffle every epoch so batches are different each time
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,     # drop the last incomplete batch to avoid batch norm issues
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # no shuffle for val/test — order doesn't matter for evaluation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,    # keep all val/test samples for accurate metrics
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    print(f"\n  Batch size  : {batch_size}")
    print(f"  num_workers : {num_workers}")
    print(f"  pin_memory  : {pin_memory}")
    print(f"\n  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    return train_loader, val_loader, test_loader


# ─── STEP 4: SANITY CHECK ─────────────────────────────────────────────────────

def run_sanity_check():
    """
    Quick sanity check — run this after preprocess.py to verify everything works.

    Checks:
        - Patches load without errors
        - Tensor shapes are correct
        - Value ranges look reasonable after normalization
        - Mask contains only 0s and 1s

    Run with:
        python src/dataset.py
    """
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("Dataset Sanity Check")
    print("=" * 60)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)

    # Grab one batch from train loader
    images, masks = next(iter(train_loader))

    print(f"\nBatch shapes:")
    print(f"  images : {images.shape}  (batch, bands, H, W)")
    print(f"  masks  : {masks.shape}   (batch, 1, H, W)")
    print(f"\nImage stats (after normalization):")
    print(f"  min  : {images.min():.4f}")
    print(f"  max  : {images.max():.4f}")
    print(f"  mean : {images.mean():.4f}  (should be close to 0)")
    print(f"  std  : {images.std():.4f}   (should be close to 1)")
    print(f"\nMask stats:")
    print(f"  unique values : {masks.unique().tolist()}  (should be [0.0, 1.0])")
    print(f"  oil pixels    : {masks.sum().item():.0f} / {masks.numel()}")
    print(f"  oil coverage  : {100 * masks.mean().item():.2f}%")

    # Visual check — plot 4 samples from the batch
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle("Sanity Check — Training Batch Samples", fontsize=14, fontweight="bold")

    for i in range(4):
        img  = images[i]   # (2, H, W)
        mask = masks[i]    # (1, H, W)

        vv_band = img[0].numpy()   # VV polarization
        vh_band = img[1].numpy()   # VH polarization
        msk     = mask[0].numpy()  # binary mask

        axes[i][0].imshow(vv_band, cmap="gray")
        axes[i][0].set_title(f"Sample {i+1} — VV band", fontsize=9)
        axes[i][0].axis("off")

        axes[i][1].imshow(vh_band, cmap="gray")
        axes[i][1].set_title(f"Sample {i+1} — VH band", fontsize=9)
        axes[i][1].axis("off")

        axes[i][2].imshow(msk, cmap="gray", vmin=0, vmax=1)
        oil_pct = 100 * msk.mean()
        axes[i][2].set_title(f"Sample {i+1} — Mask ({oil_pct:.1f}% oil)", fontsize=9)
        axes[i][2].axis("off")

    plt.tight_layout()
    os.makedirs("eda_output", exist_ok=True)
    out = "eda_output/dataset_sanity_check.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSanity check plot saved: {out}")
    print("\nDataset looks good — ready to build the model.")


if __name__ == "__main__":
    run_sanity_check()
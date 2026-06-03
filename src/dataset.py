"""
src/dataset.py
--------------
PyTorch Dataset for SAR oil spill detection.

Loads preprocessed .npz files produced by preprocess.py and returns
(image_tensor, mask_tensor) pairs ready for DataLoader consumption.

Expected .npz layout (produced by preprocess.py):
    images/<stem>.npz  — key 'data', shape (2, H, W), dtype float32, normalized
    masks/<stem>.npz   — key 'data', shape (H, W),    dtype uint8,   0 or 1
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random


class SARDataset(Dataset):
    """
    PyTorch Dataset for SAR oil spill segmentation.

    Each sample is a (image, mask) pair where:
        image : FloatTensor of shape (2, H, W) — VV and VH bands, normalized
        mask  : FloatTensor of shape (1, H, W) — 0.0 = water, 1.0 = oil spill

    Mask arrays are stored on disk as uint8 (to save storage) but returned
    as float32 tensors so they can be directly passed to BCEWithLogitsLoss.
    """

    def __init__(self,
                 image_paths: list,
                 mask_lookup:  dict,
                 augment:      bool = False,
                 patch_size:   int  = None):

        missing = [p for p in image_paths if p not in mask_lookup]
        if missing:
            raise KeyError(
                f"{len(missing)} image path(s) have no entry in mask_lookup.\n"
                f"First missing: {missing[0]}"
            )

        self.image_paths = image_paths
        self.mask_lookup  = mask_lookup
        self.augment      = augment
        self.patch_size   = patch_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __repr__(self) -> str:
        return (f"SARDataset(n={len(self)}, augment={self.augment}, "
                f"patch_size={self.patch_size})")

    def _load_npz(self, path: str) -> np.ndarray:
        """Load the 'data' array from a compressed .npz file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"NPZ file not found: {path}")
        return np.load(path)["data"]

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            image : FloatTensor (2, H, W)
            mask  : FloatTensor (1, H, W)
        """
        img_path  = self.image_paths[idx]
        mask_path = self.mask_lookup[img_path]

        image = self._load_npz(img_path).astype(np.float32, copy=False)
        mask  = self._load_npz(mask_path).astype(np.float32, copy=False)

        # Cast mask to float32 for BCEWithLogitsLoss / FocalDiceLoss
        mask = mask.astype(np.float32, copy=False)

        # Convert to tensors
        image = torch.from_numpy(image)               # (2, H, W) float32
        mask  = torch.from_numpy(mask).unsqueeze(0)   # (1, H, W) float32

        if self.patch_size is not None:
            image, mask = self._random_crop(image, mask)

        if self.augment:
            image, mask = self._augment(image, mask)

        return image, mask

    # ── Spatial transforms ────────────────────────────────────────────────────

    def _random_crop(self,
                     image: torch.Tensor,
                     mask:  torch.Tensor) -> tuple:
        """
        Extract a random (patch_size × patch_size) crop.
        Falls back to center crop if the image is smaller than patch_size.
        """
        _, H, W = image.shape
        ps = self.patch_size

        if H < ps or W < ps:
            top  = max(0, (H - ps) // 2)
            left = max(0, (W - ps) // 2)
        else:
            top  = random.randint(0, H - ps)
            left = random.randint(0, W - ps)

        image = image[:, top:top + ps, left:left + ps]
        mask  = mask[:,  top:top + ps, left:left + ps]
        return image, mask

    def _augment(self,
                 image: torch.Tensor,
                 mask:  torch.Tensor) -> tuple:
        """
        SAR-safe geometric augmentations:
            • Random horizontal flip   (p = 0.5)
            • Random vertical flip     (p = 0.5)
            • Random 90° rotation      (0 / 90 / 180 / 270°, each p = 0.25)

        No intensity / color augmentations — images are normalized radar
        backscatter; altering their distribution would corrupt the signal.
        """
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        k = random.randint(0, 3)
        if k:
            image = torch.rot90(image, k, dims=[1, 2])
            mask  = torch.rot90(mask,  k, dims=[1, 2])

        return image, mask


# ─── FACTORY HELPERS ──────────────────────────────────────────────────────────

def build_datasets(stats_file: str,
                   patch_size: int = None) -> dict:
    """
    Load train_stats.json and return all three splits.

    Returns:
        dict with keys 'train', 'val', 'test' → SARDataset instances.
    """
    with open(stats_file) as f:
        stats = json.load(f)

    splits      = stats["splits"]
    mask_lookup = splits["masks"]

    return {
        "train": SARDataset(splits["train"], mask_lookup,
                            augment=True,  patch_size=patch_size),
        "val":   SARDataset(splits["val"],  mask_lookup,
                            augment=False, patch_size=patch_size),
        "test":  SARDataset(splits["test"], mask_lookup,
                            augment=False, patch_size=None),  # full images
    }


def build_loaders(stats_file:  str,
                  batch_size:  int = 8,
                  patch_size:  int = None,
                  num_workers: int = 4,
                  pin_memory:  bool = True) -> dict:
    """Build datasets AND wrap them in DataLoaders."""
    datasets = build_datasets(stats_file, patch_size=patch_size)

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size  = batch_size,
            shuffle     = True,
            num_workers = num_workers,
            pin_memory  = pin_memory,
            drop_last   = True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size  = 1,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = pin_memory,
        ),
    }


# ─── QUICK SANITY CHECK ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Sanity-check SARDataset against a train_stats.json file"
    )
    parser.add_argument(
        "--stats_file", required=True,
        help="Path to train_stats.json produced by preprocess.py"
    )
    parser.add_argument(
        "--patch_size", type=int, default=None,
        help="Optional patch size for random crops (default: full images)"
    )
    args = parser.parse_args()

    print(f"Loading datasets from: {args.stats_file}")
    datasets = build_datasets(args.stats_file, patch_size=args.patch_size)

    for split, ds in datasets.items():
        print(f"\n{split.upper():5s} — {ds}")
        if len(ds) == 0:
            print("  (empty split, skipping)")
            continue
        img, msk = ds[0]
        print(f"  image : {tuple(img.shape)}, dtype={img.dtype}, "
              f"min={img.min():.3f}, max={img.max():.3f}")
        print(f"  mask  : {tuple(msk.shape)}, dtype={msk.dtype}, "
              f"unique values={msk.unique().tolist()}")

    print("\nSanity check passed.")

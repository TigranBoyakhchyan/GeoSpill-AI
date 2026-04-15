"""
src/training.py
---------------
Training script for SAR oil spill segmentation.

Architecture : U-Net with pretrained ResNet34 encoder
               (segmentation_models_pytorch)
Loss         : Focal Loss + Dice Loss (combined, equal weight)
Optimizer    : AdamW with cosine annealing + linear warmup
Metrics      : IoU, Dice, Precision, Recall (per epoch, on val set)
Output       : Best model checkpoint + training log CSV

─── Google Colab usage ──────────────────────────────────────────────────────
Mount your Drive first, then run:

    !python src/training.py \
        --stats_file  /content/drive/MyDrive/data/train_stats.json \
        --output_dir  /content/drive/MyDrive/results \
        --epochs      50 \
        --batch_size  8 \
        --patch_size  256 \
        --lr          1e-4

─── Local usage ─────────────────────────────────────────────────────────────
    python src/training.py \
        --stats_file  data/train_stats.json \
        --output_dir  results \
        --epochs      50

─── Install dependencies ────────────────────────────────────────────────────
    pip install torch torchvision segmentation-models-pytorch
    pip install rasterio numpy
"""

import os
import csv
import json
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError(
        "segmentation_models_pytorch is required.\n"
        "Install it with: pip install segmentation-models-pytorch"
    )

# local import — assumes training.py lives alongside dataset.py in src/
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import build_datasets


# ─── LOSS FUNCTIONS ───────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Works on raw logits (applies sigmoid internally).
    Smooth term prevents division by zero on empty masks.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        # Flatten spatial dims
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class FocalDiceLoss(nn.Module):
    """
    Combined Focal Loss + Dice Loss.

    Focal loss addresses extreme class imbalance (tiny oil spill vs
    large ocean background) by down-weighting easy negatives.
    Dice loss directly optimizes the overlap metric.

    Args:
        alpha (float): Weight on Focal loss term.
        gamma (float): Focal loss focusing parameter (2.0 is standard).
        dice_weight (float): Weight on Dice loss term.
    """
    def __init__(self,
                 alpha:       float = 0.25,
                 gamma:       float = 2.0,
                 dice_weight: float = 1.0,
                 focal_weight: float = 1.0):
        super().__init__()
        self.gamma        = gamma
        self.alpha        = alpha
        self.dice_weight  = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss    = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ── Focal loss ────────────────────────────────────────────
        bce   = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        focal = focal.mean()

        # ── Dice loss ─────────────────────────────────────────────
        dice = self.dice_loss(logits, targets)

        return self.focal_weight * focal + self.dice_weight * dice


# ─── METRICS ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(logits: torch.Tensor,
                    targets: torch.Tensor,
                    threshold: float = 0.5) -> dict:
    """
    Compute IoU, Dice, Precision, Recall from raw logits.

    Args:
        logits  : (B, 1, H, W) raw model output
        targets : (B, 1, H, W) ground truth (0.0 / 1.0)
        threshold: probability threshold for positive prediction

    Returns:
        dict with keys: iou, dice, precision, recall
    """
    preds   = (torch.sigmoid(logits) > threshold).float()
    targets = targets.float()

    preds_f   = preds.view(-1)
    targets_f = targets.view(-1)

    tp = (preds_f * targets_f).sum().item()
    fp = (preds_f * (1 - targets_f)).sum().item()
    fn = ((1 - preds_f) * targets_f).sum().item()

    smooth = 1e-6
    iou       = tp / (tp + fp + fn + smooth)
    dice      = 2 * tp / (2 * tp + fp + fn + smooth)
    precision = tp / (tp + fp + smooth)
    recall    = tp / (tp + fn + smooth)

    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


# ─── WARMUP + COSINE SCHEDULER ────────────────────────────────────────────────

def build_scheduler(optimizer: optim.Optimizer,
                    warmup_epochs: int,
                    total_epochs:  int):
    """
    Linear warmup for `warmup_epochs`, then cosine annealing to eta_min=1e-6.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        # Cosine decay
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─── ONE EPOCH ────────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    """
    Run a single train or validation epoch.

    Returns:
        dict with mean loss and metrics for the epoch
    """
    model.train() if training else model.eval()

    total_loss = 0.0
    all_logits, all_targets = [], []
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            logits = model(images)

            loss = criterion(logits, masks)

            if training:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — stabilizes training on small datasets
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            all_logits.append(logits.detach().cpu())
            all_targets.append(masks.detach().cpu())

    all_logits  = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics     = compute_metrics(all_logits, all_targets)
    metrics["loss"] = total_loss / len(loader)

    return metrics


# ─── COLAB HELPERS ────────────────────────────────────────────────────────────

def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def print_header(args, device, model):
    """Print a clear training summary before the loop starts."""
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 65)
    print("  SAR Oil Spill Detection — Training")
    print("=" * 65)
    print(f"  Device       : {device}")
    print(f"  Architecture : U-Net + ResNet34 (pretrained ImageNet)")
    print(f"  Loss         : Focal + Dice")
    print(f"  Trainable params: {n_params:,}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Patch size   : {args.patch_size or 'full image'}")
    print(f"  LR           : {args.lr}")
    print(f"  Stats file   : {args.stats_file}")
    print(f"  Output dir   : {args.output_dir}")
    print("=" * 65)


def print_epoch(epoch, total, train_m, val_m, lr, elapsed):
    """Single-line epoch summary, readable in Colab output."""
    print(
        f"[{epoch:03d}/{total}] "
        f"loss {train_m['loss']:.4f} → {val_m['loss']:.4f} | "
        f"IoU {val_m['iou']:.4f} | "
        f"Dice {val_m['dice']:.4f} | "
        f"Prec {val_m['precision']:.4f} | "
        f"Rec {val_m['recall']:.4f} | "
        f"LR {lr:.2e} | "
        f"{elapsed:.0f}s"
    )


# ─── MAIN TRAINING LOOP ───────────────────────────────────────────────────────

def train(args):
    # ── Device ────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )

    # ── Datasets & loaders ────────────────────────────────────────
    print("Loading datasets...")
    datasets = build_datasets(args.stats_file, patch_size=args.patch_size)

    # Colab tip: num_workers=2 is safe; higher values can cause issues
    nw = 2 if is_colab() else args.num_workers
    train_loader = DataLoader(
        datasets["train"], batch_size=args.batch_size,
        shuffle=True,  num_workers=nw, pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=args.batch_size,
        shuffle=False, num_workers=nw, pin_memory=(device.type == "cuda")
    )

    print(f"  Train: {len(datasets['train'])} samples  "
          f"Val: {len(datasets['val'])} samples  "
          f"Test: {len(datasets['test'])} samples")

    # ── Model ─────────────────────────────────────────────────────
    # in_channels=2 for VV+VH SAR bands
    # classes=1     for binary segmentation
    model = smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = "imagenet",
        in_channels     = 2,
        classes         = 1,
    ).to(device)

    # ── Loss, optimizer, scheduler ────────────────────────────────
    criterion = FocalDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer,
                                warmup_epochs=args.warmup_epochs,
                                total_epochs=args.epochs)

    # ── Output directory ──────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path    = os.path.join(args.output_dir, f"best_model_{timestamp}.pt")
    log_csv_path = os.path.join(args.output_dir, f"train_log_{timestamp}.csv")

    print_header(args, device, model)

    # ── CSV log setup ─────────────────────────────────────────────
    csv_fields = [
        "epoch", "lr",
        "train_loss", "train_iou", "train_dice",
        "val_loss",   "val_iou",   "val_dice",
        "val_precision", "val_recall", "elapsed_s"
    ]
    csv_file   = open(log_csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # ── Training loop ─────────────────────────────────────────────
    best_val_iou  = -1.0
    best_val_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = run_epoch(model, train_loader, criterion,
                                  optimizer, device, training=True)
        val_metrics   = run_epoch(model, val_loader,   criterion,
                                  None,      device, training=False)

        scheduler.step()
        elapsed = time.time() - t0
        lr      = scheduler.get_last_lr()[0]

        print_epoch(epoch, args.epochs, train_metrics, val_metrics, lr, elapsed)

        # ── Save best checkpoint (by val IoU) ─────────────────────
        if val_metrics["iou"] > best_val_iou:
            best_val_iou  = val_metrics["iou"]
            best_val_dice = val_metrics["dice"]
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_iou":        best_val_iou,
                "val_dice":       best_val_dice,
                "args":           vars(args),
            }, ckpt_path)
            print(f"  ✓ New best — IoU {best_val_iou:.4f}, "
                  f"Dice {best_val_dice:.4f} → saved to {ckpt_path}")

        # ── CSV row ───────────────────────────────────────────────
        csv_writer.writerow({
            "epoch":         epoch,
            "lr":            f"{lr:.6e}",
            "train_loss":    f"{train_metrics['loss']:.6f}",
            "train_iou":     f"{train_metrics['iou']:.6f}",
            "train_dice":    f"{train_metrics['dice']:.6f}",
            "val_loss":      f"{val_metrics['loss']:.6f}",
            "val_iou":       f"{val_metrics['iou']:.6f}",
            "val_dice":      f"{val_metrics['dice']:.6f}",
            "val_precision": f"{val_metrics['precision']:.6f}",
            "val_recall":    f"{val_metrics['recall']:.6f}",
            "elapsed_s":     f"{elapsed:.1f}",
        })
        csv_file.flush()  # flush every epoch so log survives Colab disconnects

    csv_file.close()

    # ── Final summary ─────────────────────────────────────────────
    print("=" * 65)
    print("Training complete.")
    print(f"  Best val IoU  : {best_val_iou:.4f}")
    print(f"  Best val Dice : {best_val_dice:.4f}")
    print(f"  Checkpoint    : {ckpt_path}")
    print(f"  Training log  : {log_csv_path}")
    print("=" * 65)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train U-Net ResNet34 for SAR oil spill segmentation"
    )

    # ── Required ──────────────────────────────────────────────────
    parser.add_argument(
        "--stats_file", required=True,
        help=(
            "Path to train_stats.json produced by preprocess.py.\n"
            "Colab example: /content/drive/MyDrive/data/train_stats.json"
        )
    )
    parser.add_argument(
        "--output_dir", required=True,
        help=(
            "Directory where checkpoints and CSV log are saved.\n"
            "Colab example: /content/drive/MyDrive/results"
        )
    )

    # ── Training hyperparameters ──────────────────────────────────
    parser.add_argument("--epochs",        type=int,   default=50)
    parser.add_argument("--batch_size",    type=int,   default=8)
    parser.add_argument("--lr",            type=float, default=1e-4,
                        help="Peak learning rate for AdamW")
    parser.add_argument("--warmup_epochs", type=int,   default=5,
                        help="Epochs for linear LR warmup before cosine decay")
    parser.add_argument("--patch_size",    type=int,   default=None,
                        help="Random crop size (None = full images)")
    parser.add_argument("--num_workers",   type=int,   default=4,
                        help="DataLoader workers (auto-set to 2 on Colab)")

    args = parser.parse_args()
    train(args)

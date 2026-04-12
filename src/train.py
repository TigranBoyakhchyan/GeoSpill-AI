"""
Full training pipeline for SAR oil spill segmentation.

Works with the on-the-fly cropping dataset — no pre-extracted patches needed.

Run locally:
    python src/train.py

Run on Colab:
    !python src/train.py --epochs 50 --batch_size 16 --crops_per_image 10
"""

import os
import csv
import argparse
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import get_dataloaders
from src.model import get_model

# ─── DEFAULTS ─────────────────────────────────────────────────────────────────
DEFAULT_EPOCHS          = 30
DEFAULT_BATCH_SIZE      = 8
DEFAULT_LR              = 1e-4
DEFAULT_MODEL_TYPE      = "smp"
DEFAULT_CROPS_PER_IMAGE = 5     # local: 46×5=230 samples; Colab: use 10 → 1200×10=12000
CHECKPOINT_DIR          = "checkpoints"
LOG_FILE                = "checkpoints/training_log.csv"


# ─── LOSS FUNCTIONS ───────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = torch.sigmoid(logits)
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        inter   = (probs * targets).sum(dim=1)
        dice    = (2.0 * inter + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """BCE + Dice loss. Handles class imbalance for oil (~4% coverage)."""
    def __init__(self, alpha: float = 0.5, pos_weight: float = 10.0):
        super().__init__()
        self.alpha = alpha
        self.dice  = DiceLoss()
        self.bce   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    def forward(self, logits, targets):
        self.bce.pos_weight = self.bce.pos_weight.to(logits.device)
        return (self.alpha * self.bce(logits, targets) +
                (1.0 - self.alpha) * self.dice(logits, targets))


# ─── METRICS ──────────────────────────────────────────────────────────────────

def compute_metrics(logits, targets, threshold=0.5):
    with torch.no_grad():
        preds   = (torch.sigmoid(logits) > threshold).float()
        preds   = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        tp = (preds * targets).sum(dim=1)
        fp = preds.sum(dim=1) - tp
        fn = targets.sum(dim=1) - tp
        iou  = (tp / (tp + fp + fn + 1e-6)).mean().item()
        dice = (2*tp / (2*tp + fp + fn + 1e-6)).mean().item()
    return {"iou": iou, "dice": dice}


# ─── TRAIN / VALIDATE ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, epoch):
    model.train()
    total_loss = total_iou = total_dice = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss   = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        m = compute_metrics(logits.detach(), masks)
        total_loss += loss.item()
        total_iou  += m["iou"]
        total_dice += m["dice"]
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{m['iou']:.4f}"})

    n = len(loader)
    return {"loss": total_loss/n, "iou": total_iou/n, "dice": total_dice/n}


def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = total_iou = total_dice = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]  ", leave=False)
    with torch.no_grad():
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            logits  = model(images)
            loss    = criterion(logits, masks)
            m       = compute_metrics(logits, masks)
            total_loss += loss.item()
            total_iou  += m["iou"]
            total_dice += m["dice"]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "iou": f"{m['iou']:.4f}"})

    n = len(loader)
    return {"loss": total_loss/n, "iou": total_iou/n, "dice": total_dice/n}


# ─── CHECKPOINTING & LOGGING ──────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_iou, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":                epoch,
        "val_iou":              val_iou,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def init_log(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch",
            "train_loss", "train_iou", "train_dice",
            "val_loss",   "val_iou",   "val_dice",
            "lr"
        ])


def log_epoch(path, epoch, train_m, val_m, lr):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            epoch,
            f"{train_m['loss']:.6f}", f"{train_m['iou']:.6f}", f"{train_m['dice']:.6f}",
            f"{val_m['loss']:.6f}",   f"{val_m['iou']:.6f}",   f"{val_m['dice']:.6f}",
            f"{lr:.8f}",
        ])


# ─── MAIN TRAINING LOOP ───────────────────────────────────────────────────────

def train(epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE,
          lr=DEFAULT_LR, model_type=DEFAULT_MODEL_TYPE,
          crops_per_image=DEFAULT_CROPS_PER_IMAGE):

    PATIENCE = 10

    print("=" * 60)
    print("SAR Oil Spill — Training Pipeline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # num_workers and pin_memory auto-set based on OS and device
    num_workers = 0 if os.name == "nt" else 4
    pin_memory  = device.type == "cuda"

    train_loader, val_loader, _ = get_dataloaders(
        batch_size      = batch_size,
        num_workers     = num_workers,
        pin_memory      = pin_memory,
        crops_per_image = crops_per_image,
    )

    print(f"\nBuilding model ({model_type})...")
    model     = get_model(model_type=model_type, in_channels=2).to(device)
    criterion = CombinedLoss(alpha=0.5, pos_weight=10.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler    = GradScaler(enabled=device.type == "cuda")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    init_log(LOG_FILE)

    best_val_iou     = 0.0
    patience_counter = 0
    best_path        = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    last_path        = os.path.join(CHECKPOINT_DIR, "last_model.pth")

    print(f"\nStarting training: {epochs} epochs, "
          f"{crops_per_image} crops/image, patience={PATIENCE}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, epoch)
        val_m   = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        log_epoch(LOG_FILE, epoch, train_m, val_m, current_lr)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train loss: {train_m['loss']:.4f}  iou: {train_m['iou']:.4f} | "
            f"Val loss: {val_m['loss']:.4f}  iou: {val_m['iou']:.4f}  "
            f"dice: {val_m['dice']:.4f} | lr: {current_lr:.2e}"
        )

        save_checkpoint(model, optimizer, epoch, val_m["iou"], last_path)

        if val_m["iou"] > best_val_iou:
            best_val_iou     = val_m["iou"]
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_m["iou"], best_path)
            print(f"  New best model saved (val IoU: {best_val_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping — no improvement for {PATIENCE} epochs.")
                break

    print("\n" + "=" * 60)
    print(f"Training complete! Best val IoU: {best_val_iou:.4f}")
    print(f"Best model: {best_path}")
    print("=" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAR oil spill U-Net")
    parser.add_argument("--epochs",          type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size",      type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",              type=float, default=DEFAULT_LR)
    parser.add_argument("--model_type",      type=str,   default=DEFAULT_MODEL_TYPE,
                        choices=["smp", "custom"])
    parser.add_argument("--crops_per_image", type=int,   default=DEFAULT_CROPS_PER_IMAGE,
                        help="Random crops per image per epoch (default 5 locally, use 10 on Colab)")
    args = parser.parse_args()

    train(
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        model_type      = args.model_type,
        crops_per_image = args.crops_per_image,
    )
"""
src/train.py
------------
Full training pipeline for SAR oil spill segmentation.

This file:
    1. Defines loss functions (Dice, BCE, Combined)
    2. Defines metric computation (IoU, Dice score)
    3. Runs the training loop with validation after each epoch
    4. Saves the best model checkpoint based on validation IoU
    5. Logs all metrics to a CSV file for later analysis

Run locally:
    python src/train.py

Run on Colab:
    !python src/train.py --epochs 50 --batch_size 16
"""

import os
import csv
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Import our own modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.dataset import get_dataloaders
from src.model import get_model

# ─── CONFIG DEFAULTS ──────────────────────────────────────────────────────────
# These are overridden by command-line arguments when running on Colab
DEFAULT_EPOCHS      = 30       # enough for local testing; use 50-100 on Colab
DEFAULT_BATCH_SIZE  = 8        # reduce to 4 if you run out of memory
DEFAULT_LR          = 1e-4     # Adam learning rate — 1e-4 is a safe default
DEFAULT_MODEL_TYPE  = "smp"    # "smp" or "custom"
CHECKPOINT_DIR      = "checkpoints"
LOG_FILE            = "checkpoints/training_log.csv"


# ─── STEP 1: LOSS FUNCTIONS ───────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.

    Dice loss directly optimizes the Dice coefficient (same as F1 score),
    which measures overlap between prediction and ground truth.

    Formula: Dice = (2 * |P ∩ G|) / (|P| + |G|)
    Loss    = 1 - Dice   (so minimizing loss = maximizing overlap)

    Why use this instead of plain BCE?
        BCE treats every pixel independently and equally.
        With 4% oil coverage, the model can get 96% accuracy by predicting
        all water — BCE won't penalize this strongly enough.
        Dice loss focuses on the OVERLAP, so it heavily penalizes
        missing oil pixels regardless of how many water pixels are correct.

    smooth=1.0 prevents division by zero when both prediction
    and target are all zeros (empty patches).

    Args:
        smooth: smoothing factor to avoid division by zero
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to convert logits → probabilities [0, 1]
        probs = torch.sigmoid(logits)

        # Flatten to (batch, pixels) for easier computation
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        # Compute per-sample intersection and union
        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum(dim=1) + targets.sum(dim=1) + self.smooth
        )

        # Return mean loss over the batch (1 - dice because we minimize)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice loss.

    Using both losses together gives better results than either alone:

    BCE (BCEWithLogitsLoss):
        - Penalizes each pixel individually
        - Sensitive to overall pixel-level accuracy
        - pos_weight compensates for class imbalance:
          if oil covers 4% of pixels, water covers 96%.
          pos_weight = 96/4 = 24 — oil pixels get 24x more penalty when missed.
          We use a conservative value of 10 to avoid over-correcting.

    Dice:
        - Penalizes poor region-level overlap
        - Directly optimizes the metric we care about
        - Handles class imbalance naturally

    alpha controls the balance:
        alpha=0.5 → equal weight to both losses
        alpha=0.3 → more emphasis on Dice (good if model ignores oil)
        alpha=0.7 → more emphasis on BCE (good if too many false positives)

    Args:
        alpha:      weight for BCE loss (1-alpha goes to Dice)
        pos_weight: how much extra weight oil pixels get in BCE
    """

    def __init__(self, alpha: float = 0.5, pos_weight: float = 10.0):
        super().__init__()
        self.alpha    = alpha
        self.dice     = DiceLoss(smooth=1.0)
        # pos_weight must be a tensor
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        # Move pos_weight tensor to same device as input
        self.bce.pos_weight = self.bce.pos_weight.to(logits.device)

        bce_loss  = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        return self.alpha * bce_loss + (1.0 - self.alpha) * dice_loss


# ─── STEP 2: METRICS ──────────────────────────────────────────────────────────

def compute_metrics(logits: torch.Tensor,
                    targets: torch.Tensor,
                    threshold: float = 0.5) -> dict:
    """
    Compute IoU and Dice score for a batch of predictions.

    Both metrics measure overlap between prediction and ground truth,
    but from slightly different angles:

    IoU (Intersection over Union / Jaccard Index):
        = TP / (TP + FP + FN)
        Strict — penalizes both false positives and false negatives equally.
        Standard metric for segmentation competitions.

    Dice Score (F1):
        = 2*TP / (2*TP + FP + FN)
        Slightly more lenient than IoU.
        Direct equivalent of F1 score from classification.

    Relationship: Dice = 2*IoU / (1 + IoU)
    They tell the same story — we track both for completeness.

    Args:
        logits:    raw model output, shape (B, 1, H, W)
        targets:   binary ground truth, shape (B, 1, H, W)
        threshold: probability cutoff to binarize predictions

    Returns:
        dict with keys "iou" and "dice", values are Python floats
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        # Flatten to (batch, pixels)
        preds   = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        tp = (preds * targets).sum(dim=1)
        fp = preds.sum(dim=1) - tp
        fn = targets.sum(dim=1) - tp

        iou  = (tp / (tp + fp + fn + 1e-6)).mean().item()
        dice = (2*tp / (2*tp + fp + fn + 1e-6)).mean().item()

    return {"iou": iou, "dice": dice}


# ─── STEP 3: ONE EPOCH OF TRAINING ───────────────────────────────────────────

def train_one_epoch(model:     nn.Module,
                    loader:    torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: nn.Module,
                    device:    torch.device,
                    scaler:    GradScaler,
                    epoch:     int) -> dict:
    """
    Run one full pass over the training set.

    Key concepts used here:

    Mixed Precision Training (autocast + GradScaler):
        By default PyTorch uses float32 everywhere (32 bits per number).
        Mixed precision uses float16 for most operations (16 bits).
        Result: ~2x faster training, ~2x less GPU memory — no accuracy loss.
        GradScaler prevents underflow (very small gradients becoming zero in float16).
        On CPU this has no effect — it only speeds things up on GPU.

    optimizer.zero_grad():
        PyTorch accumulates gradients by default — we must clear them
        before each backward pass, otherwise gradients from previous
        batches contaminate the current update.

    scaler.scale(loss).backward():
        Scales the loss before backward pass to prevent float16 underflow,
        then computes gradients for all parameters.

    scaler.step(optimizer):
        Unscales gradients back to float32 and calls optimizer.step().
        If gradients contain inf/nan (from overflow), the step is skipped.

    scaler.update():
        Adjusts the scale factor for the next iteration.

    Args:
        model:     the U-Net model
        loader:    training DataLoader
        optimizer: Adam optimizer
        criterion: CombinedLoss
        device:    "cuda" or "cpu"
        scaler:    GradScaler for mixed precision
        epoch:     current epoch number (for progress bar display)

    Returns:
        dict with average "loss", "iou", "dice" over the epoch
    """
    model.train()
    total_loss = 0.0
    total_iou  = 0.0
    total_dice = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)

    for images, masks in pbar:
        images = images.to(device)
        masks  = masks.to(device)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss   = criterion(logits, masks)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Compute metrics (detached from computation graph)
        metrics = compute_metrics(logits.detach(), masks)

        total_loss += loss.item()
        total_iou  += metrics["iou"]
        total_dice += metrics["dice"]

        # Update progress bar with current batch stats
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "iou":  f"{metrics['iou']:.4f}",
        })

    n = len(loader)
    return {
        "loss": total_loss / n,
        "iou":  total_iou  / n,
        "dice": total_dice / n,
    }


# ─── STEP 4: VALIDATION ───────────────────────────────────────────────────────

def validate(model:     nn.Module,
             loader:    torch.utils.data.DataLoader,
             criterion: nn.Module,
             device:    torch.device,
             epoch:     int) -> dict:
    """
    Evaluate the model on the validation set.

    Key differences from training:
        - model.eval(): disables dropout and makes BatchNorm use
          running statistics instead of batch statistics
        - torch.no_grad(): disables gradient computation entirely,
          saving memory and speeding up inference
        - No optimizer step — we only measure performance

    Args:
        model:     the U-Net model
        loader:    validation DataLoader
        criterion: CombinedLoss (same as training for comparable loss values)
        device:    "cuda" or "cpu"
        epoch:     current epoch number

    Returns:
        dict with average "loss", "iou", "dice" over the validation set
    """
    model.eval()
    total_loss = 0.0
    total_iou  = 0.0
    total_dice = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Val]  ", leave=False)

    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks  = masks.to(device)

            logits  = model(images)
            loss    = criterion(logits, masks)
            metrics = compute_metrics(logits, masks)

            total_loss += loss.item()
            total_iou  += metrics["iou"]
            total_dice += metrics["dice"]

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "iou":  f"{metrics['iou']:.4f}",
            })

    n = len(loader)
    return {
        "loss": total_loss / n,
        "iou":  total_iou  / n,
        "dice": total_dice / n,
    }


# ─── STEP 5: SAVE CHECKPOINT ──────────────────────────────────────────────────

def save_checkpoint(model:      nn.Module,
                    optimizer:  torch.optim.Optimizer,
                    epoch:      int,
                    val_iou:    float,
                    path:       str):
    """
    Save model weights and training state to disk.

    We save:
        - model state_dict: the learned weights
        - optimizer state_dict: momentum and adaptive learning rates
          (needed to resume training from this checkpoint)
        - epoch and val_iou: for bookkeeping

    We save only state_dict (not the whole model object) because:
        - More portable — works even if you refactor the model class
        - Smaller file size
        - Standard practice in PyTorch

    Args:
        model:     trained model
        optimizer: optimizer (saved for potential resume)
        epoch:     current epoch
        val_iou:   validation IoU at this checkpoint
        path:      file path to save to
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch":               epoch,
        "val_iou":             val_iou,
        "model_state_dict":    model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


# ─── STEP 6: LOGGING ──────────────────────────────────────────────────────────

def init_log(path: str):
    """Create the CSV log file with headers."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_iou", "train_dice",
            "val_loss",   "val_iou",   "val_dice",
            "lr"
        ])


def log_epoch(path: str, epoch: int, train_metrics: dict,
              val_metrics: dict, lr: float):
    """Append one row of metrics to the CSV log."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch,
            f"{train_metrics['loss']:.6f}",
            f"{train_metrics['iou']:.6f}",
            f"{train_metrics['dice']:.6f}",
            f"{val_metrics['loss']:.6f}",
            f"{val_metrics['iou']:.6f}",
            f"{val_metrics['dice']:.6f}",
            f"{lr:.8f}",
        ])


# ─── STEP 7: MAIN TRAINING LOOP ───────────────────────────────────────────────

def train(epochs:      int   = DEFAULT_EPOCHS,
          batch_size:  int   = DEFAULT_BATCH_SIZE,
          lr:          float = DEFAULT_LR,
          model_type:  str   = DEFAULT_MODEL_TYPE):
    """
    Full training pipeline.

    Scheduler — CosineAnnealingLR:
        Gradually reduces the learning rate following a cosine curve.
        Starts at lr, decays to near 0 by epoch T_max, then can restart.
        This prevents the model from getting stuck in a local minimum
        and often gives a small accuracy boost in the final epochs.

        Epoch:  0    10    20    30
        LR:   1e-4  ~5e-5  ~1e-5  ~0

    Early stopping patience:
        If validation IoU doesn't improve for PATIENCE consecutive epochs,
        training stops early. This prevents overfitting and saves time.
        On 46 images you may see overfitting after ~20-30 epochs.

    Args:
        epochs:     maximum number of training epochs
        batch_size: patches per batch
        lr:         initial learning rate
        model_type: "smp" or "custom"
    """
    PATIENCE = 10  # stop if no improvement for this many epochs

    print("=" * 60)
    print("SAR Oil Spill — Training Pipeline")
    print("=" * 60)

    # ── Device ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────
    # Determine num_workers and pin_memory based on device and OS
    num_workers = 0 if os.name == "nt" else 4   # 0 on Windows, 4 on Linux/Colab
    pin_memory  = device.type == "cuda"

    train_loader, val_loader, _ = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # ── Model ─────────────────────────────────────────────────
    print(f"\nBuilding model ({model_type})...")
    model = get_model(model_type=model_type, in_channels=2)
    model = model.to(device)

    # ── Loss ──────────────────────────────────────────────────
    criterion = CombinedLoss(alpha=0.5, pos_weight=10.0)

    # ── Optimizer ─────────────────────────────────────────────
    # AdamW: Adam with weight decay for regularization
    # weight_decay=1e-4 penalizes large weights, reduces overfitting
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=1e-4
    )

    # ── Scheduler ─────────────────────────────────────────────
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # ── Mixed precision scaler ────────────────────────────────
    # enabled=False on CPU (no effect), True on GPU (2x speedup)
    scaler = GradScaler(enabled=device.type == "cuda")

    # ── Logging ───────────────────────────────────────────────
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    init_log(LOG_FILE)

    # ── Training state ────────────────────────────────────────
    best_val_iou     = 0.0
    patience_counter = 0
    best_path        = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    last_path        = os.path.join(CHECKPOINT_DIR, "last_model.pth")

    print(f"\nStarting training for up to {epochs} epochs...")
    print(f"Early stopping patience: {PATIENCE} epochs")
    print("-" * 60)

    for epoch in range(1, epochs + 1):

        # ── Train ─────────────────────────────────────────────
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, epoch
        )

        # ── Validate ──────────────────────────────────────────
        val_metrics = validate(
            model, val_loader, criterion, device, epoch
        )

        # ── Step scheduler ────────────────────────────────────
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # ── Log to CSV ────────────────────────────────────────
        log_epoch(LOG_FILE, epoch, train_metrics, val_metrics, current_lr)

        # ── Print epoch summary ───────────────────────────────
        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train — loss: {train_metrics['loss']:.4f}  "
            f"iou: {train_metrics['iou']:.4f}  "
            f"dice: {train_metrics['dice']:.4f} | "
            f"Val — loss: {val_metrics['loss']:.4f}  "
            f"iou: {val_metrics['iou']:.4f}  "
            f"dice: {val_metrics['dice']:.4f} | "
            f"lr: {current_lr:.2e}"
        )

        # ── Save last checkpoint (always) ─────────────────────
        save_checkpoint(model, optimizer, epoch, val_metrics["iou"], last_path)

        # ── Save best checkpoint ───────────────────────────────
        if val_metrics["iou"] > best_val_iou:
            best_val_iou     = val_metrics["iou"]
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_metrics["iou"], best_path)
            print(f"New best model saved (val IoU: {best_val_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping — no improvement for {PATIENCE} epochs.")
                break

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"  Best val IoU : {best_val_iou:.4f}")
    print(f"  Best model   : {best_path}")
    print(f"  Training log : {LOG_FILE}")
    print("=" * 60)


# ─── COMMAND LINE INTERFACE ───────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SAR oil spill U-Net")
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--model_type",  type=str,   default=DEFAULT_MODEL_TYPE,
                        choices=["smp", "custom"])
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_type=args.model_type,
    )
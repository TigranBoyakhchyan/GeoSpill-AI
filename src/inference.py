"""
Model loading and inference for SAR oil spill detection.

Provides:
    - U-Net model construction and checkpoint loading
    - Sliding-window inference with overlap averaging
    - Demo mode threshold-based detection (for UI testing)

All functions are pure PyTorch/numpy — no Streamlit or UI dependencies.
"""

import os
import json

import numpy as np
import torch

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

DB_CLIP_MIN = -50.0
DB_CLIP_MAX =  0.0


def build_unet(in_channels: int = 2, classes: int = 1) -> torch.nn.Module:
    if smp is None:
        raise ImportError(
            "segmentation_models_pytorch is required.\n"
            "Install with: pip install segmentation-models-pytorch"
        )
    return smp.Unet(
        encoder_name    = "resnet34",
        encoder_weights = None,
        in_channels     = in_channels,
        classes         = classes,
    )


def load_checkpoint(checkpoint_path: str,
                    stats_path: str,
                    device: torch.device = None):
    if not os.path.exists(checkpoint_path):
        return None, None, f"Checkpoint not found: {checkpoint_path}"
    if not os.path.exists(stats_path):
        return None, None, f"Stats file not found: {stats_path}"

    try:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = build_unet(in_channels=2, classes=1)
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt

        model.load_state_dict(state)
        model.to(device).eval()

        with open(stats_path) as f:
            stats = json.load(f)

        return model, stats, None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def tile_positions(length: int, patch: int, stride: int) -> list:
    if length <= patch:
        return [0]
    positions = list(range(0, length - patch + 1, stride))
    if positions[-1] + patch < length:
        positions.append(length - patch)
    return positions


def predict_sliding(img, model, mean, std,
                    patch_size=256, stride=128, threshold=0.5,
                    progress_cb=None):
    device = next(model.parameters()).device

    img_norm = np.clip(img[:2], DB_CLIP_MIN, DB_CLIP_MAX).astype(np.float32)
    img_norm = (img_norm - mean[:, None, None]) / (std[:, None, None] + 1e-6)

    _, H, W = img_norm.shape
    pred_sum   = np.zeros((H, W), dtype=np.float32)
    pred_count = np.zeros((H, W), dtype=np.float32)

    ys = tile_positions(H, patch_size, stride)
    xs = tile_positions(W, patch_size, stride)
    total_tiles = len(ys) * len(xs)
    done = 0

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = img_norm[:, y:y + patch_size, x:x + patch_size]
                t = torch.from_numpy(patch).unsqueeze(0).to(device)
                logits = model(t)
                probs  = torch.sigmoid(logits).squeeze().cpu().numpy()
                pred_sum  [y:y + patch_size, x:x + patch_size] += probs
                pred_count[y:y + patch_size, x:x + patch_size] += 1.0
                done += 1
                if progress_cb:
                    progress_cb(done, total_tiles)

    avg_prob = pred_sum / np.maximum(pred_count, 1e-6)
    return (avg_prob > threshold).astype(np.float32)


def predict_demo(img, threshold_pct=8):
    vv = img[0] if img.ndim == 3 else img
    vv_clipped = np.clip(vv, DB_CLIP_MIN, DB_CLIP_MAX)
    vv_norm = (vv_clipped - vv_clipped.min()) / (vv_clipped.max() - vv_clipped.min() + 1e-6)
    threshold_val = np.percentile(vv_norm, threshold_pct)
    return (vv_norm < threshold_val).astype(np.float32)
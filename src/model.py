"""
src/model.py
------------
U-Net model for SAR oil spill binary segmentation.

Two options are provided:
    1. SMPUNet  — uses segmentation-models-pytorch (recommended)
                  pre-trained ResNet34 encoder, fastest to get good results
    2. UNet     — custom implementation built from scratch
                  good for understanding the architecture

The model takes:
    input:  tensor of shape (batch, 2, 256, 256)  — 2-band SAR image
    output: tensor of shape (batch, 1, 256, 256)  — raw logits (no sigmoid)

Sigmoid is NOT applied inside the model.
It is applied externally during loss computation and inference.
This is standard practice because:
    - BCEWithLogitsLoss is numerically more stable than BCE(sigmoid(x))
    - We can apply the threshold at inference time without recomputing

Usage:
    from src.model import get_model
    model = get_model(model_type="smp")   # or "custom"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── OPTION 1: SEGMENTATION-MODELS-PYTORCH (RECOMMENDED) ─────────────────────

def get_smp_unet(in_channels: int = 2, encoder_name: str = "resnet34") -> nn.Module:
    """
    Build a U-Net using the segmentation-models-pytorch library.

    Why use this over a custom model?
        - The encoder (ResNet34) is pre-trained on ImageNet
        - Pre-trained weights give much better starting features than
          random initialization, even though ImageNet is RGB and SAR is not.
          The low-level feature detectors (edges, textures) transfer well.
        - Saves days of training time on small datasets like ours (46 images)

    Why ResNet34?
        - Good balance of accuracy vs speed
        - Small enough to train on Colab's free GPU
        - You can upgrade to efficientnet-b3 later for better accuracy

    in_channels=2:
        Our SAR images have 2 bands (VV + VH).
        SMP handles the channel mismatch internally by adapting the
        first conv layer of ResNet34 (which normally expects 3 RGB channels).

    classes=1:
        Binary segmentation — one output channel for oil vs. not-oil.

    activation=None:
        Raw logits are returned. Sigmoid is applied externally.

    Args:
        in_channels:  number of input bands (2 for VV+VH)
        encoder_name: backbone architecture

    Returns:
        PyTorch model
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation-models-pytorch not installed.\n"
            "Run: pip install segmentation-models-pytorch"
        )

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",   # pre-trained weights
        in_channels=in_channels,
        classes=1,
        activation=None,              # raw logits
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  SMP U-Net ({encoder_name}) | Parameters: {n_params:,}")
    return model


# ─── OPTION 2: CUSTOM U-NET FROM SCRATCH ─────────────────────────────────────
# Read this section to understand how U-Net actually works.

class DoubleConv(nn.Module):
    """
    Two consecutive Conv → BatchNorm → ReLU blocks.

    This is the fundamental building block of U-Net.
    Every encoder and decoder step uses this.

    Why two convolutions?
        One convolution extracts basic features.
        The second refines them without adding a pooling step.
        This gives more representational power without losing resolution.

    Why BatchNorm?
        Normalizes activations within each batch, which:
        - Allows higher learning rates
        - Acts as a regularizer (reduces overfitting)
        - Stabilizes training

    Why ReLU?
        Introduces non-linearity so the model can learn complex patterns.
        inplace=True saves memory by modifying the tensor in-place.

    Args:
        in_channels:  number of input feature channels
        out_channels: number of output feature channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            # bias=False because BatchNorm already has a learnable bias term
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    One step of the U-Net encoder (downsampling path).

    Structure:
        DoubleConv → save as skip connection → MaxPool (halve spatial size)

    The skip connection output is saved BEFORE pooling so it retains
    the full spatial resolution — this is what gets passed to the decoder.

    MaxPool2d(2) halves both height and width:
        256×256 → 128×128 → 64×64 → 32×32 → 16×16

    Args:
        in_channels:  input feature channels
        out_channels: output feature channels
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple:
        skip = self.conv(x)    # full resolution — saved for skip connection
        down = self.pool(skip) # half resolution — passed to next encoder step
        return skip, down


class DecoderBlock(nn.Module):
    """
    One step of the U-Net decoder (upsampling path).

    Structure:
        ConvTranspose2d (double spatial size) → concatenate skip → DoubleConv

    ConvTranspose2d is a learnable upsampling operation.
    It doubles height and width (the reverse of MaxPool).

    Skip connection concatenation:
        The skip connection from the corresponding encoder step is
        concatenated along the channel dimension.
        This gives the decoder access to fine-grained spatial details
        that were lost during downsampling — this is the key innovation
        of U-Net over plain encoder-decoder networks.

    Why concatenate and not add?
        Concatenation preserves both sets of features independently.
        Addition would mix them, potentially losing information.

    Size mismatch handling:
        Due to integer rounding in pooling, the upsampled tensor may be
        1 pixel off from the skip connection. We handle this with
        F.interpolate as a safety measure.

    Args:
        in_channels:  channels coming into this block (from below)
        out_channels: channels after this block
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # ConvTranspose2d doubles spatial resolution
        # in_channels → out_channels, kernel=2, stride=2
        self.up   = nn.ConvTranspose2d(in_channels, out_channels,
                                       kernel_size=2, stride=2)
        # After concatenation with skip, channels double → need DoubleConv to reduce
        self.conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle potential size mismatch (1-pixel difference from rounding)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)

        # Concatenate skip connection along channel dimension
        x = torch.cat([skip, x], dim=1)  # channels double here
        return self.conv(x)


class UNet(nn.Module):
    """
    Custom U-Net for binary SAR segmentation.

    Architecture overview:
        Input (2, 256, 256)
            ↓ Encoder
        E1: DoubleConv → skip1 (64, 256, 256)  → pool → (64, 128, 128)
        E2: DoubleConv → skip2 (128, 128, 128) → pool → (128, 64, 64)
        E3: DoubleConv → skip3 (256, 64, 64)   → pool → (256, 32, 32)
        E4: DoubleConv → skip4 (512, 32, 32)   → pool → (512, 16, 16)
            ↓ Bottleneck
        B:  DoubleConv (1024, 16, 16)
            ↓ Decoder
        D4: Up → cat(skip4) → DoubleConv (512, 32, 32)
        D3: Up → cat(skip3) → DoubleConv (256, 64, 64)
        D2: Up → cat(skip2) → DoubleConv (128, 128, 128)
        D1: Up → cat(skip1) → DoubleConv (64, 256, 256)
            ↓ Output head
        1×1 Conv → (1, 256, 256)  raw logits

    Args:
        in_channels: number of input bands (2 for VV+VH SAR)
        features:    list of feature channel counts at each encoder level
    """

    def __init__(self, in_channels: int = 2,
                 features: list = [64, 128, 256, 512]):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(EncoderBlock(ch, f))
            ch = f

        # ── Bottleneck ───────────────────────────────────────────
        # Deepest part of the network — no spatial change, just more channels
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        ch = features[-1] * 2   # 1024

        # ── Decoder ──────────────────────────────────────────────
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.decoders.append(DecoderBlock(ch, f))
            ch = f

        # ── Output head ──────────────────────────────────────────
        # 1×1 conv collapses 64 channels → 1 channel (binary mask logits)
        self.output_conv = nn.Conv2d(features[0], 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encoder pass — collect skip connections ───────────────
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)

        # ── Bottleneck ───────────────────────────────────────────
        x = self.bottleneck(x)

        # ── Decoder pass — use skip connections in reverse order ──
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # ── Output ───────────────────────────────────────────────
        return self.output_conv(x)   # raw logits, shape (B, 1, H, W)


# ─── MODEL FACTORY ───────────────────────────────────────────────────────────

def get_model(model_type: str = "smp",
              in_channels: int = 2) -> nn.Module:
    """
    Return the chosen model.

    Args:
        model_type:  "smp" for pre-trained U-Net (recommended)
                     "custom" for the from-scratch implementation
        in_channels: number of input bands

    Returns:
        PyTorch model (not yet moved to device)
    """
    if model_type == "smp":
        return get_smp_unet(in_channels=in_channels)
    elif model_type == "custom":
        model = UNet(in_channels=in_channels)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Custom U-Net | Parameters: {n_params:,}")
        return model
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose 'smp' or 'custom'.")


# ─── SANITY CHECK ─────────────────────────────────────────────────────────────

def run_sanity_check():
    """
    Verify both models accept the correct input shape and produce
    the correct output shape without crashing.

    Run with:
        python src/model.py
    """
    print("=" * 60)
    print("Model Sanity Check")
    print("=" * 60)

    # Fake batch: 2 samples, 2 bands, 256x256 patches
    dummy_input = torch.randn(2, 2, 256, 256)
    print(f"\nDummy input shape: {dummy_input.shape}")

    for model_type in ["custom", "smp"]:
        print(f"\n── {model_type.upper()} U-Net ──")
        model = get_model(model_type=model_type, in_channels=2)
        model.eval()

        with torch.no_grad():
            output = model(dummy_input)

        print(f"  Output shape : {output.shape}")
        print(f"  Output range : [{output.min():.3f}, {output.max():.3f}]  (raw logits)")

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(output)
        print(f"  After sigmoid: [{probs.min():.3f}, {probs.max():.3f}]  (should be 0-1)")

        # Verify output shape is correct
        assert output.shape == (2, 1, 256, 256), \
            f"Expected (2, 1, 256, 256) but got {output.shape}"
        print(f"  Shape check  : passed")

    print("\nBoth models working correctly — ready for training.")


if __name__ == "__main__":
    run_sanity_check()
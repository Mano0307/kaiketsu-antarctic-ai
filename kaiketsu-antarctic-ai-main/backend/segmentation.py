"""
Module 2: Sea-Ice Segmentation — U-Net (water vs sea-ice mask)
Kaiketsu | SIH 2026 | Problem ID 26059
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── U-Net Architecture ────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU blocks."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class Down(nn.Module):
    """MaxPool + DoubleConv (encoder step)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)


class Up(nn.Module):
    """Bilinear upsample + skip connection + DoubleConv (decoder step)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # Pad if needed
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        x  = F.pad(x, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """
    U-Net for sea-ice segmentation.
    Output channels:
      0 → Open Water
      1 → Sea Ice
      2 → Ice Lead / Polynya
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 3, base_features: int = 64):
        super().__init__()
        f = base_features
        self.inc   = DoubleConv(in_channels, f)
        self.down1 = Down(f,   f*2)
        self.down2 = Down(f*2, f*4)
        self.down3 = Down(f*4, f*8)
        self.down4 = Down(f*8, f*16)
        self.up1   = Up(f*16 + f*8, f*8)
        self.up2   = Up(f*8  + f*4, f*4)
        self.up3   = Up(f*4  + f*2, f*2)
        self.up4   = Up(f*2  + f,   f)
        self.outc  = nn.Conv2d(f, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


# ── Segmentation Pipeline ─────────────────────────────────────────────────────
CLASS_NAMES = {0: "Open Water", 1: "Sea Ice", 2: "Ice Lead/Polynya"}
CLASS_COLORS = {0: [5,13,31], 1: [0,180,216], 2: [0,245,212]}


def segment_image(
    image: np.ndarray,
    model: UNet = None,
    device: str = "cpu",
    threshold: float = 0.5,
) -> Tuple[np.ndarray, Dict]:
    """
    Run U-Net sea-ice segmentation on a preprocessed SAR image.

    Args:
        image: 2D float32 numpy array (H, W) — output from preprocessing module.
        model: Pre-trained UNet instance. If None, uses untrained (random) weights.
        device: 'cpu' or 'cuda'.
        threshold: Confidence threshold for class assignment.

    Returns:
        mask: 2D int array (H, W) with class IDs 0=water, 1=ice, 2=lead.
        stats: Dict with percentage per class.
    """
    if model is None:
        logger.warning("No trained model provided. Using untrained U-Net for demo.")
        model = UNet(in_channels=1, num_classes=3)
    model.eval()

    # Normalize
    img_norm = (image - image.mean()) / (image.std() + 1e-8)
    tensor   = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

    with torch.no_grad():
        logits = model(tensor.to(device))            # (1, 3, H, W)
        probs  = torch.softmax(logits, dim=1)        # (1, 3, H, W)
        mask   = probs.argmax(dim=1).squeeze(0)      # (H, W)
    mask_np = mask.cpu().numpy().astype(np.uint8)

    total = mask_np.size
    stats = {
        CLASS_NAMES[c]: float((mask_np == c).sum() / total * 100)
        for c in range(3)
    }
    logger.info(f"Segmentation complete: {stats}")
    return mask_np, stats


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert class mask to RGB visualization."""
    H, W = mask.shape
    rgb  = np.zeros((H, W, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[mask == cls] = color
    return rgb


# ── Training Helper ───────────────────────────────────────────────────────────
def train_unet(
    model: UNet,
    train_loader,
    num_epochs: int = 20,
    learning_rate: float = 1e-4,
    device: str = "cpu",
) -> UNet:
    """
    Train U-Net sea-ice segmentation model.

    Args:
        model: UNet instance.
        train_loader: PyTorch DataLoader yielding (image, mask) tuples.
            image shape: (B,1,H,W), mask shape: (B,H,W) with class IDs.
        num_epochs: Training epochs.
        learning_rate: Adam optimizer learning rate.
        device: 'cpu' or 'cuda'.
    Returns:
        Trained model.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images = images.to(device, dtype=torch.float32)
            masks  = masks.to(device,  dtype=torch.long)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / max(len(train_loader), 1)
        logger.info(f"[U-Net] Epoch {epoch+1}/{num_epochs} — Loss: {avg:.4f}")

    return model


def save_model(model: UNet, path: str = "models/unet_seaice.pth"):
    """Save trained model weights."""
    import os; os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    logger.info(f"U-Net model saved: {path}")


def load_model(path: str = "models/unet_seaice.pth", device: str = "cpu") -> UNet:
    """Load trained U-Net model weights."""
    model = UNet(in_channels=1, num_classes=3)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    logger.info(f"U-Net model loaded: {path}")
    return model


# ── CLI Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Kaiketsu Sea-Ice Segmentation Test ===")
    from preprocessing import generate_synthetic_sar, preprocess_sar_image

    raw  = generate_synthetic_sar(shape=(256, 256))
    proc, _ = preprocess_sar_image(raw, filter_size=5)

    model = UNet(in_channels=1, num_classes=3)
    mask, stats = segment_image(proc, model=model)

    print(f"[+] Segmentation mask shape: {mask.shape}")
    print(f"[+] Class distribution: {stats}")
    rgb = mask_to_rgb(mask)
    print(f"[+] RGB mask shape: {rgb.shape}")

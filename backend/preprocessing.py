"""
Module 1: Preprocessing — SAR Image Calibration, Lee Filter & Geocoding
Kaiketsu | SIH 2026 | Problem ID 26059
"""

import numpy as np
import cv2
import os
import logging
from typing import Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Lee Filter (Speckle Noise Reduction) ──────────────────────────────────────
def lee_filter(image: np.ndarray, filter_size: int = 7) -> np.ndarray:
    """
    Apply Lee speckle filter to a SAR image.
    Reduces coherent speckle noise inherent in synthetic aperture radar imagery.

    Args:
        image: 2D numpy array (single band SAR image, float32).
        filter_size: Kernel size (odd integer). Default 7.
    Returns:
        Filtered image (float32).
    """
    if image.ndim == 3:
        image = image[:, :, 0]
    image = image.astype(np.float32)

    half = filter_size // 2
    rows, cols = image.shape
    output = np.zeros_like(image)

    # Pad image to handle borders
    padded = np.pad(image, half, mode='reflect')

    for r in range(rows):
        for c in range(cols):
            window = padded[r:r + filter_size, c:c + filter_size]
            mean   = np.mean(window)
            var    = np.var(window)
            noise_var = np.mean(image) ** 2 / max(np.var(image), 1e-8)
            weight = var / (var + noise_var)
            output[r, c] = mean + weight * (image[r, c] - mean)

    logger.info(f"Lee filter applied (kernel={filter_size}x{filter_size})")
    return output


def fast_lee_filter(image: np.ndarray, filter_size: int = 7) -> np.ndarray:
    """
    Fast Lee filter using vectorized OpenCV operations.
    Preferred for production — much faster than the reference loop above.
    """
    if image.ndim == 3:
        image = image[:, :, 0].astype(np.float32)
    else:
        image = image.astype(np.float32)

    img2    = image ** 2
    mean    = cv2.boxFilter(image, -1, (filter_size, filter_size))
    sq_mean = cv2.boxFilter(img2,   -1, (filter_size, filter_size))
    var     = sq_mean - mean ** 2

    noise_var = np.mean(var)
    weight    = var / (var + noise_var + 1e-8)
    filtered  = mean + weight * (image - mean)
    logger.info("Fast Lee filter applied.")
    return filtered


# ── Radiometric Calibration ───────────────────────────────────────────────────
def radiometric_calibration(
    image: np.ndarray,
    calibration_factor: float = 1.0,
    offset_db: float = -83.0
) -> np.ndarray:
    """
    Calibrate raw SAR digital numbers (DN) to sigma-naught (σ°) in dB.

    For Sentinel-1:
        σ°(dB) = 10 * log10(DN²) + offset_db

    Args:
        image: Raw DN image (float32).
        calibration_factor: Sensor-specific factor (default 1.0).
        offset_db: Calibration offset in dB (Sentinel-1: ~-83 dB).
    Returns:
        Calibrated sigma-naught image in dB.
    """
    image = image.astype(np.float32)
    image = np.clip(image, 1e-6, None)  # avoid log(0)
    sigma = 10.0 * np.log10(image ** 2 * calibration_factor) + offset_db
    logger.info(f"Radiometric calibration done. σ° range: [{sigma.min():.2f}, {sigma.max():.2f}] dB")
    return sigma


# ── Geocoding (Simulated — requires GDAL for real geolocation) ────────────────
def geocode_image(
    image: np.ndarray,
    target_epsg: int = 4326,
    pixel_size_m: float = 10.0
) -> Tuple[np.ndarray, dict]:
    """
    Apply basic geocoding metadata to SAR image.
    Real implementation uses GDAL/SNAP for slant-range to ground-range projection.

    Returns:
        (geocoded_image, metadata_dict)
    """
    metadata = {
        "epsg": target_epsg,
        "pixel_size_m": pixel_size_m,
        "rows": image.shape[0],
        "cols": image.shape[1],
        "sensor": "Sentinel-1 SAR (C-band)",
        "polarization": "VV+VH",
        "projection": f"EPSG:{target_epsg}",
    }
    # Spatial resampling (demonstration — actual uses GDAL warp)
    geocoded = cv2.resize(image, (image.shape[1], image.shape[0]))
    logger.info(f"Geocoding complete. EPSG:{target_epsg}, pixel_size={pixel_size_m}m")
    return geocoded, metadata


# ── Contrast Enhancement ──────────────────────────────────────────────────────
def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE contrast-limited adaptive histogram equalization."""
    norm = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(norm)
    return enhanced.astype(np.float32) / 255.0


# ── Full Preprocessing Pipeline ───────────────────────────────────────────────
def preprocess_sar_image(
    image: np.ndarray,
    filter_size: int = 7,
    calibration_factor: float = 1.0,
    calibrate: bool = True,
    geocode: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Full SAR preprocessing pipeline:
    1. Lee filter (speckle removal)
    2. Radiometric calibration → σ° dB
    3. Contrast enhancement
    4. Geocoding metadata

    Args:
        image: Raw SAR image array.
        filter_size: Lee filter kernel size.
        calibration_factor: Radiometric calibration factor.
        calibrate: Apply radiometric calibration.
        geocode: Attach geocoding metadata.
    Returns:
        (processed_image, info_dict)
    """
    info = {}
    logger.info("=== Preprocessing Module START ===")

    # Step 1: Lee Filter
    filtered = fast_lee_filter(image, filter_size=filter_size)
    info["noise_reduction_applied"] = True
    info["filter"] = f"Lee filter ({filter_size}x{filter_size})"

    # Step 2: Calibration
    if calibrate:
        calibrated = radiometric_calibration(filtered, calibration_factor)
        info["calibration"] = "σ° dB (Sentinel-1 standard)"
        info["sigma_min_db"] = float(calibrated.min())
        info["sigma_max_db"] = float(calibrated.max())
    else:
        calibrated = filtered
        info["calibration"] = "skipped"

    # Step 3: Contrast enhancement
    enhanced = enhance_contrast(calibrated)

    # Step 4: Geocoding
    if geocode:
        enhanced, geo_meta = geocode_image(enhanced)
        info.update(geo_meta)

    logger.info("=== Preprocessing Module DONE ===")
    return enhanced, info


# ── Load Image Helper ─────────────────────────────────────────────────────────
def load_sar_image(path: str) -> Optional[np.ndarray]:
    """Load SAR image from file path (.tif, .png, .jpg). Returns float32 array."""
    if not os.path.exists(path):
        logger.error(f"File not found: {path}")
        return None
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error(f"Failed to read image: {path}")
        return None
    return img.astype(np.float32)


# ── Synthetic Test Image ──────────────────────────────────────────────────────
def generate_synthetic_sar(
    shape: Tuple[int,int] = (512, 512),
    num_icebergs: int = 5,
    add_ship: bool = True,
    seed: int = 42
) -> np.ndarray:
    """
    Generate a synthetic SAR-like image for testing.
    Dark background (ocean), bright blobs (icebergs/ships), speckle noise.
    """
    np.random.seed(seed)
    img = np.random.gamma(1.5, 10, shape).astype(np.float32)  # speckle

    # Sea ice patches
    for _ in range(6):
        cx, cy = np.random.randint(50,462), np.random.randint(50,462)
        for x in range(max(0,cx-60), min(512,cx+60)):
            for y in range(max(0,cy-40), min(512,cy+40)):
                if ((x-cx)/60)**2 + ((y-cy)/40)**2 < 1:
                    img[y,x] += 80 + np.random.randn()*10

    # Icebergs (bright)
    for _ in range(num_icebergs):
        cx, cy = np.random.randint(30,480), np.random.randint(30,480)
        r = np.random.randint(8,22)
        Y,X = np.ogrid[:512,:512]
        mask = (X-cx)**2 + (Y-cy)**2 < r**2
        img[mask] += 200 + np.random.randn() * 15

    # Ship (smaller, different signature)
    if add_ship:
        cx, cy = np.random.randint(100,400), np.random.randint(100,400)
        img[cy-4:cy+4, cx-10:cx+10] += 220  # very bright, small, elongated

    return np.clip(img, 0, 255)


# ── CLI Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Kaiketsu Preprocessing Module Test ===")

    # Synthetic image
    raw_img = generate_synthetic_sar(shape=(512, 512), num_icebergs=5, add_ship=True)
    print(f"[+] Synthetic SAR image: shape={raw_img.shape}, "
          f"min={raw_img.min():.1f}, max={raw_img.max():.1f}")

    # Run full preprocessing
    processed, info = preprocess_sar_image(raw_img, filter_size=7)
    print(f"[+] Preprocessed image: shape={processed.shape}")
    print(f"[+] Info: {info}")

    # Save result
    out_path = "preprocessed_sar.png"
    cv2.imwrite(out_path, (processed * 255).astype(np.uint8))
    print(f"[+] Saved: {out_path}")

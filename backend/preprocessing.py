"""
Module 1: Preprocessing — SAR Image Calibration, Lee Filter & Geocoding
Kaiketsu | SIH 2026 | Problem ID 26059
"""

import numpy as np
import cv2
import os
import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict

try:
    from dataset_loader import PROJECT_DATA_DIR
except ModuleNotFoundError:  # pragma: no cover - local direct execution fallback
    PROJECT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

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


def collect_dataset_images(dataset_dir: str) -> List[str]:
    """Collect image files inside a dataset folder recursively."""
    if not dataset_dir or not os.path.exists(dataset_dir):
        return []

    supported_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    files: List[str] = []
    for root, _, filenames in os.walk(dataset_dir):
        for filename in filenames:
            if Path(filename).suffix.lower() in supported_exts:
                files.append(os.path.join(root, filename))
    return sorted(files)


def process_dataset_folder(
    dataset_dir: str,
    output_dir: str,
    filter_size: int = 7,
    calibrate: bool = True,
    geocode: bool = True,
) -> Dict[str, object]:
    """Preprocess every image in a dataset folder and save the processed results.

    Each dataset gets its own output subfolder. The function processes all images
    and returns a summary with the exact count of images processed.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    image_paths = collect_dataset_images(dataset_dir)
    summary: Dict[str, object] = {
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "total_images": len(image_paths),
        "processed": 0,
        "failed": 0,
        "saved_paths": [],
    }

    for image_path in image_paths:
        try:
            image = load_sar_image(image_path)
            if image is None:
                summary["failed"] += 1
                continue

            processed, _ = preprocess_sar_image(
                image,
                filter_size=filter_size,
                calibrate=calibrate,
                geocode=geocode,
            )

            rel_path = os.path.relpath(image_path, dataset_dir)
            base_name = os.path.splitext(os.path.basename(rel_path))[0]
            dataset_output_dir = os.path.join(output_dir, os.path.basename(dataset_dir))
            os.makedirs(dataset_output_dir, exist_ok=True)
            output_path = os.path.join(dataset_output_dir, f"{base_name}_preprocessed.png")
            cv2.imwrite(output_path, (processed * 255).astype(np.uint8))

            summary["processed"] += 1
            summary["saved_paths"].append(output_path)
            logger.info(f"Processed {image_path} -> {output_path}")
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.exception(f"Error while preprocessing {image_path}: {exc}")
            summary["failed"] += 1

    return summary


# ── CLI Test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Kaiketsu Preprocessing Module Test ===")

    dataset_root = os.environ.get("KAIKETSU_DATASET_DIR", PROJECT_DATA_DIR)
    output_root = os.path.join(os.path.dirname(__file__), "preprocessed_output")

    if os.path.exists(dataset_root):
        image_files = collect_dataset_images(dataset_root)
        print(f"[+] Detected dataset images under {dataset_root}: {len(image_files)} files")

        if image_files:
            dataset_dirs = [
                os.path.join(dataset_root, name)
                for name in sorted(os.listdir(dataset_root))
                if os.path.isdir(os.path.join(dataset_root, name))
            ]

            if not dataset_dirs:
                dataset_dirs = [dataset_root]

            all_processed = 0
            all_failed = 0
            all_saved = []

            for dataset_dir in dataset_dirs:
                summary = process_dataset_folder(
                    dataset_dir=dataset_dir,
                    output_dir=output_root,
                    filter_size=7,
                    calibrate=True,
                    geocode=True,
                )
                all_processed += int(summary["processed"])
                all_failed += int(summary["failed"])
                all_saved.extend(summary["saved_paths"])
                print(f"[+] Dataset: {dataset_dir}")
                print(f"    total images: {summary['total_images']}")
                print(f"    processed: {summary['processed']}")
                print(f"    failed: {summary['failed']}")
                print(f"    output folder: {summary['output_dir']}")

            print(f"[+] TOTAL images processed: {all_processed}")
            print(f"[+] TOTAL images failed: {all_failed}")
            print(f"[+] TOTAL saved outputs: {len(all_saved)}")
        else:
            print("[!] No image files found in dataset directory. Falling back to synthetic demo image.")
            raw_img = generate_synthetic_sar(shape=(512, 512), num_icebergs=5, add_ship=True)
            processed, info = preprocess_sar_image(raw_img, filter_size=7)
            out_path = os.path.join(os.path.dirname(__file__), "preprocessed_sar.png")
            cv2.imwrite(out_path, (processed * 255).astype(np.uint8))
            print(f"[+] Synthetic image processed and saved to: {out_path}")
            print(f"[+] Info: {info}")
    else:
        print(f"[!] Dataset directory not found: {dataset_root}")
        print("[+] Running synthetic demo image instead.")
        raw_img = generate_synthetic_sar(shape=(512, 512), num_icebergs=5, add_ship=True)
        processed, info = preprocess_sar_image(raw_img, filter_size=7)
        out_path = os.path.join(os.path.dirname(__file__), "preprocessed_sar.png")
        cv2.imwrite(out_path, (processed * 255).astype(np.uint8))
        print(f"[+] Synthetic image processed and saved to: {out_path}")
        print(f"[+] Info: {info}")

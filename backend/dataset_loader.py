"""
Dataset Loader & Downloader Engine — Antarctic Ocean & Ice Data Registry
Kaiketsu | SIH 2026 | Problem ID 26059

This module keeps the project's dataset catalog and supports simulated downloads
for the materials used by the Antarctic navigation AI pipeline.
"""

import os
import logging
import shutil
import subprocess
from urllib.parse import urlparse
from urllib.request import urlretrieve
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")

REAL_DATASET_URLS = {
    "circ_antarctic_iceberg": [
        "https://www.nature.com/articles/s41561-024-01472-2",
    ],
    "grounded_iceberg": [
        "https://nsidc.org/",
    ],
    "sentinel1_sar": [
        "https://dataspace.copernicus.eu/",
    ],
    "amsr2_osi_saf": [
        "https://osi-saf.org/",
    ],
    "era5": [
        "https://cds.climate.copernicus.eu/",
    ],
    "copernicus_marine": [
        "https://marine.copernicus.eu/",
    ],
    "gebco": [
        "https://www.gebco.net/data_and_products/gridded_bathymetry_data/",
    ],
    "statoil_iceberg_kaggle": [
        "https://www.kaggle.com/c/statoil-iceberg-classifier-challenge",
    ],
}

LOCAL_DATASET_PATHS = {
    "circ_antarctic_iceberg": os.path.join(PROJECT_DATA_DIR, "circ_antarctic_iceberg"),
    "grounded_iceberg": os.path.join(PROJECT_DATA_DIR, "grounded_iceberg"),
    "sentinel1_sar": os.path.join(PROJECT_DATA_DIR, "sentinel1_sar"),
    "amsr2_osi_saf": os.path.join(PROJECT_DATA_DIR, "amsr2_osi_saf"),
    "era5": os.path.join(PROJECT_DATA_DIR, "era5"),
    "copernicus_marine": os.path.join(PROJECT_DATA_DIR, "copernicus_marine"),
    "gebco": os.path.join(PROJECT_DATA_DIR, "gebco"),
    "statoil_iceberg_kaggle": os.path.join(PROJECT_DATA_DIR, "statoil_iceberg_kaggle"),
}

DATASET_REGISTRY = {
    "circ_antarctic_iceberg": {
        "name": "Circum-Antarctic iceberg dataset",
        "source": "Public polar / iceberg survey datasets",
        "purpose": "Iceberg location and size labels",
        "url": REAL_DATASET_URLS["circ_antarctic_iceberg"][0],
        "size": "~2.5GB",
        "format": "CSV / GeoJSON / image annotations",
        "description": "Provides iceberg positions and size labels used for detection training and validation.",
        "download_type": "url",
        "kaggle_slug": None,
        "copernicus_product": None,
        "local_source": LOCAL_DATASET_PATHS["circ_antarctic_iceberg"],
    },
    "grounded_iceberg": {
        "name": "Grounded iceberg dataset",
        "source": "Polar field / remote sensing repositories",
        "purpose": "Small and stationary iceberg detection",
        "url": REAL_DATASET_URLS["grounded_iceberg"][0],
        "size": "~600MB",
        "format": "Raster + labels",
        "description": "Used to detect grounded, small, and stationary icebergs that may be missed by general models.",
        "download_type": "url",
        "kaggle_slug": None,
        "copernicus_product": None,
        "local_source": LOCAL_DATASET_PATHS["grounded_iceberg"],
    },
    "sentinel1_sar": {
        "name": "Sentinel-1 SAR images",
        "source": "Copernicus / ESA",
        "purpose": "Raw training images",
        "url": REAL_DATASET_URLS["sentinel1_sar"][0],
        "size": "~4GB+",
        "format": "GeoTIFF / Sentinel-1 products",
        "description": "Primary SAR imagery for sea-ice and iceberg analysis.",
        "download_type": "copernicus",
        "kaggle_slug": None,
        "copernicus_product": "COPERNICUS_SENTINEL_1",
        "local_source": LOCAL_DATASET_PATHS["sentinel1_sar"],
    },
    "amsr2_osi_saf": {
        "name": "AMSR2 / OSI SAF sea-ice concentration",
        "source": "EUMETSAT / OSI SAF",
        "purpose": "Sea-ice concentration",
        "url": REAL_DATASET_URLS["amsr2_osi_saf"][0],
        "size": "~1.5GB",
        "format": "NetCDF / GRIB",
        "description": "Provides sea-ice concentration data for environmental context and forecasting.",
        "download_type": "url",
        "kaggle_slug": None,
        "copernicus_product": None,
        "local_source": LOCAL_DATASET_PATHS["amsr2_osi_saf"],
    },
    "era5": {
        "name": "ERA5 atmospheric reanalysis",
        "source": "ECMWF",
        "purpose": "Wind, air temperature, and pressure",
        "url": REAL_DATASET_URLS["era5"][0],
        "size": "~800MB",
        "format": "NetCDF",
        "description": "Supports meteorological features used in risk and route planning models.",
        "download_type": "copernicus",
        "kaggle_slug": None,
        "copernicus_product": "ERA5_LAND",
        "local_source": LOCAL_DATASET_PATHS["era5"],
    },
    "copernicus_marine": {
        "name": "Copernicus Marine data",
        "source": "Copernicus Marine Service",
        "purpose": "Currents, waves, and sea-surface temperature",
        "url": REAL_DATASET_URLS["copernicus_marine"][0],
        "size": "~1GB",
        "format": "NetCDF / GRIB",
        "description": "Ocean state variables used for hazard and trajectory models.",
        "download_type": "copernicus",
        "kaggle_slug": None,
        "copernicus_product": "GLOBAL_ANALYSIS_FORECAST_PHY_001_024",
        "local_source": LOCAL_DATASET_PATHS["copernicus_marine"],
    },
    "gebco": {
        "name": "GEBCO bathymetry",
        "source": "GEBCO",
        "purpose": "Water depth and grounding risk",
        "url": REAL_DATASET_URLS["gebco"][0],
        "size": "~600MB",
        "format": "GeoTIFF / gridded bathymetry",
        "description": "Bathymetry metadata for grounding risk and safe route planning.",
        "download_type": "url",
        "kaggle_slug": None,
        "copernicus_product": None,
        "local_source": LOCAL_DATASET_PATHS["gebco"],
    },
    "statoil_iceberg_kaggle": {
        "name": "Statoil Iceberg Kaggle dataset",
        "source": "Kaggle",
        "purpose": "Synthetic and real iceberg classification training data",
        "url": REAL_DATASET_URLS["statoil_iceberg_kaggle"][0],
        "size": "~2.5GB",
        "format": "CSV + images",
        "description": "Kaggle-hosted iceberg recognition dataset used for training detection and classification models.",
        "download_type": "kaggle",
        "kaggle_slug": "statoil-iceberg-classifier-challenge",
        "copernicus_product": None,
        "local_source": LOCAL_DATASET_PATHS["statoil_iceberg_kaggle"],
    },
}


def list_datasets() -> Dict[str, Dict]:
    """Return the complete dataset registry."""
    return DATASET_REGISTRY


def _copy_local_dataset(source_path: str, target_dir: str) -> str:
    """Copy a local folder or file into the project data directory."""
    os.makedirs(target_dir, exist_ok=True)

    if os.path.isdir(source_path):
        dest = os.path.join(target_dir, os.path.basename(source_path))
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(source_path, dest)
        return dest

    filename = os.path.basename(source_path)
    destination = os.path.join(target_dir, filename)
    shutil.copy2(source_path, destination)
    return destination


def _download_via_url(source_url: str, target_dir: str) -> str:
    """Download a single file from a URL into the target directory."""
    os.makedirs(target_dir, exist_ok=True)
    parsed = urlparse(source_url)
    filename = os.path.basename(parsed.path) or "downloaded_file.bin"
    target_path = os.path.join(target_dir, filename)
    logger.info(f"Downloading from URL: {source_url}")
    urlretrieve(source_url, target_path)
    return target_path


def _download_from_kaggle(dataset_key: str, save_dir: str, kaggle_slug: Optional[str]) -> Dict:
    """Download a Kaggle dataset using kaggle CLI if credentials are available."""
    if not kaggle_slug:
        return {"status": "metadata_only", "reason": "No Kaggle slug defined for this dataset."}

    kaggle_cmd = shutil.which("kaggle")
    if not kaggle_cmd:
        return {"status": "metadata_only", "reason": "Kaggle CLI is not installed. Install with: pip install kaggle"}

    username = os.getenv("KAGGLE_USERNAME")
    api_key = os.getenv("KAGGLE_KEY")
    if not username or not api_key:
        return {"status": "metadata_only", "reason": "KAGGLE_USERNAME and KAGGLE_KEY are not set."}

    os.makedirs(save_dir, exist_ok=True)
    command = [kaggle_cmd, "datasets", "download", "-d", kaggle_slug, "-p", save_dir, "--unzip"]
    logger.info(f"Running Kaggle download: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return {"status": "failed", "reason": result.stderr.strip() or result.stdout.strip() or "Kaggle download failed."}

    return {"status": "downloaded", "path": save_dir, "reason": "Kaggle dataset downloaded successfully."}


def _download_from_copernicus(dataset_key: str, save_dir: str, product_name: Optional[str]) -> Dict:
    """Attempt a Copernicus-style API download when credentials are available."""
    if not product_name:
        return {"status": "metadata_only", "reason": "No Copernicus product configured for this dataset."}

    username = os.getenv("COPERNICUS_USERNAME")
    password = os.getenv("COPERNICUS_PASSWORD")
    if not username or not password:
        return {"status": "metadata_only", "reason": "COPERNICUS_USERNAME and COPERNICUS_PASSWORD are not set."}

    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Copernicus product configured: {product_name}. Use a real API client to fetch the dataset.")
    return {
        "status": "metadata_only",
        "reason": "Copernicus API credentials are set, but a real product-call implementation is still required for the exact product endpoint.",
        "product_name": product_name,
    }


def _download_from_url_list(save_dir: str, url_list: Optional[List[str]]) -> List[Dict]:
    """Download one or more files from a list of URLs."""
    results = []
    if not url_list:
        return results

    for url in url_list:
        try:
            target_dir = os.path.join(save_dir, "url_downloads")
            downloaded = _download_via_url(url, target_dir)
            results.append({"status": "downloaded", "path": downloaded, "url": url})
        except Exception as exc:
            results.append({"status": "failed", "url": url, "reason": str(exc)})
    return results


def download_dataset(
    dataset_key: str,
    save_dir: str = "data/",
    local_source: Optional[str] = None,
    url_list: Optional[List[str]] = None,
) -> Dict:
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    ds = DATASET_REGISTRY[dataset_key]
    os.makedirs(save_dir, exist_ok=True)

    source_path = local_source or ds.get("local_source")
    target_dir = os.path.join(save_dir, dataset_key)

    if source_path and os.path.exists(source_path):
        copied_path = _copy_local_dataset(source_path, target_dir)
        logger.info(f"Copied local dataset: {ds['name']} -> {copied_path}")
        return {
            "status": "downloaded",
            "dataset": ds["name"],
            "key": dataset_key,
            "source": ds["source"],
            "purpose": ds["purpose"],
            "size": ds["size"],
            "path": copied_path,
            "download_url": ds.get("url"),
            "source_type": "local_folder",
        }

    if url_list:
        from_url = _download_from_url_list(save_dir, url_list)
        return {
            "status": "downloaded" if any(item["status"] == "downloaded" for item in from_url) else "failed",
            "dataset": ds["name"],
            "key": dataset_key,
            "source": ds["source"],
            "purpose": ds["purpose"],
            "size": ds["size"],
            "path": save_dir,
            "source_type": "url_list",
            "files": from_url,
        }

    if ds.get("download_type") == "kaggle":
        result = _download_from_kaggle(dataset_key, target_dir, ds.get("kaggle_slug"))
        if result.get("status") == "downloaded":
            result["dataset"] = ds["name"]
            result["key"] = dataset_key
            result["source_type"] = "kaggle"
            result["path"] = target_dir
            return result
        return {
            "status": "metadata_only",
            "dataset": ds["name"],
            "key": dataset_key,
            "source": ds["source"],
            "purpose": ds["purpose"],
            "size": ds["size"],
            "path": target_dir,
            "download_url": ds.get("url"),
            "source_type": "kaggle",
            "reason": result.get("reason", "Kaggle API credentials are required."),
        }

    if ds.get("download_type") == "copernicus":
        result = _download_from_copernicus(dataset_key, target_dir, ds.get("copernicus_product"))
        if result.get("status") == "downloaded":
            result["dataset"] = ds["name"]
            result["key"] = dataset_key
            result["source_type"] = "copernicus"
            result["path"] = target_dir
            return result
        return {
            "status": "metadata_only",
            "dataset": ds["name"],
            "key": dataset_key,
            "source": ds["source"],
            "purpose": ds["purpose"],
            "size": ds["size"],
            "path": target_dir,
            "download_url": ds.get("url"),
            "source_type": "copernicus",
            "reason": result.get("reason", "Copernicus credentials are required."),
        }

    if ds.get("url") and "example.com" not in ds["url"]:
        try:
            downloaded_path = _download_via_url(ds["url"], target_dir)
            logger.info(f"Downloaded dataset: {ds['name']} -> {downloaded_path}")
            return {
                "status": "downloaded",
                "dataset": ds["name"],
                "key": dataset_key,
                "source": ds["source"],
                "purpose": ds["purpose"],
                "size": ds["size"],
                "path": downloaded_path,
                "download_url": ds.get("url"),
                "source_type": "url",
            }
        except Exception as exc:
            logger.warning(f"URL download failed for {dataset_key}: {exc}")

    logger.info(f"No valid dataset source found for {dataset_key}. Returning metadata only.")
    output_path = os.path.join(save_dir, dataset_key)
    return {
        "status": "metadata_only",
        "dataset": ds["name"],
        "key": dataset_key,
        "source": ds["source"],
        "purpose": ds["purpose"],
        "size": ds["size"],
        "path": output_path,
        "download_url": ds.get("url"),
        "source_type": "metadata",
    }


def download_all_datasets(save_dir: str = "data/", source_map: Optional[Dict[str, str]] = None, url_map: Optional[Dict[str, List[str]]] = None) -> List[Dict]:
    results = []
    source_map = source_map or {}
    url_map = url_map or {}
    for key in DATASET_REGISTRY:
        local_source = source_map.get(key)
        url_list = url_map.get(key)
        results.append(download_dataset(key, save_dir, local_source=local_source, url_list=url_list))
    return results


if __name__ == "__main__":
    datasets = list_datasets()
    print("[+] Available datasets:")
    for key, meta in datasets.items():
        print(f"  - {key}: {meta['name']} | purpose={meta['purpose']} | source={meta['source']} | size={meta['size']} | type={meta.get('download_type')}")

    print("\n[+] Project data directory:", PROJECT_DATA_DIR)
    print("\n[+] Example local-folder copy config:")
    print("    result = download_dataset('sentinel1_sar', save_dir='data', local_source=r'C:\\datasets\\sentinel1_sar')")
    print("\n[+] Example URL list config:")
    print("    urls = ['https://dataspace.copernicus.eu/', 'https://osi-saf.org/']")
    print("    result = download_dataset('sentinel1_sar', save_dir='data', url_list=urls)")

    print("\n[+] Checking dataset sources...")
    results = download_all_datasets(save_dir="data")
    for result in results:
        print(f"  - {result['key']}: {result['status']} | path={result['path']} | source_type={result.get('source_type')}")

def _copy_local_dataset(source_path: str, target_dir: str) -> str:
    """Copy a local folder or file into the project data directory."""
    os.makedirs(target_dir, exist_ok=True)

    if os.path.isdir(source_path):
        dest = os.path.join(target_dir, os.path.basename(source_path))
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(source_path, dest)
        return dest

    filename = os.path.basename(source_path)
    destination = os.path.join(target_dir, filename)
    shutil.copy2(source_path, destination)
    return destination


def _download_via_url(source_url: str, target_dir: str) -> str:
    """Download a single file from a URL into the target directory."""
    os.makedirs(target_dir, exist_ok=True)
    parsed = urlparse(source_url)
    filename = os.path.basename(parsed.path) or "downloaded_file.bin"
    target_path = os.path.join(target_dir, filename)
    logger.info(f"Downloading from URL: {source_url}")
    urlretrieve(source_url, target_path)
    return target_path


def _download_from_kaggle(dataset_key: str, save_dir: str, kaggle_slug: Optional[str]) -> Dict:
    """Download a Kaggle dataset using kaggle CLI if credentials are available."""
    if not kaggle_slug:
        return {
            "status": "metadata_only",
            "reason": "No Kaggle slug defined for this dataset.",
        }

    kaggle_cmd = shutil.which("kaggle")
    if not kaggle_cmd:
        return {
            "status": "metadata_only",
            "reason": "Kaggle CLI is not installed. Install with: pip install kaggle",
        }

    username = os.getenv("KAGGLE_USERNAME")
    api_key = os.getenv("KAGGLE_KEY")
    if not username or not api_key:
        return {
            "status": "metadata_only",
            "reason": "KAGGLE_USERNAME and KAGGLE_KEY are not set.",
        }

    os.makedirs(save_dir, exist_ok=True)
    command = [
        kaggle_cmd,
        "datasets",
        "download",
        "-d",
        kaggle_slug,
        "-p",
        save_dir,
        "--unzip",
    ]
    logger.info(f"Running Kaggle download: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        return {
            "status": "failed",
            "reason": result.stderr.strip() or result.stdout.strip() or "Kaggle download failed.",
        }

    return {
        "status": "downloaded",
        "path": save_dir,
        "reason": "Kaggle dataset downloaded successfully.",
    }


def _download_from_copernicus(dataset_key: str, save_dir: str, product_name: Optional[str]) -> Dict:
    """Attempt a Copernicus-style API download when credentials are available."""
    if not product_name:
        return {
            "status": "metadata_only",
            "reason": "No Copernicus product configured for this dataset.",
        }

    username = os.getenv("COPERNICUS_USERNAME")
    password = os.getenv("COPERNICUS_PASSWORD")
    if not username or not password:
        return {
            "status": "metadata_only",
            "reason": "COPERNICUS_USERNAME and COPERNICUS_PASSWORD are not set.",
        }

    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Copernicus product configured: {product_name}. Use a real API client to fetch the dataset.")
    return {
        "status": "metadata_only",
        "reason": "Copernicus API credentials are set, but this demo loader requires the real product download call to be implemented for the exact dataset endpoint.",
        "product_name": product_name,
    }


def _download_from_url_list(save_dir: str, url_list: Optional[List[str]]) -> List[Dict]:
    """Download one or more files from a given list of URLs."""
    results = []
    if not url_list:
        return results

    for url in url_list:
        try:
            target_dir = os.path.join(save_dir, "url_downloads")
            downloaded = _download_via_url(url, target_dir)
            results.append({"status": "downloaded", "path": downloaded, "url": url})
        except Exception as exc:
            results.append({"status": "failed", "url": url, "reason": str(exc)})
    return results


def download_dataset(
    dataset_key: str,
    save_dir: str = "data/",
    local_source: Optional[str] = None,
    url_list: Optional[List[str]] = None,
) -> Dict:
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    ds = DATASET_REGISTRY[dataset_key]
    os.makedirs(save_dir, exist_ok=True)

    source_path = local_source or ds.get("local_source")
    target_dir = os.path.join(save_dir, dataset_key)

    if source_path and os.path.exists(source_path):
        copied_path = _copy_local_dataset(source_path, target_dir)
        logger.info(f"Copied local dataset: {ds['name']} -> {copied_path}")
        return {
            "status": "downloaded",
            "dataset": ds["name"],
            "key": dataset_key,
            "source": ds["source"],
            "purpose": ds["purpose"],
            "size": ds["size"],
            "path": copied_path,
            "download_url": ds.get("url"),
            "source_type": "local_folder",
        }

    if url_list:
        from_url = _download_from_url_list(save_dir, url_list)
        return {
            "status": "downloaded" if any(item["status"] == "downloaded" for item in from_url) else "failed",
            "dataset": ds["name"],
            "key": dataset_key,
            "source": ds["source"],
            "purpose": ds["purpose"],
            "size": ds["size"],
            "path": save_dir,
            "source_type": "url_list",
            "files": from_url,
        }

    if ds.get("download_type") == "kaggle":
        result = _download_from_kaggle(dataset_key, target_dir, ds.get("kaggle_slug"))
        if result.get("status") == "downloaded":
            result["dataset"] = ds["name"]
            result["key"] = dataset_key
            result["source_type"] = "kaggle"
            result["path"] = target_dir
            return result
        return {
            "status": "metadata_only",
            "dataset": ds["name"],
            "key": dataset_key,
            "source": ds["source"],
            "purpose": ds["purpose"],
            "size": ds["size"],
            "path": target_dir,
            "download_url": ds.get("url"),
            "source_type": "kaggle",
            "reason": result.get("reason", "Kaggle API credentials are required."),
        }

    if ds.get("download_type") == "copernicus":
        result = _download_from_copernicus(dataset_key, target_dir, ds.get("copernicus_product"))
        if result.get("status") == "downloaded":
            result["dataset"] = ds["name"]
            result["key"] = dataset_key
            result["source_type"] = "copernicus"
            result["path"] = target_dir
            return result
        return {
            "status": "metadata_only",
            "dataset": ds["name"],
            "key": dataset_key,
            "source": ds["source"],
            "purpose": ds["purpose"],
            "size": ds["size"],
            "path": target_dir,
            "download_url": ds.get("url"),
            "source_type": "copernicus",
            "reason": result.get("reason", "Copernicus credentials are required."),
        }

    if ds.get("url") and "example.com" not in ds["url"]:
        try:
            downloaded_path = _download_via_url(ds["url"], target_dir)
            logger.info(f"Downloaded dataset: {ds['name']} -> {downloaded_path}")
            return {
                "status": "downloaded",
                "dataset": ds["name"],
                "key": dataset_key,
                "source": ds["source"],
                "purpose": ds["purpose"],
                "size": ds["size"],
                "path": downloaded_path,
                "download_url": ds.get("url"),
                "source_type": "url",
            }
        except Exception as exc:
            logger.warning(f"URL download failed for {dataset_key}: {exc}")

    logger.info(f"No valid dataset source found for {dataset_key}. Returning metadata only.")
    output_path = os.path.join(save_dir, dataset_key)
    return {
        "status": "metadata_only",
        "dataset": ds["name"],
        "key": dataset_key,
        "source": ds["source"],
        "purpose": ds["purpose"],
        "size": ds["size"],
        "path": output_path,
        "download_url": ds.get("url"),
        "source_type": "metadata",
    }


def download_all_datasets(save_dir: str = "data/", source_map: Optional[Dict[str, str]] = None, url_map: Optional[Dict[str, List[str]]] = None) -> List[Dict]:
    results = []
    source_map = source_map or {}
    url_map = url_map or {}
    for key in DATASET_REGISTRY:
        local_source = source_map.get(key)
        url_list = url_map.get(key)
        results.append(download_dataset(key, save_dir, local_source=local_source, url_list=url_list))
    return results


if __name__ == "__main__":
    datasets = list_datasets()
    print("[+] Available datasets:")
    for key, meta in datasets.items():
        print(f"  - {key}: {meta['name']} | purpose={meta['purpose']} | source={meta['source']} | size={meta['size']} | type={meta.get('download_type')}")

    print("\n[+] Checking dataset sources...")
    print("Note: local folder and API downloads are supported when credentials or paths are provided.")
    results = download_all_datasets(save_dir="data")
    for result in results:
        print(f"  - {result['key']}: {result['status']} | path={result['path']} | source_type={result.get('source_type')}")

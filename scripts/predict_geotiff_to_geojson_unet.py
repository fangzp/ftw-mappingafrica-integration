#!/usr/bin/env python3
"""
Run UNet inference on GeoTIFF images and convert 3-class semantic output to GeoJSON.

Mirrors the Mask2Former pipeline output (per-image GeoJSON, cell_id, acq_date, image_path)
so ag-seg's aggregate_cell_predictions, compute_field_statistics, etc. work unchanged.

Usage:
    python scripts/predict_geotiff_to_geojson_unet.py \
        --checkpoint /path/to/last.ckpt \
        --input /path/to/geotiffs \
        --output /path/to/output \
        --chip-size 501 \
        --overlap 100 \
        --normalization-stats '{"mean":[2160.11,2720.38,2156.53,4174.56],"std":[1499.02,1508.71,1408.98,1532.44]}' \
        --min-area 100 \
        --max-area 90000 \
        --edge-buffer 5 \
        --iou-threshold 0.7
"""

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from shapely.geometry import box, mapping
from tqdm import tqdm

# ftw_ma imports
from ftw_ma.inference import load_model, _prepare_image_batch
from ftw_ma.normalize import normalize_image
from ftw_ma.uganda_inference_geo import (
    semantic_3class_to_geodataframe,
    filter_edge_polygons_geo,
    resolve_overlaps,
    merge_overlapping_fields,
)

logger = logging.getLogger(__name__)

_MONTH = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_cell_id_and_date(geotiff_path: Path, input_path: Path):
    """Parse cell_id and acq_date from path (same convention as ag-seg pipeline)."""
    cell_id, acq_date = None, None
    for part in (geotiff_path.parent.name, geotiff_path.stem):
        if part.startswith("cell_") and "_" in part[5:]:
            try:
                cell_id = int(part.split("_")[1])
                break
            except (IndexError, ValueError):
                pass
        elif part.startswith("cell_"):
            try:
                cell_id = int(part[5:])
                break
            except ValueError:
                pass
    stem = geotiff_path.stem
    m = re.search(
        r"(\d{2})(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2})",
        stem, re.I
    )
    if m:
        yy, mon, dd = m.group(1), m.group(2).upper(), m.group(3)
        yyyy = 2000 + int(yy) if int(yy) < 50 else 1900 + int(yy)
        month = _MONTH.get(mon)
        if month is not None:
            try:
                day = int(dd)
                if 1 <= day <= 31 and month <= 12:
                    acq_date = f"{yyyy}-{month:02d}-{day:02d}"
            except ValueError:
                pass
    return (cell_id, acq_date)


def chip_image(image: np.ndarray, chip_size: int, overlap: int):
    """Split image (C, H, W) into chips; return list of (chip, (x_start, y_start))."""
    C, height, width = image.shape
    effective = chip_size - overlap
    chips, coords = [], []
    for y in range(0, height, effective):
        for x in range(0, width, effective):
            y_end = min(y + chip_size, height)
            x_end = min(x + chip_size, width)
            y_start = max(0, y_end - chip_size)
            x_start = max(0, x_end - chip_size)
            chip = image[:, y_start:y_end, x_start:x_end]
            if chip.shape[1] < chip_size or chip.shape[2] < chip_size:
                padded = np.zeros((C, chip_size, chip_size), dtype=image.dtype)
                padded[:, : chip.shape[1], : chip.shape[2]] = chip
                chip = padded
            chips.append(chip)
            coords.append((x_start, y_start))
    return chips, coords


def process_single_geotiff(
    geotiff_path: Path,
    model_net,
    device: torch.device,
    chip_size: int,
    overlap: int,
    global_stats: dict,
    min_area: float,
    max_area: float,
    edge_buffer: float,
    iou_threshold: float,
    output_dir: Path,
    input_path: Path,
) -> dict:
    """Process one GeoTIFF: chip, run UNet, vectorize, postprocess; return features + metadata."""
    with rasterio.open(geotiff_path) as src:
        image = src.read()  # (C, H, W)
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

    # Normalize full image (z_value per band)
    mean = np.array(global_stats["mean"], dtype=np.float64)
    std = np.array(global_stats["std"], dtype=np.float64)
    std[std == 0] = 1e-6
    image_norm = np.zeros_like(image, dtype=np.float32)
    for b in range(image.shape[0]):
        image_norm[b] = (image[b].astype(np.float32) - mean[b]) / std[b]

    chips, coords = chip_image(image_norm, chip_size, overlap)
    chip_geodataframes = []

    for (chip, (x_start, y_start)) in tqdm(
        list(zip(chips, coords)), desc=geotiff_path.stem, leave=False
    ):
        with torch.inference_mode():
            tensor = torch.from_numpy(chip).float().to(device)  # (C, H, W)
            batch = _prepare_image_batch(tensor, device, band_order=None)
            logits = model_net(batch)
            pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        chip_transform = rasterio.Affine(
            transform[0], transform[1],
            transform[2] + x_start * transform[0] + y_start * transform[1],
            transform[3], transform[4],
            transform[5] + x_start * transform[3] + y_start * transform[4],
        )
        chip_gdf = semantic_3class_to_geodataframe(
            pred, chip_transform, crs,
            min_area=min_area, max_area=max_area,
        )
        if chip_gdf.empty:
            continue

        x_end = min(x_start + chip_size, image.shape[2])
        y_end = min(y_start + chip_size, image.shape[1])
        chip_bounds_geo = box(
            transform[2] + x_start * transform[0],
            transform[5] + y_end * transform[4],
            transform[2] + x_end * transform[0],
            transform[5] + y_start * transform[4],
        )
        chip_gdf = filter_edge_polygons_geo(
            chip_gdf, chip_bounds_geo, edge_buffer=edge_buffer
        )
        if not chip_gdf.empty:
            chip_geodataframes.append(chip_gdf)

    if not chip_geodataframes:
        return {"features": [], "crs": crs, "bounds": bounds, "source_file": geotiff_path.name}

    combined = pd.concat(chip_geodataframes, ignore_index=True)
    combined = resolve_overlaps(
        combined, overlap_threshold=iou_threshold, keep_higher_confidence=True
    )
    combined = merge_overlapping_fields(combined, iou_threshold=iou_threshold)

    features = []
    for _, row in combined.iterrows():
        props = {k: v for k, v in row.items() if k != "geometry"}
        if "score" not in props and "confidence" in props:
            props["score"] = props["confidence"]
        features.append({
            "type": "Feature",
            "geometry": mapping(row.geometry),
            "properties": props,
        })

    return {
        "features": features,
        "crs": crs,
        "bounds": bounds,
        "source_file": geotiff_path.name,
    }


def main():
    parser = argparse.ArgumentParser(
        description="UNet inference on GeoTIFFs → per-image GeoJSON (Uganda pipeline)."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to UNet checkpoint (.ckpt)")
    parser.add_argument("--input", required=True, help="Path to GeoTIFF file or directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--chip-size", type=int, default=501)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument(
        "--normalization-stats",
        required=True,
        help='JSON with "mean" and "std" arrays (4 bands) e.g. from uganda-unet.yaml',
    )
    parser.add_argument("--min-area", type=float, default=100.0)
    parser.add_argument("--max-area", type=float, default=90000.0)
    parser.add_argument("--edge-buffer", type=float, default=5.0)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    global_stats = json.loads(args.normalization_stats)
    if "mean" not in global_stats or "std" not in global_stats:
        raise ValueError("normalization-stats must contain 'mean' and 'std' arrays")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model_net = load_model(args.checkpoint, device)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        geotiff_files = [input_path]
    else:
        geotiff_files = sorted(
            list(input_path.rglob("*.tif")) + list(input_path.rglob("*.tiff"))
        )
        geotiff_files = list(dict.fromkeys(geotiff_files))

    all_metadata = []
    image_bounds_features = []

    for geotiff_file in geotiff_files:
        geotiff_file = Path(geotiff_file)
        try:
            cell_id, acq_date = parse_cell_id_and_date(geotiff_file, input_path)
            image_path_str = (
                str(geotiff_file.relative_to(input_path))
                if input_path.is_dir() else geotiff_file.name
            )

            result = process_single_geotiff(
                geotiff_file,
                model_net,
                device,
                args.chip_size,
                args.overlap,
                global_stats,
                args.min_area,
                args.max_area,
                args.edge_buffer,
                args.iou_threshold,
                output_dir,
                input_path,
            )

            if input_path.is_file():
                out_geojson = output_dir / f"{geotiff_file.stem}.geojson"
            else:
                rel = geotiff_file.relative_to(input_path)
                out_geojson = output_dir / rel.with_suffix(".geojson")
            out_geojson.parent.mkdir(parents=True, exist_ok=True)

            features = result.get("features", [])
            for f in features:
                f["properties"]["source_file"] = result.get("source_file", geotiff_file.name)
                if cell_id is not None:
                    f["properties"]["cell_id"] = cell_id
                if acq_date:
                    f["properties"]["acq_date"] = acq_date
                f["properties"]["image_path"] = image_path_str
                if "score" not in f["properties"] and "confidence" in f["properties"]:
                    f["properties"]["score"] = f["properties"]["confidence"]

            crs = result.get("crs")
            epsg = "4326"
            if crs:
                try:
                    epsg = str(crs).split(":")[-1] if ":" in str(crs) else str(crs)
                except Exception:
                    pass

            if features:
                geojson = {
                    "type": "FeatureCollection",
                    "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}},
                    "features": features,
                }
                with open(out_geojson, "w") as f:
                    json.dump(geojson, f, indent=2)
                logger.info("Saved %d features to %s", len(features), out_geojson)

            all_metadata.append({
                "file": geotiff_file.name,
                "output": str(out_geojson),
                "crs": str(crs or ""),
                "num_features": len(features),
            })

            if "bounds" in result and result["bounds"]:
                b = result["bounds"]
                bounds_polygon = box(b[0], b[1], b[2], b[3])
                image_bounds_features.append({
                    "type": "Feature",
                    "geometry": mapping(bounds_polygon),
                    "properties": {
                        "source_file": result.get("source_file", geotiff_file.name),
                        "filename": geotiff_file.name,
                        "num_predictions": len(features),
                        "crs": str(crs or "unknown"),
                    },
                })

        except Exception as e:
            logger.error("Error processing %s: %s", geotiff_file, e, exc_info=True)

    with open(output_dir / "processing_summary.json", "w") as f:
        json.dump({
            "total_files_processed": len(all_metadata),
            "total_features": sum(m["num_features"] for m in all_metadata),
            "files": all_metadata,
        }, f, indent=2)

    if image_bounds_features:
        epsg = "4326"
        if all_metadata and all_metadata[0].get("crs"):
            try:
                epsg = str(all_metadata[0]["crs"]).split(":")[-1] if ":" in all_metadata[0]["crs"] else "4326"
            except Exception:
                pass
        with open(output_dir / "image_bounds.geojson", "w") as f:
            json.dump({
                "type": "FeatureCollection",
                "crs": {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{epsg}"}},
                "features": image_bounds_features,
            }, f, indent=2)
        logger.info("Saved image_bounds.geojson with %d features", len(image_bounds_features))


if __name__ == "__main__":
    main()

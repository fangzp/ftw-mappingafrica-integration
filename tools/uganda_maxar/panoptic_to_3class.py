#!/usr/bin/env python3
"""
Convert panoptic PNGs (from ag-seg geojson_to_coco) to FTW 3-class GeoTIFFs.

Reads panoptic_{split}/*.png, decodes instance IDs from RGB (COCO rgb2id),
computes 1-pixel boundary per instance (erosion), and writes:
  0 = background, 1 = field interior, 2 = boundary.

Uses transform/CRS from the corresponding image in train/ or test/ when available.
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

# Match geojson_to_coco defaults (R, G, B)
BACKGROUND_COLOR = [153, 30, 76]


def rgb2id(rgb: np.ndarray) -> int:
    """COCO panopticapi formula: R + 256*G + 256^2*B."""
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return r + 256 * g + 256 * 256 * b


def panoptic_to_3class_raster(panoptic_rgb: np.ndarray, background_id: int) -> np.ndarray:
    """
    panoptic_rgb: (H, W, 3) uint8 RGB.
    Returns (H, W) uint8: 0=background, 1=interior, 2=boundary.
    """
    r, g, b = panoptic_rgb[..., 0], panoptic_rgb[..., 1], panoptic_rgb[..., 2]
    ids = r.astype(np.int64) + 256 * g.astype(np.int64) + 256 * 256 * b.astype(np.int64)

    out = np.zeros(ids.shape, dtype=np.uint8)
    unique_ids = np.unique(ids)
    unique_ids = unique_ids[unique_ids != background_id]

    for seg_id in unique_ids:
        mask = ids == seg_id
        if not np.any(mask):
            continue
        interior = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(2, 1))
        boundary = mask & ~interior
        out[interior] = 1
        out[boundary] = 2  # boundary overwrites so boundary wins at overlaps

    return out


def main():
    p = argparse.ArgumentParser(description="Panoptic PNGs -> FTW 3-class GeoTIFFs")
    p.add_argument("coco_root", type=Path, help="Path to coco/ (has panoptic_train, panoptic_test, train, test)")
    p.add_argument("--output_dir", type=Path, default=None, help="Root for semantic_3class_{split}/ (default: coco_root)")
    p.add_argument("--splits", nargs="+", default=["train", "test"], help="Splits to process")
    p.add_argument("--background_rgb", nargs=3, type=int, default=BACKGROUND_COLOR, metavar=("R", "G", "B"))
    args = p.parse_args()

    coco_root = args.coco_root.resolve()
    output_root = args.output_dir.resolve() if args.output_dir else coco_root
    background_id = rgb2id(np.array(args.background_rgb))

    for split in args.splits:
        panoptic_dir = coco_root / f"panoptic_{split}"
        image_dir = coco_root / split
        out_dir = output_root / f"semantic_3class_{split}"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not panoptic_dir.exists():
            print(f"Skip {split}: {panoptic_dir} not found")
            continue

        pngs = sorted(panoptic_dir.glob("*.png"))
        for png_path in tqdm(pngs, desc=split):
            stem = png_path.stem
            # Load panoptic RGB (PIL gives (H,W,3))
            img = np.array(Image.open(png_path))
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            raster_3class = panoptic_to_3class_raster(img, background_id)

            # Georeference from same-named TIF in train/ or test/
            ref_tif = image_dir / f"{stem}.tif"
            if ref_tif.exists():
                with rasterio.open(ref_tif) as src:
                    profile = src.profile.copy()
                    profile.update(dtype=rasterio.uint8, count=1, nodata=0)
                with rasterio.open(out_dir / f"{stem}.tif", "w", **profile) as dst:
                    dst.write(raster_3class, 1)
            else:
                profile = {
                    "driver": "GTiff",
                    "height": raster_3class.shape[0],
                    "width": raster_3class.shape[1],
                    "count": 1,
                    "dtype": rasterio.uint8,
                    "nodata": 0,
                }
                with rasterio.open(out_dir / f"{stem}.tif", "w", **profile) as dst:
                    dst.write(raster_3class, 1)

    print("Done. 3-class TIFs under", output_root / "semantic_3class_*")


if __name__ == "__main__":
    main()

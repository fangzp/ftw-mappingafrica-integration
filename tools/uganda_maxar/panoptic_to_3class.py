#!/usr/bin/env python3
"""
Convert panoptic PNGs (from ag-seg geojson_to_coco) to FTW 3-class GeoTIFFs.

Iterates annotations/instances_{split}.json, decodes each image's panoptic PNG
(COCO rgb2id), computes a 1-pixel boundary per instance (erosion), and writes
semantic_3class_{split}/<panoptic stem>.tif:
  0 = background, 1 = field interior, 2 = boundary.

Georeferenced from the image tif the COCO record names.
"""

import argparse
import json
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
    p = argparse.ArgumentParser(description="Panoptic PNGs -> FTW 3-class GeoTIFFs, one per COCO image")
    p.add_argument("coco_root", type=Path, help="Dataset root with annotations/, panoptic_{split}/ and {split}/")
    p.add_argument("--output_dir", type=Path, required=True, help="Root for semantic_3class_{split}/")
    p.add_argument("--splits", nargs="+", default=["train", "test"], help="Splits to process")
    p.add_argument("--background_rgb", nargs=3, type=int, default=BACKGROUND_COLOR, metavar=("R", "G", "B"))
    args = p.parse_args()

    coco_root = args.coco_root.resolve()
    output_root = args.output_dir.resolve()
    if output_root == coco_root or coco_root in output_root.parents:
        raise ValueError(f"output_dir {output_root} must be outside the source dataset {coco_root}")
    background_id = rgb2id(np.array(args.background_rgb))

    for split in args.splits:
        images = json.loads((coco_root / "annotations" / f"instances_{split}.json").read_text())["images"]
        out_dir = output_root / f"semantic_3class_{split}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # One mask per COCO image: labelers sharing a tif get separate masks named by the panoptic stem.
        for im in tqdm(images, desc=split):
            png_path = coco_root / im["pan_seg_file_name"]
            img = np.array(Image.open(png_path).convert("RGB"))
            if img.shape[:2] != (im["height"], im["width"]):
                raise ValueError(f"{png_path}: shape {img.shape[:2]} != COCO {(im['height'], im['width'])}")
            raster_3class = panoptic_to_3class_raster(img, background_id)

            with rasterio.open(coco_root / split / im["file_name"]) as src:
                profile = src.profile.copy()
            profile.update(dtype=rasterio.uint8, count=1, nodata=None, compress="deflate")
            with rasterio.open(out_dir / f"{png_path.stem}.tif", "w", **profile) as dst:
                dst.write(raster_3class, 1)

    print("Done. 3-class TIFs under", output_root / "semantic_3class_*")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Derive instance and 2-class semantic masks for the Uganda Maxar UNet layout.

Inputs:
- coco_root: COCO-style root with panoptic_train/test and train/test TIFFs
  (e.g. /nfs/hpc/share/fangzha/data/uganda_coco_norm/coco/)
- uganda_root: Option-B layout root
  (e.g. /nfs/hpc/share/fangzha/data/uganda_maxar_unet/)
- chips_parquet: chips_uganda.parquet under uganda_root

Outputs under uganda_root:
- label_masks/instance/<id>.tif      (per-instance labels, 0 = background)
- label_masks/semantic_2class/<id>.tif  (0 = background, 1 = field)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from tqdm import tqdm


def rgb2id(rgb: np.ndarray) -> int:
    """COCO panopticapi formula: R + 256*G + 256^2*B."""
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    return r + 256 * g + 256 * 256 * b


def panoptic_png_to_instance(png_path: Path) -> np.ndarray:
    """Read panoptic PNG and return instance ID map as int64 array."""
    img = np.array(Image.open(png_path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    ids = (
        r.astype(np.int64)
        + 256 * g.astype(np.int64)
        + 256 * 256 * b.astype(np.int64)
    )
    # keep raw ids; 0 will typically be background
    return ids


def main():
    ap = argparse.ArgumentParser(
        description="Make instance + semantic_2class masks for Uganda Maxar layout"
    )
    ap.add_argument("coco_root", type=Path, help="Path to coco/ root")
    ap.add_argument("uganda_root", type=Path, help="Path to uganda_maxar_unet root")
    ap.add_argument(
        "--chips_parquet",
        type=Path,
        default=None,
        help="chips_uganda.parquet (default: uganda_root/chips_uganda.parquet)",
    )
    args = ap.parse_args()

    coco_root = args.coco_root.resolve()
    uganda_root = args.uganda_root.resolve()
    chips_parquet = (
        args.chips_parquet.resolve()
        if args.chips_parquet
        else (uganda_root / "chips_uganda.parquet")
    )

    df = pd.read_parquet(chips_parquet)
    aoi_ids = df["aoi_id"].tolist()

    sem3_dir = uganda_root / "label_masks" / "semantic_3class"
    inst_dir = uganda_root / "label_masks" / "instance"
    sem2_dir = uganda_root / "label_masks" / "semantic_2class"
    inst_dir.mkdir(parents=True, exist_ok=True)
    sem2_dir.mkdir(parents=True, exist_ok=True)

    for aoi in tqdm(aoi_ids, desc="chips"):
        sem3_path = sem3_dir / f"{aoi}.tif"
        if not sem3_path.exists():
            print(f"Skip {aoi}: missing semantic_3class {sem3_path}")
            continue

        # Decide whether this chip comes from train or test in coco_root
        pan_train = coco_root / "panoptic_train" / f"{aoi}.png"
        pan_test = coco_root / "panoptic_test" / f"{aoi}.png"
        if pan_train.exists():
            pan_path = pan_train
        elif pan_test.exists():
            pan_path = pan_test
        else:
            print(f"Skip {aoi}: no panoptic PNG in train/test")
            continue

        # Instance map from panoptic PNG
        inst_map = panoptic_png_to_instance(pan_path)

        # Semantic 3-class -> 2-class: (1,2) -> 1
        with rasterio.open(sem3_path) as src:
            sem3 = src.read(1)
            profile = src.profile.copy()
        sem2 = (sem3 > 0).astype(np.uint8)

        # Write instance mask (uint32 to be safe)
        inst_profile = profile.copy()
        inst_profile.update(dtype="uint32", count=1, nodata=0)
        inst_out = inst_dir / f"{aoi}.tif"
        with rasterio.open(inst_out, "w", **inst_profile) as dst:
            dst.write(inst_map.astype("uint32"), 1)

        # Write 2-class semantic mask
        sem2_profile = profile.copy()
        sem2_profile.update(dtype="uint8", count=1, nodata=0)
        sem2_out = sem2_dir / f"{aoi}.tif"
        with rasterio.open(sem2_out, "w", **sem2_profile) as dst:
            dst.write(sem2, 1)

    print("Done. Wrote instance + semantic_2class masks under", uganda_root)


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Build Option B folder layout and chips parquet for Uganda Maxar UNet training.

Mask-driven, with stem normalization so we can use all chips even when the
image filenames keep the full Maxar product name while masks use a shortened
<tile>_<YYMMMDD> stem.

Expects:
  - coco_root/train/, coco_root/test/ (normalized GeoTIFFs)
  - coco_root/semantic_3class_train/, coco_root/semantic_3class_test/
    (from panoptic_to_3class.py)

Creates under uganda_root:
  - s2_images/window_a/<aoi_id>.tif
  - label_masks/semantic_3class/<aoi_id>.tif
  - chips_uganda.parquet  (columns: aoi_id, split)
"""

import argparse
from pathlib import Path

import pandas as pd

import numpy as np
import rasterio

TARGET_SIZE = 251

def ensure_size(src_path: Path, dst_path: Path, size: int = TARGET_SIZE):
    """Write dst_path as a size×size raster, padding with edge values if needed."""
    import shutil

    with rasterio.open(src_path) as src:
        data = src.read()  # (bands, H, W)
        bands, h, w = data.shape

        # If already the right size, just copy once
        if h == size and w == size:
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
            return

        if h > size or w > size:
            raise ValueError(f"{src_path} is larger than target {size}×{size}: {h}×{w}")

        pad_h = size - h
        pad_w = size - w

        # Pad at the bottom and right using edge values
        pad_widths = ((0, 0), (0, pad_h), (0, pad_w))
        data_padded = np.pad(data, pad_widths, mode="edge")

        profile = src.profile
        profile.update(height=size, width=size)

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data_padded)


def shorten_stem(stem: str) -> str:
    """
    Normalize a Maxar-style stem to <tile>_<YYMMMDD>.

    Examples
    --------
    - '10_21OCT11'                      -> '10_21OCT11'
    - '9500_22APR06081711-M2AS-...'    -> '9500_22APR06'
    """
    parts = stem.split("_")
    if len(parts) < 2:
        return stem
    tile = parts[0]
    rest = parts[1]
    if len(rest) >= 7:
        date7 = rest[:7]
        return f"{tile}_{date7}"
    return stem


def main():
    p = argparse.ArgumentParser(description="Build Uganda Maxar Option B layout and parquet")
    p.add_argument("coco_root", type=Path, help="Path to coco/ (has train/, test/, semantic_3class_*)")
    p.add_argument("uganda_root", type=Path, help="Output root for uganda_maxar layout")
    p.add_argument("--copy", action="store_true", help="Copy files instead of symlinking")
    p.add_argument(
        "--keep_test_split",
        action="store_true",
        help="Keep coco_root/test as split='test' instead of mapping it to 'validate'.",
    )
    args = p.parse_args()

    coco_root = args.coco_root.resolve()
    uganda_root = args.uganda_root.resolve()
    do_symlink = not args.copy

    img_dir = uganda_root / "s2_images" / "window_a"
    mask_dir = uganda_root / "label_masks" / "semantic_3class"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    split_map = {"train": "train", "test": ("test" if args.keep_test_split else "validate")}

    for src_split, dst_split in split_map.items():
        img_split_dir = coco_root / src_split
        mask_split_dir = coco_root / f"semantic_3class_{src_split}"

        if not img_split_dir.exists() or not mask_split_dir.exists():
            print(f"Skip {src_split}: missing {img_split_dir} or {mask_split_dir}")
            continue

        # Build mapping from shortened stem -> image path
        short_to_img: dict[str, Path] = {}
        tile_to_imgs: dict[str, list[Path]] = {}
        for tif_path in sorted(img_split_dir.glob("*.tif")):
            full_stem = tif_path.stem
            short_stem = shorten_stem(full_stem)
            if short_stem in short_to_img and short_to_img[short_stem] != tif_path:
                print(f"Warning: multiple images map to {short_stem}, keeping first: {short_to_img[short_stem]}, skipping {tif_path}")
                continue
            short_to_img[short_stem] = tif_path
            tile = full_stem.split("_")[0]
            tile_to_imgs.setdefault(tile, []).append(tif_path)

        # Drive off masks to ensure we only include chips with labels
        for mask_path in sorted(mask_split_dir.glob("*.tif")):
            mask_stem = mask_path.stem  # canonical aoi_id
            img_src = short_to_img.get(mask_stem)
            if img_src is None:
                # Fallback: same tile, single image (mask date may differ from image date)
                tile = mask_stem.split("_")[0]
                candidates = tile_to_imgs.get(tile, [])
                if len(candidates) == 1:
                    img_src = candidates[0]
                    print(f"Warning: date mismatch for {mask_stem}, using only image for tile {tile}: {img_src.name}")
                else:
                    print(f"Skip {mask_stem}: no image in {img_split_dir} matching shortened stem (tile {tile}: {len(candidates)} images)")
                    continue

            img_dst = img_dir / f"{mask_stem}.tif"
            mask_dst = mask_dir / f"{mask_stem}.tif"

            # if do_symlink:
            #     if not img_dst.exists():
            #         img_dst.symlink_to(img_src.resolve())
            #     if not mask_dst.exists():
            #         mask_dst.symlink_to(mask_path.resolve())
            # else:
            #     import shutil
            #     if not img_dst.exists():
            #         shutil.copy2(img_src, img_dst)
            #     if not mask_dst.exists():
            #         shutil.copy2(mask_path, mask_dst)

            ensure_size(img_src, img_dst)     # image padded / copied to 251×251
            ensure_size(mask_path, mask_dst)  # mask padded / copied to 251×251

            rows.append({"aoi_id": mask_stem, "split": dst_split})

    df = pd.DataFrame(rows)
    parquet_path = uganda_root / "chips_uganda.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Wrote {parquet_path} with {len(df)} rows. Layout under {uganda_root}")


if __name__ == "__main__":
    main()

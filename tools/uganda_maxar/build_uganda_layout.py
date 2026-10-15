#!/usr/bin/env python3
"""
Build the FTWMapAfrica catalog for Uganda Maxar UNet training from a COCO dataset.

One catalog row per COCO image (tile, date, labeler), so a tif shared by several
labelers yields several samples. Chips are used at native size.

Expects:
  - coco_root/annotations/instances_{train,test}.json, coco_root/{train,test}/*.tif
  - mask_root/semantic_3class_{train,test}/<panoptic stem>.tif (from panoptic_to_3class.py)

Creates under uganda_root:
  - images_{train,test} -> coco_root/{train,test} (directory symlinks)
  - semantic_3class_{train,test} -> mask_root/semantic_3class_{train,test} (when mask_root != uganda_root)
  - uganda-maxar-catalog.csv (name, window_a, mask, split, coco_image_id, tile_id, labeler)
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def link_dir(link: Path, target: Path) -> None:
    if link.is_symlink() or link.exists():
        if link.resolve() != target.resolve():
            raise FileExistsError(f"{link} exists and does not point at {target}")
        return
    link.symlink_to(target, target_is_directory=True)


def main():
    p = argparse.ArgumentParser(description="Build Uganda Maxar catalog from COCO annotations")
    p.add_argument("coco_root", type=Path, help="COCO dataset root (annotations/, train/, test/)")
    p.add_argument("uganda_root", type=Path, help="Output root; becomes the trainer's data_dir")
    p.add_argument("--mask_root", type=Path, default=None, help="Root holding semantic_3class_{split}/ (default: uganda_root)")
    p.add_argument("--full_pan_only", action="store_true", help="Keep only images with pansharp.pan_coverage == 1.0")
    p.add_argument(
        "--keep_test_split",
        action="store_true",
        help="Keep coco_root/test as split='test' instead of mapping it to 'validate'.",
    )
    args = p.parse_args()

    coco_root = args.coco_root.resolve()
    uganda_root = args.uganda_root.resolve()
    mask_root = (args.mask_root or args.uganda_root).resolve()
    uganda_root.mkdir(parents=True, exist_ok=True)

    rows = []
    split_map = {"train": "train", "test": ("test" if args.keep_test_split else "validate")}
    for src_split, dst_split in split_map.items():
        images = json.loads((coco_root / "annotations" / f"instances_{src_split}.json").read_text())["images"]
        if args.full_pan_only:
            images = [im for im in images if float(im["pansharp"]["pan_coverage"]) == 1.0]

        link_dir(uganda_root / f"images_{src_split}", coco_root / src_split)
        mask_dir_name = f"semantic_3class_{src_split}"
        if mask_root != uganda_root:
            link_dir(uganda_root / mask_dir_name, mask_root / mask_dir_name)

        for im in images:
            stem = Path(im["pan_seg_file_name"]).stem
            img_rel = f"images_{src_split}/{im['file_name']}"
            mask_rel = f"{mask_dir_name}/{stem}.tif"
            for rel in (img_rel, mask_rel):
                if not (uganda_root / rel).exists():
                    raise FileNotFoundError(uganda_root / rel)
            rows.append({
                "name": stem,
                "window_a": img_rel,
                "mask": mask_rel,
                "split": dst_split,
                "coco_image_id": im["id"],
                "tile_id": im["tile_id"],
                "labeler": im.get("labeler"),
            })

    df = pd.DataFrame(rows)
    out = uganda_root / "uganda-maxar-catalog.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}: {df['split'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()

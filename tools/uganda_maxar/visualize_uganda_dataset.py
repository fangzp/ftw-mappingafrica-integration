#!/usr/bin/env python3
"""
Visualize a few Uganda Maxar chips from the Option-B layout.

Shows:
- window_a RGB
- semantic_2class (0/1)
- semantic_3class (0/1/2)
- instance mask colored by instance ID
"""

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from skimage.color import label2rgb


def load_rgb_window_a(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        img = src.read()
    # Uganda Maxar chips are normalized 0-10000; scale to [0,1] for display
    rgb = img[0:3, :, :].transpose(1, 2, 0).astype(np.float32) / 10000.0
    return np.clip(rgb, 0, 1)


def plot_sample(root: Path, aoi_id: str, output_dir: Path) -> Path | None:
    img_path = root / "s2_images" / "window_a" / f"{aoi_id}.tif"
    sem3_path = root / "label_masks" / "semantic_3class" / f"{aoi_id}.tif"
    sem2_path = root / "label_masks" / "semantic_2class" / f"{aoi_id}.tif"
    inst_path = root / "label_masks" / "instance" / f"{aoi_id}.tif"

    if not img_path.exists():
        print(f"Missing image for {aoi_id}: {img_path}")
        return None

    rgb = load_rgb_window_a(img_path)

    sem3 = None
    if sem3_path.exists():
        with rasterio.open(sem3_path) as src:
            sem3 = src.read(1)

    sem2 = None
    if sem2_path.exists():
        with rasterio.open(sem2_path) as src:
            sem2 = src.read(1)

    inst_rgb = None
    if inst_path.exists():
        with rasterio.open(inst_path) as src:
            inst = src.read(1)
        labels = np.unique(inst)
        colors = [
            (np.random.rand(), np.random.rand(), np.random.rand())
            for _ in labels
        ]
        inst_rgb = label2rgb(
            inst,
            bg_label=0,
            bg_color=(0, 0, 0),
            colors=colors,
        )

    print(f"Sample {aoi_id}")
    print("  window_a:", img_path)
    print("  sem3:", sem3_path if sem3_path.exists() else "missing")
    print("  sem2:", sem2_path if sem2_path.exists() else "missing")
    print("  instance:", inst_path if inst_path.exists() else "missing")

    n_cols = 1 + (sem2 is not None) + (sem3 is not None) + (inst_rgb is not None)
    fig, axs = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    axs = np.atleast_1d(axs)
    col = 0

    axs[col].imshow(rgb)
    axs[col].set_title("window_a RGB")
    axs[col].axis("off")
    col += 1

    if sem2 is not None:
        axs[col].imshow(sem2, vmin=0, vmax=1, cmap="viridis")
        axs[col].set_title("semantic_2class")
        axs[col].axis("off")
        col += 1

    if sem3 is not None:
        axs[col].imshow(sem3, vmin=0, vmax=2, cmap="viridis")
        axs[col].set_title("semantic_3class")
        axs[col].axis("off")
        col += 1

    if inst_rgb is not None:
        axs[col].imshow(inst_rgb)
        axs[col].set_title("instance")
        axs[col].axis("off")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{aoi_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Visualize Uganda Maxar UNet dataset")
    ap.add_argument(
        "uganda_root",
        type=Path,
        help="uganda_maxar_unet root (Option-B layout)",
    )
    ap.add_argument(
        "--chips_parquet",
        type=Path,
        default=None,
        help="chips_uganda.parquet (default: uganda_root/chips_uganda.parquet)",
    )
    ap.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of random samples to visualize",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validate", "test"],
        help="Which split to sample from (by chips_uganda.parquet split column)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling",
    )
    ap.add_argument(
        "--output_dir",
        "-o",
        type=Path,
        required=True,
        help="Directory to save figure images (unique filename per sample: <aoi_id>.png)",
    )
    args = ap.parse_args()

    root = args.uganda_root.resolve()
    chips_parquet = (
        args.chips_parquet.resolve()
        if args.chips_parquet
        else (root / "chips_uganda.parquet")
    )

    df = pd.read_parquet(chips_parquet)
    df_split = df[df["split"] == args.split]
    ids = df_split["aoi_id"].tolist()
    if not ids:
        print(f"No samples found for split '{args.split}'")
        return

    random.seed(args.seed)
    sample_ids = random.sample(ids, min(args.num_samples, len(ids)))

    output_dir = args.output_dir.resolve()
    for aoi_id in sample_ids:
        plot_sample(root, aoi_id, output_dir)


if __name__ == "__main__":
    main()
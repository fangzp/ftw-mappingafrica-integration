#!/usr/bin/env python3
"""
Build uganda-maxar-catalog.csv from chips_uganda.parquet for ftw_ma training.

Paths in the CSV are relative to data_dir (Uganda root), so the catalog is portable.
Columns: name, window_a, mask, split (window_b omitted for windowA-only).
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Parquet -> catalog CSV for FTWMapAfrica")
    p.add_argument("parquet", type=Path, help="chips_uganda.parquet path")
    p.add_argument("-o", "--output", type=Path, default=None, help="Output CSV (default: same dir as parquet, uganda-maxar-catalog.csv)")
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)
    if "aoi_id" not in df.columns or "split" not in df.columns:
        raise ValueError("Parquet must have columns aoi_id, split")

    # Paths relative to data_dir (Uganda root)
    df["name"] = df["aoi_id"]
    df["window_a"] = df["aoi_id"].apply(lambda x: f"s2_images/window_a/{x}.tif")
    df["mask"] = df["aoi_id"].apply(lambda x: f"label_masks/semantic_3class/{x}.tif")
    df = df.rename(columns={"split": "split"})
    out_df = df[["name", "window_a", "mask", "split"]]

    out_path = args.output or (args.parquet.parent / "uganda-maxar-catalog.csv")
    out_path = out_path.resolve()
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(out_df)} rows.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Write per-chip 3-class softmax rasters for a catalog split, for the bakeoff scorer.

Each unique window_a tif is predicted once (labelers sharing a tif share it) and
written as <output>/<tif name>: float32, 3 bands (background, field, boundary),
georeferenced like the input. The instance step (fbp) runs in the scorer.

    python scripts/predict_coco_probs_unet.py \
        --config configs/uganda_maxar/bakeoff/unet-ms.yaml \
        --checkpoint /path/to/last.ckpt --split validate --output /path/to/probs
"""

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
import yaml

from ftw_ma.datamodule import FTWMapAfricaDataModule
from ftw_ma.trainers import CustomSemanticSegmentationTask


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True, help="Training config (data section defines inputs)")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="validate", help="Catalog split (the COCO test split is 'validate')")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    dm = FTWMapAfricaDataModule(**cfg["data"]["init_args"])
    dm.setup("test" if args.split == "test" else "validate")
    ds = dm.test_dataset if args.split == "test" else dm.val_dataset

    model = CustomSemanticSegmentationTask.load_from_checkpoint(args.checkpoint, map_location="cpu")
    model = model.to(args.device).eval()

    args.output.mkdir(parents=True, exist_ok=True)
    done = {}
    with torch.inference_mode():
        for i, f in enumerate(ds.filenames):
            src_path = Path(f["window_a"])
            if src_path.name in done:
                continue
            x = ds[i]["image"][None].to(args.device)
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy().astype(np.float32)
            with rasterio.open(src_path) as src:
                profile = src.profile.copy()
            profile.update(count=3, dtype="float32", nodata=None, compress="deflate", predictor=3)
            out_path = args.output / src_path.name
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(probs)
                dst.set_band_description(1, "background")
                dst.set_band_description(2, "field")
                dst.set_band_description(3, "boundary")
            done[src_path.name] = out_path.name

    manifest = {
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "split": args.split,
        "date": datetime.date.today().isoformat(),
        "bands": ["background", "field", "boundary"],
        "files": done,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(done)} probability rasters to {args.output}")


if __name__ == "__main__":
    main()

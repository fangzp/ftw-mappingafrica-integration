# Uganda Maxar UNet dataset preparation

Scripts to convert geojson_to_coco outputs to the Option B layout and catalog for training a PRUE-like UNet in this repo (branch `ug-maxar`). Dataset lives at `/nfs/hpc/share/fangzha/data/uganda_coco_norm/coco/`.

## 1. Panoptic → 3-class rasters

From `ftw-mappingafrica-integration` root:

```bash
python tools/uganda_maxar/panoptic_to_3class.py /nfs/hpc/share/fangzha/data/uganda_coco_norm/coco/
```

Writes `semantic_3class_train/` and `semantic_3class_test/` under the same coco root. Uses transform/CRS from `train/*.tif` and `test/*.tif` when present.

## 2. Build Option B layout and parquet

```bash
python tools/uganda_maxar/build_uganda_layout.py \
  /nfs/hpc/share/fangzha/data/uganda_coco_norm/coco/ \
  /nfs/hpc/share/fangzha/data/uganda_maxar
```

Creates under `uganda_maxar/`: `s2_images/window_a/`, `label_masks/semantic_3class/`, `chips_uganda.parquet`.\n+\n+By default this maps `coco/test` → split `validate` (since you said your current \"test\" is mainly for overfitting checks). If you later create a real test set, use `--keep_test_split`.

## 3. Parquet → catalog CSV

```bash
python tools/uganda_maxar/parquet_to_catalog.py \
  /nfs/hpc/share/fangzha/data/uganda_maxar/chips_uganda.parquet \
  -o /nfs/hpc/share/fangzha/data/uganda_maxar/uganda-maxar-catalog.csv
```

## 4. Train

```bash
ftw_ma model fit -c configs/uganda_maxar/uganda-unet.yaml
```

Ensure `data_dir` and `catalog` in the config match your paths (e.g. above).

## 5. Large-scale inference (GeoTIFF → GeoJSON)

To run UNet on full GeoTIFFs (chip 501×501, vectorize, same output as Mask2Former pipeline):

```bash
python scripts/predict_geotiff_to_geojson_unet.py \
  --checkpoint /path/to/last.ckpt \
  --input /path/to/geotiffs \
  --output /path/to/output \
  --chip-size 501 --overlap 100 \
  --normalization-stats '{"mean":[2160.11,2720.38,2156.53,4174.56],"std":[1499.02,1508.71,1408.98,1532.44]}' \
  --min-area 100 --max-area 90000 --edge-buffer 5 --iou-threshold 0.7
```

Output: per-image GeoJSON (mirror input layout), `image_bounds.geojson`, `processing_summary.json`. Use ag-seg’s `aggregate_cell_predictions.py` and field stats on the output dir. SLURM: `sbatch slurm/run_uganda_maxar_inference_unet_dgx2.sh` (or `_dgxh.sh`); override `INPUT_DIR`, `OUTPUT_DIR`, `CHECKPOINT` via env.

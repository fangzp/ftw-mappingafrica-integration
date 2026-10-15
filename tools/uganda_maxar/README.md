# Uganda Maxar UNet dataset preparation

Scripts to turn an ag-seg COCO export into the catalog this repo's UNet trains on. Current dataset: `/nfs/hpc/share/fangzha/data/uganda_coco_20261014/{multispectral,pansharpened}/` (read-only). Derived data: `/nfs/hpc/share/fangzha/data/uganda_coco_20261014_derived/`.

Run everything with the `ftw` env active:

```bash
source /nfs/stak/a1/rhel5apps/conda/24.3/etc/profile.d/conda.sh
conda activate /nfs/hpc/share/fangzha/envs/ftw
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH   # pandas needs the env's libstdc++
```

## 1. Panoptic -> 3-class rasters

One mask per COCO image (tile, date, labeler), named by its panoptic stem (`10041_16NOV06_tim.tif`), from `panoptic_{split}/` (not `panoptic_semseg_*`, which has no boundary class). 0 background, 1 interior, 2 boundary (1-pixel erosion edge per instance).

```bash
S=/nfs/hpc/share/fangzha/data/uganda_coco_20261014
O=/nfs/hpc/share/fangzha/data/uganda_coco_20261014_derived/unet_3class
python tools/uganda_maxar/panoptic_to_3class.py $S/multispectral --output_dir $O/multispectral
python tools/uganda_maxar/panoptic_to_3class.py $S/pansharpened  --output_dir $O/pansharpened_fullpan
```

## 2. Catalog

One row per COCO image, so a tif shared by several labelers yields several samples. Chips stay at native size (256 px MS, 1024 px PS). `images_{split}` are directory symlinks into the source dataset. COCO `test` becomes catalog split `validate` (test doubles as the monitoring set; the reported model is `last.ckpt`).

```bash
python tools/uganda_maxar/build_uganda_layout.py $S/multispectral $O/multispectral
python tools/uganda_maxar/build_uganda_layout.py $S/pansharpened  $O/pansharpened_fullpan --full_pan_only
```

Counts: MS train 995 / validate 148; PS (`pan_coverage == 1.0`) train 816 / validate 118.

## 3. Train

Configs in `configs/uganda_maxar/bakeoff/`:

| Config | Input | Init |
|---|---|---|
| `unet-ms.yaml` | MS 2 m, z-score with train stats | ImageNet |
| `unet-ps.yaml` | PS 0.5 m, z-score with train stats | ImageNet |
| `unet-ms-ftpb3.yaml` | MS 2 m, `/10000` | `ftp-b3`, first conv 8 -> 4 by summing the two window halves |

- Stats: `pixel_stats.json` beside the derived data (keys `ms`, `ps_fullpan`), read via `global_stats_file` / `global_stats_key`.
- Nodata: a pixel is nodata only when all 4 bands are 0 (`nodata_all_bands`); it is zeroed after normalization and its mask set to `ignore_index` 3.
- No `RandomResizedCrop` (`random_resized_crop: false`). Fixed `max_epochs`; checkpoints keep the latest epoch and `last.ckpt`.

```bash
CONFIG_FILE=configs/uganda_maxar/bakeoff/unet-ms.yaml sbatch --export=ALL slurm/train_uganda_unet_dgxh.sh
```

### `ftp-b3` weights (CC-BY-NC-4.0)

`taylor-geospatial/ftp-b3` (Fields of the Planet, smp UNet EfficientNet-B3, 8 channels `[winB BGRN, winA BGRN]`), downloaded to `/nfs/hpc/share/fangzha/weights/ftp-b3/`:

```bash
huggingface-cli download taylor-geospatial/ftp-b3 --local-dir /nfs/hpc/share/fangzha/weights/ftp-b3
```

**The weights are licensed CC-BY-NC-4.0: non-commercial use only.** Setting `model.init_args.weights` to a `.safetensors` path loads them through `CustomSemanticSegmentationTask.load_two_window_weights`, which fails unless exactly the first conv mismatches and it is 8 -> 4 channels, and fails on any missing or unexpected key.

## 4. Test-split probabilities for the bakeoff scorer

```bash
CONFIG_FILE=configs/uganda_maxar/bakeoff/unet-ms.yaml CHECKPOINT=<default_root_dir>/lightning_logs/version_<slurm job id>/checkpoints/last.ckpt \
OUTPUT_DIR=<out> sbatch --export=ALL slurm/predict_uganda_unet_probs_dgxh.sh
```

Output, one file per unique test tif (labelers sharing a tif share it): `<out>/<tif name>`, float32 GeoTIFF, 3 bands = softmax (background, field, boundary), same grid and CRS as the input chip. `<out>/manifest.json` records config, checkpoint, split, and the file list. The instance step is `fbp` in the scorer (`seg = argmax`, `probs` = the 3 bands).

## 5. Large-scale inference (GeoTIFF -> GeoJSON)

For settlement maps, not the bakeoff. It still merges boundary into field before connected components and uses the old stats.

```bash
python scripts/predict_geotiff_to_geojson_unet.py \
  --checkpoint /path/to/last.ckpt \
  --input /path/to/geotiffs \
  --output /path/to/output \
  --chip-size 501 --overlap 100 \
  --normalization-stats '{"mean":[2160.11,2720.38,2156.53,4174.56],"std":[1499.02,1508.71,1408.98,1532.44]}' \
  --min-area 100 --max-area 90000 --edge-buffer 5 --iou-threshold 0.7
```

Output: per-image GeoJSON (mirror input layout), `image_bounds.geojson`, `processing_summary.json`. SLURM: `sbatch slurm/run_uganda_maxar_inference_unet_dgx2.sh` (or `_dgxh.sh`); override `INPUT_DIR`, `OUTPUT_DIR`, `CHECKPOINT` via env.

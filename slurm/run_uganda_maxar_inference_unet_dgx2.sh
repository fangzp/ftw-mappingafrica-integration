#!/bin/bash -l
#SBATCH -J uganda_unet_inference
#SBATCH -p dgx2
#SBATCH -A ceoas
#SBATCH -o /nfs/stak/users/fangzha/projects/ftw-mappingafrica-integration/slurm/logs/%j_uganda_unet_inference.out
#SBATCH -e /nfs/stak/users/fangzha/projects/ftw-mappingafrica-integration/slurm/logs/%j_uganda_unet_inference.err
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --qos=normal

# =============================================================================
# Uganda Maxar UNet Inference - GPU (dgx2)
# =============================================================================
# Runs predict_geotiff_to_geojson_unet.py (ftw-mappingafrica-integration).
# Output layout matches Mask2Former pipeline for aggregate_cell_predictions.py.
#
# Usage:
#   sbatch slurm/run_uganda_maxar_inference_unet_dgx2.sh
#   INPUT_DIR=/path/to/geotiffs OUTPUT_DIR=/path/to/out CHECKPOINT=/path/to/last.ckpt sbatch ...
# =============================================================================

set -e

FTW_PROJECT_DIR="${FTW_PROJECT_DIR:-/nfs/stak/users/fangzha/projects/ftw-mappingafrica-integration}"
SCRATCH_DIR="${SCRATCH_DIR:-/nfs/hpc/share/fangzha}"

echo "Loading modules..."
module load conda 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
module load cudnn/8.9_cuda12 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda deactivate 2>/dev/null || true
# Activate env where ftw_ma is installed (e.g. ftw, or same as used for ftw_ma model predict)
CONDA_ENV="/nfs/hpc/share/fangzha/envs/ftw"
conda activate "${CONDA_ENV}" || { echo "Conda env ${CONDA_ENV} not found; using current env."; }
PYTHON="${CONDA_ENV}/bin/python"

# INPUT_DIR="${INPUT_DIR:-${SCRATCH_DIR}/data/uganda_maxar_inference/sampling_region_basemap}"
# OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_DIR}/outputs/uganda_maxar_maps_unet/sampling_region_basemap}"
CHECKPOINT="${CHECKPOINT:-${SCRATCH_DIR}/outputs/uganda-maxar-unet/lightning_logs/version_20061532/checkpoints/last.ckpt}"
INPUT_DIR="${INPUT_DIR:-${SCRATCH_DIR}/data/uganda_coco_norm/coco/test}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_DIR}/outputs/uganda_coco_test_inference/unet}"

# Uganda UNet normalization (from configs/uganda_maxar/uganda-unet.yaml)
NORMALIZATION_STATS="${NORMALIZATION_STATS:-{\"mean\":[2160.112107499254,2720.384320220083,2156.533516309466,4174.567795159017],\"std\":[1499.029369788556,1508.715890626181,1408.9864076781628,1532.448615017464]}}"
CHIP_SIZE="${CHIP_SIZE:-501}"
OVERLAP="${OVERLAP:-100}"
# Align with M2F for comparable evaluation: min-area 100 m², iou 0.7
MIN_AREA="${MIN_AREA:-100.0}"
MAX_AREA="${MAX_AREA:-90000.0}"
EDGE_BUFFER="${EDGE_BUFFER:-5.0}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.7}"
GPU="${GPU:-0}"

echo "=============================================="
echo "Uganda Maxar UNet Inference"
echo "=============================================="
echo "  Input: ${INPUT_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Checkpoint: ${CHECKPOINT}"
echo "  Chip size: ${CHIP_SIZE}  Overlap: ${OVERLAP}"
echo "=============================================="

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi
if [[ ! -d "${INPUT_DIR}" ]] && [[ ! -f "${INPUT_DIR}" ]]; then
    echo "ERROR: Input not found: ${INPUT_DIR}"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cd "${FTW_PROJECT_DIR}"
"${PYTHON}" scripts/predict_geotiff_to_geojson_unet.py \
    --checkpoint "${CHECKPOINT}" \
    --input "${INPUT_DIR}" \
    --output "${OUTPUT_DIR}" \
    --chip-size "${CHIP_SIZE}" \
    --overlap "${OVERLAP}" \
    --normalization-stats "${NORMALIZATION_STATS}" \
    --min-area "${MIN_AREA}" \
    --max-area "${MAX_AREA}" \
    --edge-buffer "${EDGE_BUFFER}" \
    --iou-threshold "${IOU_THRESHOLD}" \
    --gpu "${GPU}"

EXIT_CODE=$?
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}

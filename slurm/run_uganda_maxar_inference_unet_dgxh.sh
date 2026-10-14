#!/bin/bash -l
#SBATCH -J uganda_unet_inference
#SBATCH -p dgxh
#SBATCH -A dmv
#SBATCH -o /nfs/stak/users/fangzha/projects/ftw-mappingafrica-integration/slurm/logs/%j_uganda_unet_inference.out
#SBATCH -e /nfs/stak/users/fangzha/projects/ftw-mappingafrica-integration/slurm/logs/%j_uganda_unet_inference.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --qos=normal

# Same as run_uganda_maxar_inference_unet_dgx2.sh but for dgxh (H100) partition.
# Usage: sbatch slurm/run_uganda_maxar_inference_unet_dgxh.sh
# Override: INPUT_DIR=... OUTPUT_DIR=... CHECKPOINT=... sbatch ...

set -e

FTW_PROJECT_DIR="${FTW_PROJECT_DIR:-/nfs/stak/users/fangzha/projects/ftw-mappingafrica-integration}"
SCRATCH_DIR="${SCRATCH_DIR:-/nfs/hpc/share/fangzha}"

echo "Loading modules..."
module load conda 2>/dev/null || true
module load cuda/12.1 2>/dev/null || true
module load cudnn/8.9_cuda12 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda deactivate 2>/dev/null || true
CONDA_ENV="${CONDA_ENV:-ftw}"
conda activate "${CONDA_ENV}" || true

INPUT_DIR="${INPUT_DIR:-${SCRATCH_DIR}/data/uganda_maxar_inference/pagrinya_basemap}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_DIR}/outputs/uganda_maxar_maps_unet/pagrinya_basemap}"
CHECKPOINT="${CHECKPOINT:-${SCRATCH_DIR}/outputs/uganda-maxar-unet/lightning_logs/version_0/checkpoints/last.ckpt}"
NORMALIZATION_STATS="${NORMALIZATION_STATS:-{\"mean\":[2160.112107499254,2720.384320220083,2156.533516309466,4174.567795159017],\"std\":[1499.029369788556,1508.715890626181,1408.9864076781628,1532.448615017464]}}"
CHIP_SIZE="${CHIP_SIZE:-501}"
OVERLAP="${OVERLAP:-100}"
MIN_AREA="${MIN_AREA:-100.0}"
MAX_AREA="${MAX_AREA:-90000.0}"
EDGE_BUFFER="${EDGE_BUFFER:-5.0}"
IOU_THRESHOLD="${IOU_THRESHOLD:-0.7}"
GPU="${GPU:-0}"

[[ -f "${CHECKPOINT}" ]] || { echo "ERROR: Checkpoint not found: ${CHECKPOINT}"; exit 1; }
[[ -d "${INPUT_DIR}" || -f "${INPUT_DIR}" ]] || { echo "ERROR: Input not found: ${INPUT_DIR}"; exit 1; }
mkdir -p "${OUTPUT_DIR}"

cd "${FTW_PROJECT_DIR}"
python scripts/predict_geotiff_to_geojson_unet.py \
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

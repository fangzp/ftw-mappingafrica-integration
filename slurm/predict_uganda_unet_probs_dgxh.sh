#!/bin/bash -l
#SBATCH -J uganda_unet_probs
#SBATCH -p dgxh
#SBATCH -A dmv
#SBATCH -o /nfs/stak/users/fangzha/projects/ftw-mappingafrica-integration/slurm/logs/%j_uganda_unet_probs.out
#SBATCH -e /nfs/stak/users/fangzha/projects/ftw-mappingafrica-integration/slurm/logs/%j_uganda_unet_probs.err
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --qos=normal

# 3-class softmax rasters on the COCO test split (catalog split 'validate') for the bakeoff scorer.
# Usage: CONFIG_FILE=... CHECKPOINT=.../last.ckpt OUTPUT_DIR=... sbatch --export=ALL slurm/predict_uganda_unet_probs_dgxh.sh

set -e

FTW_PROJECT_DIR="${FTW_PROJECT_DIR:-/nfs/stak/users/fangzha/projects/ftw-mappingafrica-integration}"
CONFIG_FILE="${CONFIG_FILE:?CONFIG_FILE is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate /nfs/hpc/share/fangzha/envs/ftw
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

cd "${FTW_PROJECT_DIR}"
srun --ntasks=1 python scripts/predict_coco_probs_unet.py \
    --config "${CONFIG_FILE}" --checkpoint "${CHECKPOINT}" --split validate --output "${OUTPUT_DIR}"

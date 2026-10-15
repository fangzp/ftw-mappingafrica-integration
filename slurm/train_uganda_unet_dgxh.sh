#!/bin/bash -l
#SBATCH -J uganda_unet_train
#SBATCH -p dgxh
#SBATCH -A dmv
#SBATCH -o /nfs/stak/users/fangzha/projects/ftw-mappingafrica-integration/slurm/logs/%j_uganda_unet_train.out
#SBATCH -e /nfs/stak/users/fangzha/projects/ftw-mappingafrica-integration/slurm/logs/%j_uganda_unet_train.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --exclude=dgxh-1
#SBATCH --time=12:00:00
#SBATCH --qos=normal

# Uganda bakeoff UNet training, 1 GPU. dgxh-1 is MIG-partitioned (40 GB slices); excluded.
# Usage:
#   CONFIG_FILE=configs/uganda_maxar/bakeoff/unet-ms.yaml sbatch --export=ALL slurm/train_uganda_unet_dgxh.sh
# Resume: CKPT_PATH=<.../last.ckpt>. Extra LightningCLI args (space-separated) go in EXTRA_ARGS,
# exported and submitted with a bare --export=ALL (sbatch --export= splits on whitespace).

set -e

FTW_PROJECT_DIR="${FTW_PROJECT_DIR:-/nfs/stak/users/fangzha/projects/ftw-mappingafrica-integration}"
CONFIG_FILE="${CONFIG_FILE:?CONFIG_FILE is required}"
CKPT_PATH="${CKPT_PATH:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate /nfs/hpc/share/fangzha/envs/ftw
# Env's libstdc++ (pandas needs GLIBCXX_3.4.29, newer than the system's).
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

cd "${FTW_PROJECT_DIR}"
echo "Host: $(hostname)  Config: ${CONFIG_FILE}  Resume: ${CKPT_PATH:-<none>}  Extra: ${EXTRA_ARGS:-<none>}"
nvidia-smi --query-gpu=name,memory.total --format=csv

CMD=(ftw_ma model fit -c "${CONFIG_FILE}")
if [[ -n "${CKPT_PATH}" ]]; then
  CMD+=(--ckpt_path "${CKPT_PATH}")
fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  read -r -a EXTRA_ARR <<< "${EXTRA_ARGS}"
  # "--" ends click's options; the rest goes to LightningCLI.
  CMD+=(-- "${EXTRA_ARR[@]}")
fi

srun --cpu-bind=cores --ntasks=1 --cpus-per-task=8 "${CMD[@]}"
echo "Training job completed."

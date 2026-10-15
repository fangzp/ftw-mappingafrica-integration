#!/bin/bash -l
#SBATCH -J smoke_unet
#SBATCH -p dgxh
#SBATCH -A dmv
#SBATCH -o /nfs/stak/users/fangzha/projects/ftw-mappingafrica-integration/slurm/logs/%j_smoke_unet.out
#SBATCH -e /nfs/stak/users/fangzha/projects/ftw-mappingafrica-integration/slurm/logs/%j_smoke_unet.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --exclude=dgxh-1
#SBATCH --time=00:30:00
#SBATCH --qos=normal

# Phase-2 smoke: MS 200 steps, PS 50 steps, ftp-b3 init 20 steps, then MS probs on the test split.
set -e
P=/nfs/stak/users/fangzha/projects/ftw-mappingafrica-integration
OUT=/nfs/hpc/share/fangzha/outputs/uganda_bakeoff/smoke_unet/${SLURM_JOB_ID}
mkdir -p "${OUT}"
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda deactivate || true
conda activate /nfs/hpc/share/fangzha/envs/ftw
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
cd "${P}"
nvidia-smi --query-gpu=name,memory.total --format=csv
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 2 > "${OUT}/gpu_mem.csv" &
SMI=$!

phase() { echo "PHASE $1 START $(date +%s)" | tee -a "${OUT}/phases.txt"; }
end() { echo "PHASE $1 END $(date +%s)" | tee -a "${OUT}/phases.txt"; }

phase ms
ftw_ma model fit -c configs/uganda_maxar/bakeoff/unet-ms.yaml -- --trainer.max_steps=200 --trainer.default_root_dir="${OUT}/ms" --trainer.enable_progress_bar=false --trainer.log_every_n_steps=10
end ms
phase ps
ftw_ma model fit -c configs/uganda_maxar/bakeoff/unet-ps.yaml -- --trainer.max_steps=50 --trainer.limit_val_batches=5 --trainer.default_root_dir="${OUT}/ps" --trainer.enable_progress_bar=false
end ps
phase ftpb3
ftw_ma model fit -c configs/uganda_maxar/bakeoff/unet-ms-ftpb3.yaml -- --trainer.max_steps=20 --trainer.limit_val_batches=2 --trainer.default_root_dir="${OUT}/ftpb3" --trainer.enable_progress_bar=false
end ftpb3
phase probs
python scripts/predict_coco_probs_unet.py --config configs/uganda_maxar/bakeoff/unet-ms.yaml \
  --checkpoint "${OUT}/ms/lightning_logs/version_${SLURM_JOB_ID}/checkpoints/last.ckpt" --split validate --output "${OUT}/ms_probs"
end probs
kill ${SMI}
ls "${OUT}"/*/lightning_logs/version_0/checkpoints/

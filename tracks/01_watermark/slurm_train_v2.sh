#!/bin/bash
# Single exclusive-node job (--gres=gpu:4) running pixelseal (GPU 0) & trustmark (GPU 1) in parallel,
# keeping existing work_bassim/test_v2_stegastamp.npz (0.6750 OOF mIoU), then assembling submission_v2.pt.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:22:00
#SBATCH --output=/p/scratch/training2666/ouaziz1/TRYON/Hackathon_Watermark_Localisation_Task/output/%u_%j.out

set -euo pipefail
TASK_DIR="/p/scratch/training2666/ouaziz1/TRYON/Hackathon_Watermark_Localisation_Task"
export HF_HOME="$TASK_DIR/.cache/huggingface"
export TORCH_HOME="$TASK_DIR/.cache/torch"
export XDG_CACHE_HOME="$TASK_DIR/.cache"
cd "$TASK_DIR"
source .venv/bin/activate
export PYTHONPATH="$TASK_DIR/work_bassim:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

CUDA_VISIBLE_DEVICES=0 python work_bassim/train_v2.py --method pixelseal  --epochs 15 --bs 24 > "output/${USER}_${SLURM_JOB_ID}_pixelseal.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 python work_bassim/train_v2.py --method trustmark --epochs 15 --bs 24 > "output/${USER}_${SLURM_JOB_ID}_trustmark.log" 2>&1 &
p1=$!

wait "$p0" "$p1"
python work_bassim/postprocess_submit_v2.py --out output/submission_v2.pt

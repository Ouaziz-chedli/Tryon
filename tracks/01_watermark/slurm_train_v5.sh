#!/bin/bash
# Phase 3 single exclusive-node job (--gres=gpu:4) utilizing ALL 4 A100 GPUs in parallel:
#   GPU 0: pixelseal (v5, 7-fold, seed=13)
#   GPU 1: trustmark (v5, 7-fold, v2-style no-aux/no-ms, seed=13)
#   GPU 2: stegastamp (v5, 7-fold, ep1=12/ep2=7, seed=13)
#   GPU 3: pixelseal (v5b, 7-fold, no-aux diversity, seed=99)
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:15:00
#SBATCH --output=/p/scratch/training2666/ouaziz1/TRYON/Hackathon_Watermark_Localisation_Task/output/%u_%j.out

set -euo pipefail
TASK_DIR="/p/scratch/training2666/ouaziz1/TRYON/Hackathon_Watermark_Localisation_Task"
export HF_HOME="$TASK_DIR/.cache/huggingface"
export TORCH_HOME="$TASK_DIR/.cache/torch"
export XDG_CACHE_HOME="$TASK_DIR/.cache"
export HF_HUB_OFFLINE=1
cd "$TASK_DIR"
source .venv/bin/activate
export PYTHONPATH="$TASK_DIR/work_bassim:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

# Ensure Phase 2 submission_v5_rect.pt is built with updated grid & auto-submitted if better
python work_bassim/postprocess_submit_v4.py --out output/submission_v5_rect.pt
python work_bassim/auto_submit.py output/submission_v5_rect.pt || true

CUDA_VISIBLE_DEVICES=0 python work_bassim/train_v5.py --tag v5  --method pixelseal  --folds 7 --ep1 9  --ep2 7 --bs 24 --seed 13 > "output/${USER}_${SLURM_JOB_ID}_v5_ps.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 python work_bassim/train_v5.py --tag v5  --method trustmark  --folds 7 --ep1 10 --ep2 8 --bs 24 --seed 13 > "output/${USER}_${SLURM_JOB_ID}_v5_tm.log" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=2 python work_bassim/train_v5.py --tag v5  --method stegastamp --folds 7 --ep1 12 --ep2 7 --bs 24 --seed 13 > "output/${USER}_${SLURM_JOB_ID}_v5_ss.log" 2>&1 &
p2=$!
CUDA_VISIBLE_DEVICES=3 python work_bassim/train_v5.py --tag v5b --method pixelseal  --folds 7 --ep1 9  --ep2 7 --bs 24 --seed 99 --no_aux > "output/${USER}_${SLURM_JOB_ID}_v5b_ps.log" 2>&1 &
p3=$!

wait "$p0" "$p1" "$p2" "$p3"
python work_bassim/postprocess_submit_v4.py --out output/submission_v5.pt
python work_bassim/auto_submit.py output/submission_v5.pt || true

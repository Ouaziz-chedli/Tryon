#!/bin/bash
# Phase 5 single exclusive-node job (--gres=gpu:4): Hole Specialist on all 3 methods in parallel
# (+ PixelSeal seed=99 diversity on GPU 3), then blend with generalist ensemble on full_acc >= 0.72.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:55:00
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

CUDA_VISIBLE_DEVICES=0 python work_bassim/train_holes.py --tag hole --method pixelseal  --ep1 8  --ep2 8 --seed 31 > "output/${USER}_${SLURM_JOB_ID}_hole_ps.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 python work_bassim/train_holes.py --tag hole --method trustmark  --ep1 10 --ep2 8 --seed 31 > "output/${USER}_${SLURM_JOB_ID}_hole_tm.log" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=2 python work_bassim/train_holes.py --tag hole --method stegastamp --ep1 12 --ep2 8 --seed 31 > "output/${USER}_${SLURM_JOB_ID}_hole_ss.log" 2>&1 &
p2=$!
CUDA_VISIBLE_DEVICES=3 python work_bassim/train_holes.py --tag hole2 --method pixelseal --ep1 8  --ep2 8 --seed 77 > "output/${USER}_${SLURM_JOB_ID}_hole2_ps.log" 2>&1 &
p3=$!

for pid in "$p0" "$p1" "$p2" "$p3"; do
    wait "$pid" || echo "WARN: worker pid $pid exited non-zero"
done
python work_bassim/postprocess_submit_v4.py --out output/submission_holes.pt
python work_bassim/auto_submit.py output/submission_holes.pt || true

#!/bin/bash
# Phase 4 single exclusive-node job (--gres=gpu:4): transductive self-training with
# triple-filtered test pseudo-labels across all 3 methods in parallel, then final ensemble.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:05:00
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

CUDA_VISIBLE_DEVICES=0 python work_bassim/pseudo_label.py --method pixelseal  > "output/${USER}_${SLURM_JOB_ID}_p4_ps.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 python work_bassim/pseudo_label.py --method trustmark  > "output/${USER}_${SLURM_JOB_ID}_p4_tm.log" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=2 python work_bassim/pseudo_label.py --method stegastamp > "output/${USER}_${SLURM_JOB_ID}_p4_ss.log" 2>&1 &
p2=$!
CUDA_VISIBLE_DEVICES=3 python work_bassim/train_v5.py --tag v5 --method stegastamp --folds 7 --ep1 12 --ep2 7 --bs 24 --seed 13 > "output/${USER}_${SLURM_JOB_ID}_v5_ss.log" 2>&1 &
p3=$!

wait "$p0" "$p1" "$p2" "$p3"
python work_bassim/postprocess_submit_v4.py --out output/submission_v6p.pt
python work_bassim/auto_submit.py output/submission_v6p.pt || true

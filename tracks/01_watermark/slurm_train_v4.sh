#!/bin/bash
# Single exclusive-node job (--gres=gpu:4) running v4.1 (--tag v41, reusing cached ms_maps)
# on GPU 0/1/2 in parallel, then assembling ensemble v2+v3+v4+v41 into submission_v41.pt.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
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

CUDA_VISIBLE_DEVICES=0 python work_bassim/train_v4.py --tag v41 --method pixelseal  --ep1 6  --ep2 5 --bs 24 > "output/${USER}_${SLURM_JOB_ID}_pixelseal.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 python work_bassim/train_v4.py --tag v41 --method trustmark  --ep1 6  --ep2 5 --bs 24 > "output/${USER}_${SLURM_JOB_ID}_trustmark.log" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=2 python work_bassim/train_v4.py --tag v41 --method stegastamp --ep1 12 --ep2 6 --bs 24 > "output/${USER}_${SLURM_JOB_ID}_stegastamp.log" 2>&1 &
p2=$!

wait "$p0" "$p1" "$p2"
python work_bassim/postprocess_submit_v4.py --out output/submission_v41.pt

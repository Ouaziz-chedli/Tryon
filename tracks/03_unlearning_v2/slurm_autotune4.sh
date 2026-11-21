#!/bin/bash
#SBATCH --job-name=tryon-autotune
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:25:00
#SBATCH --output=output/autotune/slurm_%j.out
#SBATCH --error=output/autotune/slurm_%j.err

set -euo pipefail
cd /p/scratch/training2666/ouaziz1/TRYON/unlearning
mkdir -p output/autotune

export PYTHONPATH="/p/scratch/training2666/ouaziz1/TRYON/unlearning:/p/scratch/training2666/ouaziz1/TRYON/unlearning/harness"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u harness/m1_soup.py --probe-bound 3.5
echo "M1_SOUP_DONE"
for gid in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gid .venv/bin/python -u harness/auto_tune_4gpu.py --worker-id $gid --num-workers 4 &
done
wait
echo "AUTOTUNE_ALL_DONE"

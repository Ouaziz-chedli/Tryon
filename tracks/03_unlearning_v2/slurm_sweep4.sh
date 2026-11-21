#!/bin/bash
#SBATCH --job-name=tryon-ul-b4
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:15:00
#SBATCH --output=output/bounded_sweep/slurm_%j.out
#SBATCH --error=output/bounded_sweep/slurm_%j.err

set -euo pipefail
cd /p/scratch/training2666/ouaziz1/TRYON/unlearning
mkdir -p output/bounded_sweep

export PYTHONPATH="/p/scratch/training2666/ouaziz1/TRYON/unlearning:/p/scratch/training2666/ouaziz1/TRYON/unlearning/harness"
for gid in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gid .venv/bin/python -u harness/sweep_bounded_4gpu.py --worker-id $gid --num-workers 4 &
done
wait
echo "ALL_4_WORKERS_DONE"

#!/bin/bash
#SBATCH --job-name=sweep_grid
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --array=0-5%4
#SBATCH --output=output/sweep_grid/slurm_%A_%a.out
#SBATCH --error=output/sweep_grid/slurm_%A_%a.err

# General-partition (no reservation) AT grid + multi-seed soups, isolated in harness_grid/
# so the shared harness/ used by other in-flight jobs is never touched.
# Grid:  6 array tasks x 4 workers = 24 GPU slots, 2 configs each via --worker-id/--num-workers.
# Soup:  resubmit with --array=0 --time=00:20:00 --dependency=afterok:<grid_job_id>
#        and --export=ALL,SLURM_PHASE=soup (4 soup workers, eval-only).

set -euo pipefail
cd /p/scratch/training2666/ouaziz1/TRYON/unlearning
mkdir -p output/sweep_grid

export PYTHONPATH="/p/scratch/training2666/ouaziz1/TRYON/unlearning/harness_grid:/p/scratch/training2666/ouaziz1/TRYON/unlearning"
PHASE="${SLURM_PHASE:-grid}"
NUM_TASKS="${SLURM_ARRAY_TASK_COUNT:-6}"
BASE_ID=$((SLURM_ARRAY_TASK_ID * 4))

for gid in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gid .venv/bin/python -u harness_grid/sweep_grid.py \
        --phase "$PHASE" \
        --worker-id $((BASE_ID + gid)) \
        --num-workers $((4 * NUM_TASKS)) &
done
wait
echo "SWEEPGRID_TASK_DONE phase=$PHASE task=$SLURM_ARRAY_TASK_ID"

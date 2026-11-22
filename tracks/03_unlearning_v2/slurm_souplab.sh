#!/bin/bash
#SBATCH --job-name=soup_lab
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:45:00
#SBATCH --output=output/soup_lab/slurm_%A.out
#SBATCH --error=output/soup_lab/slurm_%A.err

# SoupLab: 16-soup combinatorial queue over the in-window sweep_grid runs, ONE job.
# Requires the grid rerun complete: output/sweep_grid/SG*/{model.pt,metrics.json}.
# Partition: dc-gpu-devel by default (idle nodes); override with
#   sbatch --partition=<idle-partition> slurm_souplab.sh
# Resume-safe: soups with an existing metrics.json are skipped on resubmit.
# Output: output/soup_lab/SL_*/{model.pt,metrics.json} + ranking.json
#         (submission priority order, orchestrator submits the top-2 at 15-min cadence).

set -euo pipefail
cd /p/scratch/training2666/ouaziz1/TRYON/unlearning
mkdir -p output/soup_lab

export PYTHONPATH="/p/scratch/training2666/ouaziz1/TRYON/unlearning/harness_grid:/p/scratch/training2666/ouaziz1/TRYON/unlearning"

for gid in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gid .venv/bin/python -u harness_grid/run_soup_lab.py \
        --phase soup \
        --worker-id $gid \
        --num-workers 4 &
done
wait

# CPU-only final pass: aggregate every soup row into ranking.json + print the queue.
.venv/bin/python -u harness_grid/run_soup_lab.py --phase rank
echo "SOUPLAB_DONE"

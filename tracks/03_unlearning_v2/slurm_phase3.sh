#!/bin/bash
#SBATCH --job-name=phase3_views
#SBATCH --account=training2666
#SBATCH --partition=PLACEHOLDER_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=01:00:00
#SBATCH --output=output/phase3_views/slurm_%j.out
#SBATCH --error=output/phase3_views/slurm_%j.err

# Phase 3 two-view asymmetric grid (rupture plan 3.1/3.2): 10 runs
# (8 two-view: forget_view x ce{0.85,0.95} x seed{42,101} + 2 mono-view controls ce0.90),
# 4 workers on 4 GPUs of ONE task (no array needed). Each run ends with the per-half
# exact/altered diagnostic as a subprocess on the same GPU.
#
# Prereq on the login node BEFORE sbatch (SBATCH opens the log at submit time):
#   cd $HOME/tryon_ws/unl && mkdir -p output/phase3_views
#   # deploy this harness copy as harness_phase3/ (isolated, never touch shared harness/)
#   # partition: pick one with idle nodes (sinfo; often dc-gpu-devel) and replace
#   # PLACEHOLDER_PARTITION above.
#
# Expected: ~3 min train + ~1 min eval + ~6 min per-half per run, 2-3 runs per worker.

set -euo pipefail
cd "$HOME/tryon_ws/unl"

export PYTHONPATH="$HOME/tryon_ws/unl/harness_phase3:$HOME/tryon_ws/unl"
for gid in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gid .venv/bin/python -u harness_phase3/run_phase3_views.py \
        --worker-id $gid \
        --num-workers 4 &
done
wait
echo "PHASE3_TASK_DONE"

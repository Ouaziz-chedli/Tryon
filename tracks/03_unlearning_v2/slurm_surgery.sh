#!/bin/bash
#SBATCH --job-name=head_surgery
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=output/surgery/slurm_%A.out
#SBATCH --error=output/surgery/slurm_%A.err

# Escanade-guided weight surgery (no training): 8 dosed rank-1 / head-scaling configs on
# the moderate in-window candidate SG_ce0.90_b2.0_kl2.8_s101, isolated in harness_surgery/
# so the shared harness/ used by other in-flight jobs is never touched. Eval-only, 1 GPU.
# Doctrine: before sbatch run `sinfo -h -o '%P %t %D %N' | grep -v alloc` and retarget
# --partition to an idle one (often dc-gpu-devel, MaxTime=UNLIMITED).

set -euo pipefail
cd /p/scratch/training2666/ouaziz1/TRYON/unlearning
mkdir -p output/surgery

export PYTHONPATH="/p/scratch/training2666/ouaziz1/TRYON/unlearning/harness_surgery:/p/scratch/training2666/ouaziz1/TRYON/unlearning"
.venv/bin/python -u harness_surgery/run_head_surgery.py \
    --candidate output/sweep_grid/SG_ce0.90_b2.0_kl2.8_s101
echo "HEAD_SURGERY_DONE"

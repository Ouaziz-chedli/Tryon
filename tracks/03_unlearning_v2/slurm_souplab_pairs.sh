#!/bin/bash
#SBATCH --job-name=soup_pairs
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=output/soup_lab/slurm_pairs_%A.out
#SBATCH --error=output/soup_lab/slurm_pairs_%A.err

# SoupLab custom pairs: the two JurecaSweep off-grid recommendations (journal 08h45).
#   SL_custom_PA2xAT14_5050   = PA2 (diamond4 scratch) x #14 ($HOME/tryon_ws/pack) 50/50
#   SL_custom_PA1xDIAhigh_6040 = PA1 x DIA_highstd (diamond4 scratch) 60/40
# NO sweep_grid dependency: safe to run while the grid rerun is still queued.
# Partition: dc-gpu-devel by default; override with
#   sbatch --partition=<idle-partition> slurm_souplab_pairs.sh
# Output: output/soup_lab/SL_custom_*/{model.pt,metrics.json} + ranking.json (scratch doctrine).
# Resume-safe: soups with metrics.json AND a >1MB checkpoint are skipped on resubmit.
# Paths sanity-check BEFORE sbatch:
#   ls output/diamond4/PA1_AT_ce085_b17_k20_kl26_s77.pt \
#      output/diamond4/PA2_AT_ce095_b20_k22_kl30_s123.pt \
#      output/diamond4/DIA_highstd_AT_ce120_b12_k20_kl26.pt \
#      $HOME/tryon_ws/pack/AT_ce0.80_b1.9_k20_kl28.pt

set -euo pipefail
cd /p/scratch/training2666/ouaziz1/TRYON/unlearning
mkdir -p output/soup_lab

export PYTHONPATH="/p/scratch/training2666/ouaziz1/TRYON/unlearning/harness_grid:/p/scratch/training2666/ouaziz1/TRYON/unlearning"

.venv/bin/python -u harness_grid/run_soup_lab.py --phase pairs
.venv/bin/python -u harness_grid/run_soup_lab.py --phase rank
echo "SOUPLAB_PAIRS_DONE"

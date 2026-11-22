#!/bin/bash
#SBATCH --job-name=vgt_rank
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --output=output/vgt_rank/slurm_%j.out
#SBATCH --error=output/vgt_rank/slurm_%j.err

set -euo pipefail
jutil env activate -p training2666
WS_UNL=$HOME/tryon_ws/unl
HARNESS=$WS_UNL/harness
VENV_PYTHON=$WS_UNL/.venv/bin/python
ORIG_MODEL=$WS_UNL/model.pt
ORIG_IMAGES=$WS_UNL/forget_images.npz
cd "$HARNESS"
mkdir -p output/vgt_rank
PYTHONPATH="$HARNESS" "$VENV_PYTHON" -u vgt_rank_all.py \
  --orig "$ORIG_MODEL" \
  --images "$ORIG_IMAGES" \
  --ckpts "$HOME"/tryon_ws/pack/*.pt \
  --max-images 3500 \
  --output output/vgt_rank/vgt_rank_report.json
echo VGT_RANK_DONE

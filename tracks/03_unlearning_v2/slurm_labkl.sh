#!/bin/bash
#SBATCH --job-name=labkl
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=output/lab_kl/slurm_%j.out
#SBATCH --error=output/lab_kl/slurm_%j.err

# P4.1 lab-KL: build the reference pair (orig_lab saturated on F + gold_lab on a retain-proxy
# without F) then validate the local judge on the 11 server-scored checkpoints in ~/tryon_ws/pack/.
# Prereqs in $HOME/tryon_ws/unl/harness/: eval_harness.py, contracts.py, task_template.py,
# lab/server_metrics.py (+ lab/__init__.py) and lab_kl/ at the harness root. General partition
# (doctrine: never queue inside the cispahack reservation).

set -euo pipefail
jutil env activate -p training2666
WS_UNL=$HOME/tryon_ws/unl
HARNESS=$WS_UNL/harness
VENV_PYTHON=$WS_UNL/.venv/bin/python
ORIG_MODEL=$WS_UNL/model.pt
ORIG_IMAGES=$WS_UNL/forget_images.npz
cd "$HARNESS"
mkdir -p output/lab_kl

PYTHONPATH="$HARNESS" "$VENV_PYTHON" -u lab_kl/build_lab_kl.py \
  --orig "$ORIG_MODEL" \
  --images "$ORIG_IMAGES" \
  --output-dir output/lab_kl

PYTHONPATH="$HARNESS" "$VENV_PYTHON" -u lab_kl/run_lab_kl_judge.py \
  --images "$ORIG_IMAGES" \
  --orig-lab output/lab_kl/orig_lab.pt \
  --gold-lab output/lab_kl/gold_lab.pt \
  --ckpts "$HOME"/tryon_ws/pack/*.pt \
  --output output/lab_kl/lab_kl_judge_report.json

echo LABKL_DONE

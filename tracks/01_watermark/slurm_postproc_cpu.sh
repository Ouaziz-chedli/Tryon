#!/bin/bash
# Postprocess v5 on an idle dc-cpu node (fallback if the devel GPU job is killed
# before its in-job postprocess step). Produces submission_rects.pt + auto-submit.
#SBATCH --account=training2666
#SBATCH --partition=dc-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --output=/p/scratch/training2666/ouaziz1/TRYON/Hackathon_Watermark_Localisation_Task/output/%u_%j.out

set -euo pipefail
TASK_DIR="/p/scratch/training2666/ouaziz1/TRYON/Hackathon_Watermark_Localisation_Task"
cd "$TASK_DIR"
source .venv/bin/activate
export PYTHONPATH="$TASK_DIR/work_bassim:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

mkdir -p output work_bassim
rm -f work_bassim/partial_*.npz /tmp/partial_*.npz
df -h "$TASK_DIR" || true

python work_bassim/postprocess_submit_v5.py --out output/submission_rects.pt
python work_bassim/auto_submit.py output/submission_rects.pt || true

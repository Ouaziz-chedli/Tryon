#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:30:00
#SBATCH --output=output/%j.out

set -euo pipefail

TASK_DIR="${TASK_DIR:-/p/scratch/training2666/$USER/unlearning}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p output
source "$TASK_DIR/.venv/bin/activate"

PYTHONPATH="$SCRIPT_DIR:$TASK_DIR" srun python "$SCRIPT_DIR/run_unlearn.py" \
  --model-path "$TASK_DIR/model.pt" \
  --forget-path "$TASK_DIR/forget_images.npz" \
  --output-path "$TASK_DIR/submission.pt" \
  --steps 150 \
  --batch-size 64 \
  --lr 2e-5 \
  --npo-beta 0.1 \
  --retain-kl-weight 2.0 \
  --l2-weight 1e-3 \
  --salun-keep-ratio 0.2 \
  --target-forget-loss 1.0

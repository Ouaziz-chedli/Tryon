#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:45:00
#SBATCH --output=/p/scratch/training2666/ouaziz1/TRYON/unlearning/output/benchmark_%j.out

set -euo pipefail

TASK_DIR="/p/scratch/training2666/ouaziz1/TRYON/unlearning"
HARNESS_DIR="$TASK_DIR/harness"

cd "$TASK_DIR"
mkdir -p "$TASK_DIR/output/benchmark"
source "$TASK_DIR/.venv/bin/activate"

PYTHONPATH="$HARNESS_DIR:$TASK_DIR" srun python "$HARNESS_DIR/benchmark.py" \
  --model-path "$TASK_DIR/model.pt" \
  --forget-path "$TASK_DIR/forget_images.npz" \
  --out-dir "$TASK_DIR/output/benchmark_v3" \
  --save-checkpoints

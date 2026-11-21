#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --job-name=tryon-sel3f
#SBATCH --output=/p/scratch/training2666/ouaziz1/TRYON/unlearning/output/sel3f_%j.out

set -euo pipefail
source /p/scratch/training2666/ouaziz1/TRYON/unlearning/.venv/bin/activate

TASK_DIR="/p/scratch/training2666/ouaziz1/TRYON/unlearning"
HARNESS_DIR="$TASK_DIR/harness_v2"
OUT_DIR="$TASK_DIR/output/selective_strong"
mkdir -p "$OUT_DIR"
cd "$TASK_DIR"

export PYTHONPATH="$HARNESS_DIR:$TASK_DIR"
python3 "$HARNESS_DIR/tests/test_selective_benchmark.py"

for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$gpu python3 -u "$HARNESS_DIR/selective_benchmark.py" \
    --worker-id "$gpu" \
    --num-workers 4 \
    --num-folds 3 \
    --out-dir "$OUT_DIR" &
done
wait
echo "ALL_4_WORKERS_DONE"

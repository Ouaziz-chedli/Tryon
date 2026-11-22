#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=00:40:00
#SBATCH --job-name=tryon-t3-4gpu
#SBATCH --output=/p/scratch/training2666/ouaziz1/TRYON/unlearning/output/t3_4gpu_%j.out

set -euo pipefail
source /p/scratch/training2666/ouaziz1/TRYON/unlearning/.venv/bin/activate

TASK_DIR="/p/scratch/training2666/ouaziz1/TRYON/unlearning"
HARNESS_DIR="$TASK_DIR/harness_v2"
OUT_H="$TASK_DIR/output/track_h"
OUT_SEL="$TASK_DIR/output/selective_bell"
mkdir -p "$OUT_H" "$OUT_SEL"
cd "$TASK_DIR"

export PYTHONPATH="$HARNESS_DIR:$TASK_DIR"

echo "=== Running Unit Tests ==="
python3 -c "import tests.test_eval_harness as t; [getattr(t, fn)() for fn in dir(t) if fn.startswith('test_')]; print('EVAL_HARNESS_TESTS_OK')"
python3 "$HARNESS_DIR/tests/test_selective_benchmark.py"
python3 "$HARNESS_DIR/tests/test_track_h.py"

echo "=== Launching 4-GPU Parallel Workload ==="
CUDA_VISIBLE_DEVICES=0 python3 -u "$HARNESS_DIR/run_track_h.py" \
  --round 7 --eval-seed 777 --out-dir "$OUT_H" &

for w in 0 1 2; do
  gpu=$((w + 1))
  CUDA_VISIBLE_DEVICES=$gpu python3 -u "$HARNESS_DIR/selective_benchmark.py" \
    --worker-id "$w" --num-workers 3 --num-folds 3 --out-dir "$OUT_SEL" &
done

wait
echo "ALL_4_GPUS_DONE"

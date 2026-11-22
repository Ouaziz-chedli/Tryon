#!/bin/bash
#SBATCH --job-name=tryon-dia4
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:35:00
#SBATCH --output=output/diamond4/slurm_%j.out
#SBATCH --error=output/diamond4/slurm_%j.err

set -euo pipefail
cd /p/scratch/training2666/ouaziz1/TRYON/unlearning
mkdir -p output/diamond4

# CPU/GPU heartbeat once per minute
(
  while true; do
    echo "[hb $(date +%H:%M:%S)] load:$(cut -d' ' -f1-3 /proc/loadavg) | gpu:$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
    sleep 60
  done
) &
HB=$!

export PYTHONPATH="/p/scratch/training2666/ouaziz1/TRYON/unlearning:/p/scratch/training2666/ouaziz1/TRYON/unlearning/harness"
for gid in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gid .venv/bin/python -u harness/diamond4_sweep.py --worker-id $gid --num-workers 4 &
done
wait
kill $HB 2>/dev/null || true
echo "ALL_4_WORKERS_DONE"

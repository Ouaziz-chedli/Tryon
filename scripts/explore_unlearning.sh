#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:10:00
#SBATCH --output=/p/scratch/training2666/ouaziz1/TRYON/unlearning/output/explore_%j.out

set -euo pipefail

TASK_DIR="/p/scratch/training2666/ouaziz1/TRYON/unlearning"
cd "$TASK_DIR"
source "$TASK_DIR/.venv/bin/activate"
PYTHONPATH="$TASK_DIR:$TASK_DIR/explore" srun python "$TASK_DIR/explore/explore_unlearning.py"

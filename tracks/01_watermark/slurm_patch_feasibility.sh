#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/p/scratch/training2666/ouaziz1/TRYON/Hackathon_Watermark_Localisation_Task/output/%u_%j.out

set -euo pipefail
TASK_DIR="/p/scratch/training2666/ouaziz1/TRYON/Hackathon_Watermark_Localisation_Task"
cd "$TASK_DIR"
source .venv/bin/activate
srun python work_bassim/patch_feasibility.py

#!/bin/bash
# Fetch logs/artifacts of cancelled job 15905508 from scratch to home
# (login node has no /p/scratch mount).
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00

T=/p/scratch/training2666/ouaziz1/TRYON/Hackathon_Watermark_Localisation_Task
D=/p/home/jusers/bassim1/jureca/stage_wm/fetch_15905508
mkdir -p "$D"
cp "$T/output/bassim1_15905508.out" "$D/" 2>/dev/null
cp "$T"/output/*_15905508_*.log "$D/" 2>/dev/null
ls -la "$T"/output/*15905508* > "$D/artifacts.txt" 2>&1
echo "DUMP DONE" > "$D/ok.txt"

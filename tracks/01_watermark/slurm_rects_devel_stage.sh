#!/bin/bash
# Phase 6/7 on dc-gpu-devel (no reservation) WITH staging sync prologue.
# Staging (login node has no /p/scratch mount): *.py/*.sh are scp'd to
# $HOME/stage_wm/ first; this job copies them into work_bassim/ on scratch,
# then runs the standard rects_devel workflow:
#   GPU0: pixelseal  rect_search_v2 ; train_rects
#   GPU1: trustmark  rect_search_v2 ; train_rects
#   GPU2: stegastamp rect_search_v2
#   GPU3: stegastamp train_rects
# then postprocess_submit_v5 (rect override + rect/hole blends) + auto_submit.
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:50:00
#SBATCH --output=/p/scratch/training2666/ouaziz1/TRYON/Hackathon_Watermark_Localisation_Task/output/%u_%j.out

set -euo pipefail
TASK_DIR="/p/scratch/training2666/ouaziz1/TRYON/Hackathon_Watermark_Localisation_Task"
STAGE_DIR="$HOME/stage_wm"
export HF_HOME="$TASK_DIR/.cache/huggingface"
export TORCH_HOME="$TASK_DIR/.cache/torch"
export XDG_CACHE_HOME="$TASK_DIR/.cache"
export HF_HUB_OFFLINE=1

# 0. Sync staged code (login node cannot see /p/scratch, compute nodes can).
cp "$STAGE_DIR"/*.py "$STAGE_DIR"/*.sh "$TASK_DIR/work_bassim/"
echo "STAGE SYNC DONE: $(ls "$STAGE_DIR"/*.py "$STAGE_DIR"/*.sh | wc -l) files -> $TASK_DIR/work_bassim/"

cd "$TASK_DIR"
source .venv/bin/activate
export PYTHONPATH="$TASK_DIR/work_bassim:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

mkdir -p output work_bassim
# Quota guard (job 15905297 died on "Disk quota exceeded"): drop stale
# per-fold partials from previous runs (final oof_*/test_*.npz are kept).
rm -f work_bassim/partial_*.npz /tmp/partial_*.npz
df -h "$TASK_DIR" || true

CUDA_VISIBLE_DEVICES=0 bash -c "python work_bassim/rect_search_v2.py --method pixelseal --max_windows 1100 > output/${USER}_${SLURM_JOB_ID}_r2_ps.log 2>&1; python work_bassim/train_rects.py --tag rect --method pixelseal --seed 31 >> output/${USER}_${SLURM_JOB_ID}_r2_ps.log 2>&1" &
p0=$!
CUDA_VISIBLE_DEVICES=1 bash -c "python work_bassim/rect_search_v2.py --method trustmark --max_windows 900 > output/${USER}_${SLURM_JOB_ID}_r2_tm.log 2>&1; python work_bassim/train_rects.py --tag rect --method trustmark --seed 31 >> output/${USER}_${SLURM_JOB_ID}_r2_tm.log 2>&1" &
p1=$!
CUDA_VISIBLE_DEVICES=2 bash -c "python work_bassim/rect_search_v2.py --method stegastamp --max_windows 700 > output/${USER}_${SLURM_JOB_ID}_r2_ss.log 2>&1" &
p2=$!
CUDA_VISIBLE_DEVICES=3 bash -c "python work_bassim/train_rects.py --tag rect --method stegastamp --seed 31 > output/${USER}_${SLURM_JOB_ID}_rt_ss.log 2>&1" &
p3=$!

for pid in "$p0" "$p1" "$p2" "$p3"; do
    wait "$pid" || echo "WARN: worker pid $pid exited non-zero"
done
python work_bassim/postprocess_submit_v5.py --out output/submission_rects.pt
python work_bassim/auto_submit.py output/submission_rects.pt || true

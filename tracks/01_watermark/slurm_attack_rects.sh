#!/bin/bash
# ATTACK THE RECT HOLE (A+B+C) on dc-gpu-devel (no reservation).
#   GPU 0: diag_rect (all) -> rect_search_v3 pixelseal -> train_rects pixelseal
#   GPU 1: rect_search_v3 trustmark (smoke pre-flight, then full)
#   GPU 2: rect_search_v3 stegastamp
#   GPU 3: train_rects stegastamp --hf_residue (6ch model)
# then postprocess_submit_v6 (prefers bbox_v7, confidence-gated rect blend,
# end-to-end variant selection) + auto-submit.
# Job name is explicit so the team can identify it (tryon-t1-rects).
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 --job-name=tryon-t1-rects
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:55:00
#SBATCH --output=/p/scratch/training2666/ouaziz1/TRYON/Hackathon_Watermark_Localisation_Task/output/%u_%j.out

set -euo pipefail
TASK_DIR="/p/scratch/training2666/ouaziz1/TRYON/Hackathon_Watermark_Localisation_Task"
STAGE_DIR="$HOME/stage_wm"
export HF_HOME="$TASK_DIR/.cache/huggingface"
export TORCH_HOME="$TASK_DIR/.cache/torch"
export XDG_CACHE_DIR="$TASK_DIR/.cache"
export HF_HUB_OFFLINE=1

# 0. Sync staged code (login node has no /p/scratch mount, compute nodes do).
cp "$STAGE_DIR"/*.py "$STAGE_DIR"/*.sh "$TASK_DIR/work_bassim/"
echo "STAGE SYNC DONE: $(ls "$STAGE_DIR"/*.py "$STAGE_DIR"/*.sh | wc -l) files -> $TASK_DIR/work_bassim/"

cd "$TASK_DIR"
source .venv/bin/activate
export PYTHONPATH="$TASK_DIR/work_bassim:${PYTHONPATH:-}"
export OMP_NUM_THREADS=16

mkdir -p output work_bassim
# Quota guard (jobs 15905297/15905508 died on "Disk quota exceeded"): drop stale
# per-fold partials from previous runs (final oof_*/test_*.npz are kept).
rm -f work_bassim/partial_*.npz /tmp/partial_*.npz
df -h "$TASK_DIR" || true

CUDA_VISIBLE_DEVICES=0 bash -c "python work_bassim/diag_rect.py > output/${USER}_${SLURM_JOB_ID}_diag.log 2>&1; python work_bassim/rect_search_v3.py --method pixelseal --max_windows 1100 >> output/${USER}_${SLURM_JOB_ID}_v3_ps.log 2>&1; python work_bassim/train_rects.py --tag rect --method pixelseal --seed 31 >> output/${USER}_${SLURM_JOB_ID}_v3_ps.log 2>&1" &
p0=$!
CUDA_VISIBLE_DEVICES=1 bash -c "python work_bassim/rect_search_v3.py --method trustmark --smoke > output/${USER}_${SLURM_JOB_ID}_v3_tm_smoke.log 2>&1 && python work_bassim/rect_search_v3.py --method trustmark --max_windows 900 > output/${USER}_${SLURM_JOB_ID}_v3_tm.log 2>&1" &
p1=$!
CUDA_VISIBLE_DEVICES=2 bash -c "python work_bassim/rect_search_v3.py --method stegastamp --max_windows 700 > output/${USER}_${SLURM_JOB_ID}_v3_ss.log 2>&1" &
p2=$!
CUDA_VISIBLE_DEVICES=3 bash -c "python work_bassim/train_rects.py --tag rect --method stegastamp --seed 31 --hf_residue > output/${USER}_${SLURM_JOB_ID}_rt_ss_hf.log 2>&1" &
p3=$!

for pid in "$p0" "$p1" "$p2" "$p3"; do
    wait "$pid" || echo "WARN: worker pid $pid exited non-zero"
done
python work_bassim/postprocess_submit_v6.py --out output/submission_attack.pt
python work_bassim/auto_submit.py output/submission_attack.pt || true

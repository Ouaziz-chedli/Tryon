#!/bin/bash
#SBATCH --job-name=scrub2
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --array=0-6%7
#SBATCH --output=output/scrub_r2/slurm_%A_%a.out
#SBATCH --error=output/scrub_r2/slurm_%A_%a.err

# Partition rule: dc-gpu by default (works with --reservation=cispahack); NEVER dc-hwai
# (closed to our account; pairing it with the reservation is rejected as invalid).
# SCRUB+R rounds 2-3: PD-depth grids, 1 GPU per candidate, round via SLURM_ROUND (default 2).
# Round 2: 7 round2_candidates() -- array=0-6%7, out output/scrub_r2.
# Round 3: 3 round3_candidates() (SR8_balanced/push_mu/high_pd, 242-288 ascent steps,
#          rewind target mu+sig 4.2). Submit on the reservation:
#   mkdir -p output/scrub_r3
#   sbatch --array=0-2%3 --time=01:00:00 --export=ALL,SLURM_ROUND=3 \
#       --output=output/scrub_r3/slurm_%A_%a.out \
#       --error=output/scrub_r3/slurm_%A_%a.err slurm_scrub2.sh
# Round 4 (GO once lab-KL returns): 5 round4_candidates() = 3 SR9 rest-KL pins
#   (SR9_rest02/rest035/rest05) + 2 SR10 per-image PD-target mimics (SR10_pdt05/pdt10);
#   upload methods/scrub_r_rest.py next to methods/scrub_r.py first. Both families go
#   together at GO. Two waves on one 4-GPU node (reservation mandatory, generals
#   saturated): wave 1 = ids 0-3 on GPUs 0-3, wave 2 = id 4 (SR10_pdt10) on GPU 0.
#   mkdir -p output/scrub_r4
#   sbatch --reservation=cispahack --partition=dc-gpu --nodes=1 --ntasks=1 \
#       --gres=gpu:4 --time=02:00:00 --job-name=scrub4 \
#       --output=output/scrub_r4/slurm_%j.out --error=output/scrub_r4/slurm_%j.err \
#       --wrap='cd $HOME/tryon_ws/unl; H=$PWD/harness; export PYTHONPATH=$H:$PWD; \
#         .venv/bin/python $H/tests/test_scrub_r2.py; \
#         for k in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$k .venv/bin/python -u \
#           $H/run_scrub_r2.py --round 4 --candidate-id $k --out-dir output/scrub_r4 & done; wait; \
#         CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u \
#           $H/run_scrub_r2.py --round 4 --candidate-id 4 --out-dir output/scrub_r4'
# Run mkdir BEFORE sbatch (slurm opens the output path at submit) and submit from
# the workspace root so relative paths resolve. Verdict: min L1 toward (1.70, 2.25),
# in_window (S [3.6, 4.4] with sigma >= 2.1), rho_PD >= 0.45, gap logged.
#
# Round 5 (SR11 gate-plate): CLUSTER CUT by organizers (~07:35, slurm associations
# removed) -- the sbatch recipes above are DEAD; round 5 runs on Colab A100 instead
# (single GPU per session, runner auto-sufficient, checkpoints exported by download):
#   cd <dir with model.pt + forget_images.npz + harness files>
#   python3 -u run_scrub_r2.py --round 5 --candidate-id <0-3|-1> --out-dir output/scrub_r5
#   # 2 sessions: session A = ids 0 then 1, session B = ids 2 then 3 (&& chained).

set -euo pipefail
cd "$HOME/tryon_ws/unl"
ROUND="${SLURM_ROUND:-2}"
mkdir -p "output/scrub_r${ROUND}"

# run_scrub_r2.py + methods/ live in harness/ next to the round-1 harness.
HARNESS_DIR="${SLURM_HARNESS_DIR:-$PWD/harness}"
export PYTHONPATH="$HARNESS_DIR:$PWD"

if [[ "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
    .venv/bin/python "$HARNESS_DIR/tests/test_scrub_r2.py"
fi

.venv/bin/python -u "$HARNESS_DIR/run_scrub_r2.py" \
    --round "$ROUND" \
    --candidate-id "${SLURM_ARRAY_TASK_ID:-0}" \
    --out-dir "output/scrub_r${ROUND}"

echo "SCRUB2_TASK_DONE id=${SLURM_ARRAY_TASK_ID:-0}"

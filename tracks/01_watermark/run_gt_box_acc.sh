#!/bin/bash
cd /p/scratch/training2666/ouaziz1/TRYON/Hackathon_Watermark_Localisation_Task || exit 1
export PYTHONPATH=.
for m in pixelseal trustmark stegastamp; do
    echo "=== $m ==="
    .venv/bin/python work_bassim/gt_box_acc.py "$m"
    echo "EXIT:$?"
done
echo "=== DONE ==="

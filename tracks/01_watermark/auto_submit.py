"""Submit a .pt file ONLY if its accompanying .oof score beats output/best_oof.txt.
Respects the 10-minute Track 1 server cooldown via output/last_submit_ts.txt.

    python work_bassim/auto_submit.py output/submission_v5_rect.pt
"""
import os
import sys
import time
import requests
import torch

BASE_URL = "http://34.147.229.55"
TASK_ID = "34-watermark-localisation-image"
BEST_FILE = "output/best_oof.txt"
TS_FILE = "output/last_submit_ts.txt"
KEY_FILE = "work_bassim/.api_key"


def main():
    pt_path = sys.argv[1]
    oof_path = f"{pt_path}.oof"
    if not os.path.exists(pt_path) or not os.path.exists(oof_path):
        print(f"auto_submit: missing {pt_path} or {oof_path}, skipping", flush=True)
        return
    oof = float(open(oof_path).read().strip())
    best = float(open(BEST_FILE).read().strip()) if os.path.exists(BEST_FILE) else 0.6329
    print(f"auto_submit: {pt_path} OOF={oof:.6f} vs best={best:.6f}", flush=True)
    if oof <= best + 1e-4:
        print("auto_submit: did not beat best OOF, skipping upload", flush=True)
        return

    # Validate .pt integrity before consuming a submission slot
    d = torch.load(pt_path, weights_only=True)
    assert d["masks"].shape == (1020, 128, 128) and d["masks"].dtype == torch.bool
    assert os.path.getsize(pt_path) < 20_000_000

    if os.path.exists(TS_FILE):
        last_ts = float(open(TS_FILE).read().strip())
        wait_s = 615.0 - (time.time() - last_ts)
        if wait_s > 0:
            print(f"auto_submit: waiting {wait_s:.0f}s for 10-min cooldown...", flush=True)
            time.sleep(wait_s)

    key = os.environ.get("TRYON_API_KEY") or open(KEY_FILE).read().strip()
    try:
        with open(pt_path, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": key},
                files={"file": (os.path.basename(pt_path), f, "application/octet-stream")},
                timeout=(10, 600),
            )
        print("auto_submit HTTP", r.status_code, r.text[:400], flush=True)
        if r.status_code == 200:
            open(BEST_FILE, "w").write(f"{oof:.6f}\n")
            open(TS_FILE, "w").write(f"{time.time():.1f}\n")
    except Exception as exc:
        print(f"auto_submit upload failed (likely offline compute node; submit from login node): {exc}", flush=True)


if __name__ == "__main__":
    main()

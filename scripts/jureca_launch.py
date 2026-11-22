#!/usr/bin/env python3
"""One-command JURECA job helpers (Team TRYON) — basename-safe, auto-idle partition.

Usage:
  python3 scripts/jureca_launch.py push-extract <tarball> <remote_dir> [--strip N]
  python3 scripts/jureca_launch.py sbatch <remote_slurm_path> [--partition auto|<part>]
  python3 scripts/jureca_launch.py status <job_id>
  python3 scripts/jureca_launch.py collect <job_id> <remote_glob> <local_dir>

'auto' partition = first VALID partition with idle nodes (sinfo), else dc-gpu.
All network via scripts/jureca_xfer.py (TOTP + drift margin).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jureca_xfer import scp_put, ssh_exec  # noqa: E402

VALID_PARTITIONS = ["dc-gpu-devel-sw", "dc-gpu-devel", "dc-gpu", "dc-gpu-large"]


def push_extract(tarball: str, remote_dir: str, strip: int) -> None:
    """scp keeping the basename (no renaming traps), then extract in place."""
    name = Path(tarball).name
    out = scp_put([tarball], "/tmp/", timeout=600)
    if "100%" not in out:
        raise SystemExit(f"scp failed: {out[-200:]}")
    cmd = f"mkdir -p {remote_dir} && tar -xzf /tmp/{name} -C {remote_dir}"
    if strip:
        cmd += f" --strip-components={strip}"
    print(ssh_exec(cmd + f" && echo EXTRACT_OK && ls {remote_dir} | head -5", timeout=120))


def pick_idle_partition() -> str:
    out = ssh_exec("sinfo -h -o '%P|%t|%D' | grep -E '^dc-gpu'", timeout=60)
    for line in out.splitlines():
        part, state, cnt = (line.strip().split("|") + ["?", "?", "?"])[:3]
        if state in ("idle", "mix") and cnt.isdigit() and int(cnt) > 0 and part in VALID_PARTITIONS:
            return part
    return "dc-gpu"


def sbatch(slurm_path: str, partition: str) -> None:
    part = pick_idle_partition() if partition == "auto" else partition
    cmd = (
        f"sed -i 's|#SBATCH --partition=.*|#SBATCH --partition={part}|' {slurm_path}; "
        f"sbatch {slurm_path} 2>&1; squeue -u ouaziz1 -h -o '%i %j %T %R' | head -6"
    )
    print(ssh_exec(cmd, timeout=90))


def status(job_id: str) -> None:
    cmd = f"sacct -n -X -j {job_id} -o JobID%22,State,Elapsed,ExitCode; squeue -j {job_id} -o '%i %T %M %N %R' | tail -2"
    print(ssh_exec(cmd, timeout=60))


def collect(job_id: str, remote_glob: str, local_dir: str) -> None:
    out = ssh_exec(f"sacct -n -X -j {job_id} -o State%12 | head -1", timeout=60)
    if "COMPLETED" not in out:
        print(f"[collect] job not COMPLETED yet: {out.strip()[:80]}")
        return
    ls = ssh_exec(f"ls {remote_glob} 2>/dev/null", timeout=60)
    files = [f for f in ls.split() if f.endswith((".json", ".pt"))]
    if not files:
        print("[collect] no matching files yet")
        return
    os.makedirs(local_dir, exist_ok=True)
    scp_put(files, local_dir + "/", timeout=900)
    print(f"[collect] {len(files)} files -> {local_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="One-command JURECA job helpers.")
    sub = parser.add_subparsers(dest="mode", required=True)
    p1 = sub.add_parser("push-extract")
    p1.add_argument("tarball"); p1.add_argument("remote_dir"); p1.add_argument("--strip", type=int, default=0)
    p2 = sub.add_parser("sbatch")
    p2.add_argument("slurm_path"); p2.add_argument("--partition", default="auto")
    p3 = sub.add_parser("status"); p3.add_argument("job_id")
    p4 = sub.add_parser("collect")
    p4.add_argument("job_id"); p4.add_argument("remote_glob"); p4.add_argument("local_dir")
    args = parser.parse_args()
    if args.mode == "push-extract":
        push_extract(args.tarball, args.remote_dir, args.strip)
    elif args.mode == "sbatch":
        sbatch(args.slurm_path, args.partition)
    elif args.mode == "status":
        status(args.job_id)
    elif args.mode == "collect":
        collect(args.job_id, args.remote_glob, args.local_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

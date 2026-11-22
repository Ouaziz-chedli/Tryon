#!/usr/bin/env python3
"""Fleet scheduler v2 — exploits JURECA GPU capacity permanently (Chef directive).

Fixes from Antigravity review 09h10: scp_get for collect (was inverted upload),
FLEET_IDS interpolated into the remote command (was a local env var, never
forwarded), rebalance measured from submitted_at (Slurm %M is 0:00 while
PENDING), arrays aggregated over ALL sacct lines (a _0 COMPLETED no longer
marks the whole array done), resubmit-failure falls back to pending.

Loop (~every 180 s): snapshot (squeue+sacct+sinfo, one ssh) -> submit gated
specs on idle VALID partitions -> rebalance stuck PENDING -> collect COMPLETED.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jureca_xfer import scp_get, ssh_exec  # noqa: E402

VALID_PARTITIONS = ["dc-gpu-devel-sw", "dc-gpu-devel", "dc-gpu", "dc-gpu-large"]
PENDING_REBALANCE_MIN = 20.0


def log(msg: str) -> None:
    print(f"[fleet {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_queue(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
GENERAL_EXCLUDE = "--exclude=jrc[0198,0215,0217,0231-0233,0245,0257-0258,0283,0314,0321,0323-0324,0327,0329,0347,0349-0350,0362-0363,0374,0387,0394,0396,0410,0414,0423,0442,0444]"  # admin 10h05: general jobs backfill onto resa nodes unless excluded; NEVER combine with --reservation (excludes ALL resa nodes)


def save_queue(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _base_id(token: str) -> str:
    return token.split("_")[0]


def snapshot(tracked_ids: list[str]) -> tuple[dict, list, dict]:
    """One ssh: our live jobs, sacct lines for tracked ids, idle partitions."""
    ids = ",".join(i for i in tracked_ids if i.isdigit()) or "1"
    remote = (
        "squeue -u ouaziz1 -h -o '%i|%T|%M|%P|%R'; echo ===SACCT; "
        f"sacct -n -X -j {ids} -o JobID%22,State,Elapsed,ExitCode 2>/dev/null; echo ===SINFO; "
        "sinfo -h -o '%P|%t|%D' | grep -E '^dc-gpu'"
    )
    out = ssh_exec(remote, timeout=60)
    jobs, sacct, idle = {}, [], {}
    section = None
    for line in out.splitlines():
        clean = line.strip()
        if clean == "===SACCT":
            section = "sacct"; continue
        if clean == "===SINFO":
            section = "sinfo"; continue
        if section is None and "|" in clean and clean.split("|")[0].isdigit():
            jid, state, elapsed, part, reason = (clean.split("|") + [""] * 5)[:5]
            jobs[_base_id(jid)] = {"state": state, "elapsed": elapsed, "partition": part, "reason": reason}
        elif section == "sacct" and clean and not clean.startswith("---"):
            sacct.append(clean)
        elif section == "sinfo" and "|" in clean:
            part, st, cnt = clean.split("|")[:3]
            if st in ("idle", "mix") and cnt.isdigit() and part in VALID_PARTITIONS:
                idle[part] = max(idle.get(part, 0), int(cnt))
    return jobs, sacct, idle


def gate_open(entry: dict, sacct: list) -> tuple[bool, str]:
    """'after:<jobid>' waits for ALL sacct lines of the dependency to be terminal."""
    gate = entry.get("gate", "idle")
    if gate.startswith("after:"):
        dep = gate.split(":", 1)[1]
        lines = [l for l in sacct if l.split() and _base_id(l.split()[0]) == dep]
        if not lines:
            return False, f"dep {dep} not in sacct yet"
        states = " ".join(lines)
        if "RUNNING" in states or "PENDING" in states:
            return False, f"dep {dep} still active"
        return True, f"dep {dep} terminal"
    if gate.startswith("file:"):
        path = gate.split(":", 1)[1]
        out = ssh_exec(f"test -s {path} && echo GATE_FILE_OK", timeout=40)
        return ("GATE_FILE_OK" in out), f"file {path}"
    return True, "always open"


def submit(entry: dict, partition: str) -> str:
    sbatch_cmd = entry["sbatch_cmd"].replace("__PARTITION__", partition)
    # General-queue submissions must not backfill onto the 30 reserved nodes;
    # reserved submissions (--reservation) must NEVER carry the exclude list.
    if "reservation" not in sbatch_cmd and "exclude" not in sbatch_cmd:
        sbatch_cmd = sbatch_cmd.replace("sbatch ", f"sbatch {GENERAL_EXCLUDE} ")
    out = ssh_exec(sbatch_cmd, timeout=60)
    for token in out.split():
        if token.isdigit() and len(token) >= 7:
            return token
    return ""


def collect(entry: dict) -> str:
    """scp_get results back, one file at a time into per-run subdirs (same
    basenames like metrics.json must never collide), verified distinctly."""
    remote_ls = ssh_exec(f"ls {entry['collect_glob']} 2>/dev/null", timeout=40)
    files = [f for f in remote_ls.split() if f.endswith((".json", ".pt"))]
    if not files:
        return "nothing to collect"
    local_dir = entry["local_dir"]
    landed = 0
    for f in files:
        parts = f.rstrip("/").split("/")
        run = parts[-2] if len(parts) >= 2 else "root"
        sub = os.path.join(local_dir, run)
        os.makedirs(sub, exist_ok=True)
        scp_get([f], sub + "/", timeout=600)
        if os.path.exists(os.path.join(sub, os.path.basename(f))):
            landed += 1
    if landed < len(files):
        return f"PARTIAL {landed}/{len(files)}"
    return f"collected {len(files)} files"


def cycle(queue_path: Path) -> None:
    data = load_queue(queue_path)
    jobs_state = data.get("jobs", {})
    tracked = [e["job_id"] for e in jobs_state.values() if e.get("status") == "submitted" and e.get("job_id")]
    snap_jobs, sacct, idle = snapshot(tracked)
    changed = False

    for name, entry in jobs_state.items():
        st = entry.get("status")
        if st == "done":
            continue
        if st == "pending":
            ok, why = gate_open(entry, sacct)
            if not ok:
                continue
            part = entry.get("partition_pref", "dc-gpu")
            chosen = next((p for p in [part] + VALID_PARTITIONS if idle.get(p, 0) > 0), part)
            jid = submit(entry, chosen)
            if jid:
                entry.update(status="submitted", job_id=jid, partition=chosen, submitted_at=time.time())
                changed = True
                log(f"SUBMITTED {name} -> {jid} on {chosen} (gate: {why})")
            else:
                log(f"{name}: sbatch failed, staying pending")
        elif st == "submitted":
            jid = entry["job_id"]
            lines = [l for l in sacct if l.split() and _base_id(l.split()[0]) == jid]
            live = snap_jobs.get(jid, {})
            if live.get("state"):
                mins = (time.time() - entry.get("submitted_at", time.time())) / 60.0
                if live["state"] == "PENDING" and mins > PENDING_REBALANCE_MIN \
                        and "__PARTITION__" in entry["sbatch_cmd"]:
                    alt = next((p for p in VALID_PARTITIONS if idle.get(p, 0) > 0 and p != entry.get("partition")), None)
                    if alt:
                        ssh_exec(f"scancel {jid}", timeout=40)
                        new_jid = submit(entry, alt)
                        if new_jid:
                            log(f"REBALANCED {name}: {jid} -> {new_jid} on {alt}")
                            entry.update(job_id=new_jid, partition=alt, submitted_at=time.time())
                            changed = True
                        else:
                            entry.update(status="pending")
                            entry.pop("job_id", None)
                            log(f"REBALANCE_LOST {name}: scancel done, resubmit failed -> back to pending")
                            changed = True
                continue
            if not lines:
                log(f"WARNING {name} ({jid}): MISSING_IN_SQUEUE_AND_SACCT")
                continue
            states = " ".join(lines)
            if "RUNNING" in states or "PENDING" in states:
                continue
            if "FAILED" in states or "CANCELLED" in states or "TIMEOUT" in states:
                entry.update(status="failed", sacct=lines[0])
                changed = True
                log(f"FAILED {name} ({jid}): {lines[0]}")
            elif "COMPLETED" in states:
                note = collect(entry)
                entry.update(status="done", collected=note)
                changed = True
                log(f"DONE {name} ({jid}) — {note}")
    if changed:
        save_queue(queue_path, data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Permanent JURECA fleet scheduler (TRYON).")
    parser.add_argument("--queue", type=Path, default=Path("tracks/03_unlearning_v2/fleet_queue.json"))
    parser.add_argument("--interval", type=int, default=180)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            cycle(args.queue)
        except Exception as exc:  # noqa: BLE001 — fleet must survive transient ssh/TOTP failures
            log(f"cycle error (continuing): {type(exc).__name__}: {exc}")
        if args.once:
            return 0
        time.sleep(max(args.interval, 61))


if __name__ == "__main__":
    raise SystemExit(main())

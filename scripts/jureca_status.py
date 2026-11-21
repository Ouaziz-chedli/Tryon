#!/usr/bin/env python3
"""Live JURECA Slurm & reservation monitor for Team TRYON."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import pty
import re
import select
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from totp import SECRET, compute_totp  # noqa: E402

TEAM_USERS = ("ouaziz1", "sangkhavongs1", "bassim1")
SSH_HOST = "ouaziz1@jureca.fz-juelich.de"
ROOT_DIR = Path(__file__).resolve().parent.parent
SSH_KEY = ROOT_DIR / "infra" / "juelich_ed25519"
SQUEUE_FMT = "%i|%u|%T|%D|%b|%V|%S|%M|%l|%Z"


@dataclass(frozen=True)
class JobInfo:
    job_id: str
    user: str
    state: str
    nodes: int
    gpus_per_node: int
    submit_time: str
    start_time: str
    elapsed: str
    limit: str
    workdir: str


@dataclass(frozen=True)
class ReservationInfo:
    name: str
    state: str
    node_cnt: int
    total_gpus: int
    start_time: str
    end_time: str


def _ssh_exec(remote_cmd: str, timeout: int = 25) -> str:
    """Run a single remote command on JURECA via SSH + TOTP pty."""
    code, remaining = compute_totp(SECRET)
    if remaining <= 2:
        time.sleep(remaining + 0.5)
        code, _ = compute_totp(SECRET)
    cmd = [
        "ssh", "-i", str(SSH_KEY),
        "-o", "MACs=hmac-sha2-256-etm@openssh.com",
        "-o", "ControlMaster=no", "-o", "ControlPath=none",
        "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no",
        SSH_HOST, remote_cmd,
    ]
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(cmd[0], cmd)
    return _read_pty_until_exit(pid, fd, code, timeout)


def _read_pty_until_exit(pid: int, fd: int, code: str, timeout: int) -> str:
    """Feed TOTP prompt once and collect SSH stdout until process exits."""
    out = bytearray()
    sent = False
    deadline = time.time() + timeout
    while time.time() < deadline:
        ready, _, _ = select.select([fd], [], [], 1.0)
        if fd in ready:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            out.extend(chunk)
            if not sent and (b"Verification code:" in out or b"OTP" in out or b"Password" in out):
                os.write(fd, (code + "\n").encode())
                sent = True
        if os.waitpid(pid, os.WNOHANG)[0] != 0:
            break
    os.close(fd)
    return out.decode(errors="replace")


def _extract_section(raw: str, header: str) -> list[str]:
    """Extract lines between === HEADER === and the next === marker."""
    lines, active = [], False
    for line in raw.splitlines():
        clean = line.strip()
        if clean == f"=== {header} ===":
            active = True
            continue
        if active and clean.startswith("=== ") and clean.endswith(" ==="):
            break
        if active and clean:
            lines.append(clean)
    return lines


def _parse_reservation(lines: list[str]) -> ReservationInfo:
    """Parse `scontrol show reservation cispahack` output."""
    blob = " ".join(lines)
    node_m = re.search(r"NodeCnt=(\d+)", blob)
    state_m = re.search(r"State=(\S+)", blob)
    start_m = re.search(r"StartTime=(\S+)", blob)
    end_m = re.search(r"EndTime=(\S+)", blob)
    nodes = int(node_m.group(1)) if node_m else 0
    return ReservationInfo(
        name="cispahack",
        state=state_m.group(1) if state_m else "UNKNOWN",
        node_cnt=nodes,
        total_gpus=nodes * 4,
        start_time=start_m.group(1) if start_m else "N/A",
        end_time=end_m.group(1) if end_m else "N/A",
    )


def _parse_gpus(tres: str) -> int:
    m = re.search(r"gpu:(\d+)", tres)
    return int(m.group(1)) if m else 0


def _parse_jobs(lines: list[str]) -> list[JobInfo]:
    """Parse pipe-delimited squeue lines into JobInfo records."""
    jobs = []
    for line in lines:
        parts = line.split("|")
        if len(parts) < 10 or not parts[0].isdigit():
            continue
        jobs.append(JobInfo(
            job_id=parts[0], user=parts[1], state=parts[2],
            nodes=int(parts[3]) if parts[3].isdigit() else 1,
            gpus_per_node=_parse_gpus(parts[4]),
            submit_time=parts[5][-8:], start_time=parts[6][-8:] if parts[6] != "N/A" else "N/A",
            elapsed=parts[7], limit=parts[8], workdir=parts[9],
        ))
    return jobs


def fetch_snapshot() -> tuple[ReservationInfo, list[JobInfo], list[JobInfo]]:
    """Fetch reservation info, all reservation jobs, and Team TRYON jobs."""
    users = ",".join(TEAM_USERS)
    remote = (
        "echo '=== RES ==='; scontrol show reservation cispahack; "
        f"echo '=== ALL ==='; squeue -r --reservation=cispahack -h -o '{SQUEUE_FMT}'; "
        f"echo '=== TEAM ==='; squeue -u {users} -h -o '{SQUEUE_FMT}'"
    )
    raw = _ssh_exec(remote)
    res = _parse_reservation(_extract_section(raw, "RES"))
    all_jobs = _parse_jobs(_extract_section(raw, "ALL"))
    team_jobs = _parse_jobs(_extract_section(raw, "TEAM"))
    return res, all_jobs, team_jobs


_GENERIC_DIRS = {"unlearning", "watermark", "hackathon", "queue", "depot", "code", "kit"}


def _infer_team(job: JobInfo) -> str:
    """Infer team/group name from JURECA workdir, falling back to owner/user."""
    wd = job.workdir
    if wd.startswith("/p/project1/training2666/"):
        return wd.removeprefix("/p/project1/training2666/").split("/")[0]
    if wd.startswith("/p/scratch/training2666/"):
        parts = wd.removeprefix("/p/scratch/training2666/").split("/")
        if len(parts) >= 2 and parts[1].lower() not in _GENERIC_DIRS and not parts[1].startswith("Hackathon"):
            return parts[1]
        return parts[0]
    return job.user


def _short_team(workdir: str) -> str:
    return workdir.replace("/p/scratch/training2666/", "").replace("/p/project1/training2666/", "")


def _print_team_summary(all_jobs: list[JobInfo]) -> None:
    """Group reservation jobs by team and print sorted by nodes/GPUs descending."""
    groups: dict[str, list[JobInfo]] = {}
    for j in all_jobs:
        groups.setdefault(_infer_team(j), []).append(j)
    rows = []
    for team, jobs in groups.items():
        active = [j for j in jobs if j.state != "PENDING"]
        pend = [j for j in jobs if j.state == "PENDING"]
        users = ",".join(sorted({j.user for j in jobs}))
        nodes = sum(j.nodes for j in active)
        gpus = sum(j.nodes * j.gpus_per_node for j in active)
        rows.append((nodes, gpus, len(pend), team, len(active), users))
    rows.sort(key=lambda r: (-r[0], -r[1], -r[2], r[3].lower()))
    print(f"\n=== Par Team ({len(rows)} équipes) ===")
    print(f"{'TEAM':<20} {'ACT':>3} {'PEND':>4} {'NODES':>5} {'GPUs':>4}  USERS")
    for nodes, gpus, n_pend, team, n_act, users in rows:
        print(f"{team:<20} {n_act:>3} {n_pend:>4} {nodes:>5} {gpus:>4}  {users}")


def _print_jobs_table(title: str, jobs: list[JobInfo]) -> None:
    print(f"\n=== {title} ({len(jobs)}) ===")
    if not jobs:
        print("  (aucun job)")
        return
    print(f"{'TEAM':<18} {'JOBID':>9} {'USER':<15} {'STATE':<11} {'N':>2} {'GPU':>3} {'START':>8} {'ELAPSED':>8} {'LIMIT':>8}")
    for j in sorted(jobs, key=lambda x: (_infer_team(x).lower(), x.state, x.job_id)):
        print(f"{_infer_team(j):<18} {j.job_id:>9} {j.user:<15} {j.state:<11} {j.nodes:>2} "
              f"{j.gpus_per_node:>3} {j.start_time:>8} {j.elapsed:>8} {j.limit:>8}")


def render_report(res: ReservationInfo, all_jobs: list[JobInfo], team_jobs: list[JobInfo], show_all: bool) -> int:
    """Print Team TRYON occupancy check and reservation summary sorted by team."""
    counts = Counter(j.state for j in all_jobs)
    alloc_nodes = sum(j.nodes for j in all_jobs if j.state != "PENDING")
    req_gpus = sum(j.nodes * j.gpus_per_node for j in all_jobs if j.state != "PENDING")
    team_nodes = sum(j.nodes for j in team_jobs)
    team_gpus = sum(j.nodes * j.gpus_per_node for j in team_jobs)
    status = "FREE" if not team_jobs else ("OK (1 slot)" if len(team_jobs) == 1 else "VIOLATION (>1 job!)")

    print(f"[Team TRYON] status={status} | jobs={len(team_jobs)} | nodes={team_nodes} | gpus={team_gpus}")
    _print_jobs_table("Jobs Équipe TRYON", team_jobs)
    print(f"\n[cispahack {res.state}] window={res.start_time} -> {res.end_time}")
    print(f"  Nodes: {alloc_nodes}/{res.node_cnt} | GPUs actifs: {req_gpus}/{res.total_gpus} | States: {dict(counts)}")
    _print_team_summary(all_jobs)
    if show_all:
        _print_jobs_table("Tous les jobs cispahack (triés par team)", all_jobs)
    return 2 if len(team_jobs) > 1 else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Live JURECA cispahack & Team TRYON resource checker.")
    parser.add_argument("--all", action="store_true", help="Afficher tous les jobs de la réservation cispahack.")
    parser.add_argument("--json", action="store_true", help="Sortie JSON structurée.")
    parser.add_argument("--strict", action="store_true", help="Exit code 2 si l'équipe a >1 job actif/en file.")
    parser.add_argument("--watch", type=int, default=0, help="Rafraîchir toutes les N secondes (>=31s recommandé pour TOTP).")
    args = parser.parse_args()

    while True:
        res, all_jobs, team_jobs = fetch_snapshot()
        if args.json:
            payload = {"reservation": asdict(res), "team_jobs": [asdict(j) for j in team_jobs], "all_jobs": [asdict(j) for j in all_jobs]}
            print(json.dumps(payload, indent=2))
            rc = 2 if len(team_jobs) > 1 else 0
        else:
            rc = render_report(res, all_jobs, team_jobs, show_all=args.all)
        if args.watch <= 0:
            return rc if args.strict else 0
        time.sleep(max(args.watch, 31))


if __name__ == "__main__":
    raise SystemExit(main())

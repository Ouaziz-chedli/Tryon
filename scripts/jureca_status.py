#!/usr/bin/env python3
"""Live JURECA monitor for Team TRYON: reservation vs general partition, host/IP, team jobs.

Sections: [CANAL] pinned host + IPs · [TRYON réservation] (règle 1 slot) ·
[TRYON partition générale] (hors résa, illimité — doctrine Chef) ·
[cispahack] état global + par team · [Charge partitions GPU].
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import re
import socket
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jureca_xfer import BASE_OPTS, SSH_HOST, SSH_KEY, feed_run  # noqa: E402

TEAM_USERS = ("ouaziz1", "sangkhavongs1", "bassim1")
RESERVATION = "cispahack"
SQUEUE_FMT = "%i|%u|%T|%D|%b|%V|%S|%M|%l|%Z|%P"


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
    partition: str


@dataclass(frozen=True)
class ReservationInfo:
    name: str
    state: str
    node_cnt: int
    total_gpus: int
    start_time: str
    end_time: str


@dataclass(frozen=True)
class ChannelInfo:
    pinned_host: str
    local_ip: str
    remote_hostname: str
    remote_ip: str


def _resolve_ip(host: str) -> str:
    try:
        return socket.getaddrinfo(host, None, socket.AF_INET)[0][4][0]
    except OSError:
        return "unresolved"


def _ssh_exec(remote_cmd: str, timeout: int = 40) -> str:
    """Single JURECA command via the robust TOTP feeder (multi-code, drift margin)."""
    return feed_run(["ssh", "-i", str(SSH_KEY), *BASE_OPTS, SSH_HOST, remote_cmd], timeout=timeout)


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
        name=RESERVATION,
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
    """Parse pipe-delimited squeue lines (11 fields incl. partition) into JobInfo."""
    jobs = []
    for line in lines:
        parts = line.split("|")
        if len(parts) < 10 or not parts[0].isdigit():
            continue
        partition = parts[10] if len(parts) > 10 else "?"
        jobs.append(JobInfo(
            job_id=parts[0], user=parts[1], state=parts[2],
            nodes=int(parts[3]) if parts[3].isdigit() else 1,
            gpus_per_node=_parse_gpus(parts[4]),
            submit_time=parts[5][-8:], start_time=parts[6][-8:] if parts[6] != "N/A" else "N/A",
            elapsed=parts[7], limit=parts[8], workdir=parts[9], partition=partition,
        ))
    return jobs


def fetch_snapshot() -> tuple[ChannelInfo, ReservationInfo, list[JobInfo], list[JobInfo], list[str]]:
    """One ssh: host, reservation, all reservation jobs, team jobs, GPU partitions load."""
    users = ",".join(TEAM_USERS)
    remote = (
        "echo '=== HOST ==='; hostname; hostname -i 2>/dev/null; hostname -I 2>/dev/null; "
        f"echo '=== RES ==='; scontrol show reservation {RESERVATION} 2>&1; "
        f"echo '=== ALL ==='; squeue -r --reservation={RESERVATION} -h -o '{SQUEUE_FMT}'; "
        f"echo '=== TEAM ==='; squeue -u {users} -h -o '{SQUEUE_FMT}'; "
        "echo '=== SINFO ==='; sinfo -h -o '%P %D %t' | grep -E '^dc-.*gpu|^gpu' | head -12"
    )
    raw = _ssh_exec(remote)
    if "=== HOST ===" not in raw:
        raise RuntimeError(f"SSH FAILED (sortie vide ou TOTP rejeté) — pas de snapshot faux: {raw[-200:]}")
    host_lines = _extract_section(raw, "HOST")
    remote_hostname = host_lines[0] if host_lines else "?"
    remote_ip = next((ln for ln in host_lines[1:] if re.match(r"^[0-9a-f.:]+$", ln)), "?")
    channel = ChannelInfo(
        pinned_host=SSH_HOST.split("@")[-1],
        local_ip=_resolve_ip(SSH_HOST.split("@")[-1]),
        remote_hostname=remote_hostname,
        remote_ip=remote_ip,
    )
    res = _parse_reservation(_extract_section(raw, "RES"))
    all_jobs = _parse_jobs(_extract_section(raw, "ALL"))
    team_jobs = _parse_jobs(_extract_section(raw, "TEAM"))
    sinfo = _extract_section(raw, "SINFO")
    return channel, res, all_jobs, team_jobs, sinfo


def _split_reservation(team_jobs: list[JobInfo], all_jobs: list[JobInfo]) -> tuple[list[JobInfo], list[JobInfo]]:
    """Team jobs inside the reservation vs outside (partition générale)."""
    res_ids = {j.job_id.split("_")[0] for j in all_jobs} | {
        j.job_id.split("_")[0] for j in team_jobs
        if j.partition and "cispahack" not in str(j.partition)
    }
    in_res = [j for j in team_jobs if j.job_id.split("_")[0] in {a.job_id.split("_")[0] for a in all_jobs}]
    out_res = [j for j in team_jobs if j.job_id.split("_")[0] not in {a.job_id.split("_")[0] for a in all_jobs}]
    return in_res, out_res


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
    if wd.startswith("/p/home/jusers/"):
        seg = wd.split("/")[4] if len(wd.split("/")) > 4 else job.user
        return seg if seg else job.user
    return job.user


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
    print(f"\n=== Par Team dans {RESERVATION} ({len(rows)} équipes) ===")
    print(f"{'TEAM':<20} {'ACT':>3} {'PEND':>4} {'NODES':>5} {'GPUs':>4}  USERS")
    for nodes, gpus, n_pend, team, n_act, users in rows:
        print(f"{team:<20} {n_act:>3} {n_pend:>4} {nodes:>5} {gpus:>4}  {users}")


def _print_jobs_table(title: str, jobs: list[JobInfo]) -> None:
    print(f"\n=== {title} ({len(jobs)}) ===")
    if not jobs:
        print("  (aucun job)")
        return
    print(f"{'TEAM':<14} {'JOBID':>11} {'USER':<11} {'STATE':<11} {'N':>2} {'GPU':>3} "
          f"{'ELAPSED':>9} {'LIMIT':>8} {'PARTITION':<12} WORKDIR")
    for j in sorted(jobs, key=lambda x: (x.state, x.job_id)):
        wd = j.workdir.replace("/p/home/jusers/", "~u/").replace("/p/scratch/training2666/", "S:").replace("/p/project1/training2666/", "P:")
        print(f"{_infer_team(j):<14} {j.job_id:>11} {j.user:<11} {j.state:<11} {j.nodes:>2} "
              f"{j.gpus_per_node:>3} {j.elapsed:>9} {j.limit:>8} {j.partition:<12} {wd}")


def render_report(channel: ChannelInfo, res: ReservationInfo, all_jobs: list[JobInfo],
                  team_jobs: list[JobInfo], sinfo: list[str], show_all: bool) -> int:
    """Print channel, team jobs split reservation/general, reservation summary, GPU load."""
    in_res, out_res = _split_reservation(team_jobs, all_jobs)
    counts = Counter(j.state for j in all_jobs)
    alloc_nodes = sum(j.nodes for j in all_jobs if j.state != "PENDING")
    req_gpus = sum(j.nodes * j.gpus_per_node for j in all_jobs if j.state != "PENDING")
    status = "FREE" if not in_res else ("OK (1 slot résa)" if len(in_res) == 1 else "VIOLATION RÉSA (>1 job!)")

    print(f"[CANAL] {channel.pinned_host} -> {channel.local_ip} | login node: "
          f"{channel.remote_hostname} ({channel.remote_ip})")
    print(f"[TRYON réservation {RESERVATION}] {status} | jobs={len(in_res)}")
    _print_jobs_table(f"Jobs TRYON DANS la réservation {RESERVATION}", in_res)
    print(f"\n[TRYON partition générale] jobs={len(out_res)} (hors résa — illimité, doctrine Chef)")
    _print_jobs_table("Jobs TRYON HORS réservation (partition générale)", out_res)
    print(f"\n[{res.name} {res.state}] window={res.start_time} -> {res.end_time}")
    print(f"  Nodes: {alloc_nodes}/{res.node_cnt} | GPUs actifs: {req_gpus}/{res.total_gpus} | States: {dict(counts)}")
    _print_team_summary(all_jobs)
    if sinfo:
        print("\n=== Charge partitions GPU (alloc/mixed/idle) ===")
        for line in sinfo:
            print(f"  {line}")
    if show_all:
        _print_jobs_table(f"Tous les jobs {RESERVATION} (triés par team)", all_jobs)
    return 2 if len(in_res) > 1 else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Live JURECA monitor: résera vs générale, canal + IP, team TRYON.")
    parser.add_argument("--all", action="store_true", help="Afficher aussi tous les jobs de la réservation.")
    parser.add_argument("--json", action="store_true", help="Sortie JSON structurée.")
    parser.add_argument("--strict", action="store_true", help="Exit 2 si >1 job RÉSERVATION pour l'équipe (la partition générale ne compte pas).")
    parser.add_argument("--watch", type=int, default=0, help="Rafraîchir toutes les N secondes (>=31s recommandé).")
    args = parser.parse_args()

    while True:
        try:
            channel, res, all_jobs, team_jobs, sinfo = fetch_snapshot()
        except RuntimeError as exc:
            print(f"[ERREUR CANAL] {exc}", file=sys.stderr)
            return 1
        if args.json:
            in_res, out_res = _split_reservation(team_jobs, all_jobs)
            payload = {
                "channel": asdict(channel),
                "reservation": asdict(res),
                "team_jobs_reservation": [asdict(j) for j in in_res],
                "team_jobs_general": [asdict(j) for j in out_res],
                "all_jobs": [asdict(j) for j in all_jobs],
                "gpu_partitions": sinfo,
            }
            print(json.dumps(payload, indent=2))
            rc = 2 if len(in_res) > 1 else 0
        else:
            rc = render_report(channel, res, all_jobs, team_jobs, sinfo, show_all=args.all)
        if args.watch <= 0:
            return rc if args.strict else 0
        import time
        time.sleep(max(args.watch, 31))


if __name__ == "__main__":
    raise SystemExit(main())

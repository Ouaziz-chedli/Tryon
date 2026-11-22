#!/usr/bin/env python3
"""JURECA ssh/scp wrapper with robust TOTP feeding (handles code rejections).

Single-owner network channel for Team TRYON: every remote op goes through
this script so TOTP windows are never contested between agents.
"""

from __future__ import annotations

import base64
import os
import pty
import select
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from totp import SECRET, compute_totp  # noqa: E402

SSH_HOST = "ouaziz1@jureca05.fz-juelich.de"  # pinned: alias round-robin lands on broken nodes (org notice 05-22)
SSH_KEY = str(Path(__file__).resolve().parent.parent / "infra" / "juelich_ed25519")
BASE_OPTS = [
    "-o", "MACs=hmac-sha2-256-etm@openssh.com",
    "-o", "ControlMaster=no", "-o", "ControlPath=none",
    "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
]


def _fresh_code(min_remaining: float = 20.0) -> str:
    # shortcut: local WSL2 clock runs ~10 s behind JSC (measured 2026-11-22);
    # 20 s margin keeps the code valid server-side despite the drift.
    code, remaining = compute_totp(SECRET)
    if remaining < min_remaining:
        time.sleep(remaining + 1.0)
        code, _ = compute_totp(SECRET)
    return code


def feed_run(argv: list[str], timeout: float = 120.0, max_feeds: int = 4) -> str:
    """Run argv under a pty, feeding a fresh TOTP code at each prompt."""
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(argv[0], argv)
    out = bytearray()
    fed, deadline = 0, time.time() + timeout
    prompt_seen_at = 0.0
    while time.time() < deadline:
        ready, _, _ = select.select([fd], [], [], 1.0)
        if fd in ready:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            out += chunk
            tail = bytes(out[-300:])
            if b"not accepted" in tail:
                # Rejected code: wait for the NEXT TOTP window before re-feeding
                # (same-window codes are anti-replay rejected instantly).
                _, rem = compute_totp(SECRET)
                time.sleep(rem + 1.0)
                out = bytearray()
            elif b"Verification code" in tail and time.time() - prompt_seen_at > 1.0:
                prompt_seen_at = time.time()
                if fed >= max_feeds:
                    break
                os.write(fd, (_fresh_code() + "\r\n").encode())
                fed += 1
                out = bytearray()
    try:
        _, status = os.waitpid(pid, os.WNOHANG)
        if status == 0:
            time.sleep(min(10.0, max(0.0, deadline - time.time())))
            os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    return out.decode(errors="replace")


def ssh_exec(remote_cmd: str, timeout: float = 120.0) -> str:
    argv = ["ssh", "-i", SSH_KEY, *BASE_OPTS, SSH_HOST, remote_cmd]
    return feed_run(argv, timeout=timeout)


def scp_put(local_paths: list[str], remote_dir: str, timeout: float = 1800.0) -> str:
    argv = ["scp", "-i", SSH_KEY, *BASE_OPTS, *local_paths, f"{SSH_HOST}:{remote_dir}"]
    return feed_run(argv, timeout=timeout)


def scp_get(remote_paths: list[str], local_dir: str, timeout: float = 900.0) -> str:
    """Pull remote files to a local dir (collect direction — upload is scp_put)."""
    argv = ["scp", "-i", SSH_KEY, *BASE_OPTS, *[f"{SSH_HOST}:{p}" for p in remote_paths], local_dir]
    return feed_run(argv, timeout=timeout)


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: jureca_xfer.py ssh '<remote cmd>' | scp <local...> <remote_dir>", file=sys.stderr)
        return 2
    if sys.argv[1] == "ssh":
        print(ssh_exec(sys.argv[2], timeout=float(sys.argv[3]) if len(sys.argv) > 3 else 120.0))
        return 0
    if sys.argv[1] == "scp":
        print(scp_put(sys.argv[2:-1], sys.argv[-1]))
        return 0
    print("unknown mode", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

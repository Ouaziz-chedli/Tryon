#!/usr/bin/env python3
"""Run one remote command on JURECA via TOTP pty; optionally stream a local file as base64 stdin."""

import argparse
import base64
import os
import pty
import select
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from totp import SECRET, compute_totp  # noqa: E402

KEY = Path(__file__).resolve().parent.parent / "infra" / "juelich_ed25519"
HOST = "ouaziz1@jureca.fz-juelich.de"


def main() -> int:
    ap = argparse.ArgumentParser(description="Single JURECA command over TOTP pty.")
    ap.add_argument("remote_cmd")
    ap.add_argument("--stdin-file", help="local file streamed as base64 to remote stdin")
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()

    code, remaining = compute_totp(SECRET)
    if remaining <= 2:
        time.sleep(remaining + 0.5)
        code, _ = compute_totp(SECRET)

    cmd = [
        "ssh", "-i", str(KEY),
        "-o", "MACs=hmac-sha2-256-etm@openssh.com",
        "-o", "ControlMaster=no", "-o", "ControlPath=none",
        "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no",
        HOST, args.remote_cmd,
    ]
    payload = base64.encodebytes(Path(args.stdin_file).read_bytes()) if args.stdin_file else None
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(cmd[0], cmd)

    out = bytearray()
    sent_totp = sent_payload = False
    pos = 0
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        ready, _, _ = select.select([fd], [], [], 1.0)
        if ready:
            try:
                chunk = os.read(fd, 4096)
            except OSError:
                break
        if sent_totp and payload is not None and not sent_payload and b"READY_FOR_STDIN" in out:
            pos = os.write(fd, payload[:16384])
            sent_payload = True
        elif sent_payload and pos < len(payload):
            wready, _, _ = select.select([], [fd], [], 1.0)
            if wready:
                pos += os.write(fd, payload[pos:pos + 16384])
        elif sent_payload and pos >= len(payload):
            os.write(fd, b"\x04")  # pty EOF so the remote reader terminates
            sent_payload = False  # send exactly once
            payload = None
        if not sent_totp and (b"Verification code:" in out or b"OTP" in out or b"Password" in out):
            os.write(fd, (code + "\n").encode())
            sent_totp = True
        if os.waitpid(pid, os.WNOHANG)[0] != 0:
            break
    try:
        os.close(fd)
    except OSError:
        pass
    sys.stdout.buffer.write(bytes(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

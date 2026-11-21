#!/usr/bin/env python3
"""Generate the current 6-digit TOTP code for the shared Jülich JSC account."""

import base64
import hashlib
import hmac
import struct
import time

SECRET = "XYOFKN26KL2RATEUZJ7D7WDPKRE4S2ZH2PB3GZTHDOYRZ2MPPWSMH34POGTKXVQJ74SNAV6QCQUC4WJL"


def compute_totp(secret: str) -> tuple[str, int]:
    """Return (6-digit code, seconds remaining in current 30s window)."""
    padded = secret.upper() + "=" * ((8 - len(secret) % 8) % 8)
    key = base64.b32decode(padded)
    now = int(time.time())
    msg = struct.pack(">Q", now // 30)
    digest = hmac.new(key, msg, hashlib.sha1).digest()
    offset = digest[19] & 0x0F
    code = (struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF) % 1_000_000
    return f"{code:06d}", 30 - (now % 30)


if __name__ == "__main__":
    code, remaining = compute_totp(SECRET)
    print(f"JSC TOTP code: {code} (valid for {remaining}s)")

#!/usr/bin/env python3
"""Probe free team slots: fire N concurrent tiny requests at a ref endpoint
(unlimited budget, 5 tokens each) and count 200 (slot was free) vs 429 Retry-After
(all 4 team slots busy). Refuses to touch ep-*.

Usage: python probe_slots.py [n_requests]"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

API_KEY = "0895eb0a5610b64740af893f4abefbc4"
BASE_URL = "http://34.147.229.55/33-model-equality/v1"


def probe(i: int) -> str:
    try:
        r = requests.post(f"{BASE_URL}/chat/completions", timeout=60,
                           headers={"X-API-Key": API_KEY},
                           json={"model": "ref-01", "max_tokens": 5,
                                 "messages": [{"role": "user", "content": "Hi"}]})
    except requests.RequestException as exc:
        return f"erreur reseau ({type(exc).__name__})"
    if r.status_code == 200:
        return "200 (slot libre)"
    if r.status_code == 429 and "Retry-After" in r.headers:
        return f"429 busy (Retry-After {r.headers['Retry-After']}s)"
    return f"HTTP {r.status_code}: {r.text[:120]}"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    print(f"Sonde: {n} requetes concurrentes vers ref-01 (aucun cout reel)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(probe, range(n)))
    ok = sum(1 for r in results if r.startswith("200"))
    busy = sum(1 for r in results if r.startswith("429"))
    print("\n".join(f"  requete {i + 1}: {r}" for i, r in enumerate(results)))
    print(f"\nResultat en {time.time() - t0:.0f}s: {ok} slots etaient libres, "
          f"{busy} requetes mises en file ({n - ok - busy} autres/erreurs)")
    print("(4 slots max par equipe, cle API commune aux 3 tasks)")


if __name__ == "__main__":
    main()

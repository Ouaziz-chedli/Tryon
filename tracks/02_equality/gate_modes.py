#!/usr/bin/env python3
"""GATE v4: mode-matching — do the two endpoints agree on the DOMINANT answer?

Invariant to decoding flatness (temp 1.0 keeps the same mode, just noisier);
sensitive to weight flips (fp8/LoRA change which value dominates near-ties).
sim(A,B) = fraction of prompts where A's modal value == B's modal value.
"""
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import importlib.util

spec = importlib.util.spec_from_file_location("gate_values", Path(__file__).with_name("gate_values.py"))
# reuse parse_value without executing the module body: copy minimal parts instead
ANSWERS = Path(__file__).with_name("reference_answers.jsonl")
S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]
T = {"ref-02": "fp8", "ref-09": "lora0.3", "ref-04": "lora0.7", "ref-07": "lora1.5"}


def normalize(s: str) -> str:
    lines = s.strip().splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].lstrip().startswith("```"):
            lines = lines[:-1]
    return " ".join(l.strip() for l in lines if l.strip()).lower()


def parse_value(pid: int, text: str) -> str:
    t = normalize(text)
    if pid in (101, 102, 109, 110):
        m = re.search(r"-?\d+(?:[.,]\d+)?", t)
        return m.group(0).replace(",", ".") if m else ""
    if pid in (100, 105):
        m = re.search(r"\b(yes|no)\b", t)
        return m.group(1) if m else ""
    if pid == 111:
        m = re.search(r"\b(same|iron|feathers)\b", t)
        return m.group(1) if m else ""
    if pid == 103:
        m = re.search(r"\bletter\s+(?:is\s+)?([a-z])\b", t) or re.search(r"^([a-z])\b", t)
        return m.group(1) if m else ""
    if pid == 104:
        for w in ("decrease", "reduce", "less", "smaller", "decline", "down"):
            if w in t:
                return w
        return ""
    if pid == 106:
        m = re.search(r"salt\s+and\s+([a-z]+)", t)
        return m.group(1) if m else ""
    if pid == 107:
        return "h2o" if re.search(r"(h2o|h₂o|h20)", t) else ""
    if pid == 108:
        return "thanks" if re.search(r"thank\s*you|thanks", t) else ""
    if pid in (112, 0):
        m = re.search(r"return\s+([^;\n]+)", t)
        return m.group(1).strip() if m else ""
    if pid in (1,):                                  # 17*23 (v1, 15 reps)
        m = re.search(r"=\s*(-?\d+)", t)
        return m.group(1) if m else ""
    return t[:40]


answers = defaultdict(lambda: defaultdict(list))
for line in ANSWERS.open(encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    pid = int(r["id"])
    if pid >= 100 or pid in (0, 1):
        v = parse_value(pid, r["response"])
        if v:
            answers[r["endpoint"]][pid].append(v)

prompt_ids = sorted({pid for ep in answers.values() for pid in ep})
eps = sorted(answers)
modes = {ep: {pid: Counter(vals).most_common(1)[0][0] if vals else None
              for pid, vals in answers[ep].items()} for ep in eps}

print("== MODES par endpoint (differences marquees *)")
for pid in prompt_ids:
    row = {ep: modes[ep].get(pid) for ep in eps}
    ref_mode = row.get("ref-01")
    marks = {ep: ("*" if row[ep] != ref_mode else " ") for ep in eps}
    print(f"p{pid}: " + "  ".join(f"{ep[-2:]}={row[ep]!r}{marks[ep]}" for ep in eps))


def sim(a, b):
    shared = [pid for pid in prompt_ids if modes[a].get(pid) and modes[b].get(pid)]
    if not shared:
        return 0.0
    return sum(modes[a][pid] == modes[b][pid] for pid in shared) / len(shared)


print("\n== matrice mode-match (1 = modes identiques partout)")
sims = {(a, b): sim(a, b) for a, b in combinations(eps, 2)}


def get(a, b):
    return sims[(min(a, b), max(a, b))]


print("     " + " ".join(f"{e[-2:]}" for e in eps))
for a in eps:
    print(f"{a[-2:]}  " + " ".join(f"{get(a, b):.2f}" if a != b else " 1. " for b in eps))
intra = [get(a, b) for a, b in combinations(S, 2)]
st = [get(a, b) for a in S for b in T]
tt = [get(a, b) for a, b in combinations(sorted(T), 2)]
print(f"intra-S: min={min(intra):.3f} mean={sum(intra)/len(intra):.3f}")
print(f"S-T    : max={max(st):.3f} mean={sum(st)/len(st):.3f}  (fp8={max(get(a, 'ref-02') for a in S):.2f}, lora03={max(get(a, 'ref-09') for a in S):.2f})")
print(f"T-T    : min={min(tt):.3f}")
gap = min(intra) - max(st)
print(f"MARGE: {gap:+.3f} " + ("✓ SEPARABLE" if gap > 0 else "✗ chevauchement"))

#!/usr/bin/env python3
"""GATE v3: compare PARSED ANSWER VALUES (not raw text).

For each prompt, extract the answer value (number / yes-no / first content word)
from every repeat, build the per-endpoint histogram, and compare histograms
between endpoints. Verbose personas change the wrapper, not the value; weight
perturbations (fp8/LoRA) flip near-tie values -> histogram shifts.
"""
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

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
    """Extract the answer value from a response, per prompt type."""
    t = normalize(text)
    if pid in (101, 102, 109, 110):               # numeric answers
        m = re.search(r"-?\d+(?:[.,]\d+)?", t)
        return m.group(0).replace(",", ".") if m else ""
    if pid in (100, 105):                          # yes/no
        m = re.search(r"\b(yes|no)\b", t)
        return m.group(1) if m else ""
    if pid == 111:                                 # iron/feathers/same
        m = re.search(r"\b(same|iron|feathers)\b", t)
        return m.group(1) if m else ""
    if pid == 103:                                 # 15th letter
        m = re.search(r"\bletter\s+(?:is\s+)?([a-z])\b", t)
        if m:
            return m.group(1)
        m = re.search(r"^([a-z])\b", t)
        return m.group(1) if m else ""
    if pid == 104:                                 # opposite of increase
        for w in ("decrease", "decrease.", "reduce", "less", "smaller", "decline", "down"):
            if w.rstrip(".") in t:
                return w.rstrip(".")
        return ""
    if pid == 106:                                 # salt and ___
        m = re.search(r"salt\s+and\s+([a-z]+)", t)
        return m.group(1) if m else ""
    if pid == 107:                                 # water formula
        m = re.search(r"(h2o|h₂o|h20)", t)
        return "h2o" if m else ""
    if pid == 108:                                 # merci -> ?
        for w in ("thank you", "thanks", "thankyou"):
            if w in t:
                return "thanks"
        return ""
    if pid == 112:                                 # def double(x): -> ?
        m = re.search(r"return\s+([^;\n]+)", t)
        return m.group(1).strip() if m else ""
    if pid == 0:                                   # def add(a, b)
        m = re.search(r"return\s+([^;\n]+)", t)
        return m.group(1).strip() if m else ""
    return t[:40]                                  # fallback: first 40 chars


def tv(ha, hb):
    """Total variation distance between two histograms (dicts of counts)."""
    ka, kb = sum(ha.values()), sum(hb.values())
    if ka == 0 and kb == 0:
        return 0.0
    if ka == 0 or kb == 0:
        return 1.0
    keys = set(ha) | set(hb)
    return 0.5 * sum(abs(ha.get(k, 0) / ka - hb.get(k, 0) / kb) for k in keys)


answers = defaultdict(lambda: defaultdict(list))
for line in ANSWERS.open(encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    pid = int(r["id"])
    if pid >= 100 or pid in (0,):
        v = parse_value(pid, r["response"])
        if v:
            answers[r["endpoint"]][pid].append(v)

prompt_ids = sorted({pid for ep in answers.values() for pid in ep})
eps = sorted(answers)

# Per-prompt sanity: what values do the endpoints give?
for pid in prompt_ids:
    print(f"\n--- prompt {pid}: valeurs majoritaires par endpoint")
    for ep in eps:
        vals = answers[ep].get(pid, [])
        if vals:
            top = defaultdict(int)
            for v in vals:
                top[v] += 1
            best = sorted(top.items(), key=lambda x: -x[1])[:3]
            print(f"  {ep}: {len(vals)} reps -> " + ", ".join(f"{v!r}({c})" for v, c in best))

# Pairwise TV distance averaged over prompts
print("\n== distance TV des histogrammes de valeurs (moyenne sur prompts)")
sims = {}
for a, b in combinations(eps, 2):
    vals = [tv(defaultdict(int, {v: answers[a][pid].count(v) for v in set(answers[a][pid])}),
               defaultdict(int, {v: answers[b][pid].count(v) for v in set(answers[b][pid])}))
            for pid in prompt_ids if answers[a].get(pid) and answers[b].get(pid)]
    # simpler: build counts via Counter
    sims[(a, b)] = None

from collections import Counter
sims = {}
for a, b in combinations(eps, 2):
    vals = []
    for pid in prompt_ids:
        ha, hb = answers[a].get(pid), answers[b].get(pid)
        if ha and hb:
            vals.append(tv(Counter(ha), Counter(hb)))
    sims[(a, b)] = sum(vals) / len(vals) if vals else 1.0


def get(a, b):
    return sims[(min(a, b), max(a, b))]


print("     " + " ".join(f"{e[-2:]}" for e in eps))
for a in eps:
    print(f"{a[-2:]}  " + " ".join(f"{get(a, b):.2f}" if a != b else " 0. " for b in eps))
intra = [get(a, b) for a, b in combinations(S, 2)]
st = [get(a, b) for a in S for b in T]
tt = [get(a, b) for a, b in combinations(sorted(T), 2)]
print(f"intra-S: max={max(intra):.3f} mean={sum(intra)/len(intra):.3f}   (bas = jumeaux)")
print(f"S-T    : min={min(st):.3f} mean={sum(st)/len(st):.3f}   (haut = differents)")
print(f"T-T    : min={min(tt):.3f} max={max(tt):.3f}")
gap = min(st) - max(intra)
print(f"MARGE (min S-T) - (max intra-S): {gap:+.3f} " + ("✓ SEPARABLE" if gap > 0 else "✗ chevauchement"))

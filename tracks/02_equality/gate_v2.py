#!/usr/bin/env python3
"""GATE v2: do the near-tie prompts (ids 100+) separate S from fp8/LoRA0.3?

Prints the consensus-similarity matrix on v2 prompts, the S/T margins per prompt,
and (if a threshold works) writes practice_answer.csv for score_practice.py.
"""
import json
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


def consensus(strings):
    if not strings:
        return ""
    n = max(len(s) for s in strings)
    out = []
    for i in range(n):
        votes = defaultdict(int)
        for s in strings:
            if i < len(s):
                votes[s[i]] += 1
        out.append(max(votes, key=votes.get) if votes else "")
    return "".join(out)


def dice(a, b, n=3):
    if len(a) < n or len(b) < n:
        return 1.0 if a == b else 0.0
    ga = {a[i:i + n] for i in range(len(a) - n + 1)}
    gb = {b[i:i + n] for i in range(len(b) - n + 1)}
    return 2 * len(ga & gb) / (len(ga) + len(gb))


answers = defaultdict(lambda: defaultdict(list))
for line in ANSWERS.open(encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    if int(r["id"]) >= 100:                       # v2 prompts only
        answers[r["endpoint"]][r["id"]].append(normalize(r["response"]))

prompt_ids = sorted({pid for ep in answers.values() for pid in ep})
eps = sorted(answers)
print(f"v2: {len(prompt_ids)} prompts, endpoints: {', '.join(eps)}")
print(f"repetitions par (endpoint, prompt): "
      f"min={min(len(v) for ep in answers.values() for v in ep.values())} "
      f"max={max(len(v) for ep in answers.values() for v in ep.values())}")

cons = {ep: {pid: consensus(answers[ep][pid]) for pid in prompt_ids} for ep in eps}


def pair_sim(a, b, metric):
    vals = [metric(cons[a][pid], cons[b][pid]) for pid in prompt_ids
            if cons[a][pid] and cons[b][pid]]
    return sum(vals) / len(vals) if vals else 0.0


for name, metric in [("poswise", lambda a, b: sum(1 for i in range(min(len(a), len(b))) if a[i] == b[i]) / max(len(a), len(b), 1)),
                     ("dice-3g", dice)]:
    sims = {(a, b): pair_sim(a, b, metric) for a, b in combinations(eps, 2)}

    def get(a, b):
        return sims[(min(a, b), max(a, b))]

    intra = [get(a, b) for a, b in combinations(S, 2)]
    st = [get(a, b) for a in S for b in T]
    tt = [get(a, b) for a, b in combinations(sorted(T), 2)]
    print(f"\n== {name} (moyenne sur les {len(prompt_ids)} prompts v2)")
    print("     " + " ".join(f"{e[-2:]}" for e in eps))
    for a in eps:
        print(f"{a[-2:]}  " + " ".join(f"{get(a, b):.2f}" if a != b else " 1. " for b in eps))
    print(f"intra-S: min={min(intra):.3f} mean={sum(intra)/len(intra):.3f}")
    print(f"S-T    : max={max(st):.3f} mean={sum(st)/len(st):.3f}  (detail fp8={max(get(a,'ref-02') for a in S):.2f}, lora03={max(get(a,'ref-09') for a in S):.2f})")
    print(f"T-T    : min={min(tt):.3f} max={max(tt):.3f}")
    gap = min(intra) - max(st)
    print(f"MARGE intra-S vs S-T: {gap:+.3f} " + ("✓ SEPARABLE" if gap > 0 else "✗ chevauchement"))

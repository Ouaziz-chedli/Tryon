#!/usr/bin/env python3
"""Candidate: distribution overlap between repeat SETS (alignment-free).

sim(A,B,p) = symmetrized mean of best-match dice-3gram between A's and B's repeats.
Same model + different decoding -> overlapping supports -> high.
Also test: trimmed confident-prefix consensus (truncate at first low-agreement position).
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


def dice(a, b, n=3):
    if len(a) < n or len(b) < n:
        return 1.0 if a == b else 0.0
    ga = {a[i:i + n] for i in range(len(a) - n + 1)}
    gb = {b[i:i + n] for i in range(len(b) - n + 1)}
    return 2 * len(ga & gb) / (len(ga) + len(gb))


def set_match(A: list, B: list) -> float:
    """Mean best-match similarity, both directions."""
    if not A or not B:
        return 0.0
    ab = sum(max(dice(a, b) for b in B) for a in A) / len(A)
    ba = sum(max(dice(a, b) for a in A) for b in B) / len(B)
    return (ab + ba) / 2


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


def agreement(strings):
    """Per-position repeat agreement rate."""
    if not strings:
        return []
    n = max(len(s) for s in strings)
    out = []
    for i in range(n):
        votes = defaultdict(int)
        seen = 0
        for s in strings:
            if i < len(s):
                votes[s[i]] += 1
                seen += 1
        out.append(max(votes.values()) / seen if seen else 0.0)
    return out


def confident_prefix(strings, thresh=0.8, min_len=8):
    """Consensus truncated at the first position where agreement < thresh."""
    c = consensus(strings)
    a = agreement(strings)
    k = 0
    while k < len(a) and a[k] >= thresh:
        k += 1
    return c[:k] if k >= min_len else ""


answers = defaultdict(lambda: defaultdict(list))
for line in ANSWERS.open(encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    answers[r["endpoint"]][r["id"]].append(normalize(r["response"]))

prompt_ids = sorted({pid for ep in answers.values() for pid in ep})
eps = list(answers)


def pair_set(a, b):
    vals = [set_match(answers[a].get(pid, []), answers[b].get(pid, [])) for pid in prompt_ids]
    return sum(vals) / len(vals) if vals else 0.0


def pair_conf(a, b):
    vals = []
    for pid in prompt_ids:
        pa, pb = confident_prefix(answers[a].get(pid, [])), confident_prefix(answers[b].get(pid, []))
        if pa and pb:
            vals.append(dice(pa, pb))
    return sum(vals) / len(vals) if vals else 0.0


def report(name, pair_fn):
    sims = {}
    for a, b in combinations(sorted(eps), 2):
        sims[(a, b)] = pair_fn(a, b)

    def get(a, b):
        return sims[(min(a, b), max(a, b))]

    intra_S = [get(a, b) for a, b in combinations(S, 2)]
    st = [get(a, b) for a in S for b in T]
    tt = [get(a, b) for a, b in combinations(sorted(T), 2)]
    print(f"\n== {name}")
    print("     " + " ".join(f"{e[-2:]}" for e in sorted(eps)))
    for a in sorted(eps):
        print(f"{a[-2:]}  " + " ".join(f"{get(a, b):.2f}" if a != b else " 1. " for b in sorted(eps)))
    print(f"intra-S min={min(intra_S):.3f}  S-T max={max(st):.3f}  T-T max={max(tt):.3f}  "
          f"MARGE={min(intra_S) - max(st):+.3f} " + ("✓" if min(intra_S) > max(st) else "✗"))


report("set-match dice (moyenne tous prompts)", pair_set)
report("confident-prefix dice", pair_conf)

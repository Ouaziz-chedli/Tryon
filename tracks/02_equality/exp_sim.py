#!/usr/bin/env python3
"""Throwaway experiment: which similarity separates the ref ground truth?

S (same model, disguised) = {01,03,05,06,08}; T (different models) = {02,04,07,09}.
A candidate is perfect iff min(intra-S sim) > max(S-to-T sim).
"""
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ANSWERS = Path(__file__).with_name("reference_answers.jsonl")
S = {"ref-01", "ref-03", "ref-05", "ref-06", "ref-08"}
T = {"ref-02", "ref-04", "ref-07", "ref-09"}


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


def stable_prefix(strings):
    """Longest prefix shared by ALL repeats (the confident region)."""
    if not strings:
        return ""
    p = strings[0]
    for s in strings[1:]:
        n = min(len(p), len(s))
        i = 0
        while i < n and p[i] == s[i]:
            i += 1
        p = p[:i]
    return p


def dice_ngrams(a, b, n=3):
    if len(a) < n or len(b) < n:
        return 1.0 if a == b else 0.0
    ga = {a[i:i + n] for i in range(len(a) - n + 1)}
    gb = {b[i:i + n] for i in range(len(b) - n + 1)}
    return 2 * len(ga & gb) / (len(ga) + len(gb))


def prefix_sim(a, b):
    if not a and not b:
        return 1.0
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b), 1)


def containment_sim(a, b):
    """Shorter consensus contained in the longer one (fuzzy on word boundaries)."""
    if a == b:
        return 1.0
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if not short:
        return 0.0
    return 1.0 if short.rstrip(". ") in long_ else 0.0


# load
answers = defaultdict(lambda: defaultdict(list))
for line in ANSWERS.open(encoding="utf-8"):
    if not line.strip():
        continue
    r = json.loads(line)
    answers[r["endpoint"]][r["id"]].append(normalize(r["response"]))

prompt_ids = sorted({pid for ep in answers.values() for pid in ep})
cons = {ep: {pid: consensus(answers[ep].get(pid, [])) for pid in prompt_ids} for ep in answers}
stab = {ep: {pid: stable_prefix(answers[ep].get(pid, [])) for pid in prompt_ids} for ep in answers}

CANDIDATES = {
    "poswise (actuel)": lambda a, b, pid: prefix_sim(cons[a][pid], cons[b][pid]),
    "dice-3g consensus": lambda a, b, pid: dice_ngrams(cons[a][pid], cons[b][pid]),
    "dice-3g prefixe stable": lambda a, b, pid: dice_ngrams(stab[a][pid], stab[b][pid]) if stab[a][pid] and stab[b][pid] else None,
    "containment consensus": lambda a, b, pid: containment_sim(cons[a][pid], cons[b][pid]),
}


def evaluate(name, pair_fn):
    """pair_fn returns per-prompt sim or None to skip."""
    def sim(a, b):
        vals = [pair_fn(a, b, pid) for pid in prompt_ids]
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    intra_S = [sim(a, b) for a, b in combinations(sorted(S), 2)]
    s_t = [sim(a, b) for a in sorted(S) for b in sorted(T)]
    t_t = [sim(a, b) for a, b in combinations(sorted(T), 2)]
    print(f"\n== {name}")
    print(f"  intra-S : min={min(intra_S):.3f} mean={sum(intra_S)/len(intra_S):.3f}")
    print(f"  S-T     : max={max(s_t):.3f} mean={sum(s_t)/len(s_t):.3f}")
    print(f"  T-T     : max={max(t_t):.3f} mean={sum(t_t)/len(t_t):.3f}")
    gap = min(intra_S) - max(s_t)
    print(f"  MARGE   : {gap:+.3f} " + ("✓ SÉPARABLE" if gap > 0 else "✗ chevauchement"))


for name, fn in CANDIDATES.items():
    evaluate(name, fn)

#!/usr/bin/env python3
"""Per-prompt discrimination: which prompts separate S from T, and from whom?

For each prompt: intra-S min similarity vs max similarity to each T member.
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


def prefix_sim(a, b):
    if not a and not b:
        return 1.0
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b), 1)


def dice_ngrams(a, b, n=3):
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
    answers[r["endpoint"]][r["id"]].append(normalize(r["response"]))

prompt_ids = sorted({pid for ep in answers.values() for pid in ep})
cons = {ep: {pid: consensus(answers[ep].get(pid, [])) for pid in prompt_ids} for ep in answers}

PROMPTS = {
    "0": "code add", "1": "17*23", "2": "primes", "3": "alphabet inv", "4": "fer Fe",
    "5": "tri mots", "6": "suite 2,4,8", "7": "reverse s", "8": "traduction",
    "9": "json bob", "10": "capitale France", "11": "cycle eau", "12": "haiku",
    "13": "histoire renard", "14": "decris-toi", "15": "SQL",
}

print(f"{'prompt':<15} {'intra-S min':>11} " + " ".join(f"{v:>8}" for v in T.values()) + "   (sim max S->T)")
for pid in prompt_ids:
    sims_S = [prefix_sim(cons[a][pid], cons[b][pid]) for a, b in combinations(S, 2)]
    line = f"{PROMPTS.get(pid, pid):<15} {min(sims_S):>11.2f} "
    for t in T:
        sims = [prefix_sim(cons[a][pid], cons[t][pid]) for a in S]
        line += f"{max(sims):>8.2f} "
    print(line)

print("\n--- detail: consensus par endpoint pour les prompts ou intra-S < 1.0 ---")
for pid in prompt_ids:
    sims_S = [prefix_sim(cons[a][pid], cons[b][pid]) for a, b in combinations(S, 2)]
    if min(sims_S) < 0.99:
        print(f"\nprompt {pid} ({PROMPTS.get(pid, pid)}):")
        for ep in S + list(T):
            c = cons[ep][pid]
            print(f"  {ep} ({T.get(ep, 'S')}): {c[:80]!r}")

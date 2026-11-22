#!/usr/bin/env python3
"""Phase-1 first submission: cluster ep-01..33 from answers.jsonl (5 prompts x 2 reps).

Composite per pair, calibrated on the refs:
  - value agreement: parsed answer values equal / histogram TV
  - text similarity: dice-3g of the (normalized) consensus
Merge model only if BOTH are high (conservative: refs taught that fine variants
sit close, so we accept documented FP risk on fp8/LoRA-fine).
Families: looser threshold on the same composite.
"""
import csv
import json
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from analyze import partition_rows

ANSWERS = Path(__file__).with_name("answers.jsonl")
OUT = Path(__file__).with_name("submission.csv")

# seuils calibres sur les refs (gate jaune): jumeaux vus a >=0.8 mode-match,
# variants fins a 0.8-0.93 -> on fusionne confident et on documente le risque.
TAU_MODEL = 0.85
TAU_FAMILY = 0.60


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
    if pid == 112:
        m = re.search(r"return\s+([^;\n]+)", t)
        return m.group(1).strip() if m else ""
    return t[:40]


def dice(a, b, n=3):
    if len(a) < n or len(b) < n:
        return 1.0 if a == b else 0.0
    ga = {a[i:i + n] for i in range(len(a) - n + 1)}
    gb = {b[i:i + n] for i in range(len(b) - n + 1)}
    return 2 * len(ga & gb) / (len(ga) + len(gb))


def tv(ha: Counter, hb: Counter) -> float:
    ka, kb = sum(ha.values()), sum(hb.values())
    if ka == 0 and kb == 0:
        return 0.0
    if ka == 0 or kb == 0:
        return 1.0
    return 0.5 * sum(abs(ha.get(k, 0) / ka - hb.get(k, 0) / kb)
                     for k in set(ha) | set(hb))


def main() -> None:
    answers = defaultdict(lambda: defaultdict(list))
    with ANSWERS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            answers[r["endpoint"]][r["id"]].append(normalize(r["response"]))

    eps = sorted(answers)
    prompt_ids = sorted({pid for ep in answers.values() for pid in ep})
    vals = {ep: {pid: [v for v in (parse_value(int(pid), t) for t in answers[ep].get(pid, [])) if v]
                 for pid in prompt_ids} for ep in eps}
    cons = {ep: {pid: answers[ep].get(pid, [""])[0] for pid in prompt_ids} for ep in eps}  # 2 reps: consensus trivial

    def sim(a, b) -> float:
        """Composite: 0.5 * value agreement + 0.5 * consensus dice."""
        agree = [vals[a][pid][:1] == vals[b][pid][:1]
                 for pid in prompt_ids if vals[a].get(pid) and vals[b].get(pid)]
        val_sim = sum(agree) / len(agree) if agree else 0.0
        txt = [dice(cons[a][pid], cons[b][pid]) for pid in prompt_ids]
        txt_sim = sum(txt) / len(txt) if txt else 0.0
        return 0.5 * val_sim + 0.5 * txt_sim

    sims = {(a, b): sim(a, b) for a, b in combinations(eps, 2)}

    def get(a, b):
        return sims[(min(a, b), max(a, b))]

    print("matrice composite (phase 1):")
    print("     " + " ".join(f"{e[-2:]}" for e in eps))
    for a in eps:
        print(f"{a[-2:]}  " + " ".join(f"{get(a, b):.2f}" if a != b else " 1. " for b in eps))

    def cluster(tau):
        parent = {ep: ep for ep in eps}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (a, b), s in sims.items():
            if s >= tau:
                parent[find(a)] = find(b)
        return {ep: find(ep) for ep in eps}

    models_map = cluster(TAU_MODEL)
    families = cluster(TAU_FAMILY)
    rows = partition_rows(models_map, families)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["endpoint", "family", "model"])
        writer.writerows(rows)
    model_groups = defaultdict(list)
    family_groups = defaultdict(set)
    for ep, family, model in rows:
        model_groups[model].append(ep)
        family_groups[family].add(model)
    print(f"\n{len(family_groups)} familles, {len(model_groups)} modeles -> {OUT}")
    for family, models in family_groups.items():
        groups = [model_groups[model] for model in sorted(models)]
        print(f"famille {family}: " + " | ".join("{" + ",".join(g) + "}" for g in groups))


if __name__ == "__main__":
    main()

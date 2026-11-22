#!/usr/bin/env python3
"""Calibration on ref-01..09: consensus-Hamming analysis of reference_answers.jsonl.

Method (v1):
  1. Per prompt & endpoint, build the consensus response = majority vote per character
     position across repeats. Sampling-only disguises (watermark, decoding) move individual
     samples but leave confident positions' consensus intact; weight changes (fp8, LoRA)
     move the consensus itself.
  2. Pairwise similarity between endpoints = mean over prompts of consensus agreement
     (matched positions / max consensus length).
  3. Cluster by single-linkage over a threshold, write practice_answer.csv, print the
     matrix so the threshold can be eyeballed.

Usage: python analyze.py [threshold] ; then: python score_practice.py practice_answer.csv
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
def normalize(s: str) -> str:
    """Strip code fences and surrounding whitespace: formatting does not track identity."""
    lines = s.strip().splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
        if lines and lines[-1].lstrip().startswith("```"):
            lines = lines[:-1]
    return "\n".join(l.rstrip() for l in lines if l.strip()).strip()


from pathlib import Path


ANSWERS = Path(__file__).with_name("reference_answers.jsonl")
OUT = Path(__file__).with_name("practice_answer.csv")


def consensus(strings: list[str]) -> str:
    """Majority-vote consensus, one char per position."""
    if not strings:
        return ""
    n = max(len(s) for s in strings)
    out = []
    for i in range(n):
        votes = Counter(s[i] for s in strings if i < len(s))
        out.append(votes.most_common(1)[0][0] if votes else "")
    return "".join(out)


def similarity(a: str, b: str) -> float:
    """Fraction of matching positions over the longer consensus."""
    n = max(len(a), len(b))
    if n == 0:
        return 1.0
    return sum(1 for i in range(min(len(a), len(b))) if a[i] == b[i]) / n


def read_prompt_csv(path: Path) -> list[tuple[str, str]]:
    """Reject truncated CSV rows before any request can consume an endpoint budget."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if len(reader.fieldnames or ()) != 2 or set(reader.fieldnames) != {"id", "prompt"}:
            raise ValueError(f"{path}: CSV header must contain exactly id,prompt")
        prompts, seen = [], set()
        for line, row in enumerate(reader, 2):
            if None in row or not row["id"] or not row["prompt"]:
                raise ValueError(f"{path}:{line}: expected two columns; quote prompts containing commas")
            pid = row["id"].strip()
            if not pid or not row["prompt"].strip() or pid in seen:
                raise ValueError(f"{path}:{line}: empty prompt or duplicate/empty id {pid!r}")
            seen.add(pid)
            prompts.append((pid, row["prompt"]))
    return prompts


def partition_rows(models: dict, families: dict) -> list[tuple[str, str, str]]:
    """Serialize endpoint partitions without changing their same/different relation.

    Both maps must cover exactly the same endpoints, and every model group must
    belong to one family. Labels follow sorted endpoints, not arbitrary roots.
    """
    if models.keys() != families.keys():
        raise ValueError("Model and family partitions must cover the same endpoints")
    family_labels, model_labels, model_families = {}, {}, {}
    rows = []
    for endpoint in sorted(models):
        model, family = models[endpoint], families[endpoint]
        if model in model_families and model_families[model] != family:
            raise ValueError(f"Model group {model!r} spans multiple families")
        model_families[model] = family
        family_label = family_labels.setdefault(family, f"F{len(family_labels)}")
        model_label = model_labels.setdefault(model, f"M{len(model_labels)}")
        rows.append((endpoint, family_label, model_label))
    return rows


def main() -> None:
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9

    # answers[endpoint][prompt_id] = list of responses
    answers = defaultdict(lambda: defaultdict(list))
    with ANSWERS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            answers[r["endpoint"]][r["id"]].append(normalize(r["response"]))

    endpoints = sorted(answers)
    prompt_ids = sorted({pid for ep in answers.values() for pid in ep})
    cons = {ep: {pid: consensus(answers[ep].get(pid, [])) for pid in prompt_ids}
            for ep in endpoints}

    # Self-agreement per prompt (how deterministic the endpoint is) — diagnostic only.
    print(f"{'prompt':>7} | " + " ".join(f"{ep[-2:]}" for ep in endpoints) + "  (mean pairwise similarity of repeats)")
    for pid in prompt_ids:
        row = []
        for ep in endpoints:
            reps = answers[ep].get(pid, [])
            if len(reps) < 2:
                row.append("  --")
                continue
            sims = [similarity(a, b) for a, b in combinations(reps, 2)]
            row.append(f"{sum(sims) / len(sims):.2f}")
        print(f"{pid:>7} | " + " ".join(row))

    # Pairwise consensus similarity, averaged over prompts.
    print("\nConsensus similarity matrix (mean over prompts):")
    pair_sim = {}
    for a, b in combinations(endpoints, 2):
        sims = [similarity(cons[a][pid], cons[b][pid]) for pid in prompt_ids
                if cons[a][pid] and cons[b][pid]]
        pair_sim[(a, b)] = sum(sims) / len(sims) if sims else 0.0
    header = "       " + " ".join(f"{ep[-2:]}" for ep in endpoints)
    print(header)
    for a in endpoints:
        row = []
        for b in endpoints:
            if a == b:
                row.append(" 1.0")
            else:
                key = (a, b) if a < b else (b, a)
                row.append(f"{pair_sim[key]:.2f}")
        print(f"ref-{a[-2:]} " + " ".join(row))

    # Single-linkage clustering at the threshold.
    parent = {ep: ep for ep in endpoints}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), s in pair_sim.items():
        if s >= threshold:
            parent[find(a)] = find(b)
    groups = defaultdict(list)
    for ep in endpoints:
        groups[find(ep)].append(ep)

    models = {ep: find(ep) for ep in endpoints}
    rows = partition_rows(models, models)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["endpoint", "family", "model"])
        writer.writerows(rows)
    print(f"\nThreshold {threshold:.2f} -> {len(groups)} groups:")
    for _, eps in sorted(groups.items()):
        print("  " + ", ".join(sorted(eps)))
    print(f"Wrote {OUT}; score with: python score_practice.py {OUT.name}")


if __name__ == "__main__":
    main()

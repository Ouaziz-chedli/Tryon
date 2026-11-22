#!/usr/bin/env python3
"""Collect v2 near-tie prompts (ids 100+) on the critical refs, appended to the same jsonl.

ref-04/07 (LoRA 0.7/1.5) are already separated by v1 data, so v2 repeats are
not spent on them. Reprise-safe: answers already in the jsonl are skipped.
Run: .venv/bin/python collect2.py
"""
from pathlib import Path

import query_reference as qr

REPEATS = 15
ENDPOINTS = ["ref-01", "ref-02", "ref-03", "ref-05", "ref-06", "ref-08", "ref-09"]

if __name__ == "__main__":
    prompts = qr.read_prompts(Path("prompts2.csv"))
    qr.query(prompts, output=Path("reference_answers.jsonl"),
             endpoints=ENDPOINTS, repeats=REPEATS)

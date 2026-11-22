#!/usr/bin/env python3
"""Phase 2 real collection: all 12 calibrated prompts x 7 more repeats x 33 eps.

Deepens the same answers.jsonl (phase-1 rows skipped automatically).
Total spend after this run: 94 of 100 per endpoint; ~2 500 queries overnight.
"""
from pathlib import Path

import task_template as tt

REPEATS = 7

if __name__ == "__main__":
    prompts = tt.read_prompts(Path("prompts2.csv"))          # ids 100-112
    tt.query(prompts, output=Path("answers.jsonl"), endpoints=tt.ENDPOINTS, repeats=REPEATS)

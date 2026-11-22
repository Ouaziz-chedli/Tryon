#!/usr/bin/env python3
"""Phase 3: targeted deepening of AMBIGUOUS endpoints only.

The 7-block {ep-01,04,12,14,17,18,25} already sits at 0.85-0.94 similarity
with 3-4 reps -> leave it for a later light top-up. Everything else is
uncertain at this rep level -> push to the per-endpoint cap (100).
Budget is per endpoint (non-transferable): this only redirects depth, not budget.
"""
from pathlib import Path

import task_template as tt

CLEAR = {"ep-01", "ep-04", "ep-12", "ep-14", "ep-17", "ep-18", "ep-25"}
AMBIGUOUS = [ep for ep in tt.ENDPOINTS if ep not in CLEAR]
REPEATS = 9  # planned 13x9=117/ep, capped at the 100 budget by the server-side planner

if __name__ == "__main__":
    prompts = tt.read_prompts(Path("prompts2.csv"))
    tt.query(prompts, output=Path("answers.jsonl"), endpoints=AMBIGUOUS, repeats=REPEATS)

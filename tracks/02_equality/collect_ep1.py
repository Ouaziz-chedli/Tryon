#!/usr/bin/env python3
"""Phase 1 real collection: 5 calibrated prompts x 2 repeats x 33 endpoints.

Subset of the phase-2 plan (same ids) -> zero budget waste: phase 2 will deepen
these very rows. Output: answers.jsonl (the template's file for ep data).
Budget used: 10 of 100 per endpoint.
"""
from pathlib import Path

import task_template as tt

PROMPTS = [
    ("100", "Is 51 a prime number? Answer with yes or no only:"),
    ("101", "Compute 47 * 83. Answer with the number only."),
    ("109", "How many days in a leap year? Answer with the number only."),
    ("110", "What is 30% of 90? Answer with the number only."),
    ("112", "Complete the Python function in one line: def double(x):"),
]

if __name__ == "__main__":
    tt.query(PROMPTS, output=Path("answers.jsonl"), endpoints=tt.ENDPOINTS, repeats=2)

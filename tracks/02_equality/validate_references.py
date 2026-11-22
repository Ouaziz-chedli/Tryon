#!/usr/bin/env python3
"""Calibrate on disjoint reference repeats; never query endpoints or public scores."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
from analyze_ep2 import PROMPT_IDS


REFERENCES = tuple(f"ref-{i:02d}" for i in range(1, 10))
PROMPTS = tuple(sorted(PROMPT_IDS))
THRESHOLDS = tuple(i / 100 for i in range(35, 96, 5))
MODE_WEIGHTS = (1.0, 0.5, 0.0)
BASE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_truth(path):
    truth = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"endpoint", "family", "model", "kind"} <= set(reader.fieldnames or ()):
            raise ValueError("Truth CSV needs endpoint,family,model,kind columns")
        for line, row in enumerate(reader, 2):
            endpoint = (row.get("endpoint") or "").strip()
            values = tuple((row.get(key) or "").strip() for key in ("family", "model", "kind"))
            if endpoint not in REFERENCES or endpoint in truth or not all(values):
                raise ValueError(f"Invalid, duplicate, or unexpected truth reference at line {line}")
            truth[endpoint] = values
    missing = sorted(set(REFERENCES) - truth.keys())
    if missing:
        raise ValueError("Truth CSV missing references: " + ", ".join(missing))
    return truth


def split_answers(answers, repeats, seed):
    """Equal sample depth, with disjoint original-record indices retained for provenance."""
    missing = []
    coverage = {}
    for endpoint in REFERENCES:
        coverage[endpoint] = {}
        for prompt in PROMPTS:
            count = len(answers.get(endpoint, {}).get(prompt, []))
            coverage[endpoint][str(prompt)] = count
            if count < 2 * repeats:
                missing.append(f"{endpoint} prompt {prompt}: {count}/{2 * repeats} responses (need {2 * repeats - count} more)")
    if missing:
        raise ValueError("Incomplete reference coverage; collect missing reference responses before calibration:\n  "
                         + "\n  ".join(missing))
    rng = random.Random(seed)
    training = {endpoint: {} for endpoint in REFERENCES}
    held_out = {endpoint: {} for endpoint in REFERENCES}
    selection = {endpoint: {} for endpoint in REFERENCES}
    for endpoint in REFERENCES:
        for prompt in PROMPTS:
            responses = answers[endpoint][prompt]
            indices = list(range(len(responses)))
            rng.shuffle(indices)
            train_indices = indices[:repeats]
            held_out_indices = indices[repeats:2 * repeats]
            training[endpoint][prompt] = [responses[i] for i in train_indices]
            held_out[endpoint][prompt] = [responses[i] for i in held_out_indices]
            selection[endpoint][str(prompt)] = {
                "training_indices": train_indices,
                "held_out_indices": held_out_indices,
                "unused_count": len(responses) - 2 * repeats,
            }
    return training, held_out, coverage, selection


def prediction_rows(answers, mode_weight, threshold):
    # Imports stay lazy so importing this CLI cannot trigger legacy analyzers.
    from analyze import partition_rows
    from analyze_ep2 import cluster, pair_similarities

    models = cluster(REFERENCES, pair_similarities(answers, mode_weight), threshold)
    # All nine references share a family; do not leak their model truth into labels.
    return partition_rows(models, {endpoint: "reference" for endpoint in REFERENCES})


def report(rows, truth):
    mine = {endpoint: (family, model) for endpoint, family, model in rows}
    if set(mine) != set(REFERENCES) or len(rows) != len(REFERENCES):
        raise ValueError("partition_rows must emit exactly one row for each of the nine references")
    confusion = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
    by_kind = {}
    wrong = []
    for a, b in itertools.combinations(REFERENCES, 2):
        actual = truth[a][:2] == truth[b][:2]
        predicted = mine[a] == mine[b]
        cell = "tp" if actual and predicted else "fn" if actual else "fp" if predicted else "tn"
        confusion[cell] += 1
        kinds = tuple(sorted((truth[a][2], truth[b][2])))
        counts = by_kind.setdefault(kinds, dict.fromkeys(confusion, 0))
        counts[cell] += 1
        if actual != predicted:
            wrong.append({"endpoints": [a, b], "kinds": [truth[a][2], truth[b][2]],
                          "truth_same": actual, "predicted_same": predicted})
    tp, fp, fn, tn = (confusion[key] for key in ("tp", "fp", "fn", "tn"))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    score = max(0.0, (tp * tn - fp * fn) / denominator) if denominator else 0.0
    return {"score": score, "confusion": confusion, "wrong_pairs": wrong,
            "by_kind_pair": [{"kinds": list(kinds), "confusion": counts}
                             for kinds, counts in sorted(by_kind.items())],
            "rows": [{"endpoint": endpoint, "family": family, "model": model}
                     for endpoint, family, model in rows]}


def select_parameters(training, truth):
    """Rank training MCC, mode-only simplicity, then widest stable partition plateau."""
    from analyze import partition_rows
    from analyze_ep2 import cluster, pair_similarities

    candidates = []
    for weight in MODE_WEIGHTS:
        sims = pair_similarities(training, weight)
        for threshold in THRESHOLDS:
            models = cluster(REFERENCES, sims, threshold)
            rows = partition_rows(models, {endpoint: "reference" for endpoint in REFERENCES})
            evaluation = report(rows, truth)
            labels = {endpoint: (family, model) for endpoint, family, model in rows}
            partition = tuple(labels[a] == labels[b] for a, b in itertools.combinations(REFERENCES, 2))
            candidates.append({"mode_weight": weight, "threshold": threshold,
                               "score": evaluation["score"], "partition": partition,
                               "report": evaluation})
    best_score = max(candidate["score"] for candidate in candidates)
    best_weight = next(weight for weight in MODE_WEIGHTS
                       if any(c["mode_weight"] == weight and c["score"] == best_score for c in candidates))
    eligible = [c for c in candidates if c["mode_weight"] == best_weight]
    plateaus = []
    run = []
    for candidate in eligible:
        if run and (candidate["score"] != best_score or candidate["partition"] != run[-1]["partition"]):
            plateaus.append(run)
            run = []
        if candidate["score"] == best_score:
            run.append(candidate)
    if run:
        plateaus.append(run)
    # A wider region is less sensitive to one grid step. On ties choose the
    # higher threshold (conservative about merging distinct model identities).
    plateau = max(plateaus, key=lambda items: (len(items), items[-1]["threshold"]))
    chosen = plateau[len(plateau) // 2]
    selected = {"mode_weight": chosen["mode_weight"], "threshold": chosen["threshold"],
                "stable_threshold_region": [plateau[0]["threshold"], plateau[-1]["threshold"]],
                "selection_rule": "training MCC; mode-only then 0.5 then overlap-only; widest contiguous identical-partition optimum; higher region and upper midpoint on ties"}
    grid = [{key: c[key] for key in ("mode_weight", "threshold", "score")} for c in candidates]
    return selected, chosen["report"], grid


def score_csv(rows, truth_path):
    """Exercise the actual submission CSV parser and real practice scorer."""
    scorer = BASE / "score_practice.py"
    with tempfile.TemporaryDirectory(prefix="reference-validation-") as directory:
        csv_path = Path(directory) / "held_out.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("endpoint", "family", "model"))
            writer.writerows(rows)
        result = subprocess.run([sys.executable, str(scorer), str(csv_path), str(truth_path.resolve())],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            raise ValueError(f"score_practice.py failed ({result.returncode}): {result.stderr.strip()}")
        match = re.search(r"Practice score \(MCC over (\d+) pairs, same model or not\): ([0-9.]+)", result.stdout)
        if not match or int(match.group(1)) != 36:
            raise ValueError("Cannot parse score_practice.py output for all 36 reference pairs")
        return {"score": float(match.group(2)), "pairs": int(match.group(1)),
                "command": ["score_practice.py", "<temporary-held_out.csv>", str(truth_path.resolve())],
                "stdout": result.stdout.strip()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=BASE / "reference_answers.jsonl")
    parser.add_argument("--truth", type=Path, default=BASE / "practice_answers.csv")
    parser.add_argument("--output", type=Path, default=BASE / "calibration.json")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2666)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.output.resolve() in {args.answers.resolve(), args.truth.resolve()}:
        parser.error("--output must not overwrite either input")
    try:
        from analyze_ep2 import load_answers

        paths = {"answers": args.answers, "truth": args.truth,
                 "analyzer": BASE / "analyze_ep2.py", "partition_exporter": BASE / "analyze.py",
                 "scorer": BASE / "score_practice.py", "validator": Path(__file__).resolve()}
        provenance = {name: {"path": str(path.resolve()), "sha256": sha256(path)}
                      for name, path in paths.items()}
        truth = read_truth(args.truth)
        answers = load_answers(args.answers)
        training, held_out, coverage, split = split_answers(answers, args.repeats, args.seed)
        selected, train_report, grid = select_parameters(training, truth)
        # Selection is now frozen; held-out responses enter the pipeline only here.
        held_out_rows = prediction_rows(held_out, selected["mode_weight"], selected["threshold"])
        held_out_report = report(held_out_rows, truth)
        held_out_report["official_csv_score"] = score_csv(held_out_rows, args.truth)
        if abs(held_out_report["score"] - held_out_report["official_csv_score"]["score"]) > 0.000501:
            raise ValueError("Internal MCC and actual CSV scorer disagree")
        for name, path in paths.items():
            if sha256(path) != provenance[name]["sha256"]:
                raise ValueError(f"{name} changed during calibration; rerun on a stable snapshot")
        result = {"selected": selected, "train": train_report, "held_out": held_out_report,
                  "training_grid": grid, "coverage": coverage,
                  "provenance": {"files": provenance, "seed": args.seed, "repeats_per_split": args.repeats,
                                 "prompts": list(PROMPTS), "references": list(REFERENCES),
                                 "split_algorithm": "Python random.Random(seed), sorted reference/prompt order, shuffle original zero-based response indices; first repeats training, next repeats held-out",
                                 "python_version": sys.version.split()[0], "selected_response_indices": split,
                                 "ignored_endpoints": sorted(set(answers) - set(REFERENCES)),
                                 "endpoint_queries": 0, "public_feedback_used": False}}
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Calibration failed: {error}\n")
    print(f"Coverage: 9 references x {len(PROMPTS)} usable prompts; {args.repeats} training + {args.repeats} disjoint held-out responses per cell")
    print(f"Selected on training only: mode_weight={selected['mode_weight']:.1f}, threshold={selected['threshold']:.2f}, stable region={selected['stable_threshold_region']}")
    print(f"Training MCC: {train_report['score']:.6f}; held-out CSV MCC (official): {held_out_report['official_csv_score']['score']:.3f}")
    for name, evaluation in (("Training", train_report), ("Held-out", held_out_report)):
        print(f"{name} confusion: {evaluation['confusion']}; wrong pairs: {len(evaluation['wrong_pairs'])}")
        for wrong in evaluation["wrong_pairs"]:
            print(f"  {wrong['endpoints'][0]} ({wrong['kinds'][0]}) / {wrong['endpoints'][1]} ({wrong['kinds'][1]}): predicted {'same' if wrong['predicted_same'] else 'different'}, truth {'same' if wrong['truth_same'] else 'different'}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()

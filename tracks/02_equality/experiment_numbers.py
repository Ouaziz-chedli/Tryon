#!/usr/bin/env python3
"""Numeric-response distributions with joint nuisance removal; reference-only selection.

Uses existing arithmetic probes. Values remain values, including incorrect answers;
no text-generation, model-query or submission clients are imported by this module.
"""
import argparse
from itertools import combinations
import json
import math
from pathlib import Path
import re
import unicodedata

import numpy as np

import experiment_metric as common
from analyze_ep2 import cluster, write_candidate

ROOT = Path(__file__).resolve().parent
TARGETS = {1: 391, 101: 3901, 102: 494, 109: 366, 110: 27}
BANDWIDTHS = (0.25, 1.0)


def numeric_value(text):
    text = unicodedata.normalize("NFKC", text).replace("−", "-")
    number = r"[+-]?\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?"
    final = re.search(r"\bFINAL\s*:\s*([^\n]*)", text, re.I)
    if final:
        # Generated follow-up tasks are not answers to the requested task.
        scalar = re.fullmatch(r"[\s*`]*(" + number + r")[\s*`]*[.!]?", final.group(1))
        if not scalar:
            return None
        token = scalar.group(1)
    else:
        literal_end = r"(?!\d|[.,]\d|[eE][+-]?\d|\s*[+*/^=]|\s*-\s*\d)"
        explicit = re.findall(r"(?:=|\b(?:answer|result|value|equals|gives|is)\s*(?:is|:|=)?)\s*(" + number + r")" + literal_end, text, re.I)
        tokens = re.findall(number, text)
        if not explicit and len(tokens) != 1:
            return None
        token = explicit[-1] if explicit else tokens[0]
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", token):
        token = token.replace(",", "")
    else:
        token = token.replace(",", ".")
    try:
        value = float(token)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def values_for(texts, pid):
    target = TARGETS[pid]
    return [(value - target) / max(abs(target), 1)
            for text in texts if (value := numeric_value(text)) is not None]


def fit_kernel(answers, pid, bandwidth):
    values = [v for ep in common.REFERENCES for v in values_for(answers[ep][pid], pid)]
    if not values:
        raise ValueError(f"probe {pid}: no numeric responses")
    landmarks = np.unique(np.r_[np.quantile(values, np.linspace(0, 1, 9)), 0.0])
    scale = max(float(np.quantile(values, .9) - np.quantile(values, .1)), 1e-3)
    return {"landmarks": landmarks.tolist(), "scale": scale * bandwidth}


def kernel_centroids(answers, pid, parameters, endpoints):
    landmarks = np.asarray(parameters["landmarks"])
    result = []
    for ep in endpoints:
        texts = answers.get(ep, {}).get(pid, [])
        values = values_for(texts, pid)
        if not texts or not values:
            raise ValueError(f"{ep}/{pid}: no usable numeric evidence")
        delta = (np.asarray(values)[:, None] - landmarks) / parameters["scale"]
        # Cauchy features preserve proximity between different wrong numbers.
        features = 1 / (1 + delta * delta)
        # Missing parses contribute no equality evidence; never average only successes.
        vector = features.sum(axis=0) / len(texts) / np.sqrt(len(landmarks))
        if np.linalg.norm(vector) < 1e-12:
            raise ValueError(f"{ep}/{pid}: numeric responses outside fitted support")
        result.append(vector)
    return np.vstack(result)


def nuisance_basis(matrix, projection):
    if not projection:
        return np.empty((0, matrix.shape[1]))
    same = matrix[[common.REFERENCES.index(ep) for ep in sorted(common.SAME)]]
    differences = same - same.mean(axis=0)
    _, singular, right = np.linalg.svd(differences, full_matrices=False)
    tolerance = max(differences.shape) * np.finfo(float).eps * max(float(singular[0]), 1)
    return right[singular > tolerance]


def similarities_from(matrix, basis, endpoints):
    if basis.size:
        matrix = matrix - (matrix @ basis.T) @ basis
    squares = (matrix * matrix).sum(axis=1)
    if np.any(squares < 1e-24):
        raise ValueError("zero projected numeric centroid")
    distance = np.maximum(squares[:, None] + squares[None, :] - 2 * matrix @ matrix.T, 0)
    similarity = 1 - np.clip(distance, 0, 2) / 2
    return {(a, b): float(similarity[i, j]) for i, a in enumerate(endpoints)
            for j, b in enumerate(endpoints) if i < j}


def fit_metric(answers, ids, bandwidth, projection):
    kernels = {pid: fit_kernel(answers, pid, bandwidth) for pid in ids}
    matrix = np.concatenate([kernel_centroids(answers, pid, kernels[pid], common.REFERENCES)
                             for pid in ids], axis=1) / np.sqrt(len(ids))
    return {"kernels": kernels, "basis": nuisance_basis(matrix, projection).tolist()}


def pair_similarities(answers, fitted):
    endpoints = sorted(answers)
    ids = sorted(map(int, fitted["kernels"]))
    if any(len(answers.get(ep, {}).get(pid, ())) != common.REPLIES for ep in endpoints for pid in ids):
        raise ValueError("need exactly four responses per endpoint/probe")
    matrices = [kernel_centroids(answers, pid, fitted["kernels"].get(pid, fitted["kernels"].get(str(pid))), endpoints)
                for pid in ids]
    matrix = np.concatenate(matrices, axis=1) / np.sqrt(len(ids))
    basis = np.asarray(fitted["basis"])
    return similarities_from(matrix, basis, endpoints)


def source_hashes():
    return {name: common.sha256(ROOT / name) for name in
            ("experiment_numbers.py", "experiment_metric.py", "analyze_ep2.py", "analyze.py")}


def select(data, splits=12, seed=2666):
    groups = common.grouped_rows(data)
    ids = [pid for pid in TARGETS if all(len(groups.get(ep, {}).get(pid, ())) >= 8 for ep in common.REFERENCES)]
    if not ids:
        raise ValueError("no arithmetic probe has eight clean responses at every reference")
    split_data = [common.split_training(groups, ids, seed, i) for i in range(splits)]
    leaderboard = []
    rejected = []
    for count in range(1, min(4, len(ids)) + 1):
        for subset in combinations(ids, count):
            for bandwidth in BANDWIDTHS:
                for projection in (False, True):
                    training_vectors, validation_vectors = [], []
                    try:
                        for train, validation, _ in split_data:
                            fitted = fit_metric(train, subset, bandwidth, projection)
                            training_vectors.append(list(pair_similarities(train, fitted).values()))
                            validation_vectors.append(list(pair_similarities(validation, fitted).values()))
                    except ValueError as error:
                        rejected.append({"ids": subset, "bandwidth": bandwidth, "projection": projection, "error": str(error)})
                        continue
                    best = None
                    for threshold in common.THRESHOLDS:
                        reports = [common.evaluate_vector(row, threshold) for row in validation_vectors]
                        candidate = {"prompt_ids": list(subset), "representation": "numeric", "bandwidth": bandwidth,
                                     "projection": projection, "threshold": threshold,
                                     "validation": common.summarize_reports(reports)}
                        if best is None or common.candidate_rank(candidate) > common.candidate_rank(best):
                            best = candidate
                    best["training"] = common.summarize_reports([common.evaluate_vector(row, best["threshold"]) for row in training_vectors])
                    best["accepted"] = best["validation"]["mean_clipped_mcc"] >= .1 and best["validation"]["positive_fraction"] >= .75
                    leaderboard.append(best)
    if not leaderboard:
        raise ValueError(f"all numeric candidates rejected: {rejected}")
    accepted = [c for c in leaderboard if c["accepted"]]
    selected = max(accepted or leaderboard, key=common.candidate_rank)
    answers = common.training_answers(data, selected["prompt_ids"])
    fitted = fit_metric(answers, selected["prompt_ids"], selected["bandwidth"], selected["projection"])
    return selected, fitted, leaderboard, rejected


def load_frozen(path):
    config = json.loads(Path(path).read_text())
    saved = config.pop("config_sha256")
    if common.canonical_hash(config) != saved:
        raise ValueError("numeric configuration hash mismatch")
    config["config_sha256"] = saved
    if config["source_sha256"] != source_hashes() or common.sha256(config["training_file"]) != config["training_sha256"]:
        raise ValueError("numeric source or immutable training changed")
    if not config["selected"]["accepted"]:
        raise ValueError("numeric inner validation rejected")
    return config, config["fitted"]


def validate(config, fitted, data):
    ids = config["selected"]["prompt_ids"]
    keys = {tuple(key) for key in config["training_keys"]}
    if keys & {common.record_key(row) for row in data["rows"] if row["id"] in ids}:
        raise ValueError("numeric holdout reuses training keys")
    groups = common.grouped_rows(data)
    if set(groups) != set(common.REFERENCES):
        raise ValueError("holdout needs all nine references")
    answers = {}
    for ep in common.REFERENCES:
        answers[ep] = {}
        for pid in ids:
            if data["prompts"].get(pid) != config["prompts"][str(pid)]:
                raise ValueError(f"prompt {pid} changed")
            rows = groups[ep].get(pid, ())[:common.REPLIES]
            if len(rows) != common.REPLIES:
                raise ValueError(f"{ep}/{pid}: incomplete fresh numeric holdout")
            answers[ep][pid] = [row["response"] for row in rows]
    sims = pair_similarities(answers, fitted)
    return common.identity_report(cluster(answers, sims, config["selected"]["threshold"]), sims)


def self_check():
    assert numeric_value("17 * 23 = 391") == 391
    assert numeric_value("17 and 23") is None
    assert numeric_value("FINAL: 3,901") == 3901
    assert numeric_value("The answer is 3881.") == 3881
    assert numeric_value("No result available") is None
    assert numeric_value("FINAL: 3 * 4") is None
    assert numeric_value("FINAL: 7\n\nTask: another question\nFINAL: 9") == 7
    assert numeric_value("FINAL: 1e999") is None
    fitted = {"landmarks": [0., .1], "scale": .1}
    answers = {"a": {101: ["3901"] * 4}, "b": {101: ["3902"] * 4}, "c": {101: ["8000"] * 4}}
    matrix = kernel_centroids(answers, 101, fitted, list(answers))
    sims = similarities_from(matrix, np.empty((0, 2)), list(answers))
    assert sims["a", "b"] > sims["a", "c"]
    print("numeric self-check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not args.output or bool(args.frozen) != bool(args.validation) or bool(args.train) == bool(args.frozen):
        parser.error("use --train or --frozen/--validation, and --output")
    inputs = {p.resolve() for p in (args.train, args.frozen, args.validation) if p}
    inputs.update((ROOT / name).resolve() for name in source_hashes())
    if args.frozen:
        inputs.add(Path(json.loads(args.frozen.read_text())["training_file"]).resolve())
    csv_path = args.output.with_suffix(".csv")
    if args.output.resolve() in inputs or csv_path.resolve() in inputs or args.output.resolve() == csv_path.resolve():
        parser.error("outputs must be distinct and must not overwrite inputs")
    initial_sources = source_hashes()
    try:
        if args.train:
            train_hash = common.sha256(args.train)
            data = common.load_responses(args.train)
            selected, fitted, leaderboard, rejected = select(data)
            result = {"selected": selected, "fitted": fitted, "leaderboard": leaderboard, "rejected": rejected,
                      "training_file": str(args.train.resolve()), "training_sha256": train_hash,
                      "training_keys": [list(common.record_key(row)) for row in data["rows"] if row["id"] in selected["prompt_ids"]],
                      "prompts": {str(pid): data["prompts"][pid] for pid in selected["prompt_ids"]},
                      "source_sha256": initial_sources, "warning": "Inner selection only; fresh confirmation still required."}
            if common.sha256(args.train) != train_hash:
                raise ValueError("training file changed")
            result["config_sha256"] = common.canonical_hash(result)
            print("Numeric inner selection:", selected["prompt_ids"], "mean MCC", selected["validation"]["mean_clipped_mcc"], "accepted", selected["accepted"])
        else:
            config, fitted = load_frozen(args.frozen)
            validation_hash = common.sha256(args.validation)
            report = validate(config, fitted, common.load_responses(args.validation))
            if common.sha256(args.validation) != validation_hash:
                raise ValueError("holdout changed")
            result = {"frozen_sha256": config["config_sha256"], "report": report,
                      "validation_sha256": validation_hash, "selected": config["selected"]}
            write_candidate(csv_path, report["models"], {ep: "reference" for ep in common.REFERENCES})
            print("Numeric fresh MCC:", report["clipped_mcc"], report["confusion"])
        if source_hashes() != initial_sources:
            raise ValueError("source changed during run")
        common.save_json(args.output, result)
        print("Saved", args.output)
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
        parser.exit(2, f"Numeric experiment failed: {error}\n")


if __name__ == "__main__":
    main()

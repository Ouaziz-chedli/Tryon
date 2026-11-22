#!/usr/bin/env python3
"""Apply a frozen metric only after two disjoint reference confirmations."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path
import random

import experiment_metric as metric
from analyze_ep2 import cluster, write_candidate

ROOT = Path(__file__).resolve().parent
MIN_CONFIRMATION_MCC = 0.5


def check_confirmations(config_id, reports):
    if len(reports) != 2:
        raise ValueError("two independent confirmation reports are required")
    used = set()
    for report in reports:
        if report["mode"] != "external_validation_no_selection" or report["frozen_config_sha256"] != config_id:
            raise ValueError("confirmation does not use this frozen configuration")
        score = report["report"]["clipped_mcc"]
        if not MIN_CONFIRMATION_MCC <= score <= 1:
            raise ValueError("both independent reference MCCs must be at least 0.5")
        keys = {(ep, int(pid), repeat) for ep, per in report["report"]["used_repeats"].items()
                for pid, repeats in per.items() for repeat in repeats}
        if not keys or keys & used:
            raise ValueError("the two confirmation blocks overlap or contain no evidence")
        used.update(keys)


def load_confirmed(config_path, confirmation_path):
    config, fitted = metric.load_frozen(config_path)
    certificate = json.loads(confirmation_path.read_text())
    if certificate["config_sha256"] != config["config_sha256"]:
        raise ValueError("confirmation certificate belongs to another configuration")
    reports, evidence = [], []
    for block in certificate["blocks"]:
        paths = {key: confirmation_path.parent / block[key] for key in ("report", "data", "csv")}
        for key, path in paths.items():
            if metric.sha256(path) != block[key + "_sha256"]:
                raise ValueError(f"confirmation {key} changed: {path}")
        report = json.loads(paths["report"].read_text())
        if report["validation_sha256"] != block["data_sha256"]:
            raise ValueError("confirmation report is bound to another raw response file")
        derived = metric.validate_external(config, fitted, metric.load_responses(paths["data"]))
        if (derived["models"] != report["report"]["models"] or
                derived["used_repeats"] != report["report"]["used_repeats"] or
                abs(derived["clipped_mcc"] - report["report"]["clipped_mcc"]) > 1e-12):
            raise ValueError("stored reference result does not match the raw evidence")
        reports.append(report)
        evidence.append({key: str(path) for key, path in paths.items()} |
                        {key + "_sha256": metric.sha256(path) for key, path in paths.items()})
    check_confirmations(config["config_sha256"], reports)
    return config, fitted, evidence


def load_evaluated(paths, config):
    prompts = {p["id"]: p["prompt"] for p in config["prompts"]}
    groups = defaultdict(lambda: defaultdict(list))
    seen = set()
    for path in paths:
        data = metric.load_responses(path)
        if any(int(key[1]) in prompts for key in data["audit"]["ambiguous_keys"]):
            raise ValueError(f"{path}: ambiguous repeats for a selected probe")
        for row in data["rows"]:
            pid = row["id"]
            if pid not in prompts:
                continue
            if row["prompt"] != prompts[pid]:
                raise ValueError(f"{path}: selected probe {pid} has another prompt text")
            key = metric.record_key(row)
            if key in seen:
                raise ValueError(f"selected repeat duplicated across inputs: {key}")
            seen.add(key)
            groups[row["endpoint"]][pid].append(row)
    expected = {f"ep-{i:02d}" for i in range(1, 34)}
    if set(groups) != expected:
        raise ValueError("need exactly ep-01..ep-33 in evaluated inputs")
    answers, manifest = {}, {}
    for ep in sorted(expected):
        answers[ep], manifest[ep] = {}, {}
        for pid in prompts:
            rows = sorted(groups[ep].get(pid, []), key=lambda row: row["repeat"])
            if len(rows) < metric.REPLIES:
                raise ValueError(f"{ep}/{pid}: need four responses, found {len(rows)}")
            chosen = rows[:metric.REPLIES]
            answers[ep][pid] = [row["response"] for row in chosen]
            manifest[ep][str(pid)] = [row["repeat"] for row in chosen]
    return answers, manifest


def require_informative(models):
    # With one predicted family, both constant model partitions guarantee MCC=0.
    if not 1 < len(set(models.values())) < len(models):
        raise ValueError("degenerate all-same/all-different model partition; do not submit")


def self_check():
    def report(start):
        return {"mode": "external_validation_no_selection", "frozen_config_sha256": "fixed",
                "report": {"clipped_mcc": .6, "used_repeats": {"ref-01": {"1": list(range(start, start + 4))}}}}
    check_confirmations("fixed", [report(100), report(104)])
    weak, other_config = report(104), report(104)
    weak["report"]["clipped_mcc"] = .49
    other_config["frozen_config_sha256"] = "not-selected"
    for reports in ([report(100), report(100)], [report(100)], [report(100), report(102)],
                    [report(100), weak], [report(100), other_config]):
        try:
            check_confirmations("fixed", reports)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid confirmation evidence was accepted")
    for models in ({"a": "x", "b": "x"}, {"a": "a", "b": "b"}):
        try:
            require_informative(models)
        except ValueError:
            pass
        else:
            raise AssertionError("a zero-information partition was accepted")
    require_informative({"a": "x", "b": "x", "c": "y"})
    print("candidate self-check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--confirmation", type=Path)
    parser.add_argument("--answers", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if not all((args.config, args.confirmation, args.answers, args.output)):
        parser.error("need --config, --confirmation, --answers and --output")
    try:
        sources = metric.code_hashes() | {Path(__file__).name: metric.sha256(__file__)}
        hashes = {str(path.resolve()): metric.sha256(path) for path in
                  [args.config, args.confirmation, *args.answers]}
        config, fitted, evidence = load_confirmed(args.config, args.confirmation)
        for block in evidence:
            for key in ("report", "data", "csv"):
                hashes[str(Path(block[key]).resolve())] = block[key + "_sha256"]
        hashes[str(Path(config["training_file"]["path"]).resolve())] = config["training_file"]["sha256"]
        report_path = args.output.with_suffix(".report.json")
        forbidden = set(hashes) | {str((ROOT / name).resolve()) for name in sources}
        if str(args.output.resolve()) in forbidden or str(report_path.resolve()) in forbidden:
            raise ValueError("outputs must not overwrite data, confirmations, configuration or code")
        if args.output.resolve() == report_path.resolve():
            raise ValueError("CSV and diagnostic output paths must differ")
        answers, repeats = load_evaluated(args.answers, config)
        similarities = metric.pair_similarities(answers, fitted)
        threshold = config["selected"]["threshold"]
        models = cluster(answers, similarities, threshold)
        require_informative(models)
        rng, stable = random.Random(2666), Counter()
        for _ in range(40):
            sampled = {ep: {pid: rng.choices(texts, k=len(texts)) for pid, texts in per.items()}
                       for ep, per in answers.items()}
            boot = cluster(sampled, metric.pair_similarities(sampled, fitted), threshold)
            stable.update(pair for pair in combinations(sorted(answers), 2) if boot[pair[0]] == boot[pair[1]])
        if any(metric.sha256(path) != value for path, value in hashes.items()):
            raise ValueError("input data or confirmation changed during candidate generation")
        current_sources = metric.code_hashes() | {Path(__file__).name: metric.sha256(__file__)}
        if current_sources != sources:
            raise ValueError("candidate source changed during execution")
        # ponytail: one family caps total score at 0.9; learn a family boundary only
        # with cross-family evidence. Do not let old family guesses veto model pairs.
        rows = write_candidate(args.output, models, {ep: "unresolved" for ep in models})
        result = {"configuration_id": config["config_sha256"], "selected": config["selected"],
                  "independent_confirmation": evidence, "input_sha256": hashes, "source_sha256": sources,
                  "used_repeats": repeats, "family_policy": "one family; model inference is global",
                  "models": models, "bootstrap_repeats": 40, "bootstrap_seed": 2666,
                  "pair_similarity": {f"{a},{b}": value for (a, b), value in similarities.items()},
                  "pair_stability": {f"{a},{b}": stable[a, b] / 40 for a, b in similarities},
                  "csv_sha256": metric.sha256(args.output),
                  "warning": "Reference improvement and resampling stability do not prove unknown model identities."}
        metric.save_json(report_path, result)
        groups = defaultdict(list)
        for ep, family, model in rows:
            groups[model].append(ep)
        print(f"Saved {args.output}: 33 endpoints, {len(groups)} model groups, one unresolved family; all 528 CSV pair relations checked")
        for label, members in groups.items():
            minimum = min((stable[a, b] / 40 for a, b in combinations(members, 2)), default=1.)
            print(label, ",".join(members), f"minimum resampled co-clustering={minimum:.3f}")
        print("Diagnostics:", report_path)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Candidate rejected: {error}\n")


if __name__ == "__main__":
    main()

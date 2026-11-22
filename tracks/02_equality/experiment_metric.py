#!/usr/bin/env python3
"""Offline nuisance-invariant TF-IDF experiment; no endpoint or submission client.

Select (inner validation is tuning, NOT an independent holdout)::
    python experiment_metric.py --train reference_answers.20261122T025452Z.jsonl --output metric.json
Apply unchanged settings to NEW references, with globally fresh repeat IDs::
    python experiment_metric.py --frozen metric.json --validation fresh_refs.jsonl --output metric_holdout.json
    python score_practice.py metric_holdout.csv

Reusable API: load_responses, fit_metric, pair_similarities, load_frozen.
fit_metric consumes {endpoint: {integer_prompt_id: [response_text, ...]}}.
load_frozen returns (config, fitted_transforms); pair_similarities requires exactly
four answers per selected prompt and works on arbitrary endpoint names. It does
not infer families. Existing complete-linkage and partition export are reused.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import random
import tempfile
import unicodedata

import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

from analyze_ep2 import cluster, write_candidate

ROOT = Path(__file__).resolve().parent
REFERENCES = tuple(f"ref-{i:02d}" for i in range(1, 10))
SAME = frozenset(("ref-01", "ref-03", "ref-05", "ref-06", "ref-08"))
PAIRS = tuple(combinations(REFERENCES, 2))
CANDIDATE_IDS = frozenset(range(16)) | (frozenset(range(100, 113)) - {111})
REPLIES = 4
# Fixed coarse and near-one grids: centroid distances can be much smaller than
# individual-response distances. The extra threshold represents all-singleton.
THRESHOLDS = tuple(sorted({i / 20 for i in range(1, 21)} |
                          {1 - 10 ** (-i / 4) for i in range(1, 25)})) + (float(np.nextafter(1.0, 2.0)),)
FEATURES = {"word": (1, 2), "char": (2, 4)}
MIN_INNER_MCC = 0.10
MIN_POSITIVE_FRACTION = 0.75
SCHEMA = 1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def code_hashes():
    return {name: sha256(ROOT / name) for name in
            ("experiment_metric.py", "analyze_ep2.py", "analyze.py")}


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__}


def load_responses(path):
    """Parse allowed historical IDs; discard EVERY row of an ambiguous key.

    Repeated identical rows are ambiguous too: neither copy may enter a split.
    Keys are (endpoint, integer prompt ID, nonnegative integer repeat ID).
    """
    rows, ignored = [], 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                raw_id = row["id"]
                if isinstance(raw_id, bool) or not isinstance(raw_id, (int, str)):
                    raise ValueError("id must be an integer or integer string")
                pid = int(raw_id)
                if pid not in CANDIDATE_IDS:
                    ignored += 1
                    continue
                if not all(isinstance(row[k], str) for k in ("endpoint", "prompt", "response")):
                    raise ValueError("endpoint, prompt, response must be strings")
                if not row["endpoint"].strip() or not row["prompt"].strip():
                    raise ValueError("endpoint and prompt cannot be empty")
                if type(row["repeat"]) is not int or row["repeat"] < 0:
                    raise ValueError("repeat must be a nonnegative integer")
                rows.append({k: row[k] for k in ("endpoint", "repeat", "prompt", "response")} | {"id": pid})
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    counts = Counter(record_key(row) for row in rows)
    clean = [row for row in rows if counts[record_key(row)] == 1]
    prompts = {}
    for row in clean:
        pid = row["id"]
        if pid in prompts and prompts[pid] != row["prompt"]:
            raise ValueError(f"{path}: conflicting clean prompt texts for ID {pid}")
        prompts[pid] = row["prompt"]
    clean.sort(key=record_key)
    return {"rows": clean, "prompts": prompts,
            "audit": {"usable_rows": len(clean), "ignored_rows": ignored,
                      "discarded_ambiguous_rows": len(rows) - len(clean),
                      "ambiguous_keys": [list(key) for key in sorted(counts) if counts[key] > 1]}}


def record_key(row):
    return row["endpoint"], row["id"], row["repeat"]


def grouped_rows(data):
    groups = defaultdict(lambda: defaultdict(list))
    for row in data["rows"]:
        groups[row["endpoint"]][row["id"]].append(row)
    return {ep: dict(per) for ep, per in groups.items()}


def text_answers(groups):
    return {ep: {pid: [row["response"] for row in rows] for pid, rows in per.items()}
            for ep, per in groups.items()}


def normalize(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty/non-string response has no usable text features")
    text = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return " " + text + " "  # Boundary padding gives one-character replies char n-grams too.


def vectorizer_for(representation):
    if representation not in FEATURES:
        raise ValueError(f"unknown representation {representation!r}")
    options = {"analyzer": representation, "ngram_range": FEATURES[representation],
               "max_features": 4096, "lowercase": False, "sublinear_tf": True,
               "norm": "l2", "dtype": np.float64}
    if representation == "word":
        # Retain single letters, numbers and punctuation in the short/code probes.
        options["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
    return TfidfVectorizer(**options)


def centroid_matrix(answers, pid, vectorizer, endpoints):
    if any(not answers.get(ep, {}).get(pid) for ep in endpoints):
        raise ValueError(f"missing responses for prompt {pid}")
    centroids = []
    for ep in endpoints:
        matrix = vectorizer.transform([normalize(text) for text in answers[ep][pid]])
        if np.any(matrix.getnnz(axis=1) == 0):
            raise ValueError(f"{ep}/{pid}: OOV zero-vector response under frozen vocabulary")
        centroids.append(np.asarray(matrix.mean(axis=0)).ravel())
    return np.vstack(centroids)


def fit_metric(answers, prompt_ids, representation="word", projection=True):
    """Fit vocabulary/IDF and nuisance subspace on TRAINING RESPONSES ONLY.

    Five known same-checkpoint centroids define at most four centered nuisance
    directions. Orthogonal removal is a contraction; residuals are NOT normalized
    again (which would amplify near-zero residuals). No dense document matrix.
    """
    ids = tuple(sorted(prompt_ids))
    if not 1 <= len(ids) <= 4 or len(ids) != len(set(ids)) or not set(ids) <= CANDIDATE_IDS:
        raise ValueError("need one to four distinct allowed prompt IDs")
    if set(answers) != set(REFERENCES):
        raise ValueError("transform fitting requires exactly ref-01..ref-09")
    fitted = {}
    for pid in ids:
        if any(not answers[ep].get(pid) for ep in REFERENCES):
            raise ValueError(f"incomplete fitting coverage for prompt {pid}")
        vectorizer = vectorizer_for(representation)
        vectorizer.fit([normalize(text) for ep in REFERENCES for text in answers[ep][pid]])
        means = centroid_matrix(answers, pid, vectorizer, sorted(SAME))
        directions = means - means.mean(axis=0)
        _, singular, right = np.linalg.svd(directions, full_matrices=False)
        tolerance = max(directions.shape) * np.finfo(float).eps * max(float(singular[0]), 1.0)
        basis = right[singular > tolerance] if projection else np.empty((0, means.shape[1]))
        fitted[pid] = {"vectorizer": vectorizer, "basis": basis,
                       "nuisance_rank": int(np.count_nonzero(singular > tolerance))}
    return fitted


def similarity_vector(answers, fitted, endpoints, require_four=False):
    if require_four and any(len(answers.get(ep, {}).get(pid, ())) != REPLIES
                            for ep in endpoints for pid in fitted):
        raise ValueError("application requires exactly four responses per endpoint/selected prompt")
    matrices = []
    for pid, transform in fitted.items():
        means = centroid_matrix(answers, pid, transform["vectorizer"], endpoints)
        basis = transform["basis"]
        if basis.size:
            means -= (means @ basis.T) @ basis
        # Squared Euclidean distances between distribution means, not individual responses.
        squares = np.sum(means * means, axis=1)
        if np.any(squares <= 1e-24):
            raise ValueError(f"prompt {pid}: zero centroid after nuisance projection")
        distance = np.maximum(squares[:, None] + squares[None, :] - 2 * means @ means.T, 0)
        matrices.append(1 - np.clip(distance, 0, 2) / 2)
    combined = np.mean(matrices, axis=0)
    return {(a, b): float(combined[i, j]) for i, a in enumerate(endpoints)
            for j, b in enumerate(endpoints) if i < j}


def pair_similarities(answers, fitted):
    """Frozen metric on arbitrary endpoints; caller supplies exactly four replies/cell."""
    if len(answers) < 2:
        raise ValueError("at least two endpoints are required")
    return similarity_vector(answers, fitted, sorted(answers), require_four=True)


def identity_report(models, similarities=None):
    if set(models) != set(REFERENCES):
        raise ValueError("identity scoring requires all nine reference endpoints")
    counts = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
    errors = []
    for a, b in PAIRS:
        truth, predicted = a in SAME and b in SAME, models[a] == models[b]
        counts["tp" if truth and predicted else "fn" if truth else "fp" if predicted else "tn"] += 1
        if truth != predicted:
            item = {"endpoints": [a, b], "truth_same": truth, "predicted_same": predicted}
            if similarities is not None:
                item["similarity"] = similarities[a, b]
            errors.append(item)
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    raw = (tp * tn - fp * fn) / denominator if denominator else 0.0
    return {"clipped_mcc": max(0.0, raw), "raw_mcc": raw, "confusion": counts,
            "wrong_pairs": errors, "models": dict(models)}


def evaluate_vector(values, threshold):
    similarities = dict(zip(PAIRS, map(float, values)))
    return identity_report(cluster(REFERENCES, similarities, threshold), similarities)


def split_training(groups, prompt_ids, seed, split):
    training, validation, manifest = {}, {}, {}
    for ep in REFERENCES:
        training[ep], validation[ep], manifest[ep] = {}, {}, {}
        for pid in prompt_ids:
            rows = list(groups[ep][pid])
            # Stable per-cell seeds: adding/removing a candidate does not alter other splits.
            cell_seed = int(hashlib.sha256(f"{seed}:{split}:{ep}:{pid}".encode()).hexdigest(), 16)
            random.Random(cell_seed).shuffle(rows)
            left, right = rows[:REPLIES], rows[REPLIES:2 * REPLIES]
            if len(left) != REPLIES or len(right) != REPLIES:
                raise ValueError(f"{ep}/{pid}: need eight unambiguous responses")
            training[ep][pid] = [row["response"] for row in left]
            validation[ep][pid] = [row["response"] for row in right]
            manifest[ep][str(pid)] = {"training_repeats": [r["repeat"] for r in left],
                                      "validation_repeats": [r["repeat"] for r in right]}
    return training, validation, manifest


def summarize_reports(reports):
    scores = [report["clipped_mcc"] for report in reports]
    errors = Counter(tuple(item["endpoints"]) for report in reports for item in report["wrong_pairs"])
    return {"mean_clipped_mcc": float(np.mean(scores)), "minimum_clipped_mcc": min(scores),
            "std_clipped_mcc": float(np.std(scores)),
            "positive_fraction": sum(score > 0 for score in scores) / len(scores),
            "pair_error_counts": [{"endpoints": list(pair), "errors": errors[pair],
                                   "splits": len(scores)} for pair in PAIRS]}


def candidate_rank(candidate):
    val = candidate["validation"]
    return (val["mean_clipped_mcc"], val["positive_fraction"], val["minimum_clipped_mcc"],
            -val["std_clipped_mcc"], -len(candidate["prompt_ids"]),
            not candidate["projection"], candidate["representation"] == "word",
            candidate["threshold"], tuple(-pid for pid in candidate["prompt_ids"]))


def select_candidate(cache, prompt_ids, representation, projection):
    train = np.mean([cache[pid]["training"] for pid in prompt_ids], axis=0)
    val = np.mean([cache[pid]["validation"] for pid in prompt_ids], axis=0)
    best = None
    for threshold in THRESHOLDS:
        reports = [evaluate_vector(values, threshold) for values in val]
        summary = summarize_reports(reports)
        candidate = {"prompt_ids": list(prompt_ids), "representation": representation,
                     "projection": projection, "threshold": threshold, "validation": summary}
        if best is None or candidate_rank(candidate) > candidate_rank(best):
            best = candidate
    best["training"] = summarize_reports([evaluate_vector(values, best["threshold"]) for values in train])
    best["accepted"] = (best["validation"]["mean_clipped_mcc"] >= MIN_INNER_MCC and
                        best["validation"]["positive_fraction"] >= MIN_POSITIVE_FRACTION)
    return best


def select_configuration(data, splits=12, seed=2666):
    """Four prespecified feature variants; singleton ranking, top-four pairs/top-K.

    Selection sees only repeated INNER validation. Those observations are tuning
    data, may recur across repetitions, and must never be advertised as holdout.
    Each repetition has disjoint 4+4 original repeat keys in every cell.
    """
    if splits < 2:
        raise ValueError("at least two repeated splits are required")
    groups = grouped_rows(data)
    if set(groups) != set(REFERENCES):
        raise ValueError("training must contain exactly ref-01..ref-09")
    coverage = {str(pid): {ep: len(groups[ep].get(pid, ())) for ep in REFERENCES}
                for pid in sorted(CANDIDATE_IDS)}
    usable = [pid for pid in sorted(CANDIDATE_IDS)
              if min(coverage[str(pid)].values()) >= 2 * REPLIES]
    if not usable:
        raise ValueError("no candidate prompt has eight clean responses at every reference")
    split_data = [split_training(groups, usable, seed, split) for split in range(splits)]
    cache, leaderboard, invalid = {}, [], {}
    for representation in FEATURES:
        for projection in (False, True):
            key = representation, projection
            cache[key] = {pid: {"training": [], "validation": []} for pid in usable}
        for split, (training, validation, _) in enumerate(split_data):
            for pid in usable:
                if all((representation, projection, pid) in invalid for projection in (False, True)):
                    continue
                # A failed feature is explicitly rejected, never treated as similarity evidence.
                try:
                    fitted = fit_metric(training, [pid], representation, True)
                except ValueError as error:
                    for projection in (False, True):
                        invalid[representation, projection, pid] = {"representation": representation,
                            "projection": projection, "prompt_id": pid, "split": split, "reason": str(error)}
                    continue
                for projection in (True, False):
                    if (representation, projection, pid) in invalid:
                        continue
                    view = fitted if projection else {pid: dict(fitted[pid], basis=np.empty((0, 0)))}
                    try:
                        for name, answers in (("training", training), ("validation", validation)):
                            sims = similarity_vector(answers, view, REFERENCES)
                            cache[representation, projection][pid][name].append([sims[pair] for pair in PAIRS])
                    except ValueError as error:
                        invalid[representation, projection, pid] = {"representation": representation,
                            "projection": projection, "prompt_id": pid, "split": split, "reason": str(error)}
        for projection in (False, True):
            variant = cache[representation, projection]
            singletons = [select_candidate(variant, [pid], representation, projection) for pid in usable
                          if (representation, projection, pid) not in invalid]
            leaderboard.extend(singletons)
            top = [item["prompt_ids"][0] for item in sorted(singletons, key=candidate_rank, reverse=True)[:4]]
            sets = {tuple(sorted(pair)) for pair in combinations(top, 2)}
            sets.update(tuple(sorted(top[:k])) for k in (3, 4) if len(top) >= k)
            leaderboard.extend(select_candidate(variant, ids, representation, projection) for ids in sorted(sets))
    if not leaderboard:
        raise ValueError(f"all feature/prompt candidates rejected: {list(invalid.values())}")
    accepted = [item for item in leaderboard if item["accepted"]]
    winner = max(accepted or leaderboard, key=candidate_rank)
    selected_cache = cache[winner["representation"], winner["projection"]]
    detailed = {}
    for name in ("training", "validation"):
        values = np.mean([selected_cache[pid][name] for pid in winner["prompt_ids"]], axis=0)
        detailed[name] = [evaluate_vector(row, winner["threshold"]) for row in values]
    manifests = [{ep: {str(pid): manifest[ep][str(pid)] for pid in winner["prompt_ids"]}
                  for ep in REFERENCES} for _, _, manifest in split_data]
    return winner, {"coverage": coverage, "excluded_insufficient_coverage": sorted(set(CANDIDATE_IDS) - set(usable)),
                    "leaderboard": sorted(leaderboard, key=candidate_rank, reverse=True),
                    "rejected_feature_prompts": list(invalid.values()),
                    "selected_split_reports": detailed, "selected_split_repeats": manifests,
                    "split_count": splits, "seed": seed,
                    "selection_rule": "validation mean clipped MCC, positive fraction, minimum, lower std; fewer prompts, unprojected, word, higher threshold, lower IDs on ties",
                    "search": "word(1,2)/char(2,4), projected/unprojected; all singletons, pairs among top four singletons, top-three/top-four sets; fixed coarse and near-one thresholds",
                    "acceptance": {"minimum_mean_inner_clipped_mcc": MIN_INNER_MCC,
                                   "minimum_positive_inner_fraction": MIN_POSITIVE_FRACTION},
                    "warning": "All inner validation is hyperparameter tuning, NOT independent evidence; repeated splits overlap across repetitions."}


def training_answers(data, ids):
    groups = grouped_rows(data)
    if set(groups) != set(REFERENCES):
        raise ValueError("immutable training file must contain exactly the nine references")
    return text_answers({ep: {pid: groups[ep][pid] for pid in ids} for ep in REFERENCES})


def load_frozen(config_path, train_path=None):
    """Check config/code/runtime/training hashes and deterministically refit; no selection."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    supplied_hash = config.get("config_sha256")
    unsigned = {key: value for key, value in config.items() if key != "config_sha256"}
    if supplied_hash != canonical_hash(unsigned):
        raise ValueError("frozen config integrity mismatch")
    if config["schema"] != SCHEMA or config["code_sha256"] != code_hashes():
        raise ValueError("analyzer/dependency source changed since freezing")
    if config["versions"] != versions():
        raise ValueError("Python/numpy/scipy/sklearn versions changed since freezing")
    if not config["selected"]["accepted"]:
        raise ValueError("inner validation rejected this metric; cannot apply it as accepted")
    path = Path(train_path or config["training_file"]["path"])
    if sha256(path) != config["training_file"]["sha256"]:
        raise ValueError("immutable training file SHA256 mismatch")
    data = load_responses(path)
    selected = config["selected"]
    ids = selected["prompt_ids"]
    if any(data["prompts"].get(pid) != text for pid, text in
           ((int(p["id"]), p["prompt"]) for p in config["prompts"])):
        raise ValueError("frozen prompt texts do not match immutable training")
    fitted = fit_metric(training_answers(data, ids), ids, selected["representation"], selected["projection"])
    if transform_signature(fitted) != config["transform_signature"]:
        raise ValueError("deterministic transform refit mismatch")
    if sha256(path) != config["training_file"]["sha256"] or code_hashes() != config["code_sha256"]:
        raise ValueError("training/source changed during deterministic refit")
    return config, fitted


def transform_signature(fitted):
    # Hash numerical transforms as well as sources; tolerate no silent refit drift.
    result = {}
    for pid, transform in fitted.items():
        vectorizer = transform["vectorizer"]
        result[str(pid)] = {"vocabulary_sha256": canonical_hash({term: int(index) for term, index in vectorizer.vocabulary_.items()}),
                            "idf_sha256": hashlib.sha256(vectorizer.idf_.tobytes()).hexdigest(),
                            "basis_sha256": hashlib.sha256(transform["basis"].tobytes()).hexdigest(),
                            "nuisance_rank": transform["nuisance_rank"]}
    return result


def validate_external(config, fitted, data):
    groups = grouped_rows(data)
    if set(groups) != set(REFERENCES):
        raise ValueError("external CLI validation requires exactly ref-01..ref-09; use pair_similarities API for ep-* answers")
    old_keys = {tuple(key) for key in config["training_keys"]}
    selected_ids = config["selected"]["prompt_ids"]
    reused = old_keys & {record_key(row) for row in data["rows"] if row["id"] in selected_ids}
    if reused:
        raise ValueError(f"external validation reuses {len(reused)} training repeat keys; collect with globally fresh repeat IDs")
    chosen, manifest, unused = {}, {}, {}
    for ep in REFERENCES:
        chosen[ep], manifest[ep], unused[ep] = {}, {}, {}
        for prompt in config["prompts"]:
            pid = prompt["id"]
            if data["prompts"].get(pid) != prompt["prompt"]:
                raise ValueError(f"external prompt text mismatch for ID {pid}")
            rows = groups[ep].get(pid, ())
            if len(rows) < REPLIES:
                raise ValueError(f"{ep}/{pid}: need four clean NEW responses, found {len(rows)}")
            chosen[ep][pid] = list(rows[:REPLIES])
            manifest[ep][str(pid)] = [row["repeat"] for row in rows[:REPLIES]]
            unused[ep][str(pid)] = len(rows) - REPLIES
    similarities = pair_similarities(text_answers(chosen), fitted)
    models = cluster(REFERENCES, similarities, config["selected"]["threshold"])
    return identity_report(models, similarities) | {"used_repeats": manifest, "unused_counts": unused}


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def self_check():
    """Small runnable regression, intentionally not run by the implementation worker."""
    rows = [{"endpoint": "ref-01", "id": 0, "repeat": i,
             "prompt": "test", "response": str(i)} for i in range(3)]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "answers.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in rows + [rows[0]]), encoding="utf-8")
        clean = load_responses(path)
        assert [row["repeat"] for row in clean["rows"]] == [1, 2]
        assert clean["audit"]["discarded_ambiguous_rows"] == 2
    answers = {ep: {0: ["common " + ep] * REPLIES} for ep in REFERENCES}
    fitted = fit_metric(answers, [0], "word", True)
    sims = pair_similarities(answers, fitted)
    assert all(abs(sims[a, b] - 1) < 1e-12 for a, b in combinations(sorted(SAME), 2))
    truth = {ep: "same" if ep in SAME else ep for ep in REFERENCES}
    assert identity_report(truth)["clipped_mcc"] == 1
    assert identity_report({ep: ep for ep in REFERENCES})["clipped_mcc"] == 0
    for bad_text in ("", "zzzzunseen"):
        bad = {ep: {0: list(per[0])} for ep, per in answers.items()}
        bad["ref-01"][0] = [bad_text] * REPLIES
        try:
            pair_similarities(bad, fitted)
        except ValueError:
            pass
        else:
            raise AssertionError("empty/OOV text was accepted as similarity evidence")
    print("self-check passed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, help="immutable JSONL; frozen mode defaults to saved path")
    parser.add_argument("--output", type=Path, help="selected frozen config JSON, or external validation report JSON")
    parser.add_argument("--frozen", type=Path, help="apply this config unchanged; never select using validation")
    parser.add_argument("--validation", type=Path, help="NEW reference responses; repeat IDs must not overlap training")
    parser.add_argument("--csv", type=Path, help="CSV output (default: output.inner.csv for tuning, output.csv for holdout)")
    parser.add_argument("--splits", type=int, default=None, help="selection only, default 12 repeated disjoint 4+4 splits")
    parser.add_argument("--seed", type=int, default=None, help="selection only, default 2666")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if args.self_check:
        self_check()
        return
    if args.output is None or (not args.frozen and not args.train):
        parser.error("need --output and either --train (select) or --frozen (apply)")
    if bool(args.frozen) != bool(args.validation):
        parser.error("--frozen and --validation must be supplied together; validation cannot enter selection")
    if args.frozen and (args.splits is not None or args.seed is not None):
        parser.error("--splits/--seed cannot change a frozen configuration")
    csv_path = args.csv or args.output.with_suffix(".csv" if args.frozen else ".inner.csv")
    inputs = {p.resolve() for p in (args.train, args.frozen, args.validation) if p is not None}
    inputs.update((ROOT / name).resolve() for name in code_hashes())
    if args.frozen and args.train is None:
        try:
            inputs.add(Path(json.loads(args.frozen.read_text())["training_file"]["path"]).resolve())
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"cannot read frozen training provenance: {exc}")
    if args.output.resolve() in inputs or csv_path.resolve() in inputs or args.output.resolve() == csv_path.resolve():
        parser.error("output JSON/CSV must be distinct and must not overwrite inputs or source")
    try:
        initial_code = code_hashes()
        if args.frozen:
            config, fitted = load_frozen(args.frozen, args.train)
            validation_hash = sha256(args.validation)
            if validation_hash == config["training_file"]["sha256"]:
                raise ValueError("validation is the training file")
            data = load_responses(args.validation)
            report = validate_external(config, fitted, data)
            result = {"mode": "external_validation_no_selection", "frozen_config_sha256": config["config_sha256"],
                      "selected": config["selected"], "prompts": config["prompts"],
                      "validation_sha256": validation_hash, "validation_audit": data["audit"],
                      "report": report, "csv": str(csv_path.resolve()),
                      "warning": "One independent evaluation only; do not tune this config against this holdout."}
            if sha256(args.validation) != validation_hash:
                raise ValueError("validation file changed during evaluation")
            training_path = Path(args.train or config["training_file"]["path"])
            if sha256(training_path) != config["training_file"]["sha256"]:
                raise ValueError("training file changed during evaluation")
            models = report["models"]
        else:
            training_hash = sha256(args.train)
            data = load_responses(args.train)
            selected, selection = select_configuration(data, args.splits if args.splits is not None else 12,
                                                        args.seed if args.seed is not None else 2666)
            ids = selected["prompt_ids"]
            fitted = fit_metric(training_answers(data, ids), ids, selected["representation"], selected["projection"])
            groups = grouped_rows(data)
            result = {"schema": SCHEMA, "mode": "frozen_experimental_metric", "selected": selected,
                      "prompts": [{"id": pid, "prompt": data["prompts"][pid]} for pid in ids],
                      "training_file": {"path": str(args.train.resolve()), "sha256": training_hash},
                      "code_sha256": initial_code, "versions": versions(), "training_audit": data["audit"],
                      "training_keys": [list(record_key(row)) for row in data["rows"] if row["id"] in ids],
                      "next_fresh_repeat": {ep: {str(pid): max(row["repeat"] for row in groups[ep][pid]) + 1
                                                  for pid in ids} for ep in REFERENCES},
                      "fit_protocol": "Frozen deployment transform refit on ALL clean immutable training responses for selected prompts; external centroids use four NEW responses per cell. Inner transforms use only four training responses.",
                      "metric": {"similarity": "mean_prompt(1 - squared_L2(projected_TFIDF_centroids)/2)",
                                 "projection": "all numerically nonzero centered same-checkpoint centroid directions",
                                 "max_features_per_prompt": 4096, "replies_per_prompt": REPLIES,
                                 "maximum_deployment_replies_per_endpoint": REPLIES * len(ids)},
                      "transform_signature": transform_signature(fitted), "selection": selection,
                      "inner_csv": str(csv_path.resolve()), "endpoint_queries": 0, "public_feedback_used": False}
            if sha256(args.train) != training_hash:
                raise ValueError("training file changed during selection")
            models = selection["selected_split_reports"]["validation"][0]["models"]
            result["config_sha256"] = canonical_hash(result)
        if code_hashes() != initial_code:
            raise ValueError("analyzer source changed during execution")
        # All nine references share one published family; no family inference here.
        write_candidate(csv_path, models, {ep: "reference" for ep in REFERENCES})
        save_json(args.output, result)
    except (OSError, ValueError, KeyError, TypeError, np.linalg.LinAlgError) as exc:
        parser.exit(2, f"Experiment failed: {exc}\n")
    if args.frozen:
        print(f"External frozen MCC: {report['clipped_mcc']:.6f}; confusion={report['confusion']}")
        for error in report["wrong_pairs"]:
            print(json.dumps(error, sort_keys=True))
    else:
        print(f"Selected prompts={ids}, representation={selected['representation']}, projection={selected['projection']}, threshold={selected['threshold']:.17g}")
        print(f"Inner tuning MCC mean={selected['validation']['mean_clipped_mcc']:.6f}; accepted={selected['accepted']}; NOT an independent holdout")
        print("Every split's identity-pair errors and MCC, plus aggregate pair error counts, are in the JSON.")
    print(f"Saved {args.output} and {csv_path}; run score_practice.py explicitly on the CSV.")


if __name__ == "__main__":
    main()

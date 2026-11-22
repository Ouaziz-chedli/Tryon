#!/usr/bin/env python3
"""Compare canonical answer values; calibrate with validate_references.py first."""
import argparse
import ast
import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal
from itertools import combinations
from pathlib import Path

from analyze import partition_rows

# Probe 111 was sent as only "Which is heavier" by the old malformed CSV.
# Its historical responses cannot answer the intended question; never reuse this ID.
PROMPT_IDS = set(range(100, 113)) - {111}
INVALID = "<unparsed>"
ROOT = Path(__file__).resolve().parent


def normalize(text):
    return unicodedata.normalize("NFKC", text).strip().casefold()


def code_value(text):
    """Parse generated code without execution or discarding conditional branches."""
    chunks = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", text, re.S) + [text]
    for chunk in chunks:
        try:
            tree = ast.parse(chunk.strip())
        except (SyntaxError, ValueError):
            continue
        function = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "double"), None)
        parameter, body = "x", None
        if function is not None:
            if len(function.args.args) != 1 or function.args.vararg or function.args.kwarg or function.args.kwonlyargs:
                return "code:" + ast.dump(function, include_attributes=False)
            parameter = function.args.args[0].arg
            body = function.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                body = body[1:]
        else:
            value = next((n.value for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "double" for t in n.targets)
                          and isinstance(n.value, ast.Lambda)), None)
            if value is not None and len(value.args.args) == 1:
                parameter, body = value.args.args[0].arg, [ast.Return(value=value.body)]
            elif len(tree.body) == 1 and isinstance(tree.body[0], ast.Return):
                body = tree.body
        if not body:
            continue
        for statement in body:
            for child in ast.walk(statement):
                if isinstance(child, ast.Name) and child.id == parameter:
                    child.id = "x"
        if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
            return "code:body:" + ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)
        value = body[0].value
        if isinstance(value, ast.BinOp):
            left, right = value.left, value.right
            is_x = lambda n: isinstance(n, ast.Name) and n.id == "x"
            is_two = lambda n: isinstance(n, ast.Constant) and n.value == 2
            if (isinstance(value.op, ast.Mult) and
                    ((is_x(left) and is_two(right)) or (is_two(left) and is_x(right)))):
                return "code:x*2"
            if isinstance(value.op, ast.Add) and is_x(left) and is_x(right):
                return "code:x*2"
        return "code:" + ast.dump(value, include_attributes=False)
    # Accept a prose-wrapped single return, not a fragment of a conditional function.
    returns = re.findall(r"\breturn\s+([^\n;`]+)", text)
    if len(returns) == 1 and not re.search(r"\b(if|while|else|elif)\b", text):
        try:
            value = ast.parse(returns[0].strip(), mode="eval").body
        except (SyntaxError, ValueError):
            return None
        return code_value("def double(x):\n    return " + ast.unparse(value))
    return None


def single_word(text):
    match = re.fullmatch(r"[\s\"'`*]*([a-z]+)[\s\"'`*.!]*", text)
    return match.group(1) if match else None


def parse_value(pid, text):
    t = normalize(text)
    if pid in (101, 102, 109, 110):
        number = r"[+-]?\d+(?:[.,]\d+)*"
        explicit = re.findall(r"(?:=|\b(?:answer|result|equals|is)\s*[:=]?)\s*(" + number + r")", t)
        values = explicit or re.findall(number, t)
        if not values:
            return None
        token = values[-1] if explicit else values[0]
        if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+", token):
            token = token.replace(",", "")
        else:
            token = token.replace(",", ".")
        try:
            value = Decimal(token)
        except ArithmeticError:
            return None
        return format(value.normalize(), "f") if value else "0"
    if pid in (100, 105):
        match = re.search(r"\b(yes|no)\b", t)
        return match.group(1) if match else None
    if pid == 103:
        letters = re.findall(r"\b([a-z])\b", t)
        return letters[-1] if letters else None
    if pid == 104:
        match = re.search(r"\b(decrease|reduce|decrement|reduction|decline|less|down|lower)\b", t)
        return match.group(1) if match else single_word(t)
    if pid == 106:
        match = re.search(r"\bsalt\s+and\s*[:—-]?\s*([a-z]+)\b", t)
        if not match:
            match = re.search(r":\s*([a-z]+)\b", t)
        if not match:
            match = re.match(r"[\s\"'`*]*([a-z]+)\b", t)
        word = match.group(1) if match else None
        return word if word not in {"sure", "the", "a", "an", "here", "sorry", "i", "please", "complete", "salt"} else None
    if pid == 107:
        match = re.search(r"\b([a-z]+\d+[a-z]*\d*)\b", t)
        return match.group(1) if match else single_word(t)
    if pid == 108:
        match = re.search(r"\b(thanks|thank\s*you)\b", t)
        return "thanks" if match else single_word(t)
    if pid == 112:
        return code_value(text)
    return None


def load_answers(path):
    answers = defaultdict(lambda: defaultdict(list))
    prompts, seen = {}, set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                pid = int(row["id"])
                if pid not in PROMPT_IDS:
                    continue
                ep, prompt, response = row["endpoint"], row["prompt"], row["response"]
                repeat = row["repeat"]
                if not isinstance(ep, str) or not isinstance(prompt, str) or not isinstance(response, str):
                    raise ValueError("endpoint, prompt and response must be strings")
                if not isinstance(repeat, int) or repeat < 0:
                    raise ValueError("repeat must be a non-negative integer")
                key = (ep, pid, repeat)
                if key in seen:
                    raise ValueError(f"duplicate answer {key}")
                if pid in prompts and prompts[pid] != prompt:
                    raise ValueError(f"prompt id {pid} has conflicting texts")
                seen.add(key)
                prompts[pid] = prompt
                answers[ep][pid].append(response)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    if not answers:
        raise ValueError(f"{path}: no responses for usable prompt IDs {sorted(PROMPT_IDS)}")
    return {ep: dict(per_prompt) for ep, per_prompt in answers.items()}


def pair_similarities(answers, mode_weight=0.5):
    if not 0 <= mode_weight <= 1:
        raise ValueError("mode_weight must be between 0 and 1")
    counts = {ep: {pid: Counter(parse_value(pid, text) or INVALID for text in texts)
                   for pid, texts in per_prompt.items() if texts}
              for ep, per_prompt in answers.items()}
    sims = {}
    for a, b in combinations(sorted(counts), 2):
        values = []
        for pid in sorted(counts[a].keys() & counts[b].keys()):
            ha, hb = counts[a][pid], counts[b][pid]
            na, nb = ha.total(), hb.total()
            # Two parsing failures are not positive evidence of model equality.
            valid = (ha.keys() | hb.keys()) - {INVALID}
            overlap = sum(min(ha[v] / na, hb[v] / nb) for v in valid)
            ma = {v for v in valid if ha[v] == max(ha.values())}
            mb = {v for v in valid if hb[v] == max(hb.values())}
            mode = len(ma & mb) / len(ma | mb) if ma | mb else 0.0
            values.append(mode_weight * mode + (1 - mode_weight) * overlap)
        sims[a, b] = sum(values) / len(values) if values else 0.0
    return sims


def cluster(endpoints, sims, threshold):
    """Complete linkage: every cross-pair must pass; no single-linkage bridges."""
    groups = [(ep,) for ep in sorted(endpoints)]
    while len(groups) > 1:
        best = None
        for i, left in enumerate(groups):
            for j in range(i + 1, len(groups)):
                right = groups[j]
                score = min(sims[min(a, b), max(a, b)] for a in left for b in right)
                if score >= threshold and (best is None or score > best[0]):
                    best = (score, i, j)
        if best is None:
            break
        _, i, j = best
        groups[i] = tuple(sorted(groups[i] + groups[j]))
        groups.pop(j)
    return {ep: group[0] for group in groups for ep in group}


def write_candidate(path, models, families):
    rows = partition_rows(models, families)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["endpoint", "family", "model"])
        writer.writerows(rows)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    labels = {row["endpoint"]: (row["family"], row["model"]) for row in saved}
    if len(saved) != len(models) or set(labels) != set(models):
        raise ValueError("CSV endpoint coverage does not match the analyzed partition")
    for a, b in combinations(sorted(models), 2):
        if ((labels[a] == labels[b]) != (models[a] == models[b]) or
                (labels[a][0] == labels[b][0]) != (families[a] == families[b])):
            raise ValueError(f"CSV changed the predicted relationship for {a}, {b}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, default=ROOT / "answers.jsonl")
    parser.add_argument("--calibration", type=Path, default=ROOT / "calibration.json")
    parser.add_argument("--families", type=Path, required=True,
                        help="CSV family assignments to preserve independently of model calibration")
    parser.add_argument("--output", type=Path, default=ROOT / "submission.csv")
    parser.add_argument("--bootstrap", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2666)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error("--bootstrap must be positive")
    inputs = {p.resolve() for p in (args.answers, args.calibration, args.families)}
    if args.output.resolve() in inputs or args.output.with_suffix(".stability.json").resolve() in inputs:
        parser.error("output paths must not overwrite the answers, calibration or family inputs")
    answers = load_answers(args.answers)
    expected = {f"ep-{i:02d}" for i in range(1, 34)}
    if set(answers) != expected or any(set(per) != PROMPT_IDS for per in answers.values()):
        parser.error(f"need all ep-01..ep-33 with usable prompts {sorted(PROMPT_IDS)} before submission")
    with args.families.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    families = {row["endpoint"]: row["family"] for row in rows}
    if len(rows) != len(expected) or set(families) != expected or not all(families.values()):
        parser.error("--families must contain one non-empty family for each of the 33 endpoints")
    calibration = json.loads(args.calibration.read_text())
    if not calibration["held_out"]["official_csv_score"]["score"] > 0:
        parser.error("held-out reference MCC is zero: calibration rejected; do not generate or submit its clustering")
    selected = calibration["selected"]
    weight, threshold = selected["mode_weight"], selected["threshold"]
    if not 0 <= weight <= 1 or not 0 < threshold <= 1:
        parser.error("calibration parameters must be in [0, 1], with a positive threshold")
    for name, path in (("analyzer", Path(__file__)), ("partition_exporter", ROOT / "analyze.py")):
        if hashlib.sha256(path.read_bytes()).hexdigest() != calibration["provenance"]["files"][name]["sha256"]:
            parser.error(f"{name} changed since calibration; rerun validate_references.py")
    sims = pair_similarities(answers, weight)
    restricted = {pair: value if families[pair[0]] == families[pair[1]] else -1.0
                  for pair, value in sims.items()}
    models = cluster(answers, restricted, threshold)
    rng, stability = random.Random(args.seed), Counter()
    for _ in range(args.bootstrap):
        sampled = {ep: {pid: rng.choices(texts, k=len(texts)) for pid, texts in per.items()}
                   for ep, per in answers.items()}
        boot_sims = pair_similarities(sampled, weight)
        boot_sims = {pair: value if families[pair[0]] == families[pair[1]] else -1.0
                     for pair, value in boot_sims.items()}
        boot_models = cluster(sampled, boot_sims, threshold)
        stability.update(pair for pair in boot_sims if boot_models[pair[0]] == boot_models[pair[1]])
    written = write_candidate(args.output, models, families)
    groups = defaultdict(list)
    for ep, family, model in written:
        groups[family, model].append(ep)
    print(f"{len(answers)} endpoints; {len(set(families.values()))} preserved families; {len(groups)} models")
    print(f"Calibration only: mode_weight={weight}, threshold={threshold}; CSV reread and all 528 pairs checked")
    for (family, model), members in sorted(groups.items()):
        pairs = list(combinations(members, 2))
        rate = min((stability[pair] / args.bootstrap for pair in pairs), default=1.0)
        print(f"{family}/{model}: {','.join(members)}; minimum resampled co-clustering={rate:.2f}")
    report = {"answers_sha256": hashlib.sha256(args.answers.read_bytes()).hexdigest(),
              "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
              "mode_weight": weight, "threshold": threshold, "bootstrap": args.bootstrap,
              "families_preserved_from": str(args.families),
              "pair_stability": {f"{a},{b}": stability[a, b] / args.bootstrap for a, b in sims}}
    args.output.with_suffix(".stability.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}; resampling stability is not a probability of true model identity.")


if __name__ == "__main__":
    main()

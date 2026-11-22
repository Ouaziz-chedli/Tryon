#!/usr/bin/env python3
"""Analyze reference-only family self-identification; never make API requests.

Usage: python experiment_identity.py --output identity_report.json
Optional: --heldout reference_identity_heldout.jsonl --heldout-repeats 6
Runnable parser check for the integrator: python experiment_identity.py --self-check
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parent
REFERENCES = tuple(f"ref-{i:02d}" for i in range(1, 10))
S_REFERENCES = ("ref-01", "ref-03", "ref-05", "ref-06", "ref-08")
# Family aliases, not checkpoint identity. Providers alone are deliberately absent.
ALIASES = {
    "SmolLM": r"\bsmol[\s_-]*lm(?:[\s_-]*[123])?\b",
    "Llama": r"\b(?:meta[\s/-]*)?llama(?:[\s_-]*[234])?\b",
    "Qwen": r"\bqwen(?:[\s_-]*[123](?:\.\d)?)?\b",
    "Mistral": r"\b(?:mistral|mixtral)\b",
    "Gemma": r"\bgemma(?:[\s_-]*[123])?\b",
    "GPT": r"\b(?:chatgpt|gpt(?:[\s_-]*[2345](?:\.\d)?)?)\b",
    "Claude": r"\bclaude\b",
    "Phi": r"\bphi(?:[\s_-]*[1234])\b",
    "DeepSeek": r"\bdeepseek\b",
    "BLOOM": r"\bbloom(?:z)?\b",
    "Falcon": r"\bfalcon\b",
    "Pythia": r"\bpythia\b",
    "OLMo": r"\bolmo(?:[\s_-]*[23])?\b",
}
UNCERTAIN = re.compile(r"\b(?:unknown|not sure|cannot (?:know|identify|determine|disclose)|can't (?:know|identify|determine|disclose)|do not know|don't know|might|may be|possibly|perhaps|for example|such as)\b", re.I)
NEGATION = re.compile(r"\b(?:not|never|rather than|unlike)\b", re.I)
ATTRIBUTION = re.compile(r"\b(?:i am|i'm|i’m|i was|i belong|i come|i originate|i (?:am |was )?(?:originally )?(?:pretrained|trained|based|built)|my (?:original |underlying |pretrained )?(?:model|family)|(?:original |underlying |pretrained )?(?:language[- ]model |model )?family|this assistant (?:is|was)|the (?:original |underlying |pretrained )?(?:language[- ]model|model)(?: behind this assistant)? is)\b", re.I)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identify(text):
    """Conservative explicit self-claims only; all rejected mentions stay auditable."""
    mentions, claims = [], []
    for family, pattern in ALIASES.items():
        for match in re.finditer(pattern, text, re.I):
            mentions.append({"family": family, "text": match.group(), "span": list(match.span())})
            # ponytail: sentence-local rules abstain on complex prose; use manual review,
            # not fuzzy matching, if these conservative rules lose useful coverage.
            start = max(text.rfind("\n", 0, match.start()), text.rfind("!", 0, match.start()),
                        text.rfind("?", 0, match.start()), text.rfind(". ", 0, match.start())) + 1
            stop = len(text)
            for delimiter in ("\n", "!", "?", ". "):
                end = text.find(delimiter, match.end())
                if end >= 0:
                    stop = min(stop, end)
            sentence = text[start:stop].strip()
            before = text[start:match.start()]
            bare = bool(re.fullmatch(r"[\s*`\"']*" + pattern + r"[\s*`\"'.!]*", text, re.I))
            if not UNCERTAIN.search(sentence) and not NEGATION.search(before) and (bare or ATTRIBUTION.search(before)):
                claims.append({"family": family, "evidence": sentence, "span": list(match.span())})
    families = sorted({claim["family"] for claim in claims})
    return {"status": "IDENTIFIED" if len(families) == 1 else "CONTRADICTORY" if families else "UNKNOWN",
            "family": families[0] if len(families) == 1 else None,
            "claimed_families": families, "claims": claims, "mentions": mentions}


def load_records(path, prompts, ids):
    rows, ignored, invalid = [], 0, []
    counts, selected_lines = Counter(), 0
    for line, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if str(row.get("id")) not in ids:
            ignored += 1
            continue
        pid, endpoint, repeat = str(row.get("id")), row.get("endpoint"), row.get("repeat")
        selected_lines += 1
        if endpoint in REFERENCES and type(repeat) is int:
            counts[(endpoint, pid, repeat)] += 1
        if (endpoint not in REFERENCES or type(repeat) is not int or repeat < 0
                or not isinstance(row.get("response"), str) or not row["response"].strip()
                or row.get("prompt") != prompts[pid]):
            invalid.append({"line": line, "reason": "invalid endpoint/repeat/text or prompt mismatch"})
            continue
        rows.append({**row, "id": pid, "source_line": line})
    duplicates = [{"key": list(key), "count": count} for key, count in sorted(counts.items()) if count > 1]
    usable = [r for r in rows if counts[(r["endpoint"], r["id"], r["repeat"])] == 1]
    return usable, {"path": str(path.resolve()), "sha256": sha256(path), "selected_rows_before_deduplication": selected_lines,
                    "usable_rows": len(usable), "ignored_other_ids": ignored, "invalid_rows": invalid,
                    "duplicate_keys_excluded_entirely": duplicates}


def summarize(rows, pid, repeats):
    chosen = [r for r in rows if r["id"] == pid]
    observed = {(r["endpoint"], r["repeat"]) for r in chosen}
    expected = {(ep, repeat) for ep in REFERENCES for repeat in repeats}
    per_reference = {}
    evidence = []
    for ep in REFERENCES:
        subset = [r for r in chosen if r["endpoint"] == ep]
        classified = [identify(r["response"]) for r in subset]
        families = sorted({family for item in classified for family in item["claimed_families"]})
        per_reference[ep] = {"responses": len(subset), "identified": sum(c["status"] == "IDENTIFIED" for c in classified),
                             "unknown": sum(c["status"] == "UNKNOWN" for c in classified),
                             "contradictory_responses": sum(c["status"] == "CONTRADICTORY" for c in classified),
                             "contradictory_across_repeats": len(families) > 1, "claimed_families": families,
                             "repeat_keys": sorted(r["repeat"] for r in subset)}
        for row, classified_row in zip(subset, classified):
            evidence.append({"key": [ep, pid, row["repeat"]], "source_line": row["source_line"],
                             "response": row["response"], **classified_row})
    claimed = sorted({family for ep in per_reference.values() for family in ep["claimed_families"]})
    all_identified = bool(expected) and observed == expected and all(e["status"] == "IDENTIFIED" for e in evidence)
    return {"prompt_id": pid, "expected_responses": len(expected), "usable_responses": len(chosen),
            "missing_keys": [[ep, pid, repeat] for ep, repeat in sorted(expected - observed)],
            "unexpected_keys": [[ep, pid, repeat] for ep, repeat in sorted(observed - expected)],
            "per_reference": per_reference, "evidence": evidence,
            "identified": sum(e["status"] == "IDENTIFIED" for e in evidence),
            "unknown": sum(e["status"] == "UNKNOWN" for e in evidence),
            "contradictory_responses": sum(e["status"] == "CONTRADICTORY" for e in evidence),
            "claimed_families": claimed, "observed_false_family_split": len(claimed) > 1,
            "S_nuisance_claimed_families": {ep: per_reference[ep]["claimed_families"] for ep in S_REFERENCES},
            "passes_screen_rule": all_identified and len(claimed) == 1}


def self_check():
    assert identify("SmolLM2")['family'] == 'SmolLM'
    assert identify("I am a language model named SmolLM, trained by Hugging Face.")['family'] == 'SmolLM'
    assert identify("I was originally pretrained as Qwen2.5.")['family'] == 'Qwen'
    for text in ("UNKNOWN", "I am made by Hugging Face", "I am not SmolLM.", "I might be Llama.",
                 "For example, SmolLM or Llama.", "I am an assistant. Some models are called Llama."):
        assert identify(text)['status'] == 'UNKNOWN', text
    assert identify("I am SmolLM. I am Qwen.")['status'] == 'CONTRADICTORY'
    print("Identity parser self-check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, default=BASE / "identity_probes.csv")
    parser.add_argument("--screen", type=Path, default=BASE / "reference_identity_screen.jsonl")
    parser.add_argument("--heldout", type=Path)
    parser.add_argument("--heldout-repeats", type=int, default=6)
    parser.add_argument("--baseline", type=Path, default=BASE / "reference_answers.20261122T025452Z.jsonl")
    parser.add_argument("--output", type=Path, default=BASE / "identity_report.json")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.heldout_repeats < 1:
        parser.error("--heldout-repeats must be positive")
    with args.prompts.open(encoding="utf-8-sig", newline="") as handle:
        prompt_rows = list(csv.DictReader(handle))
    prompts = {r["id"]: r["prompt"] for r in prompt_rows}
    if set(prompts) != {"400", "401"} or len(prompt_rows) != 2:
        parser.error("Prompts must contain exactly IDs 400 and 401")
    inputs = {args.prompts.resolve(), args.screen.resolve(), args.baseline.resolve(), Path(__file__).resolve()}
    if args.heldout:
        inputs.add(args.heldout.resolve())
    if args.output.resolve() in inputs:
        parser.error("Output must not overwrite inputs")
    screen_rows, screen_source = load_records(args.screen, prompts, set(prompts))
    screens = {pid: summarize(screen_rows, pid, range(2)) for pid in sorted(prompts)}
    valid_screen = not screen_source["invalid_rows"] and not screen_source["duplicate_keys_excluded_entirely"]
    eligible = [pid for pid, report in screens.items() if valid_screen and report["passes_screen_rule"]]
    selected = eligible[0] if eligible else None  # Frozen tie-break: lower prompt ID, no public score feedback.
    baseline_rows, baseline_source = load_records(args.baseline, {"14": "Describe yourself in one sentence."}, {"14"})
    baseline_repeats = sorted({r["repeat"] for r in baseline_rows})
    result = {"prompts": {pid: {"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest()} for pid, text in prompts.items()},
              "screen": screens, "selected_prompt": selected,
              "baseline_id14": summarize(baseline_rows, "14", baseline_repeats),
              "sources": {"prompts": {"path": str(args.prompts.resolve()), "sha256": sha256(args.prompts)},
                          "screen": screen_source, "baseline": baseline_source,
                          "analyzer": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))}},
              "selection_rule": "Both screen repeats must explicitly self-identify the same one family on all nine references, including all five S nuisances; no UNKNOWN, contradiction, missing, invalid, unexpected, or duplicate key. Tie: lower prompt ID.",
              "ep_queries": 0, "reference_responses": screen_source["selected_rows_before_deduplication"],
              "protocol": {"collector": "query_reference.query", "workers": 4, "max_tokens": 256,
                           "reference_endpoints": list(REFERENCES), "max_experiment_responses": 144,
                           "screen_prompt_ids": ["400", "401"], "screen_repeat_keys": [0, 1],
                           "reserved_repeat_keys_if_qualified": list(range(2, 2 + args.heldout_repeats)),
                           "screen_collection_command": "python -c 'from pathlib import Path; import csv; import query_reference as q; p=Path(\".\"); prompts=[(r[\"id\"],r[\"prompt\"]) for r in csv.DictReader((p/\"identity_probes.csv\").open())]; assert q.PARALLEL==4; q.query(prompts,output=p/\"reference_identity_screen.jsonl\",endpoints=[f\"ref-{i:02d}\" for i in range(1,10)],repeats=2,max_tokens=256,start_repeat=0)'",
                           "analysis_command": "python experiment_identity.py --output identity_report.json"},
              "unknown_is_equality_evidence": False,
              "absence_of_observed_false_split_is_not_identification_success": True,
              "limitations": ["All reference endpoints have one true family, whose actual pretrained name/checkpoint is not supplied. Self-declarations are not checkpoint verification.",
                              "No inter-family discrimination or resistance to the same false self-declaration is measurable on these references.",
                              "No model-level identification; no public-score feedback used; no gain is promised."]}
    if args.heldout:
        if selected is None:
            parser.error("No screen prompt qualified: do not collect held-out responses")
        heldout_rows, heldout_source = load_records(args.heldout, prompts, set(prompts))
        heldout = summarize(heldout_rows, selected, range(2, 2 + args.heldout_repeats))
        heldout_valid = not heldout_source["invalid_rows"] and not heldout_source["duplicate_keys_excluded_entirely"]
        heldout_valid = heldout_valid and all(r["id"] == selected for r in heldout_rows)
        result.update(heldout=heldout)
        result["sources"]["heldout"] = heldout_source
        result["reference_responses"] += heldout_source["selected_rows_before_deduplication"]
        passed = heldout_valid and heldout["passes_screen_rule"] and heldout["claimed_families"] == screens[selected]["claimed_families"]
        result["recommendation"] = "LIMITED_CANDIDATE" if passed else "REJECT"
        result["reason"] = ("Screen and new reserved repeats agree without reference-family false splits; candidate family evidence only, not validated across families or on scored endpoints."
                            if passed else "New reserved responses fail full stable self-identification; do not recommend this prompt.")
    else:
        result["recommendation"] = "COLLECT_RESERVED_ONLY" if selected else "REJECT_BOTH"
        result["reason"] = ("A screen prompt qualified, but fresh reserved repetitions are required before recommendation."
                            if selected else "Neither prompt identifies one common family on every reference repeat; abstentions are not equality evidence. No held-out or scored-endpoint collection is justified.")
    if result["reference_responses"] > 144:
        parser.error("Experiment exceeds 144 reference responses")
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"recommendation": result["recommendation"], "selected_prompt": selected,
                      "reference_responses": result["reference_responses"], "ep_queries": 0,
                      "screen": {pid: {key: report[key] for key in ("identified", "unknown", "contradictory_responses", "claimed_families", "passes_screen_rule")} for pid, report in screens.items()},
                      "report": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

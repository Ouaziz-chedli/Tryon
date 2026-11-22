#!/usr/bin/env python3
"""CPU teacher-forced text likelihoods, not an identification/classification model.

Example (downloads/job allocation must be authorized by the operator)::
    VIRTUAL_ENV=/p/scratch/training2666/ouaziz1/TRYON/unlearning/.venv \
      uv run --no-project --with transformers python score_likelihood.py \
      --model HuggingFaceTB/SmolLM2-135M-Instruct \
      --revision 12fd25f77366fa6b3b4b768ec3050bf629380bac \
      --answers reference_answers.20261122T025452Z.jsonl \
      --output likelihood_smollm2_135m_refs.json --batch-size 4 --threads 8

SmolLM2-135M is a hypothesis, NOT the known reference checkpoint. Only a user
message is supplied; the provider's unknown system prompt/template may differ.
A small-model likelihood signal need not transfer to larger evaluated models.

Output is one atomic JSON object with metadata and a features array. Each
feature retains the original answer and its physical file/line provenance.
Every ambiguous key within a file is excluded; a key shared across files is an
error, even if identical. ID 111 is excluded. No other ID allowlist is applied.

The native chat generation prefix is concatenated with visible response text,
then retokenized with fast offsets and NO appended EOS. Tokens overlapping the
response are scored, including a boundary-straddling token (reported explicitly).
Such a token also contains prompt suffix characters: decoded text cannot reveal
the original generated tokenization, so these are retokenized-text likelihoods,
not exact API generation likelihoods. Ranks are 1 + count(logit > target logit),
i.e. best rank for ties. Quantiles use linear interpolation over observed tokens.

One offline check, requiring only torch and no model: --self-check.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile


def load_answers(paths):
    rows, manifests, owners, ignored = [], [], {}, []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        file_rows, digest, physical_rows = [], hashlib.sha256(), 0
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, 1):
                digest.update(line)
                if not line.strip():
                    continue
                physical_rows += 1
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("answer must be an object")
                    raw_id = row["id"]
                    if type(raw_id) is int:
                        pid = raw_id
                    elif isinstance(raw_id, str) and re.fullmatch(r"[+-]?[0-9]+", raw_id):
                        pid = int(raw_id)
                    else:
                        raise ValueError("id must be an integer or integer string")
                    if not all(isinstance(row[k], str) for k in ("endpoint", "prompt", "response")):
                        raise ValueError("endpoint, prompt, response must be strings")
                    if not row["endpoint"].strip() or not row["prompt"].strip():
                        raise ValueError("endpoint and prompt cannot be blank")
                    if type(row["repeat"]) is not int or row["repeat"] < 0:
                        raise ValueError("repeat must be a nonnegative integer")
                    key = row["endpoint"], pid, row["repeat"]
                    source = {"path": str(path), "line": line_number}
                    if pid == 111:
                        ignored.append(source)
                        continue
                    if key in owners and owners[key] != str(path):
                        raise ValueError(f"cross-file key collision {key}: {owners[key]} and {path}")
                    owners[key] = str(path)
                    file_rows.append({"key": list(key), "answer": row, "source": source})
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        counts = Counter(tuple(row["key"]) for row in file_rows)
        clean = [row for row in file_rows if counts[tuple(row["key"])] == 1]
        rows.extend(clean)
        manifests.append({"path": str(path), "sha256": digest.hexdigest(),
                          "input_rows": physical_rows, "usable_rows": len(clean),
                          "discarded_ambiguous_rows": len(file_rows) - len(clean),
                          "ambiguous_keys": [list(k) for k, n in sorted(counts.items()) if n > 1]})
    prompts = {}
    for row in rows:
        pid, prompt = row["key"][1], row["answer"]["prompt"]
        if pid in prompts and prompts[pid] != prompt:
            raise ValueError(f"conflicting clean prompt texts for ID {pid}")
        prompts[pid] = prompt
    if not rows:
        raise ValueError("no unambiguous answers remain after validation")
    rows.sort(key=lambda row: tuple(row["key"]))
    coverage = Counter((row["key"][0], row["key"][1]) for row in rows)
    return rows, {"inputs": manifests, "excluded_id_111": ignored,
                  "usable_rows": len(rows),
                  "coverage": [{"endpoint": ep, "id": pid, "rows": n}
                               for (ep, pid), n in sorted(coverage.items())],
                  "prompts": {str(pid): text for pid, text in sorted(prompts.items())}}


def generated_positions(offsets, boundary, text_length):
    """A straddling token belongs to the response; a token ending at the boundary does not."""
    if any(not 0 <= start <= end <= text_length for start, end in offsets):
        raise ValueError("tokenizer returned invalid character offsets")
    positions = [i for i, (start, end) in enumerate(offsets) if end > boundary and end > start]
    if positions and positions[0] == 0:
        raise ValueError("first response token has no preceding causal context")
    crossing = sum(start < boundary < end for start, end in offsets)
    return positions, crossing


def summarize(values):
    ordered = sorted(values)
    def quantile(q):
        position = (len(ordered) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {"mean": math.fsum(values) / len(values), "median": quantile(0.5),
            "min": ordered[0], "max": ordered[-1],
            "quantiles": {str(q): quantile(q) for q in (0.1, 0.25, 0.75, 0.9, 0.95)}}


def score_logits(torch, logits, input_ids, positions):
    """Read only real generated positions; position p is predicted at p - 1."""
    logprobs, ranks = [], []
    for position in positions:
        scores = logits[position - 1]
        target = scores[input_ids[position]]
        logprob = float((target - torch.logsumexp(scores, dim=-1)).item())
        if not math.isfinite(logprob):
            raise ValueError("nonfinite generated-token log probability")
        logprobs.append(logprob)
        ranks.append(int(torch.count_nonzero(scores > target).item()) + 1)
    if not logprobs:
        raise ValueError("cannot summarize an unscored response")
    return {"generated_tokens": len(positions), "logprob_sum": math.fsum(logprobs),
            "logprob": summarize(logprobs), "rank": summarize(ranks)}


def self_check(torch):
    # One boundary/causal/padding check: token 2 crosses the boundary and token 3
    # continues the answer; token 1 is prompt-only, token 4 is padding.
    offsets = [(0, 1), (1, 3), (3, 5), (5, 6), (0, 0)]
    positions, crossing = generated_positions(offsets, boundary=4, text_length=6)
    assert positions == [2, 3] and crossing == 1
    ids = torch.tensor([0, 2, 1, 0, 2])
    logits = torch.tensor([[99., 0., 0.], [0., 2., 1.], [0., 1., 2.],
                           [0., 0., 99.], [99., 0., 0.]])
    result = score_logits(torch, logits, ids, positions)
    denominator = math.log(1 + math.exp(1) + math.exp(2))
    assert result["generated_tokens"] == 2
    assert math.isclose(result["logprob_sum"], 2 - 2 * denominator, abs_tol=1e-6)
    assert result["rank"]["mean"] == 2 and result["rank"]["median"] == 2
    print("boundary / causal shift / padding self-check passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="public Hugging Face causal model repository ID")
    parser.add_argument("--revision", help="branch, tag or commit; resolved commit is recorded and pinned")
    parser.add_argument("--answers", nargs="+", type=Path, help="one or more answer JSONL files")
    parser.add_argument("--output", type=Path, help="atomic JSON output; existing output is not overwritten")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    import torch
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.self_check:
        self_check(torch)
        return
    if not args.model or not args.answers or not args.output:
        parser.error("--model, --answers and --output are required for inference")
    paths = [path.resolve() for path in args.answers]
    if len(set(paths)) != len(paths):
        parser.error("the same input file was supplied more than once")
    if args.output.resolve() in paths or args.output.exists():
        parser.error("output must be a new file, not an input or existing output")
    rows, audit = load_answers(paths)

    # Dependencies and all model loading live here; importing this module is safe.
    import transformers
    import tokenizers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    config = AutoConfig.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    resolved = getattr(config, "_commit_hash", None)
    if not resolved:
        raise ValueError("could not resolve an immutable HF revision; use a public model repository ID")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=resolved, use_fast=True,
                                              trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError("a fast tokenizer with character offsets is required")
    if not tokenizer.chat_template:
        raise ValueError("model tokenizer has no native chat template")
    template = tokenizer.get_chat_template()
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=resolved, config=config,
                                                dtype=torch.float32, trust_remote_code=False)
    model.eval()
    if any(p.device.type != "cpu" or (p.is_floating_point() and p.dtype != torch.float32)
           for p in model.parameters()):
        raise ValueError("model must be entirely on CPU with float32 floating parameters")
    limits = [value for value in (getattr(config, "max_position_embeddings", None),
                                  getattr(config, "n_positions", None), tokenizer.model_max_length)
              if isinstance(value, int) and 0 < value < 1_000_000_000]
    if not limits:
        raise ValueError("no finite model/tokenizer context limit; cannot guarantee no truncation")
    context_limit = min(limits)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("tokenizer needs a pad or EOS ID for masked right padding")
    identity = {"model": args.model, "requested_revision": args.revision,
                "resolved_revision": resolved, "tokenizer_revision": resolved,
                "dtype": "float32", "device": "cpu"}
    metadata = {"schema": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "scoring": identity, "versions": {"python": platform.python_version(),
                    "torch": torch.__version__, "transformers": transformers.__version__,
                    "tokenizers": tokenizers.__version__},
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "batch_size": args.batch_size, "threads": args.threads,
                "context_limit": context_limit, "chat_template": template,
                "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                "messages": "one user message; native add_generation_prompt=True; no supplied system message",
                "boundary_policy": "retokenize prefix+response; score tokens with offset.end > prefix length",
                "eos_policy": "no EOS or assistant terminator appended; visible response text only",
                "rank_policy": "1 + count(vocabulary logits strictly greater than target)",
                "quantiles": "linear interpolation, over generated tokens only", "audit": audit}
    prefixes = {pid: tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                   tokenize=False, add_generation_prompt=True)
                for pid, prompt in ((int(pid), text) for pid, text in audit["prompts"].items())}
    if any(not prefix for prefix in prefixes.values()):
        raise ValueError("native chat template produced an empty generation prefix")

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                         prefix=args.output.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write('{"metadata":' + json.dumps(metadata, ensure_ascii=False, allow_nan=False) + ',"features":[')
            first = True
            with torch.inference_mode():
                for start in range(0, len(rows), args.batch_size):
                    batch = rows[start:start + args.batch_size]
                    encoded, results = [], []
                    for row in batch:
                        result = dict(row, scoring=identity)
                        results.append(result)
                        response = row["answer"]["response"]
                        if response == "":
                            result.update(status="non_scorable", reason="empty_response", features=None)
                            continue
                        prefix = prefixes[row["key"][1]]
                        text = prefix + response
                        tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                                           truncation=False, padding=False)
                        ids, offsets = tokens["input_ids"], tokens["offset_mapping"]
                        if len(ids) != len(offsets) or not ids or max(end for _, end in offsets) != len(text):
                            raise ValueError(f"{row['source']}: tokenization did not cover the full untruncated text")
                        if len(ids) > context_limit:
                            raise ValueError(f"{row['source']}: {len(ids)} tokens exceed context {context_limit}; no truncation allowed")
                        positions, crossing = generated_positions(offsets, len(prefix), len(text))
                        if not positions:
                            result.update(status="non_scorable", reason="no_response_tokens", features=None)
                            continue
                        result.update(status="scored", input_tokens=len(ids),
                                      prompt_only_tokens=len(ids) - len(positions),
                                      boundary_crossing_tokens=crossing)
                        encoded.append((result, ids, positions))
                    if encoded:
                        width = max(len(ids) for _, ids, _ in encoded)
                        input_ids = torch.full((len(encoded), width), pad_id, dtype=torch.long)
                        attention_mask = torch.zeros_like(input_ids)
                        for i, (_, ids, _) in enumerate(encoded):
                            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
                            attention_mask[i, :len(ids)] = 1
                        output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                                       output_attentions=False, output_hidden_states=False, return_dict=True)
                        logits = output.logits
                        for i, (result, _, positions) in enumerate(encoded):
                            result["features"] = score_logits(torch, logits[i], input_ids[i], positions)
                        del logits, output, input_ids, attention_mask
                    for result in results:
                        if not first:
                            handle.write(",")
                        handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False))
                        first = False
                    print(f"scored/recorded {min(start + args.batch_size, len(rows))}/{len(rows)} responses", flush=True)
            handle.write("]}\n")
        # A concurrent operator must not lose a result to an accidental overwrite.
        os.link(temporary, args.output)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Wrote {args.output}; resolved model revision {resolved}", flush=True)


if __name__ == "__main__":
    main()

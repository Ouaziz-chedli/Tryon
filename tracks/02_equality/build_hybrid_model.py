"""Build Track 2 hybrid submission (Surface + WordTFIDF complete-linkage) from local caches."""

from __future__ import annotations

import collections
import csv
import itertools
import math
import random
from pathlib import Path

from eval_models_cv import (
    ALLOWED_IDS,
    _normalize,
    build_surface_vectors,
    build_tfidf_vectors,
    cluster_complete,
    eval_mcc,
    load_records,
    load_truth,
    pairwise_dist,
    split_folds,
)

PIDS = ["1", "4", "100", "101", "102", "104", "105", "107", "108", "109", "110", "111", "112"]
W_SURF = 0.3
TAU_FAMILY = 0.85
TAU_MODEL = 1.02


def compute_distance_matrix(
    eps: list[str], data: dict[str, dict[str, list[str]]], pids: list[str]
) -> dict[tuple[str, str], float]:
    """Compute concatenated L2-normalized Surface + WordTFIDF pairwise distance matrix."""
    surf = build_surface_vectors(eps, data, pids)
    word = build_tfidf_vectors(eps, data, pids, use_char=False)
    vecs = {ep: _normalize([W_SURF * x for x in surf[ep]] + word[ep]) for ep in eps}
    return pairwise_dist(vecs)


def cluster_hierarchical(
    eps: list[str], data: dict[str, dict[str, list[str]]], pids: list[str]
) -> tuple[dict[str, str], dict[str, str]]:
    """Stage 1: global family clustering. Stage 2: intra-family model clustering at ref-calibrated tau."""
    d_global = compute_distance_matrix(eps, data, pids)
    raw_fam = cluster_complete(eps, d_global, TAU_FAMILY)
    fam_groups: dict[int, list[str]] = collections.defaultdict(list)
    for ep, fid in raw_fam.items():
        fam_groups[fid].append(ep)
    raw_mod: dict[str, int] = {}
    next_mid = 0
    for fid, fam_eps in sorted(fam_groups.items()):
        if len(fam_eps) == 1:
            raw_mod[fam_eps[0]] = next_mid
            next_mid += 1
            continue
        d_intra = compute_distance_matrix(sorted(fam_eps), data, pids)
        sub_mod = cluster_complete(sorted(fam_eps), d_intra, TAU_MODEL)
        for ep, mid in sub_mod.items():
            raw_mod[ep] = next_mid + mid
        next_mid += max(sub_mod.values()) + 1
    return enforce_hierarchy(eps, raw_fam, raw_mod)


def enforce_hierarchy(
    eps: list[str], fam_map: dict[str, int], mod_map: dict[str, int]
) -> tuple[dict[str, str], dict[str, str]]:
    """Ensure every model cluster belongs to a single family and relabel F0.., M0.. cleanly."""
    # Group models by (fam, mod) to guarantee strict hierarchy: same model => same family.
    pairs = sorted({(fam_map[ep], mod_map[ep]) for ep in eps})
    fams = sorted({f for f, _ in pairs})
    f_lbl = {f: f"F{i}" for i, f in enumerate(fams)}
    m_lbl = {pair: f"M{i}" for i, pair in enumerate(pairs)}
    return {ep: f_lbl[fam_map[ep]] for ep in eps}, {ep: m_lbl[(fam_map[ep], mod_map[ep])] for ep in eps}


def pair_agreement(eps: list[str], p1: dict[str, str], p2: dict[str, str]) -> float:
    """Fraction of endpoint pairs with identical same/different cluster assignment."""
    same, tot = 0, 0
    for a, b in itertools.combinations(eps, 2):
        same += (p1[a] == p1[b]) == (p2[a] == p2[b])
        tot += 1
    return same / max(1, tot)


def subsample_data(
    data: dict[str, dict[str, list[str]]], rng: random.Random, frac: float = 0.8
) -> dict[str, dict[str, list[str]]]:
    """Subsample responses per (endpoint, prompt_id) without replacement for stability checks."""
    out: dict[str, dict[str, list[str]]] = collections.defaultdict(dict)
    for ep, by_pid in data.items():
        for pid, resps in by_pid.items():
            k = max(1, int(round(len(resps) * frac)))
            out[ep][pid] = rng.sample(resps, min(k, len(resps)))
    return out


def validate_reference_cv(base: Path) -> float:
    """Run leak-free 2-fold CV on ref-01..09 using only {1, 4, 100..112}."""
    truth = load_truth(base / "practice_answers.csv")
    data = load_records(
        [base / "reference_answers.jsonl", base / "reference_metric_v1_holdout.jsonl"], ALLOWED_IDS
    )
    eps = sorted(truth)
    folds = split_folds(data, 2)
    mccs = []
    for te_i in (1, 0):
        dmat = compute_distance_matrix(eps, folds[te_i], [p for p in PIDS if p != "111"])
        mcc, tp, fp, fn, tn = eval_mcc(truth, cluster_complete(eps, dmat, TAU_MODEL))
        mccs.append(mcc)
        print(f"[REF CV fold {te_i}] MCC={mcc:.3f} (TP={tp}, FP={fp}, FN={fn}, TN={tn})")
    return sum(mccs) / len(mccs)


def write_submission(path: Path, eps: list[str], fams: dict[str, str], mods: dict[str, str]) -> None:
    """Write validated endpoint,family,model CSV."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["endpoint", "family", "model"])
        for ep in eps:
            w.writerow([ep, fams[ep], mods[ep]])


def main() -> None:
    base = Path("tracks/02_equality")
    avg_cv = validate_reference_cv(base)
    print(f"[REF CV] avg_heldout_MCC={avg_cv:.3f} at tau_model={TAU_MODEL:.2f}")

    ep_data = load_records([base / "answers.jsonl", base / "answers_metric_v1_additions.jsonl"], ALLOWED_IDS)
    eps = sorted(ep_data)
    assert len(eps) == 33, f"expected 33 endpoints, got {len(eps)}"
    dmat = compute_distance_matrix(eps, ep_data, PIDS)
    fams, mods = cluster_hierarchical(eps, ep_data, PIDS)

    rng = random.Random(42)
    fam_stab, mod_stab = [], []
    for _ in range(5):
        bf, bm = cluster_hierarchical(eps, subsample_data(ep_data, rng, 0.8), PIDS)
        fam_stab.append(pair_agreement(eps, fams, bf))
        mod_stab.append(pair_agreement(eps, mods, bm))
    print(f"[BOOTSTRAP 80%] family_pair_agree={sum(fam_stab)/5:.3f}, model_pair_agree={sum(mod_stab)/5:.3f}")

    out_csv = base / "submission_hybrid_v1.csv"
    write_submission(out_csv, eps, fams, mods)
    print(f"[WROTE] {out_csv} ({len(set(fams.values()))} families, {len(set(mods.values()))} models)")


if __name__ == "__main__":
    main()

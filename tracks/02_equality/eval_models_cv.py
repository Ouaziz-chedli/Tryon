"""Cross-validated Track 2 model benchmark on ref-01..09 using only {1, 4, 100..112}."""

from __future__ import annotations

import collections
import csv
import itertools
import json
import math
import re
from pathlib import Path

ALLOWED_IDS = {"1", "4"} | {str(i) for i in range(100, 113)}
REF_S = ("ref-01", "ref-03", "ref-05", "ref-06", "ref-08")


def load_truth(path: Path) -> dict[str, tuple[str, str]]:
    """Load practice_answers.csv -> {endpoint: (family, model)}."""
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    if rows and rows[0][:3] == ["endpoint", "family", "model"]:
        rows = rows[1:]
    return {r[0].strip(): (r[1].strip(), r[2].strip()) for r in rows if r and r[0].startswith("ref-")}


def load_records(paths: list[Path], allowed_ids: set[str]) -> dict[str, dict[str, list[str]]]:
    """Load deduplicated (endpoint, id, repeat) -> {ep: {pid: [responses]}}."""
    seen: dict[tuple[str, str, int], str] = {}
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            pid = str(r.get("id", ""))
            if pid in allowed_ids:
                seen[(r["endpoint"], pid, int(r.get("repeat", 0)))] = r["response"]
    out: dict[str, dict[str, list[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for (ep, pid, _rep), resp in sorted(seen.items()):
        out[ep][pid].append(resp)
    return out


def split_folds(
    data: dict[str, dict[str, list[str]]], seed_mod: int = 2
) -> list[dict[str, dict[str, list[str]]]]:
    """Split each (ep, pid) response list into disjoint folds of ~4-7 reps."""
    folds: list[dict[str, dict[str, list[str]]]] = [
        collections.defaultdict(lambda: collections.defaultdict(list)) for _ in range(seed_mod)
    ]
    for ep, by_pid in data.items():
        for pid, resps in by_pid.items():
            for idx, text in enumerate(resps):
                folds[idx % seed_mod][ep][pid].append(text)
    return folds


def _char_ngrams(text: str, n_min: int = 2, n_max: int = 4) -> collections.Counter:
    s = re.sub(r"\s+", " ", text.strip().lower())
    c: collections.Counter = collections.Counter()
    for n in range(n_min, n_max + 1):
        for i in range(max(0, len(s) - n + 1)):
            c[s[i : i + n]] += 1
    return c


def _word_ngrams(text: str) -> collections.Counter:
    toks = re.findall(r"[a-z0-9_]+|[^\w\s]", text.lower())
    c: collections.Counter = collections.Counter(toks)
    for a, b in zip(toks, toks[1:]):
        c[f"{a}_{b}"] += 1
    return c


def _surface_features(resps: list[str]) -> list[float]:
    """Extract length, prose-wrapper, and punctuation/numeric stats for one (ep, pid)."""
    if not resps:
        return [0.0] * 6
    lens = sorted(len(r) for r in resps)
    n = len(lens)
    med = float(lens[n // 2])
    iqr = float(lens[(3 * n) // 4] - lens[n // 4])
    words = sum(len(r.split()) for r in resps) / n
    punct = sum(1 for r in resps if r.strip().endswith((".", "!", "?"))) / n
    multi_line = sum(1 for r in resps if "\n" in r.strip()) / n
    nums = sum(len(re.findall(r"\b\d+\b", r)) for r in resps) / n
    return [math.log1p(med), math.log1p(iqr), math.log1p(words), punct, multi_line, nums]


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm > 1e-12 else vec


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def build_tfidf_vectors(
    eps: list[str], data: dict[str, dict[str, list[str]]], pids: list[str], use_char: bool = True
) -> dict[str, list[float]]:
    """Build concatenated per-prompt L2-normalized TF-IDF centroid vectors."""
    vecs: dict[str, list[float]] = {ep: [] for ep in eps}
    for pid in pids:
        ep_counts: dict[str, collections.Counter] = {}
        df: collections.Counter = collections.Counter()
        for ep in eps:
            c: collections.Counter = collections.Counter()
            for r in data[ep].get(pid, []):
                c.update(_char_ngrams(r) if use_char else _word_ngrams(r))
            ep_counts[ep] = c
            for tok in c:
                df[tok] += 1
        vocab = [tok for tok, cnt in df.items() if cnt >= 2]
        idf = {t: math.log((len(eps) + 1) / (df[t] + 1)) + 1.0 for t in vocab}
        for ep in eps:
            c = ep_counts[ep]
            tot = max(1, sum(c.values()))
            v = _normalize([(c[t] / tot) * idf[t] for t in vocab])
            vecs[ep].extend(v)
    return {ep: _normalize(v) for ep, v in vecs.items()}


def build_surface_vectors(
    eps: list[str], data: dict[str, dict[str, list[str]]], pids: list[str]
) -> dict[str, list[float]]:
    """Build standardized surface+length feature vectors across prompts."""
    raw = {ep: [f for pid in pids for f in _surface_features(data[ep].get(pid, []))] for ep in eps}
    dim = len(raw[eps[0]])
    for j in range(dim):
        vals = [raw[ep][j] for ep in eps]
        mu = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1.0
        for ep in eps:
            raw[ep][j] = (raw[ep][j] - mu) / sd
    return {ep: _normalize(v) for ep, v in raw.items()}


def project_out_nuisance(vecs: dict[str, list[float]], s_eps: tuple[str, ...]) -> dict[str, list[float]]:
    """Gram-Schmidt orthogonalize against the centered S nuisance subspace."""
    s_vecs = [vecs[ep] for ep in s_eps if ep in vecs]
    if len(s_vecs) < 2:
        return vecs
    dim = len(s_vecs[0])
    mean_s = [sum(v[j] for v in s_vecs) / len(s_vecs) for j in range(dim)]
    basis: list[list[float]] = []
    for v in s_vecs:
        u = [v[j] - mean_s[j] for j in range(dim)]
        for b in basis:
            proj = _dot(u, b)
            u = [u[j] - proj * b[j] for j in range(dim)]
        n = math.sqrt(sum(x * x for x in u))
        if n > 1e-8:
            basis.append([x / n for x in u])
    out: dict[str, list[float]] = {}
    for ep, v in vecs.items():
        r = list(v)
        for b in basis:
            proj = _dot(r, b)
            r = [r[j] - proj * b[j] for j in range(dim)]
        out[ep] = _normalize(r)
    return out


def cluster_complete(eps: list[str], dmat: dict[tuple[str, str], float], tau: float) -> dict[str, int]:
    """Complete-linkage clustering cutting at distance threshold tau."""
    clusters = [[ep] for ep in eps]
    while len(clusters) > 1:
        best_d, best_i, best_j = 1e9, -1, -1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = max(dmat[(a, b)] for a in clusters[i] for b in clusters[j])
                if d < best_d:
                    best_d, best_i, best_j = d, i, j
        if best_d > tau:
            break
        clusters[best_i] = clusters[best_i] + clusters[best_j]
        clusters.pop(best_j)
    return {ep: cid for cid, cl in enumerate(clusters) for ep in cl}


def eval_mcc(truth: dict[str, tuple[str, str]], pred: dict[str, int]) -> tuple[float, int, int, int, int]:
    """Exact clipped MCC over all endpoint pairs (same formula as score_practice.py)."""
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(sorted(truth), 2):
        t = truth[a] == truth[b]
        g = pred[a] == pred[b]
        tp += t and g
        fp += (not t) and g
        fn += t and not g
        tn += (not t) and not g
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = max(0.0, (tp * tn - fp * fn) / den) if den else 0.0
    return mcc, tp, fp, fn, tn


def pairwise_dist(vecs: dict[str, list[float]]) -> dict[tuple[str, str], float]:
    """Compute all pairwise Euclidean distances."""
    eps = sorted(vecs)
    return {(a, b): _dist(vecs[a], vecs[b]) for a in eps for b in eps}


def main() -> None:
    base = Path("tracks/02_equality")
    truth = load_truth(base / "practice_answers.csv")
    data = load_records(
        [base / "reference_answers.jsonl", base / "reference_metric_v1_holdout.jsonl"],
        ALLOWED_IDS,
    )
    eps = sorted(truth)
    pids = sorted(ALLOWED_IDS - {"111"})
    folds = split_folds(data, seed_mod=2)

    for label, builder in [
        ("M1_Surface", lambda d: build_surface_vectors(eps, d, pids)),
        ("M2_CharTFIDF_Proj", lambda d: project_out_nuisance(build_tfidf_vectors(eps, d, pids, True), REF_S)),
        ("M3_WordTFIDF_Proj", lambda d: project_out_nuisance(build_tfidf_vectors(eps, d, pids, False), REF_S)),
        ("M4_Hybrid_Surf+WordProj", lambda d: {
            ep: _normalize(
                [0.6 * x for x in build_surface_vectors(eps, d, pids)[ep]]
                + [0.8 * x for x in project_out_nuisance(build_tfidf_vectors(eps, d, pids, False), REF_S)[ep]]
            )
            for ep in eps
        }),
    ]:
        # Fit threshold on fold 0, evaluate on held-out fold 1 (and vice-versa)
        scores = []
        for tr_i, te_i in [(0, 1), (1, 0)]:
            d_tr = pairwise_dist(builder(folds[tr_i]))
            best_tau = max(
                (t / 20.0 for t in range(4, 30)),
                key=lambda tau: eval_mcc(truth, cluster_complete(eps, d_tr, tau))[0],
            )
            d_te = pairwise_dist(builder(folds[te_i]))
            mcc, tp, fp, fn, tn = eval_mcc(truth, cluster_complete(eps, d_te, best_tau))
            scores.append((mcc, best_tau, tp, fp, fn, tn))
        avg_mcc = sum(s[0] for s in scores) / 2
        print(f"{label:24s} | avg_heldout_MCC={avg_mcc:.3f} | fold0->1={scores[0][0]:.3f}(tau={scores[0][1]:.2f},fp={scores[0][3]},fn={scores[0][4]}) | fold1->0={scores[1][0]:.3f}(tau={scores[1][1]:.2f},fp={scores[1][3]},fn={scores[1][4]})")


if __name__ == "__main__":
    main()

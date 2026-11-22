"""CPU pure-logic unit tests for run_soup_lab.py (no torch: queue construction only)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_soup_lab import (
    CUSTOM_PAIRS,
    ORIG,
    QUEUE_TARGET,
    _build_queue,
    _crossce_specs,
    _in_window,
    _is_path_ref,
    _pairs_specs,
    _rank_rows,
)


def _row(name: str, seed: int, ce: float, beta: float, kl: float, pkl: float, in_window: bool = True) -> dict:
    """Synthetic sweep_grid metrics row for one SG run."""
    mu_sigma = 3.85 if in_window else 2.50
    row = {
        "name": name, "valid": True,
        "cfg": {"seed": seed, "target_forget_ce": ce, "npo_beta": beta, "retain_kl_weight": kl},
        "ce_mean": 1.0, "ce_std": mu_sigma - 1.0, "ce_p50": 0.95, "frac_gt_95": 0.20,
        "proxy_kl_t2": pkl, "absent_classes_kl_t2": 0.03, "delta_l2_total": 4.0,
        "calibrated_proxy": 0.12, "mu_sigma": mu_sigma, "in_window": in_window,
    }
    if not in_window:
        row["ce_p50"], row["frac_gt_95"] = 0.30, 0.05
    return row


def _nine_in_window_rows() -> list[dict]:
    """The expected cluster table: 4 families x 2 seeds + ce0.90_b2.5_kl2.8_s101."""
    rows = []
    for ce, beta in ((0.90, 2.0), (1.00, 2.0), (0.70, 2.0), (0.70, 1.8)):
        for seed in (42, 101):
            rows.append(_row(f"SG_ce{ce:.2f}_b{beta:.1f}_kl2.8_s{seed}", seed, ce, beta, 2.8, pkl=0.045))
    rows.append(_row("SG_ce0.90_b2.5_kl2.8_s101", 101, 0.90, 2.5, 2.8, pkl=0.041))
    return rows


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_in_window_bounds() -> None:
    base = _row("SG_x", 42, 0.9, 2.0, 2.8, 0.04)
    _assert(_in_window(base), "center of window must pass")
    for key, lo, hi in (("mu_sigma", 3.65, 4.00), ("ce_p50", 0.80, 1.20)):
        base[key] = lo
        _assert(_in_window(base), f"{key} lower bound inclusive")
        base[key] = hi
        _assert(_in_window(base), f"{key} upper bound inclusive")
        base[key] = hi + 0.01
        _assert(not _in_window(base), f"{key} above window must fail")
        base["mu_sigma"], base["ce_p50"] = 3.85, 0.95
    for key, thr in (("proxy_kl_t2", 0.056), ("delta_l2_total", 4.70), ("frac_gt_95", 0.30)):
        base[key] = thr
        _assert(_in_window(base), f"{key} at threshold passes")
        base[key] = thr + 0.001
        _assert(not _in_window(base), f"{key} above threshold must fail")
        base[key] = 0.04 if key == "proxy_kl_t2" else (4.0 if key == "delta_l2_total" else 0.2)


def test_rank_rows_in_window_first_then_pkl() -> None:
    out = _row("SG_out", 42, 0.9, 2.0, 2.8, 0.01, in_window=False)
    win_hi = _row("SG_hi", 42, 0.9, 2.0, 2.8, 0.050)
    win_lo = _row("SG_lo", 42, 0.9, 2.0, 2.8, 0.041)
    ranked = _rank_rows([out, win_hi, win_lo])
    _assert([r["name"] for r in ranked] == ["SG_lo", "SG_hi", "SG_out"], "rank must be window-first, ProxyKL asc")


def test_build_queue_sixteen_on_expected_table() -> None:
    rows = _nine_in_window_rows()
    rows.append(_row("SG_out1", 42, 0.8, 2.0, 2.8, 0.02, in_window=False))
    rows.append(_row("SG_out2", 101, 0.8, 2.5, 2.2, 0.03, in_window=False))
    queue = _build_queue(rows)
    _assert(len(queue) == QUEUE_TARGET, f"expected {QUEUE_TARGET} soups, got {len(queue)}")
    kinds = {}
    for name, kind, parts in queue:
        kinds[kind] = kinds.get(kind, 0) + 1
        total_w = sum(w for w, _ in parts)
        _assert(abs(total_w - 1.0) <= 1e-5, f"{name}: weights sum to {total_w}")
        run_names = {r["name"] for r in rows}
        _assert(all(n == ORIG or n in run_names for _, n in parts), f"{name}: unknown part reference")
    _assert(kinds == {"pair": 4, "three_way": 2, "shrink": 4, "cross_ce": 4, "four_way": 2}, f"kind counts {kinds}")
    names = [s[0] for s in queue]
    _assert(len(set(names)) == len(names), "soup names must be unique")
    _assert(any(n == ORIG for _, kind, parts in queue if kind == "shrink" for _, n in parts), "shrink must pull toward ORIG")


def test_build_queue_names_match_families() -> None:
    queue = _build_queue(_nine_in_window_rows())
    pair_names = sorted(s[0] for s in queue if s[1] == "pair")
    expected = sorted(
        f"SL_pair_ce{ce:.2f}_b{beta:.1f}_kl2.8" for ce, beta in ((0.90, 2.0), (1.00, 2.0), (0.70, 2.0), (0.70, 1.8))
    )
    _assert(pair_names == expected, f"pair names {pair_names}")
    shrink_names = sorted(s[0] for s in queue if s[1] == "shrink")
    _assert(
        shrink_names
        == sorted(
            [
                "SL_shrink_l0.85_SG_ce0.90_b2.5_kl2.8_s101",
                "SL_shrink_l0.92_SG_ce0.90_b2.5_kl2.8_s101",
                "SL_shrink_l0.85_SG_ce0.90_b2.0_kl2.8_s42",
                "SL_shrink_l0.92_SG_ce0.90_b2.0_kl2.8_s42",
            ]
        ),
        f"shrink targets the 2 best-ranked runs: {shrink_names}",
    )


def test_build_queue_degrades_when_data_missing() -> None:
    two = [_row("SG_a_s42", 42, 0.90, 2.0, 2.8, 0.045), _row("SG_a_s101", 101, 0.90, 2.0, 2.8, 0.044)]
    queue = _build_queue(two)
    _assert(queue, "one pair family must still yield soups")
    kinds = [kind for _, kind, _ in queue]
    _assert(kinds.count("pair") == 1, f"pairs {kinds}")
    _assert(kinds.count("shrink") == 4, f"shrink always uses the 2 best runs: {kinds}")
    names = [s[0] for s in queue]
    _assert(len(set(names)) == len(names), "degraded queue names must stay unique")
    _assert(_build_queue([]) == [], "no in-window rows must give an empty queue")
    invalid_only = [_row("SG_inv", 42, 0.9, 2.0, 2.8, 0.04)]
    invalid_only[0]["valid"] = False
    _assert(_build_queue(invalid_only) == [], "invalid checkpoints must be excluded")


def test_crossce_prefers_primary_ce_pair() -> None:
    rows = _nine_in_window_rows()
    specs = _crossce_specs(rows, 2)
    names = [s[0] for s in specs]
    _assert(len(specs) == 2, f"budget respected: {names}")
    _assert(all("ce0.90" in n and "ce1.00" in n for n in names), f"primary ce0.90 x ce1.00 first: {names}")
    _assert(all(n.split("__")[0][-3:] == n.split("__")[1][-3:] for n in names), "cross-CE must pair the same seed")


def test_custom_pairs_defaults() -> None:
    _assert(len(CUSTOM_PAIRS) == 2, "two off-grid recommendations")
    for name, kind, parts in CUSTOM_PAIRS:
        _assert(kind == "custom", f"{name}: kind")
        _assert(abs(sum(w for w, _ in parts) - 1.0) <= 1e-5, f"{name}: weights must sum to 1")
        _assert(all(_is_path_ref(p) for _, p in parts), f"{name}: parts must be cluster-side paths")
    weights = [[w for w, _ in parts] for _, _, parts in CUSTOM_PAIRS]
    _assert(weights[0] == [0.5, 0.5], "PA2 x #14 must be 50/50")
    _assert(weights[1] == [0.6, 0.4], "PA1 x DIA_highstd must be 60/40")
    paths = {p for _, _, parts in CUSTOM_PAIRS for _, p in parts}
    _assert(any("$HOME" in p for p in paths), "#14 path uses the cluster HOME pack dir")
    _assert(any("diamond4" in p for p in paths), "PA1/PA2/DIA paths live in scratch diamond4/")
    _assert(not _is_path_ref(ORIG), "ORIG sentinel must never look like a path")


def test_pairs_file_overrides_defaults() -> None:
    import argparse
    import json
    from pathlib import Path
    import tempfile

    spec = [{"name": "SL_custom_test", "parts": [{"weight": 0.7, "path": "/tmp/a.pt"}, {"weight": 0.3, "path": "/tmp/b.pt"}]}]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(spec, fh)
        pairs_file = Path(fh.name)
    args = argparse.Namespace(pairs_file=pairs_file)
    specs = _pairs_specs(args)
    _assert(specs == [("SL_custom_test", "custom", [(0.7, "/tmp/a.pt"), (0.3, "/tmp/b.pt")])], f"file spec wins: {specs}")
    args.pairs_file = None
    _assert(_pairs_specs(args) == [(n, k, list(p)) for n, k, p in CUSTOM_PAIRS], "no file -> CUSTOM_PAIRS defaults")


if __name__ == "__main__":
    test_in_window_bounds()
    test_rank_rows_in_window_first_then_pkl()
    test_build_queue_sixteen_on_expected_table()
    test_build_queue_names_match_families()
    test_build_queue_degrades_when_data_missing()
    test_crossce_prefers_primary_ce_pair()
    test_custom_pairs_defaults()
    test_pairs_file_overrides_defaults()
    print("SOUP_LAB_TESTS_OK")

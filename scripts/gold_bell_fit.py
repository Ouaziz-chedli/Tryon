#!/usr/bin/env python3
"""Fit the L1-diamond mean_std model on the 18 server submissions (Track 3).

Model (Unlearning.pdf p.2, retain-saturated subset):
    score_i = c * mean_std_i
    mean_std_i = 1 - (|mu_i - mu_g| + |sig_i - sig_g|) / d_orig
    d_orig = |mu_g - mu_orig| + |sig_g - sig_orig| ~= mu_g + sig_g  (orig ~ 0)

`c` folds retain_score x prediction_score (plateau ~0.92 x ~0.19).
Pure stdlib (no numpy locally). Run: python3 scripts/gold_bell_fit.py
"""

from random import Random

# --- 18 submissions: (#, ckpt, score, mu, sigma, p50, proxy_kl, dtheta, quality)
# quality: clean = retain saturated (fit), degraded = retain/prediction damaged,
# guard = server anti-no-op (score forced 0), sum_only = mu+sigma known, split unknown.
SUBMISSIONS = [
    (1, "H_ssd_lora_mod_aug", 0.0287, 1.112, 3.790, None, 0.0184, 4.25, "degraded"),
    (2, "A_npo_mod_aug", 0.0843, 0.617, 1.990, 0.001, 0.0366, 2.97, "clean"),
    (3, "E_lora_bt_T5", 0.0117, 0.169, 0.330, 0.046, 0.0189, 9.48, "degraded"),
    (4, "C_ssd_plus_A_npo", 0.0106, 1.249, 3.430, 0.000, 0.0170, 17.78, "degraded"),
    (5, "B_ce0.45_lr8e-05_kl2.0", 0.1166, 1.438, 2.010, 0.380, 0.0551, 4.03, "clean"),
    (6, "model_pt_orig_probe", 0.0, 0.000, 0.000, 0.000, 0.0, 0.0, "guard"),
    (7, "AT_ce0.50_b3.0_f8", 0.0604, 0.704, 1.195, 0.116, 0.0376, 3.86, "clean"),
    (8, "AT_ce0.50_b2.0_f6", 0.1099, 1.265, 1.746, 0.397, 0.0413, 3.95, "clean"),
    (9, "AT_ce1.00_b1.8_f6", 0.1193, 1.909, 2.050, 1.214, 0.0486, 4.61, "clean"),
    (10, "H4_h123_f6_ce125", 0.0984, 2.127, 2.100, 1.468, 0.0409, 6.27, "degraded"),
    (11, "H1_h1h2_f8_ce125", 0.0813, 1.676, 1.930, 0.912, 0.0325, 5.78, "degraded"),
    (12, "I2R_f6_r4_ce18_fl085", 0.0950, 2.458, 1.980, 2.136, 0.0740, 6.01, "degraded"),
    (13, "SOUP_50_050_50_100", 0.1238, 1.794, 2.040, 1.006, 0.0558, 4.48, "clean"),
    (14, "AT_ce0.80_b1.9_k20_kl28", 0.1197, 1.958, 2.090, 1.256, 0.0569, 4.58, "clean"),
    (15, "SHRINK_88_AT9", 0.1082, 1.212, 1.710, None, 0.0353, 4.05, "clean"),
    (16, "SOUP_50_ce80kl28_50_ce90kl28", 0.1184, 1.941, 2.070, 1.258, 0.0524, 4.52, "clean"),
    (17, "SOUP3_75_AT9_25_AT2", 0.1176, 1.773, 1.990, 1.038, 0.0462, 4.32, "clean"),
    (18, "SG_ce0.90_b2.0_kl2.8_s101", 0.1234, None, None, 0.970, 0.0469, None, "sum_only"),
    (19, "SG_ce0.90_b2.5_kl2.8_s101", 0.1076, None, None, 1.091, 0.0407, None, "sum_only"),
    (21, "PA2_AT_ce095_b20_k22_kl30_s123", 0.1002, 1.560, 1.816, 0.813, 0.0438, 3.75, "clean"),
    (22, "DIA_highstd_AT_ce120_b12_k20_kl26", 0.1199, 2.222, 2.240, 1.552, 0.0530, 4.40, "clean"),
    (24, "SG_ce1.00_b2.0_kl2.8_s101", 0.1248, 1.854, 2.126, 1.016, 0.0472, 4.65, "clean"),
    (25, "SR10_pdt10", 0.0603, 0.586, 1.338, 0.066, 0.0710, 7.61, "clean"),
    (26, "SR14_champ_ref03_long", 0.0857, 0.994, 1.490, 0.247, 0.0284, 4.56, "clean"),
    (27, "SR13_npo_rest35_pd10", 0.0250, 0.203, 1.120, 0.000, 0.0279, 4.59, "clean"),
    (28, "SR15_qmap_w10", 0.0890, 1.096, 1.540, 0.352, 0.0293, 4.61, "clean"),
    (29, "SR20_gateq_b20", 0.1257, 1.772, 2.120, 0.887, 0.0462, 4.64, "clean"),
    (30, "SR21_spread115", 0.1283, 1.801, 2.160, 0.883, 0.0464, 4.64, "clean"),
]
SUM18 = 3.904  # #18: mu+sigma known from journal (06h55), split not repatriated.
SUM19 = 3.678  # #19: from grid metrics (07h15), split not repatriated.
P50_WINDOW = (0.80, 1.20)
RETAIN = 0.92  # assumed saturated retain_score (journal 03h35 diagnosis).


def clean_points():
    """(mu, sigma, score) triples with saturated retain — the fit set."""
    pts = [(s[3], s[4], s[2]) for s in SUBMISSIONS if s[8] == "clean"]
    if not pts:
        raise SystemExit("no clean points")
    return pts


def diamond_mean_std(mu, sig, mu_g, sig_g):
    """L1-diamond mean_std_score; d_orig = mu_g + sig_g (orig ~ 0, 0)."""
    d_orig = mu_g + sig_g
    return 1.0 - (abs(mu - mu_g) + abs(sig - sig_g)) / d_orig


def fit_c_for_target(pts, mu_g, sig_g):
    """Optimal linear gain c for fixed (mu_g, sig_g): closed-form LSQ."""
    f = [diamond_mean_std(mu, sig, mu_g, sig_g) for mu, sig, _ in pts]
    if any(x <= 0 for x in f):
        return 0.0, float("inf")
    ff = sum(x * x for x in f)
    if ff == 0:
        return 0.0, float("inf")
    c = sum(s * x for (_, _, s), x in zip(pts, f)) / ff
    sse = sum((s - c * x) ** 2 for (_, _, s), x in zip(pts, f))
    return c, sse


def fit_diamond(pts, mu_lo, mu_hi, sig_lo, sig_hi, steps):
    """Grid search (mu_g, sig_g) with closed-form c per cell."""
    best = (None, None, 0.0)
    best_sse = float("inf")
    for i in range(steps + 1):
        mu_g = mu_lo + (mu_hi - mu_lo) * i / steps
        for j in range(steps + 1):
            sig_g = sig_lo + (sig_hi - sig_lo) * j / steps
            c, sse = fit_c_for_target(pts, mu_g, sig_g)
            if sse < best_sse:
                best, best_sse = (mu_g, sig_g, c), sse
    if best[0] is None:
        raise SystemExit("diamond grid found no feasible cell")
    return best[0], best[1], best[2], best_sse


def fit_diamond_refined(pts):
    """Coarse grid over the plausible box, then 4 local refinements."""
    box = (1.30, 2.60, 1.60, 2.70)
    for _ in range(4):
        mu_g, sig_g, c, sse = fit_diamond(pts, *box, steps=26)
        dm = (box[1] - box[0]) / 26
        box = (mu_g - dm, mu_g + dm, sig_g - dm, sig_g + dm)
    return mu_g, sig_g, c, sse


def bootstrap_ci(pts, n_boot=400, seed=7):
    """Residual bootstrap: percentile CIs for (mu_g, sig_g, S, c)."""
    mu_g, sig_g, c, sse = fit_diamond_refined(pts)
    resid = [s - c * diamond_mean_std(mu, sig, mu_g, sig_g) for mu, sig, s in pts]
    rng = Random(seed)
    draws = ([], [], [], [])
    for _ in range(n_boot):
        boot = [
            (mu, sig, c * diamond_mean_std(mu, sig, mu_g, sig_g) + rng.choice(resid))
            for mu, sig, _ in pts
        ]
        b_mu, b_sig, b_c, _ = fit_diamond_refined(boot)
        for arr, v in zip(draws, (b_mu, b_sig, b_mu + b_sig, b_c)):
            arr.append(v)
    out = []
    for arr in draws:
        arr.sort()
        out.append((arr[int(0.025 * n_boot)], arr[int(0.975 * n_boot)]))
    return (mu_g, sig_g, c, sse), out


def fit_sum_model():
    """Reduced model score = c*(1-|S-S_g|/S_g), S=mu+sigma; uses sum_only points #18/#19."""
    pts = [(p[0] + p[1], p[2]) for p in clean_points()]
    pts.append((SUM18, 0.1234))
    pts.append((SUM19, 0.1076))
    best = (0.0, 0.0, float("inf"))
    for k in range(1, 800):  # S_g in (3.0, 8.0), closed-form c per S_g
        s_g = 3.0 + 5.0 * k / 800
        f = [1.0 - abs(s - s_g) / s_g for s, _ in pts]
        if any(x <= 0 for x in f):
            continue
        ff = sum(x * x for x in f)
        c = sum(sc * x for (_, sc), x in zip(pts, f)) / ff
        sse = sum((sc - c * x) ** 2 for (_, sc), x in zip(pts, f))
        if sse < best[2]:
            best = (s_g, c, sse)
    return best, pts


def regress_p50():
    """OLS p50 ~ a*mu + b on clean points with known p50 -> mu band for the window."""
    pts = [(s[3], s[5]) for s in SUBMISSIONS if s[8] == "clean" and s[5] is not None]
    n = len(pts)
    if n < 3:
        raise SystemExit("not enough p50 points")
    mx, my = (sum(p[i] for p in pts) / n for i in (0, 1))
    cov = sum((x - mx) * (y - my) for x, y in pts)
    var = sum((x - mx) ** 2 for x, _ in pts)
    if var == 0:
        raise SystemExit("degenerate p50 regression")
    a = cov / var
    b = my - a * mx
    lo, hi = P50_WINDOW
    return a, b, ((lo - b) / a, (hi - b) / a), pts


def implied_prediction(mu_g, sig_g, c):
    """Decomposition at the diamond peak: score_max = retain * prediction."""
    ms13 = diamond_mean_std(1.794, 2.040, mu_g, sig_g)
    pred_peak = c / RETAIN
    pred13 = 0.1238 / (RETAIN * ms13)
    return ms13, pred_peak, pred13


def main():
    pts = clean_points()
    print("=== 18 soumissions (score, mu, sigma, p50, ProxyKL, dTheta, qualite) ===")
    for s in SUBMISSIONS:
        mu_sig = f"{s[3]:.3f}+{s[4]:.3f}" if s[3] is not None else f"(sum {SUM18:.3f})"
        p50 = "  n/a " if s[5] is None else f"{s[5]:.3f}"
        dth = "  n/a" if s[7] is None else f"{s[7]:>5}"
        print(f"#{s[0]:>2} {s[1]:<30} {s[2]:.4f}  mu+sig={mu_sig}  p50={p50}"
              f"  kl={s[6]:.4f}  dth={dth}  {s[8]}")

    (mu_g, sig_g, c, sse), ci = bootstrap_ci(pts)
    print(f"\n=== Fit diamant L1 ({len(pts)} points clean) ===")
    print(f"mu_gold = {mu_g:.3f}  CI95 [{ci[0][0]:.3f}, {ci[0][1]:.3f}]")
    print(f"sig_gold = {sig_g:.3f}  CI95 [{ci[1][0]:.3f}, {ci[1][1]:.3f}]")
    print(f"S_gold = mu+sig = {mu_g + sig_g:.3f}  CI95 [{ci[2][0]:.3f}, {ci[2][1]:.3f}]")
    print(f"c (retain x mean_std x pred gain) = {c:.4f}  CI95 [{ci[3][0]:.4f}, {ci[3][1]:.4f}]")
    print(f"SSE = {sse:.6f}  (rms resid = {(sse / len(pts)) ** 0.5:.4f})")
    for mu, sig, sc in pts:
        pred_sc = c * diamond_mean_std(mu, sig, mu_g, sig_g)
        print(f"  pred {pred_sc:.4f} vs real {sc:.4f}  (mu={mu:.3f}, sig={sig:.3f})")

    (s_g, c_s, sse_s), sum_pts = fit_sum_model()
    print(f"\n=== Modele reduit en somme (inclut #18) ===")
    print(f"S_gold = {s_g:.3f}  c = {c_s:.4f}  SSE = {sse_s:.6f}")
    for s_val, sc in sum_pts:
        print(f"  sum={s_val:.3f} real={sc:.4f} pred={c_s * (1 - abs(s_val - s_g) / s_g):.4f}")

    a, b, (mu_lo, mu_hi), p50_pts = regress_p50()
    print(f"\n=== Regression p50 ~ {a:.3f}*mu + {b:.3f} ===")
    print(f"Fenetre p50 {P50_WINDOW} => mu* in [{mu_lo:.3f}, {mu_hi:.3f}]"
          f"  => sigma* in [{s_g - mu_hi:.3f}, {s_g - mu_lo:.3f}]")

    ms13, pred_peak, pred13 = implied_prediction(mu_g, sig_g, c)
    print(f"\n=== Decomposition (retain = {RETAIN}) ===")
    print(f"mean_std #13 = {ms13:.3f}  => prediction #13 = {pred13:.3f}")
    print(f"score max au pic exact = retain x pred = {c:.4f} (pred pic = {pred_peak:.3f})")

    print("\n=== 2 sondes diamant (somme egale, split oppose, p50 dans la fenetre) ===")
    for tag, mu_p, sig_p in (("A", mu_lo, s_g - mu_lo), ("B", mu_hi, s_g - mu_hi)):
        p50_p = a * mu_p + b
        pred_a = c * diamond_mean_std(mu_p, sig_p, mu_g, sig_g)
        print(f"  Sonde {tag}: mu={mu_p:.2f} sigma={sig_p:.2f} (somme {mu_p + sig_p:.2f})"
              f" p50~{p50_p:.2f}  score predit {pred_a:.4f}")
    print("  Lecture: la sonde au score le plus haut indique la direction du pic ;")
    print("  |scoreA - scoreB| > 0.006 => split (mu_g, sig_g) resolu, refit avec ces 2 points.")


if __name__ == "__main__":
    main()

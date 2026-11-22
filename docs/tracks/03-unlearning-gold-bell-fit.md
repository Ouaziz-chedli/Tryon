# Track 3 — Ajustement analytique de la cloche-or (μ_gold, σ_gold) depuis les 18 soumissions

**Date :** `2026-11-22` · **Auteur :** `GoldBellFit` (analyse locale, zéro GPU)
**Script :** [`scripts/gold_bell_fit.py`](../../scripts/gold_bell_fit.py) (stdlib pur, `py_compile` OK, exécution ~5 s)
**Données :** les 18 soumissions serveur consignées dans [03-unlearning-journal.md](03-unlearning-journal.md) + entrée `06h55` (#18).

---

## 1. Modèle

Formule officielle (`Unlearning.pdf` p. 2), avec `orig ≈ (0, 0)` (`CE = 0.0003`) :

$$\text{mean\_std}(\mu,\sigma) = 1 - \frac{|\mu - \mu_g| + |\sigma - \sigma_g|}{d_{\text{orig}}}, \qquad d_{\text{orig}} = \mu_g + \sigma_g$$

Score serveur = `retain_score × mean_std × prediction`. Sur le sous-ensemble à **retain saturé** (nos runs SalUn/NPO sains), on plie `retain × prediction` en un gain constant `c` :

$$\text{score}_i = c \cdot \text{mean\_std}(\mu_i, \sigma_i)$$

Ajustement par moindres carrés : `c` a une forme fermée pour `(μ_g, σ_g)` fixés, donc grille 2D `(μ_g, σ_g)` + 4 raffinements locaux. Intervalles : bootstrap résiduel (400 rééchantillons). Nos `(μ, σ)` locaux sont la **moitié exacte** des 3 500 images (les copies altérées cachées suivent une droite ≈ +0.3–0.7 au-dessus, cf. P0.3) : le fit est donc exprimé en **coordonnées moitié-exacte**, celles qu'on contrôle à l'entraînement.

## 2. Table nettoyée des 18 soumissions

| # | Checkpoint | Score | `μ` | `σ` | `μ+σ` | `p50` | ProxyKL | dθ | Qualité |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `H_ssd_lora_mod_aug` | 0.0287 | 1.112 | 3.790 | 4.90 | n/a | 0.0184 | 4.25 | dégradé (bimodal, Ent 0.06) |
| 2 | `A_npo_mod_aug` | 0.0843 | 0.617 | 1.990 | 2.61 | 0.001 | 0.0366 | 2.97 | **clean** |
| 3 | `E_lora_bt_T5` | 0.0117 | 0.169 | 0.330 | 0.50 | 0.046 | 0.0189 | 9.48 | dégradé (dθ 9.5) |
| 4 | `C_ssd_plus_A_npo` | 0.0106 | 1.249 | 3.430 | 4.68 | 0.000 | 0.0170 | 17.78 | dégradé (dθ 17.8) |
| 5 | `B_ce0.45_lr8e-05_kl2.0` | 0.1166 | 1.438 | 2.010 | 3.45 | 0.380 | 0.0551 | 4.03 | **clean** |
| 6 | sonde `model.pt` intact | 0.0000 | 0.000 | 0.000 | 0.00 | 0.000 | 0.0 | 0.0 | garde-fou anti-no-op |
| 7 | `AT_ce0.50_b3.0_f8` | 0.0604 | 0.704 | 1.195 | 1.90 | 0.116 | 0.0376 | 3.86 | **clean** |
| 8 | `AT_ce0.50_b2.0_f6` | 0.1099 | 1.265 | 1.746 | 3.01 | 0.397 | 0.0413 | 3.95 | **clean** |
| 9 | `AT_ce1.00_b1.8_f6` | 0.1193 | 1.909 | 2.050 | 3.96 | 1.214 | 0.0486 | 4.61 | **clean** |
| 10 | `H4_h123_f6_ce125` | 0.0984 | 2.127 | 2.100 | 4.23 | 1.468 | 0.0409 | 6.27 | dégradé (dθ 6.3) |
| 11 | `H1_h1h2_f8_ce125` | 0.0813 | 1.676 | 1.930 | 3.61 | 0.912 | 0.0325 | 5.78 | dégradé (dθ 5.8) |
| 12 | `I2R_f6_r4_ce18_fl085` | 0.0950 | 2.458 | 1.980 | 4.44 | 2.136 | 0.0740 | 6.01 | dégradé (kl 0.074) |
| 13 | `SOUP_50_050_50_100` | **0.1238** | 1.794 | 2.040 | 3.83 | 1.006 | 0.0558 | 4.48 | **clean** (record) |
| 14 | `AT_ce0.80_b1.9_k20_kl28` | 0.1197 | 1.958 | 2.090 | 4.05 | 1.256 | 0.0569 | 4.58 | **clean** |
| 15 | `SHRINK_88_AT9` | 0.1082 | 1.212 | 1.710 | 2.92 | n/a | 0.0353 | 4.05 | **clean** |
| 16 | `SOUP_50_ce80kl28_50_ce90kl28` | 0.1184 | 1.941 | 2.070 | 4.01 | 1.258 | 0.0524 | 4.52 | **clean** |
| 17 | `SOUP3_75_AT9_25_AT2` | 0.1176 | 1.773 | 1.990 | 3.76 | 1.038 | 0.0462 | 4.32 | **clean** |
| 18 | `SG_ce0.90_b2.0_kl2.8_s101` | 0.1234 | (somme 3.904) | — | 3.90 | 0.970 | 0.0469 | n/a | **sum-only** (split non rapatrié) |

Règle de filtrage : `clean` ⇔ ProxyKL ≤ 0.057 **et** dθ ≤ 4.7 **et** non bimodal. Les 10 points `clean` forment le jeu de fit ; les points `dégradés` sous-performent tous le modèle (retain/prediction endommagés par dθ ≥ 5.7, kl 0.074 ou bimodalité) — le filtre n'est donc pas un coup de patte circulaire : il est justifié par des causes mécaniques indépendantes du score.

## 3. Résultats du fit (10 points clean, diamant L1)

| Paramètre | Estimation | CI 95 % (bootstrap résiduel) |
|---|---|---|
| `μ_gold` | **1.822** | [1.704, 1.897] |
| `σ_gold` | **1.847** | [1.768, 2.070] |
| `S_gold = μ_g+σ_g` | **3.669** | [3.581, 3.802] |
| `c = retain × prediction` | **0.1312** | [0.1272, 0.1349] |

RMS résiduel **0.0039** — le diamant L1 explique le plateau `0.118–0.124` sans sur-ajustement (3 paramètres, 10 points). Le modèle réduit en somme seule (inclut #18, 11 points) confirme : **`S_gold = 3.700`, `c = 0.1280`**.

**Sensibilité (vérifiée) :** retirer #17 → `S = 3.654`, inchangé à 1.5 % près ; retirer #2 (l'ancre gauche extrême) → `σ_g` dérive vers 2.7, `S = 4.32` : **la somme `S_gold ≈ 3.65–3.80` est solide, le split `(μ_g, σ_g)` ne l'est pas** — l'identification du split repose sur #2 + l'asymétrie du plateau. C'est précisément la lacune que les 2 sondes diamant (§6) comblent.

Lecture physique : le pic n'est **pas** à `μ+σ ≈ 5.5` (hypothèse initiale du journal) mais à **`μ+σ ≈ 3.7`**, légèrement à **gauche** de #13 (3.83). Tous nos bons modèles (`σ ≈ 2.0–2.1`) sont vraisemblablement à droite du pic en σ.

## 4. Contrainte `p50` → bande cible sur μ

Régression OLS sur les 7 points clean à `p50` connu : **`p50 ≈ 0.940·μ − 0.658`** — bien calibrée sur le segment proche du pic (`μ ≥ 1.77`, résidus ≤ 0.09 : #9, #13, #14, #16, #17) ; les points à bas `μ` (#5, #8) tombent sous la droite (queues lourdes), la carte `p50 → μ` n'est donc valable que pour `μ ≥ 1.55`.
La fenêtre `p50 ∈ [0.80, 1.20]` se traduit donc en **`μ* ∈ [1.55, 1.98]`**, donc **`σ* = S* − μ* ∈ [1.72, 2.15]`**.

## 5. Décomposition `0.1238 = retain × mean_std × prediction`

Avec le fit (`retain = 0.92` supposé saturé, conformément au journal 03h35) :

- `mean_std(#13) = 1 − (0.028 + 0.193)/3.669 = 0.940` → **`prediction(#13) = 0.1238 / (0.92 × 0.940) ≈ 0.143`**
- Score maximum atteignable par placement seul (pic exact) : **`c ≈ 0.131`**, soit **+0.007 au-dessus du record 0.1238**.

**Révision importante :** l'hypothèse `mean_std ~ 0.72 / prediction ~ 0.19` du journal supposait un pic à `μ+σ ≈ 5.3` ; elle contredit l'observation décisive que le score **ne monte plus** entre 3.84 et 4.05 (#13 → #9/#16/#14). Le fit inverse la répartition : **mean_std est déjà ~0.94 (presque saturé), le goulot est prediction ≈ 0.14–0.15**. Conséquence stratégique : le placement dans la fenêtre ne vaut que +0.007 ; la route vers 0.7 (`0.92 × 0.87 × 0.87`) passe par le **levier per-image** (SCRUB+R/Patch-Dropout, ensemble de difficulté, Phase 3 vue propre sur-pondérée) — le diamant ne sert qu'à retirer le plafond mean_std avant d'attaquer prediction.

## 6. Placement optimal et 2 sondes diamant (plan §1.2)

### Cible à viser pour les prochains runs (coordonnées moitié-exacte)

$$\boxed{(\mu^*, \sigma^*) \approx (1.82,\ 1.85), \quad \mu+\sigma \in [3.65,\ 3.85],\ \mu \in [1.70,\ 1.90],\ \sigma \in [1.77,\ 2.05],\ p50 \in [0.85,\ 1.15]}$$

Concrètement : nos recettes actuelles livrent toutes `σ ≈ 2.0–2.1` pour `μ ≈ 1.8–1.96` ; la cible demande de **déplacer ~0.2 de masse de σ vers μ** — serrer la queue droite (bornage bilatéral per-sample, `max_ce ≈ 2.0`) tout en gardant la médiane haute. Le bornage NPO par échantillon (déjà implémenté `methods/common.py`) est le levier exact.

### Deux sondes maximisant l'information (splits opposés à somme égale 3.70)

| Sonde | `(μ, σ)` visés | `p50` prédit | Score prédit par le fit | Recette plausible |
|---|---|---|---|---|
| **A** | **(1.55, 2.15)** | ~0.80 | 0.1108 | AT `ce≈0.70`, kl 2.8, ou soup `0.75·AT9 + 0.25·AT8` calée sur ce split |
| **B** | **(1.98, 1.72)** | ~1.20 | 0.1211 | NPO **borné per-sample `max_ce ≈ 2.0`** (compresse σ, pousse μ) — off-manifold pour nos recettes standard, c'est le but |

Les deux sondes sont aux bornes de la fenêtre p50, à somme identique `S = 3.70` : toute différence de score ne peut venir que du **split** (direction du pic dans le plan L1), pas de la somme.

**Règle de décision :**
- `score_B > score_A + 0.006` → le pic est côté **(μ haut, σ bas)** : viser (1.98, 1.72) puis raffiner vers (μ_g, σ_g) du refit.
- `score_A > score_B + 0.006` → pic côté **(μ bas, σ haut)** : inverser.
- `|score_A − score_B| ≤ 0.006` → la somme `S ≈ 3.7` est confirmée et le split est dans la CI ; placement plat, tout le gain restant est dans `prediction` (§5).
- Après chaque sonde : ajouter la ligne dans `SUBMISSIONS` de `gold_bell_fit.py` et relancer (le bootstrap se resserre à chaque point).

Coût : 2 soumissions (30 min de cooldown), zéro GPU au-delà des runs eux-mêmes ; gain espéré : résolution du split à ±0.1 près, soit le dernier degré de liberté du terme mean_std.

## 7. Limites

1. Le fit suppose `retain × prediction = c` constant sur les points clean ; les résidus (max +0.0067 sur #17) sont compatibles avec une dérive de `c` de ±5 % — c'est aussi l'ordre de grandeur du bruit de soumission.
2. `(μ, σ)` locaux = moitié exacte seulement ; le serveur moyenne 50/50 exacte/altérée. Si le décalage exact↔altéré du modèle diffère de celui du gold, le pic en coordonnées locales est décalé d'autant (l'asymétrie P0.3 `+0.3–0.7` suggère un biais possible, non quantifiable avant les sondes).
3. #18 n'entre que par sa somme (3.904) — rapatrier son split depuis `metrics.json` du grid resserrerait la CI.
4. Le point #6 (garde-fou anti-no-op) n'informe pas le fit ; les bornes du journal (`|μ_g−1.44|+|σ_g−2.01| ≤ 2.15`) sont satisfaites par le fit mais peu contraignantes.

## 8. Recommandation chiffrée (TL;DR)

1. **Ne plus guider par `μ+σ ∈ [3.65, 3.90]` seul** : viser **`μ+σ ≈ 3.70 ± 0.10` avec le split (1.55, 2.15) vs (1.98, 1.72)** via les sondes A/B ci-dessus — c'est la seule manière de résoudre la direction du pic.
2. Si une seule soumission est possible avant les runs : soumettre la **sonde B** (prédiction la plus haute à 0.121, et la seule hors de notre manifold actuel — information maximale).
3. Le placement plafonne à **`c ≈ 0.131`** : tout budget GPU au-delà des 2 sondes doit aller au **`prediction_score ≈ 0.143`** (alignement per-image), seul chemin vers le cap 0.7.
4. Après chaque nouvelle soumission : mise à jour de `SUBMISSIONS` + relance de `scripts/gold_bell_fit.py` (le bootstrap rend les CI auto-corrigeantes).

# Track 3 — Piste Initiale (`AT` / SalUn Statique + NPO Borné & Soupes) : Journal Spécialisé

> **Journal dédié au binôme autonome `base-lead` (Orchestrateur/Chercheur) & `base-exec` (Exécuteur/GPU)** optimisant nos **techniques initiales qui détiennent tous nos records serveur** :
> - **`#13` `SOUP_50_050_50_100.pt` = `0.1238` (NOUVEAU RECORD ACTUEL, `+0.0045`)** (`CE = 1.794 ± 2.04`, $\mu+\sigma = 3.84$, `p50 = 1.006`, `>95% = 0.21`, `ProxyKL = 0.0558`, `dTheta = 4.48`, `W_SEL = 0.5678`)
> - **`#9` `AT_ce1.00_b1.8_f6.pt` = `0.1193`** (`CE = 1.909 ± 2.05`, $\mu+\sigma = 3.96$, `p50 = 1.214`, `>95% = 0.19`, `ProxyKL = 0.0486`, `dTheta = 4.61`, `W_SEL = 0.5565`)
> - **`#5` `B_ce0.45_lr8e-05_kl2.0.pt` = `0.1166`** (`CE = 1.438 ± 2.01`, $\mu+\sigma = 3.45$, `p50 = 0.380`, `>95% = 0.37`, `ProxyKL = 0.0551`, `dTheta = 4.03`)
> - **`#8` `AT_ce0.50_b2.0_f6.pt` = `0.1099` (meilleur rendement `0.0365`)** (`CE = 1.265 ± 1.75`, $\mu+\sigma = 3.01$, `p50 = 0.397`, `>95% = 0.31`, `ProxyKL = 0.0413`, `dTheta = 3.95`)

**Liens connexes :**
- Journal principal Track 3 : [03-unlearning-journal.md](03-unlearning-journal.md)
- Implémentation initiale : [auto_tune_4gpu.py](../../tracks/03_unlearning_v2/auto_tune_4gpu.py) · [salun_npo.py](../../tracks/03_unlearning_v2/methods/salun_npo.py) · [common.py](../../tracks/03_unlearning_v2/methods/common.py)
- Protocole de benchmark 3-Folds pondéré : [selective_benchmark.py](../../tracks/03_unlearning_v2/selective_benchmark.py)

---

## 1. Fenêtre exacte du sommet serveur (prouvée par nos 11 soumissions)

- **Sous-oubli (`< 0.1193`) :** $\mu + \sigma \le 3.45$ (`#7` `1.90 -> 0.0604`, `#8` `3.01 -> 0.1099`, `#5` `3.45 -> 0.1166`).
- **Sommet observé :** `#9` (`AT_ce1.00_b1.8_f6.pt`, $\mu+\sigma = 3.96$, `CE = 1.91 ± 2.05`, `p50 = 1.21`, `dTheta = 4.61` $\to$ **`0.1193`**).
- **Sur-oubli (`< 0.1193`) :** `#10` (`H4_h123_f6_ce125.pt`, $\mu+\sigma = 4.23$, `CE = 2.13 ± 2.10`, `p50 = 1.47`, `dTheta = 6.27` $\to$ `0.0984`).

**Cible chirurgicale pour dépasser `0.1193` (cibler `0.130 – 0.160+`) :**
- Garder strictement l'architecture propre qui gagne sur le serveur (**masque SalUn statique à $t=0$ sur `2 048` images**, **`freeze_until_block = 6`**, **`head_lr_mult = 1.0`**, **`weight_decay = 0.0`**, **`restore_absent_head_rows`**).
- Viser l'entre-deux exact entre `#5`/`#8` et `#9` : **`Full CE mean` $\in [1.55, 1.85]$**, **`Full mu+sigma` $\in [3.55, 3.90]$**, **`Full p50` $\in [0.65, 1.05]$**, **`Full >95% <= 0.22`**, **`ProxyKL <= 0.038`**, **`dTheta <= 4.5`**.
- Leviers autorisés :
  1. **Calibration fine `AT` (`target_forget_ce` $\in \{0.65, 0.75, 0.85, 0.90\}$, `npo_beta` $\in \{1.8, 1.9, 2.0\}$, `retain_kl_weight` $\in \{2.4, 2.8\}$, `salun_keep_ratio` $\in \{0.15, 0.20\}$)**.
  2. **Soupes convexes entre nos checkpoints initiaux déjà validés** (`AT_ce0.50_b2.0_f6.pt` `#8`, `AT_ce1.00_b1.8_f6.pt` `#9`, et les nouveaux runs `AT_ce0.75..0.90`).
  3. **Deuxième vue cohérente (`strong` ou mélange `clean + moderate` sur `forget`)** pour améliorer la moitié *altered copies* de `forget_score` sans augmenter `dTheta`.

---

## 2. Règles strictes d'isolation du binôme (`base-lead` & `base-exec`)

1. **Fichiers autorisés en écriture :**
   - `tracks/03_unlearning_v2/run_base_refine.py`
   - `tracks/03_unlearning_v2/tests/test_base_refine.py`
   - `tracks/03_unlearning_v2/output/base_refine/`
   - Ce journal : `docs/tracks/03-unlearning-base-journal.md`
   - *(Ne jamais modifier `track_h.py`, `track_i.py`, `selective_benchmark.py`, ni `03-unlearning-journal.md`.)*
2. **Session Colab dédiée :**
   - Utiliser exclusivement **`-s track3_base`** (`/home/chedli/.local/bin/colab new -s track3_base --gpu A100`), avec un *heartbeat* stdout toutes les 15 s, téléchargement des JSON et meilleurs `.pt` dans `tracks/03_unlearning_v2/output/base_refine/`, puis **`/home/chedli/.local/bin/colab stop -s track3_base`** systématique.
   - Ne jamais soumettre sur JURECA ni sur `http://34.147.229.55` (notifier `@coordinator` dès qu'un `.pt` est prêt).

---

## 3. Journal Chronologique des Expériences (`AT` / Techniques Initiales)

### 2026-11-22 (02h22) — Lancement du binôme `base-lead` + `base-exec`
- Initialisation du binôme dédié au raffinement chirurgical de la famille gagnante `AT` (`freeze_until_block = 6`, SalUn statique $t=0$, NPO borné et soupes).

### 2026-11-22 (03h14) — Round 1 (`run_base_refine.py` sur Colab A100 `track3_base`) : Diagnostic du Plancher `ProxyKL` & Soupes Convexes

Exécution complète de 8 configurations `AT` (`freeze_until_block=6`, `head_lr_mult=1.0`, `steps=210`, SalUn statique $t=0$) + 6 soupes convexes sans surcoût d'entraînement (`base_refine_round1.json`, top 3 `.pt` rapatriés dans `tracks/03_unlearning_v2/output/base_refine/`) :

| Candidat | `W_SEL ± std` | 3-Folds `F_raw / F_alt` | `Gap50` | `IntraAcc` | `InterAcc (KL)` | Full `CE ± std` ($\mu+\sigma$) | Full `p50 (>95%)` | Full `ProxyKL (AbsKL, dTheta)` | `hits_target` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`AT_ce0.80_b1.9_k20_kl28`** | **`0.5694 ± 0.020`** | `2.44 / 2.57` | `+1.38` | `74.8 %` | `99.9 % (0.0272)` | `1.958 ± 2.09 (4.05)` | `1.256 (0.19)` | `0.0569 (0.0361, 4.58)` | `False` |
| **`SOUP_50_ce80kl28_50_ce90kl28`** | `0.5679 ± 0.020` | `2.50 / 2.62` | `+1.41` | `74.5 %` | `99.9 % (0.0278)` | `1.941 ± 2.07 (4.01)` | `1.258 (0.19)` | `0.0524 (0.0329, 4.52)` | `False` |
| **`SOUP_50_050_50_100`** | `0.5678 ± 0.022` | `2.32 / 2.46` | `+1.34` | **`75.5 %`** | `99.9 % (0.0276)` | **`1.794 ± 2.04 (3.84)`** | `1.006 (0.21)` | `0.0558 (0.0347, 4.48)` | `False` |
| `SOUP_40_050_60_100` | `0.5668 ± 0.021` | `2.38 / 2.51` | `+1.36` | `75.2 %` | `99.9 % (0.0283)` | `1.814 ± 2.04 (3.86)` | `1.048 (0.20)` | `0.0539 (0.0335, 4.50)` | `False` |
| `SOUP_30_050_70_100` | `0.5655 ± 0.020` | `2.43 / 2.57` | `+1.39` | `74.8 %` | `99.9 % (0.0290)` | `1.835 ± 2.05 (3.88)` | `1.083 (0.20)` | `0.0522 (0.0325, 4.52)` | `False` |
| `AT_ce0.90_b1.8_k20_kl28` | `0.5651 ± 0.018` | `2.55 / 2.68` | `+1.44` | `73.7 %` | `99.9 % (0.0285)` | `1.926 ± 2.05 (3.97)` | `1.252 (0.19)` | `0.0504 (0.0317, 4.48)` | `False` |
| `SOUP_top3_kl28_avg` | `0.5644 ± 0.022` | `2.52 / 2.65` | `+1.42` | `73.8 %` | `99.8 % (0.0292)` | `1.899 ± 2.06 (3.95)` | `1.179 (0.19)` | `0.0549 (0.0344, 4.50)` | `False` |
| `ANCHOR_ce0.50_b2.0_f6` | `0.5642 ± 0.025` | `2.07 / 2.20` | `+1.22` | `78.6 %` | `99.9 % (0.0248)` | `1.725 ± 2.05 (3.77)` | `0.863 (0.24)` | `0.0694 (0.0432, 4.50)` | `False` |
| `SOUP_50_ce70_50_ce90_kl28` | `0.5608 ± 0.021` | `2.57 / 2.70` | `+1.44` | `73.1 %` | `99.8 % (0.0302)` | `1.870 ± 2.04 (3.91)` | `1.146 (0.20)` | `0.0552 (0.0345, 4.47)` | `False` |
| `ANCHOR_ce1.00_b1.8_f6` (`#9`) | `0.5565 ± 0.019` | `2.61 / 2.74` | `+1.47` | `73.3 %` | `99.8 % (0.0312)` | `1.909 ± 2.05 (3.96)` | `1.214 (0.19)` | `0.0486 (0.0304, 4.61)` | `False` |
| `AT_ce0.70_b2.0_k20_kl25` | `0.5558 ± 0.023` | `2.60 / 2.73` | `+1.45` | `72.2 %` | `99.8 % (0.0321)` | `1.831 ± 2.03 (3.86)` | `1.070 (0.20)` | `0.0650 (0.0406, 4.53)` | `False` |
| `AT_ce0.80_b1.9_k20_kl25` | `0.5454 ± 0.024` | `2.74 / 2.87` | `+1.51` | `70.5 %` | `99.8 % (0.0347)` | `1.820 ± 2.01 (3.83)` | `1.092 (0.20)` | `0.0544 (0.0345, 4.44)` | `False` |
| `AT_ce0.90_b1.8_k15_kl25` | `0.5371 ± 0.028` | `2.54 / 2.66` | `+1.34` | `71.8 %` | `99.7 % (0.0375)` | `1.819 ± 1.93 (3.75)` | `1.162 (0.18)` | `0.0552 (0.0353, 4.40)` | `False` |
| `AT_ce0.80_b1.9_k15_kl28` | `0.5081 ± 0.034` | `2.84 / 2.96` | `+1.42` | `66.0 %` | `99.6 % (0.0465)` | `1.865 ± 2.05 (3.92)` | `1.100 (0.21)` | `0.0646 (0.0412, 4.55)` | `False` |

**Enseignements & Diagnostic Racine Round 1 :**
1. **Pourquoi tous les runs Round 1 bloquent à `ProxyKL = 0.0486–0.0694` et `dTheta = 4.40–4.61` :**
   - Dans `auto_tune_4gpu.py` (`_adjust_controller` / `_snapshot_score`), le frein ne se déclenche qu'à `pkl > 0.048` ou `d_l2 > 4.6` sur le probe d'entraînement (`seed=42`), et `_snapshot_score` ne pénalise aucunement `pkl <= 0.048` ni `d_l2 <= 4.6`. Or `0.048` sur le probe `seed=42` correspond à `0.050–0.057` sur le probe `eval_harness` hors-échantillon (`seed=999`) !
2. **`salun_keep_ratio = 0.20` bat strictement `0.15`** (`W_SEL = 0.5694` vs `0.5081–0.5371` : concentrer l'oubli sur 15 % des poids dégrade `IntraAcc` à `66–72 %`).
3. **Effet positif des soupes et du centrage $\mu+\sigma$ :** `SOUP_50_050_50_100` place exactement $\mu+\sigma = 3.84$ (`CE = 1.794 ± 2.04`, `p50 = 1.006`, `IntraAcc = 75.5 %`, `InterAcc = 99.9 %`, `W_SEL = 0.5678`), mais il manque encore le resserrement de `ProxyKL <= 0.038` et `dTheta <= 4.4`.
4. **Plan Round 2 :**
   - **(A) Contrôleur resserré (`run_tight_at` dans `run_base_refine.py`) :** abaisser les seuils du contrôleur à `pkl > 0.032` et `d_l2 > 4.2` (pénalité snapshot dès `pkl > 0.030`, `d_l2 > 4.1`), avec `l2_weight in {4e-3, 5e-3, 6e-3}` et `retain_kl_weight in {3.2, 3.5, 3.8}` + 1 ablation isolée `aug_mode="strong"` vs `"moderate"`.
   - **(B) Soupes d'amortissement vers `orig` (`M1`-style `lambda * theta_u + (1 - lambda) * orig` pour `lambda in {0.85, 0.88}`) :** réduit `dTheta` linéairement (`~3.9–4.1`) et `ProxyKL` quadratiquement (`~0.034–0.038`) tout en ramenant `CE` et `p50` de `#9` (`1.91`, `1.21`) droit dans la fenêtre `[1.55, 1.85]` / `[0.65, 1.00]`.
   - **(C) Validation anti-overfitting (`#44`) :** vérifier les top 2 configs sur le split hold-out **`seed=777`** (jamais utilisé pendant la sélection).

### 2026-11-22 (03h30) — Soumission Serveur `#13` (`SOUP_50_050_50_100.pt`, `submission_id: 2706`) = **`0.1238` (NOUVEAU RECORD, `+0.0045`)**
- La soupe 50/50 convexe entre les deux ancres Round 1 (`ANCHOR_ce0.50_b2.0_f6` et `ANCHOR_ce1.00_b1.8_f6`) atteint `Full CE = 1.794 ± 2.04` ($\mu+\sigma = 3.84$), `p50 = 1.006`, `>95% = 0.21`, `ProxyKL = 0.0558`, `dTheta = 4.48` et **bat strictement ses deux parents sur le vrai serveur (`0.1238` vs `0.1099` et `0.1193`)**.
- Confirmation directe de notre fenêtre cible ($\mu+\sigma \approx 3.80\text{–}3.88$, `p50 \approx 0.85\text{–}1.05`, `dTheta <= 4.5`) ; Round 2 vise à combiner ce centrage exact avec **`ProxyKL <= 0.038`** et **`>95% <= 0.20`**.

### 2026-11-22 (04h01) — Round 2 (`--round 2`, Colab A100 `track3_base`) : Resserrement `run_tight_at`, Ablation `strong` & Hold-Out `seed=777`

Exécution de 6 runs `AT2_*` sous contrôleur resserré (`pkl > 0.032` / `d_l2 > 4.2`, `l2_w=4e-3..6e-3`, `kl_w=3.2..3.8`) + 2 ancres (`ANCHOR_ce1.00`, `AT_ce0.80`) + 6 soupes/amortissements `ORIG` (`base_refine_round2.json`) :

| Candidat | `W_SEL ± std` | 3-Folds `F_raw / F_alt` | `Gap50` | `IntraAcc` | `InterAcc (KL)` | Full `CE ± std` ($\mu+\sigma$) | Full `p50 (>95%)` | Full `ProxyKL (AbsKL, dTheta)` | `hits_target` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `AT_ce0.80_b1.9_k20_kl28` | `0.5694 ± 0.020` | `2.44 / 2.57` | `+1.38` | `74.8 %` | `99.9 % (0.0272)` | `1.958 ± 2.09 (4.05)` | `1.256 (0.19)` | `0.0569 (0.0361, 4.58)` | `False` |
| `AT2_ce0.85_b1.8_kl38_l26e3` | `0.5622 ± 0.019` | `2.27 / 2.43` | `+1.27` | `75.6 %` | `99.9 % (0.0257)` | `1.397 ± 1.94 (3.34)` | `0.415 (0.36)` | `0.0469 (0.0298, 3.36)` | `False` |
| **`SHRINK_88_AT9`** | `0.5602 ± 0.026` | `1.83 / 1.98` | `+1.12` | **`82.0 %`** | `99.9 % (0.0212)` | `1.212 ± 1.71 (2.92)` | `0.373 (0.32)` | **`0.0353`** `(0.0219, 4.05)` | `False` |
| `SOUP_90_AT2mix_10_ORIG` | `0.5591 ± 0.022` | `2.00 / 2.15` | `+1.15` | `77.7 %` | `99.9 % (0.0242)` | `0.863 ± 1.45 (2.31)` | `0.129 (0.44)` | **`0.0318`** `(0.0199, 3.14)` | `False` |
| `ANCHOR_ce1.00_b1.8_f6` (`#9`) | `0.5565 ± 0.019` | `2.61 / 2.74` | `+1.47` | `73.3 %` | `99.8 % (0.0312)` | `1.909 ± 2.05 (3.96)` | `1.214 (0.19)` | `0.0486 (0.0304, 4.61)` | `False` |
| `AT2_ce0.75_b1.9_kl32_l24e3` | `0.5558 ± 0.024` | `2.61 / 2.74` | `+1.43` | `70.9 %` | `99.8 % (0.0314)` | `1.363 ± 1.78 (3.14)` | `0.524 (0.29)` | `0.0433 (0.0273, 3.61)` | `False` |
| `AT2_ce0.85_b1.9_kl35_l25e3` | `0.5556 ± 0.021` | `2.54 / 2.68` | `+1.38` | `71.9 %` | `99.9 % (0.0309)` | `1.263 ± 1.78 (3.04)` | `0.364 (0.35)` | `0.0443 (0.0281, 3.34)` | `False` |
| `SOUP_50_AT2ce75_50_AT2ce95` | `0.5536 ± 0.020` | `2.64 / 2.78` | `+1.43` | `70.5 %` | `99.8 % (0.0323)` | `1.275 ± 1.76 (3.03)` | `0.398 (0.32)` | `0.0418 (0.0264, 3.45)` | `False` |
| `SOUP_45_AT2kl38_45_AT9_10_ORIG` | `0.5525 ± 0.028` | `1.77 / 1.93` | `+1.06` | `81.3 %` | `99.9 % (0.0202)` | `1.026 ± 1.59 (2.62)` | `0.217 (0.39)` | **`0.0334`** `(0.0207, 3.48)` | `False` |
| `SHRINK_85_AT9` | `0.5507 ± 0.030` | `1.65 / 1.80` | `+1.03` | `83.8 %` | `99.9 % (0.0192)` | `1.053 ± 1.60 (2.65)` | `0.255 (0.36)` | **`0.0325`** `(0.0201, 3.92)` | `False` |
| `AT2_ce0.95_b1.8_kl35_l25e3` | `0.5452 ± 0.025` | `2.70 / 2.84` | `+1.44` | `69.7 %` | `99.8 % (0.0337)` | `1.234 ± 1.78 (3.02)` | `0.326 (0.37)` | `0.0408 (0.0259, 3.32)` | `False` |
| `AT2_ce0.85_b1.8_kl32_l24e3` | `0.5446 ± 0.026` | `2.78 / 2.90` | `+1.49` | `69.2 %` | `99.7 % (0.0344)` | `1.460 ± 1.84 (3.30)` | `0.611 (0.26)` | `0.0452 (0.0286, 3.69)` | `False` |
| `SHRINK_85_R1best` | `0.5412 ± 0.025` | `1.51 / 1.66` | `+0.95` | `84.6 %` | `99.9 % (0.0169)` | `1.111 ± 1.65 (2.76)` | `0.271 (0.35)` | **`0.0374`** `(0.0234, 3.90)` | `False` |
| `AT2_ce0.85_b1.9_kl35_l25e3_strong` | `0.4703 ± 0.035` | `1.80 / 2.26` | `+0.79` | `70.0 %` | `99.5 % (0.0486)` | `0.362 ± 1.08 (1.44)` | `0.007 (0.71)` | `0.0392 (0.0222, 3.20)` | `False` |

- **Hold-out `seed=777` :** `AT2_ce0.85_b1.8_kl38_l26e3` (`W_SEL = 0.5450 ± 0.035`, `In = 67.9 %`), **`SHRINK_88_AT9` (`W_SEL = 0.5800 ± 0.006`, `In = 79.1 %`, `Out = 99.9 %`, `InterKL = 0.0169`)**.
- **3 enseignements causaux Round 2 :**
  1. **Ablation 1-facteur `aug_mode="strong"` vs `"moderate"` :** `strong` s'effondre (`Full CE = 0.362`, `>95% = 0.71`, `W_SEL = 0.4703` vs `0.5556`) — conserver strictement `aug_mode="moderate"`.
  2. **Sur-freinage de `run_tight_at` (`kl >= 3.2`, `l2 >= 4e-3`, `pkl=0.032`) :** brise `dTheta` à `3.32–3.69` (`mu+sig = 3.02–3.34`), mais conserve `ProxyKL = 0.0408–0.0469` car 210 steps à fort `kl_w` surfitent le proxy `seed=42`.
  3. **Supériorité géométrique des soupes / amortissements :** `SHRINK_88_AT9` atteint `ProxyKL = 0.0353` (`IntraAcc = 82.0 %`, `seed=777 W_SEL = 0.5800 ± 0.006`), mais `0.88` recule trop le `CE` (`1.91 -> 1.21`).
- **Plan Round 3 :**
  - Combiner **soupe multi-graines (`seed in {101, 202, 303, 42}`)** à mi-régime (`pkl_limit=0.039, d_l2_limit=4.35, l2_w=2.8e-3, kl_w=2.8`) pour annuler le bruit `seed=42` sur `ProxyKL` tout en gardant $\mu+\sigma \in [3.65, 3.88]$, plus soupes fines `AT9 + AT2` (`65/35`, `75/25`) et `SHRINK_93/95_AT9`.

### 2026-11-22 (04h42) — Round 3 (`--round 3`, Colab A100 `track3_base`) : Soupes Multi-Graines & Centrage Exact $\mu+\sigma \in [3.68, 3.77]$

Exécution de 4 runs mi-régime multi-graines `AT3_*` (`pkl_limit=0.039, d_l2_limit=4.35`) + 2 ancres (`ANCHOR_ce1.00`, `AT2_ce0.85`) + 8 soupes convexes (`base_refine_round3.json`) :

| Candidat | `W_SEL ± std` | 3-Folds `F_raw / F_alt` | `Gap50` | `IntraAcc` | `InterAcc (KL)` | Full `CE ± std` ($\mu+\sigma$) | Full `p50 (>95%)` | Full `ProxyKL (AbsKL, dTheta)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `SOUP3_multiseed_3way` | `0.5742 ± 0.024` | `2.41 / 2.54` | `+1.37` | `75.2 %` | `99.9 % (0.0248)` | `1.341 ± 1.81 (3.15)` | `0.466 (0.31)` | `0.0440 (0.0267, 3.56)` |
| `SOUP3_95_multiseed_05_ORIG` | `0.5736 ± 0.015` | `2.08 / 2.23` | `+1.22` | `78.4 %` | `99.9 % (0.0213)` | `1.104 ± 1.65 (2.75)` | `0.275 (0.36)` | `0.0381 (0.0231, 3.39)` |
| `SOUP3_4way_with_AT9` | `0.5717 ± 0.021` | `2.39 / 2.53` | `+1.36` | `75.1 %` | `99.9 % (0.0253)` | `1.353 ± 1.79 (3.14)` | `0.509 (0.28)` | `0.0414 (0.0250, 3.58)` |
| `SHRINK_93_AT9` | `0.5709 ± 0.022` | `2.15 / 2.29` | `+1.27` | `78.4 %` | `99.9 % (0.0250)` | `1.493 ± 1.87 (3.37)` | `0.671 (0.26)` | `0.0404 (0.0251, 4.29)` |
| `SHRINK_95_AT9` | `0.5708 ± 0.020` | `2.28 / 2.42` | `+1.33` | `76.2 %` | `99.9 % (0.0266)` | `1.610 ± 1.93 (3.54)` | `0.819 (0.23)` | `0.0426 (0.0265, 4.38)` |
| **`SOUP3_50_AT9_50_s101`** | `0.5680 ± 0.020` | `2.49 / 2.61` | `+1.40` | `74.2 %` | `99.9 % (0.0274)` | **`1.673 ± 2.01 (3.68)`** | **`0.861 (0.25)`** | **`0.0439 (0.0269, 3.96)`** |
| `AT3_s303_ce0.85_b1.9_kl30` | `0.5668 ± 0.032` | `2.45 / 2.61` | `+1.43` | `75.3 %` | `99.9 % (0.0267)` | `2.688 ± 3.33 (6.02)` | `1.286 (0.42)` | `0.0578 (0.0334, 4.12)` |
| `AT3_s101_ce0.90_b1.8_kl28` | `0.5634 ± 0.024` | `2.63 / 2.74` | `+1.46` | `73.6 %` | `99.8 % (0.0281)` | `1.856 ± 2.20 (4.05)` | `0.986 (0.28)` | `0.0540 (0.0336, 4.20)` |
| `AT3_s202_ce0.95_b1.8_kl26` | `0.5593 ± 0.020` | `2.63 / 2.71` | `+1.46` | `73.5 %` | `99.9 % (0.0272)` | `1.587 ± 1.93 (3.52)` | `0.739 (0.26)` | `0.0541 (0.0329, 4.13)` |
| `AT3_s42_ce0.90_b1.8_kl28` | `0.5578 ± 0.018` | `2.64 / 2.77` | `+1.47` | `72.7 %` | `99.8 % (0.0301)` | `1.612 ± 1.89 (3.50)` | `0.864 (0.23)` | `0.0453 (0.0284, 3.91)` |
| `ANCHOR_ce1.00_b1.8_f6` (`#9`) | `0.5565 ± 0.019` | `2.61 / 2.74` | `+1.47` | `73.3 %` | `99.8 % (0.0312)` | `1.909 ± 2.05 (3.96)` | `1.214 (0.19)` | `0.0486 (0.0304, 4.61)` |
| **`SOUP3_75_AT9_25_AT2`** | `0.5553 ± 0.019` | `2.63 / 2.76` | `+1.46` | `72.7 %` | `99.8 % (0.0316)` | **`1.773 ± 1.99 (3.77)`** | **`1.038 (0.21)`** | **`0.0462 (0.0288, 4.32)`** |
| **`SOUP3_65_AT9_35_AT2`** | `0.5550 ± 0.020` | `2.64 / 2.77` | `+1.46` | `72.4 %` | `99.8 % (0.0318)` | **`1.723 ± 1.97 (3.69)`** | **`0.976 (0.22)`** | **`0.0455 (0.0284, 4.22)`** |
| `AT2_ce0.85_b1.8_kl32_l24e3` | `0.5446 ± 0.026` | `2.78 / 2.90` | `+1.49` | `69.2 %` | `99.7 % (0.0344)` | `1.460 ± 1.84 (3.30)` | `0.611 (0.26)` | `0.0452 (0.0286, 3.69)` |

- **Hold-out `seed=777` :** `AT3_s303_ce0.85_b1.9_kl30` (`W_SEL = 0.5487 ± 0.040`, `In = 69.2 %`), `SOUP3_multiseed_3way` (`W_SEL = 0.5428 ± 0.036`, `In = 67.7 %`, `Out = 99.9 %`).
- **3 candidats prêts dans la zone d'or serveur (`#85` `mu+sig ∈ [3.65, 3.90]`, `p50 ∈ [0.85, 1.15]`) battant `#13` (`0.1238`) sur `ProxyKL` (`0.0439–0.0462` vs `0.0558`) et `dTheta` (`3.96–4.32` vs `4.48`) :**
  1. `SOUP3_75_AT9_25_AT2.pt` (`mu+sig=3.77, p50=1.038, >95%=0.21, ProxyKL=0.0462, dTheta=4.32`)
  2. `SOUP3_65_AT9_35_AT2.pt` (`mu+sig=3.69, p50=0.976, >95%=0.22, ProxyKL=0.0455, dTheta=4.22`)
  3. `SOUP3_50_AT9_50_s101.pt` (`mu+sig=3.68, p50=0.861, >95%=0.25, ProxyKL=0.0439, dTheta=3.96`)

### 2026-11-22 (04h44) — Soumissions `#15` (`0.1082`) & `#17` (`0.1176`) : Preuve que le Goulot est `prediction_score` (Pas `ProxyKL`)
- **`#15` (`SHRINK_88_AT9.pt`, `submission_id: 2818`) = `0.1082`** : confirme qu'en dessous de $\mu+\sigma < 3.5$ (`2.92`), même `ProxyKL = 0.0353` et `seed=777 W_SEL = 0.5800` retombent sur la pente gauche.
- **`#17` (`SOUP3_75_AT9_25_AT2.pt`, `submission_id: 2951`) = `0.1176`** : malgré `-17 %` de `ProxyKL` (`0.0462` vs `0.0558`) et `dTheta` réduit (`4.32` vs `4.48`) à $\mu+\sigma$ quasi-identique (`3.77` vs `3.84`), `#17` perd `-0.0062` face à `#13` (`0.1238`). **Conclusion :** `ProxyKL <= 0.056` et `dTheta <= 4.5` ne sont pas le goulot ; le vrai levier pour dépasser `0.1238` est **`prediction_score`** (alignement image par image `rho_PD` / Patch-Dropout & soupes hétérogènes sans sur-freinage `L2/KL`).

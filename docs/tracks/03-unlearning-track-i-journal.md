# Track 3 — Piste I : Journal Spécialisé (*Erase-Then-Heal* : Oubli Extrême à 100 % puis Réparation des Liaisons par LoRA)

> **Journal dédié à l'agent autonome `i-lora` (`Gemini 4 Hydrogen (High)`)** explorant la **Piste I** : partir d'un modèle ayant oublié à 100 % les images (`>95% -> 0.00`), puis ré-injecter les images dans une **LoRA de très faible rang ($r \in \{4, 8\}$)** avec un objectif purement structurel et relationnel pour réparer les liaisons du réseau sans ré-apprendre la connaissance spécifique par cœur.

**Liens connexes :**
- Journal principal Track 3 : [03-unlearning-journal.md](03-unlearning-journal.md)
- Journal Piste H : [03-unlearning-track-h-journal.md](03-unlearning-track-h-journal.md)
- Protocole de benchmark 3-Folds pondéré : [selective_benchmark.py](../../tracks/03_unlearning_v2/selective_benchmark.py)

---

## 1. Références à battre (Benchmark Pondéré 3-Folds `W_SEL`)

| Référence | **Score Pondéré `W_SEL ± std`** | **3-Folds `F_raw` / `F_alt`** | **3-Folds `Gap50`** | **3-Folds `IntraAcc`** | **3-Folds `InterAcc` (`InterKL`)** | **Full `.pt` `CE mean ± std` ($\mu+\sigma$)** | **Full `p50` (`>95%`)** | **Full `ProxyKL`** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`AT_ce1.00_b1.8_f6` (Serveur `#9` = `0.1193`)** | `0.5565 ± 0.019` | `2.61` / `2.74` | `+1.47` | `73.3 %` | `99.8 %` (`0.0312`) | `1.909 ± 2.05` (`3.96`) | `1.214` (`0.19`) | `0.0486` |
| **`H1_h1h2_f8_ce125` (Record Piste H)** | **`0.6041 ± 0.018`** | `3.09` / `3.18` | **`+2.06`** | `76.8 %` | `99.9 %` (`0.0199`) | `1.676 ± 1.93` (`3.61`) | `0.912` (`0.19`) | **`0.0325`** |
| **`H2_h123_f8_ce145` (Record IntraAcc Piste H)** | **`0.6023 ± 0.026`** | `2.91` / `3.00` | `+1.94` | **`78.2 %`** | `99.9 %` (`0.0215`) | `1.665 ± 1.79` (`3.46`) | `1.010` (**`0.15`**) | **`0.0324`** |

---

## 2. Architecture Mathématique de la Piste I (*Erase-Then-Heal via Bottlenecked LoRA*)

### Phase 1 — *Total Erasure* ($\theta \to \theta_{\text{extreme}}$)
- Pousser l'oubli sur $\mathcal{D}_f$ (via SalUn layer-wise `20–25 %` + NPO fort avec boost sur les images têtues `ce_i < 0.25`, `target_forget_ce` $\in [2.2, 3.0]$ ou portes `IF/THEN` modérées) jusqu'à ce que la fraction d'images saturées tombe à **`frac_gt_95 <= 0.05`** (`CE mean ~ 3.0–4.5`).
- À cet extrême, **toutes** les traces mémorisées individuelles sont brisées, mais `IntraAcc` est descendue vers `~50–60 %` et `ProxyKL` est monté vers `~0.08–0.12`.

### Phase 2 — *Structural LoRA Re-binding without Instance Knowledge* ($\theta_{\text{extreme}} + B A \to \theta_{\text{healed}}$)
- Geler complètement $\theta_{\text{extreme}}$ et `head.weight` (avec restauration exacte des 55 classes absentes `restore_absent_head_rows`).
- Injecter des adaptateurs **LoRA FP32 de très faible rang $r \in \{4, 8\}$** sur `blocks.6..11` (ou `blocks.8..11`), fusionnables en fin de Phase 2 via `merge_to_linear()`.
- Ré-injecter les images de $\mathcal{D}_f$ (et leurs mosaïques `PatchMix 4×4 + 7×7`) pendant `60 à 90` steps avec **3 pertes de réparation des liaisons SANS ré-apprentissage du par-cœur** :
  1. **Liaison de structure globale (`loss_proxy_kl`) :** `KL(p_orig || p_sub)` à $T = 2.0$ sur les mosaïques `PatchMix` pour ressouder les connexions d'attention et les 55 classes absentes.
  2. **Liaison inter-classes *Dark Knowledge* (`loss_dark_liaison`) :** sur les images de $\mathcal{D}_f$, masquer le logit top-1 ($y_i$) dans `orig_logits` et `sub_logits`, et aligner **uniquement la distribution relative des 99 autres classes à $T = 3.0$** ($D_{\text{KL}}(p_{\text{orig}, c \neq y_i}^{(T=3)} \parallel p_{\text{sub}, c \neq y_i}^{(T=3)})$) ! Cela répare toutes les liaisons sémantiques entre classes proches sans remonter la sur-confiance sur $y_i$.
  3. **Rappel plafonné vers une moyenne saine (`loss_ceiling_heal`) :** pour les images de $\mathcal{D}_f$ qui ont été *trop* détruites en Phase 1 ($\text{CE}_i > u_{\text{heal}}$, avec $u_{\text{heal}} \in [1.40, 2.00]$), appliquer un rappel doux vers un logit top-1 plafonné à $p^* \in [0.42, 0.55]$ ($\text{CE}^* \approx 0.65\text{--}0.85$), mais **mettre le gradient top-1 strictement à `0` dès que $\text{CE}_i \le u_{\text{heal}}$** ! Ainsi, aucune image ne peut redescendre vers `CE < 0.60` (`>95%` reste écrasé sous `0.08` !).

---

## 3. Règles strictes d'isolation de `i-lora`

1. **Fichiers autorisés en écriture :**
   - `tracks/03_unlearning_v2/methods/track_i.py`
   - `tracks/03_unlearning_v2/run_track_i.py`
   - `tracks/03_unlearning_v2/tests/test_track_i.py`
   - `tracks/03_unlearning_v2/output/track_i/`
   - Ce journal : `docs/tracks/03-unlearning-track-i-journal.md`
2. **Session Colab dédiée :**
   - Utiliser exclusivement **`-s track3_i`** (`/home/chedli/.local/bin/colab new -s track3_i --gpu A100`), avec un *heartbeat* stdout toutes les 15 s, téléchargement des JSON et meilleurs `.pt` dans `tracks/03_unlearning_v2/output/track_i/`, puis **`/home/chedli/.local/bin/colab stop -s track3_i`** systématique.
   - Évaluer tous les candidats avec le juge 3-Folds pondéré exact de `selective_benchmark.py` (`make_rotating_class_folds`, `score_single_fold`, `aggregate_weighted_folds`).

---

## 4. Journal Chronologique des Expériences Piste I

### 2026-11-22 (01h53) — Lancement de l'agent autonome `i-lora`
- Création du journal dédié et lancement de `i-lora` (`Gemini 4 Hydrogen (High)`).

### 2026-11-22 (02h15) — Round 1 Piste I (`track_i_round1.json`) : Phase 1 validée (`>95% = 0.005–0.039`), sur-guérison en Phase 2

| Candidat Round 1 | **Score Pondéré `W_SEL ± std`** | **3-Folds `F_raw` / `F_alt`** | **3-Folds `Gap50`** | **3-Folds `IntraAcc`** | **3-Folds `InterAcc` (`InterKL`)** | **Full `.pt` `CE mean ± std` ($\mu+\sigma$)** | **Full `p50` (`>95%`)** | **Full `ProxyKL` (`AbsKL`, `dTheta`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `I3_f8_r4_ce26_h165` | `0.1110 ± 0.013` | `0.16` / `0.18` | `+0.13` | `99.89 %` | `100.0 %` (`0.0081`) | `0.0884 ± 0.41` (`0.50`) | `0.0015` (`0.852`) | `0.0725` (`0.0512`, `14.77`) |
| `I1_f8_r8_ce24_h150` | `0.0859 ± 0.025` | `0.13` / `0.15` | `+0.10` | `99.88 %` | `100.0 %` (`0.0081`) | `0.0658 ± 0.33` (`0.39`) | `0.0009` (`0.886`) | `0.0647` (`0.0454`, `13.49`) |
| `I2_f6_r8_ce25_h140` | `0.0778 ± 0.017` | `0.11` / `0.12` | `+0.09` | `99.94 %` | `100.0 %` (`0.0088`) | `0.0836 ± 0.41` (`0.49`) | `0.0012` (`0.861`) | `0.0755` (`0.0523`, `12.49`) |
| `I4_f6_r4_ce28_h135` | `0.0667 ± 0.015` | `0.08` / `0.11` | `+0.07` | `100.0 %` | `100.0 %` (`0.0061`) | `0.1189 ± 0.49` (`0.61`) | `0.0025` (`0.817`) | `0.0873` (`0.0616`, `13.11`) |

**Diagnostic mathématique du Round 1 :**
1. **Phase 1 (*Extreme Erasure*) validée à 100 % :** Sur `I1` (et `I1–I4`), la phase 1 détruit quasi intégralement la mémorisation (`>95% = 0.005–0.039`, `I1` : `CE = 3.033`, `p50 = 2.911`, `>95% = 0.009`).
2. **Pourquoi Phase 2 a sur-guéri (`CE = 0.066–0.119`, `>95% = 0.817–0.886`, `dTheta = 12.49–14.77`) :**
   - Même si `loss_dark_liaison` masque le logit $y_i$ dans le KL et si `loss_ceiling_heal` s'annule pour $\text{CE}_i \le u_{\text{heal}}$, **`loss_proxy_kl` (`PatchMix`) et la restauration des features partagées dans `blocks.6..11`** ré-alignent les représentations internes consommées par le `head` gelé $\implies$ $z_{y_i}$ remonte mécaniquement et ré-effondre $\text{CE}_i$ vers `0.001` en l'absence d'une **barrière active de plancher** ($\text{CE}_i < \text{heal\_ce\_floor}$).
   - De plus, `heal_lr` (`4.0e-4` à `5.5e-4`) sur `70–80` steps avec `heal_l2_weight = 1e-4` a propulsé `dTheta` à `12.49–14.77` (bien au-delà du budget cible `<= 4.6`) et dégradé `ProxyKL` (`0.065–0.087`).

### 2026-11-22 (02h17) — Lancement Round 2 (`task-109` sur `track3_i` A100) : Couloir Bilatéral + Calibration `#23`
- **Corrections implémentées dans [track_i.py](../../tracks/03_unlearning_v2/methods/track_i.py) et [run_track_i.py](../../tracks/03_unlearning_v2/run_track_i.py) :**
  1. **Barrière plancher active (`compute_floor_guard_loss`) :** pour $\text{CE}_i < \text{heal\_ce\_floor}$, applique un terme `npo_loss(T=3.5, beta=1.2, max_ce=heal_ce_floor) + 0.6 * smooth_l1((heal_ce_floor - ce_i)+)` pondéré par `w_floor_guard in {1.6, 1.8}` afin de maintenir les échantillons dans le couloir `[heal_ce_floor, heal_ce_ceiling]`.
  2. **Contrôle de dérive & vitesse :** `gamma_lr_mult = 1.0`, `phase1_max_ce in [2.9, 3.3]`, `heal_lr in [1.4e-4, 1.8e-4]`, `heal_steps in {50, 55}`, sondes toutes les 5 itérations (`snap = -pkl - 0.30*f95 - 0.15*|p50 - 0.90|`).
  3. **4 candidats `I2R_*` lancés sur `track3_i` :** `I2R_f8_r8_ce19_fl085`, `I2R_f8_r4_ce20_fl095`, `I2R_f8_r4_ce16_fl075`, `I2R_f6_r4_ce18_fl085`.

### 2026-11-22 (02h37) — Résultats Round 2 Piste I (`track_i_round2.json`) vs Baselines

> **Note méthodologique (`#44`) :** `W_SEL` est un diagnostic local sur `forget_images.npz` (non un substitut au score serveur officiel ni au `val_acc >= 70 %` sur le *control set* caché). Les deux meilleurs checkpoints Round 2 (`I2R_f8_r4_ce16_fl075.pt`, `I2R_f6_r4_ce18_fl085.pt`) et `track_i_round2.json` ont été rapatriés dans `tracks/03_unlearning_v2/output/track_i/` et `track3_i` a été arrêté proprement (`colab stop -s track3_i`).

| Modèle | **Diagnostic `W_SEL ± std`** | **3-Folds `F_raw` / `F_alt`** | **3-Folds `Gap50`** | **3-Folds `IntraAcc`** | **3-Folds `InterAcc` (`InterKL`)** | **Full `.pt` `CE mean ± std` ($\mu+\sigma$)** | **Full `p50` (`>95%`)** | **Full `ProxyKL` (`AbsKL`, `dTheta`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Baseline* `AT_ce1.00_b1.8_f6` (Serveur `#9` = `0.1193`) | `0.5565 ± 0.019` | `2.61` / `2.74` | `+1.47` | `73.3 %` | `99.8 %` (`0.0312`) | `1.909 ± 2.05` (`3.96`) | `1.214` (`0.19`) | `0.0486` (`0.0304`, `4.61`) |
| *Ref* `H1_h1h2_f8_ce125` (Serveur `#11` = `0.0813`) | `0.6041 ± 0.018` | `3.09` / `3.18` | `+2.06` | `76.8 %` | `99.9 %` (`0.0199`) | `1.676 ± 1.93` (`3.61`) | `0.912` (`0.19`) | `0.0325` (`—`, `5.78`) |
| **`I2R_f8_r4_ce16_fl075`** | **`0.5684 ± 0.025`** | `1.85` / `1.95` | `+1.23` | **`83.0 %`** | `99.9 %` (`0.0226`) | `1.538 ± 1.65` (`3.19`) | `0.949` (`0.160`) | `0.0553` (`0.0371`, `5.39`) |
| `I2R_f8_r4_ce20_fl095` | `0.5417 ± 0.026` | `1.67` / `1.78` | `+1.13` | `84.5 %` | `99.9 %` (`0.0270`) | `1.876 ± 1.54` (`3.41`) | `1.565` (**`0.057`**) | `0.0568` (`0.0381`, `6.44`) |
| `I2R_f8_r8_ce19_fl085` | `0.5304 ± 0.025` | `1.59` / `1.69` | `+1.08` | `85.0 %` | `99.9 %` (`0.0277`) | `1.858 ± 1.49` (`3.35`) | `1.565` (**`0.053`**) | `0.0513` (`0.0351`, `6.58`) |
| `I2R_f6_r4_ce18_fl085` | `0.5252 ± 0.024` | `1.36` / `1.47` | `+0.95` | **`88.5 %`** | `99.9 %` (`0.0223`) | `2.458 ± 1.98` (`4.44`) | `2.136` (**`0.077`**) | `0.0740` (`0.0522`, `6.01`) |

**Analyse honnête du Round 2 :**
1. **Validation locale du couloir bilatéral (`compute_floor_guard_loss`) :** La barrière plancher empêche la ré-mémorisation constatée au Round 1 (`>95%` reste écrasé à `0.053–0.077` sur `ce18–ce20` et `0.160` sur `ce16`, au lieu de `0.82–0.89`), tout en remontant `IntraAcc` à **`83.0–88.5 %`** (vs `73.3 %` sur `AT_ce1.00_b1.8_f6`) et `W_SEL` de `0.1110` à **`0.5684`**.
2. **Écarts encore non résolus par rapport au régime serveur `#9` (`0.1193`) :**
   - `ProxyKL` (`0.0513–0.0740`) et `dTheta` (`5.39–6.58`) restent plus élevés que `AT_ce1.00_b1.8_f6` (`ProxyKL = 0.0486`, `dTheta = 4.61`) et hors de la cible `ProxyKL <= 0.035`, `dTheta <= 4.6`.
   - Round 2 tournait encore avec `saliency_refresh_frac = 0.5`, `head_lr_mult = 0.45–0.50`, `alpha_proto = 0.80–0.82`, et `f8` sur 3 candidats sur 4 (antérieurs à l'audit `#32`/`#34`).
   - Sur `I2R_f6_r4_ce18_fl085`, `Full CE = 2.458` (`p50 = 2.136`) a dépassé le plafond cible (`<= 1.00`), et aucun run n'a encore été évalué sur une graine hors-échantillon (`seed=777`).

- **Alerte serveur `@coordinator #32` (Soumission `#11` `H1_h1h2_f8_ce125` = `0.0813`) :** 3 règles critiques identifiées pour le Round 3 de la Piste I :
  1. Ne **jamais rafraîchir/rotater le masque SalUn** en Phase 1 (`saliency_refresh_frac = 0.0` au lieu de `0.5`, sinon >35 % des poids sont touchés et les poids nouvellement démasqués n'ont plus d'ancre dans `orig_params`).
  2. Ne pas brider `head.weight` en Phase 1 (`head_lr_mult = 1.0`, `alpha_proto in [0.15, 0.30]`) pour éviter de tordre le backbone et garder `dTheta <= 4.5`.
  3. Toujours utiliser **`freeze_until_block = 6` (`f6`) uniquement** (toutes les soumissions `f8` ont sous-performé `<= 0.0813` vs `f6` à `0.0984–0.1193`).

### 2026-11-22 (03h12) — Retour Serveur `#12` (`I2R_f6_r4_ce18_fl085.pt` = `0.0950`) & Lancement Round 3 (`seed=777`)
- **Soumission serveur `#12` (`submission_id: 2648`, `I2R_f6_r4_ce18_fl085.pt`) :** **`0.0950`** (bat `#11` `0.0813` et `#2` `0.0843` malgré `Full CE = 2.458 ± 1.98`, $\mu+\sigma = 4.44$, `p50 = 2.136`, `ProxyKL = 0.0740`, `dTheta = 6.01`). Confirme le transfert serveur d'*Erase-Then-Heal LoRA* et la pente droite de la courbe en cloche (`#9` `3.96 -> 0.1193`, `#10` `4.23 -> 0.0984`, `#12` `4.44 -> 0.0950`).
- **Round 3 exécuté sur `track3_i` (`L4`, `--round 3 --fold-seed 777`, `alt_seed=877`, `track_i_round3.json`) :**
  - Ablation 1-facteur-à-la-fois strictement `f6` avec masque SalUn statique (`saliency_refresh_frac = 0.0`), `head_lr_mult = 1.0`, `alpha_proto = 0.20`, et ancre L2 sur poids fusionnés `compute_merged_lora_l2_loss`.

| Modèle (`fold_seed=777`) | **Diagnostic `W_SEL ± std`** | **3-Folds `F_raw` / `F_alt`** | **3-Folds `Gap50`** | **3-Folds `IntraAcc`** | **3-Folds `InterAcc` (`InterKL`)** | **Full `.pt` `CE mean ± std` ($\mu+\sigma$)** | **Full `p50` (`>95%`)** | **Full `ProxyKL` (`AbsKL`, `dTheta`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Ref `seed=777`* `ABL_0_baseline_AT` (`#9`=`0.1193`) | `0.5505 ± 0.028` | `2.82` / `2.99` | `+1.47` | `69.5 %` | `99.9 %` (`0.0246`) | `1.909 ± 2.05` (`3.96`) | `1.214` (`0.19`) | `0.0486` (`0.0304`, `4.61`) |
| `I3R_1_p1only_f6` (Phase 1 seule, `heal_steps=0`) | **`0.5833 ± 0.006`** | `2.28` / `2.42` | `+1.37` | `75.3 %` | `99.9 %` (`0.0230`) | `2.829 ± 2.00` (`4.83`) | `2.607` (**`0.053`**) | `0.1159` (`0.0778`, `5.78`) |
| `I3R_2_lora_r4_c095_160` (`+ LoRA r=4 [0.95,1.60]`) | `0.5717 ± 0.007` | `1.91` / `2.05` | `+1.21` | `80.2 %` | `100.0 %` (`0.0208`) | `2.495 ± 1.94` (`4.44`) | `2.229` (**`0.074`**) | `0.0773` (`0.0527`, `5.79`) |
| `I3R_3_lora_r4_c105_170` (`+ Couloir [1.05,1.70]`) | `0.5752 ± 0.007` | `2.00` / `2.14` | `+1.25` | `79.0 %` | `99.9 %` (`0.0220`) | `2.535 ± 1.95` (`4.49`) | `2.286` (**`0.072`**) | `0.0788` (`0.0536`, `5.79`) |
| `I3R_4_lora_r8_c095_160` (`+ LoRA r=8 [0.95,1.60]`) | `0.5507 ± 0.008` | `1.67` / `1.78` | `+1.09` | **`83.0 %`** | `100.0 %` (`0.0201`) | `2.243 ± 1.88` (`4.12`) | `1.932` (**`0.092`**) | `0.0658` (`0.0455`, `5.80`) |

**Bilan causal propre du Round 3 (`fold_seed=777`, `alt_seed=877`) :**
1. **Effet isolé de Phase 2 LoRA (sans changer Phase 1) :**
   - `r=4` (`I3R_2` vs `I3R_1`) réduit `ProxyKL` de **`-33.3 %`** (`0.1159 -> 0.0773`), `AbsKL` de **`-32.3 %`** (`0.0778 -> 0.0527`), augmente `IntraAcc` de **`+4.9 pts`** (`75.3 % -> 80.2 %`) et baisse `InterKL` (`0.0230 -> 0.0208`), tout en gardant `>95% = 0.074`.
   - `r=8` (`I3R_4` vs `I3R_1`) réduit `ProxyKL` de **`-43.2 %`** (`0.1159 -> 0.0658`), `AbsKL` de **`-41.5 %`** (`0.0778 -> 0.0455`), augmente `IntraAcc` de **`+7.7 pts`** (`75.3 % -> 83.0 %`), ramène $\mu+\sigma$ de `4.83` à **`4.12`** (`p50 = 1.932`), tout en gardant `>95% = 0.092`.
2. **Pourquoi `ProxyKL` (`0.066–0.079`) et `dTheta` (`5.79–5.80`) n'ont pas encore atteint la fenêtre serveur `#13` (`mu+sig ~ 3.84`, `p50 ~ 1.0`, `dTheta <= 4.5`) :**
   - Phase 1 (`I3R_1`) démarre déjà très haut (`CE = 2.829`, `p50 = 2.607`, `ProxyKL = 0.1159`, `dTheta = 5.78`) et `45` steps de LoRA à `heal_lr = 1.2e-4` ne suffisent pas à ramener `dTheta` de `5.78` à `<= 4.5` ni `mu+sig` à `3.84` (et `head` est gelé aux poids déplacés de Phase 1).
   - **Prochaine hypothèse directe si Round 4 demandé :** soit démarrer Phase 1 plus près de la cible (`target_forget_ce=1.10`, `phase1_max_ce=2.0`, `dTheta <= 4.3`), soit prendre un **soup convexe** entre `orig_model` / `#8` (`0.50`) et `I3R_4` / `I3R_3` pour placer exactement $\mu+\sigma \in [3.65, 3.90]$ avec `>95% <= 0.12` et `dTheta <= 4.5`.

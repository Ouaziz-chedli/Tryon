# Track 3 — Piste H : Journal Spécialisé (Pilotage Algorithmique Temps Réel & Circuits `IF/THEN` In-Weights)

> **Journal dédié au binôme autonome `h-lead` (Orchestrateur/Chercheur) & `h-exec` (Exécuteur/GPU)** explorant la **Piste H** (hybridation algorithmie + tweaking des poids en temps réel et compilation de portes associatives `IF/THEN` dans les neurones MLP dormants du ViT-B/16).

**Liens connexes :**
- Journal principal Track 3 : [03-unlearning-journal.md](03-unlearning-journal.md)
- Protocole de benchmark 3-Folds pondéré : [selective_benchmark.py](../../tracks/03_unlearning_v2/selective_benchmark.py)

---

## 1. Référence à battre (Baseline issue du Benchmark Pondéré 3-Folds « Oubli Fort »)

Tous les candidats de la **Piste H** doivent être évalués avec **exactement le même juge impartial** (`make_rotating_class_folds`, `score_single_fold`, `aggregate_weighted_folds` de [selective_benchmark.py](../../tracks/03_unlearning_v2/selective_benchmark.py)) sur les 3 folds de 15 classes actives (`seed=42`, exact + altéré) + évaluation du checkpoint complet (`3 500` images) :

| Référence actuelle | **Score Pondéré `W_SEL ± std`** | **3-Folds `F_raw` / `F_alt`** | **3-Folds `Gap50`** | **3-Folds `IntraAcc`** | **3-Folds `InterAcc` (`InterKL`)** | **Full `.pt` `CE mean ± std` ($\mu+\sigma$)** | **Full `p50` (`>95%`)** | **Full `ProxyKL` (`dTheta`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`AT_ce1.00_b1.8_f6` (Champion #1)** | **`0.5565 ± 0.019`** | **`2.61` / `2.74`** | **`+1.47`** | **`73.3 %`** | **`99.8 %` (`0.0312`)** | **`1.909 ± 2.05` (`3.96`)** | **`1.214` (`0.19`)** | **`0.0486` (`4.61`)** |
| **`AT_ce1.40_b1.5_f8` (Champion #2)** | **`0.5451 ± 0.006`** | **`3.24` / `3.37`** | **`+1.69`** | `63.8 %` | **`99.9 %` (`0.0226`)** | **`2.547 ± 3.00` (`5.55`)** | **`1.390` (`0.31`)** | **`0.0422` (`4.56`)** |

**Objectif Piste H :** dépasser **`W_SEL > 0.5600`** en combinant la puissance d'oubli de `AT_ce1.40` ($\mu+\sigma \in [4.5, 5.8]$, `Gap50 >= +1.65`) avec une préservation Intra-Classe **`IntraAcc >= 73 %`** et Inter-Classes **`InterAcc >= 99.5 %` (`InterKL <= 0.030`)**.

---

## 2. Les 4 Axes Algorithmiques de la Piste H à expérimenter

1. **H1 — Projection Orthogonale Instance vs Classe en Temps Réel ($\nabla_{\text{instance}} = \nabla_i - \alpha_{\text{ortho}} \bar{\nabla}_c$) :**
   - Dans chaque batch d'entraînement, décomposer le gradient de la loss `forget` sur `head.weight` et `blocks.8..11` par classe active $c$ : soustraire une fraction $\alpha_{\text{ortho}} \in [0.35, 0.75]$ du gradient moyen de la classe $\bar{\nabla}_c$ afin d'empêcher l'effondrement du prototype de classe (*class collapse*, qui fait chuter `IntraAcc` de `73 %` à `63 %` quand on pousse `target_forget_ce` à `1.40`).
2. **H2 — Conductance Dynamique Temps Réel par Poids $g_i(t)$ (*Online Fisher / SalUn Recalibration & Rollback*) :**
   - Remplacer ou moduler le masque SalUn statique ($t=0$) par un ratio de conductance recalculé en temps réel tous les $K=15$ steps :
     $$g_i(t) = \frac{\text{EMA}(|\nabla_{\theta_i} \mathcal{L}_f|)}{\text{EMA}(|\nabla_{\theta_i} \mathcal{L}_r|) + \epsilon}$$
   - Si un poids dérive trop fort sur le `retain` ($g_i(t) < \tau_{\text{lock}}$), couper son gradient et effectuer un rollback partiel vers $\theta_{\text{orig}, i}$.
3. **H3 — Tweaking Algorithmique Direct sur `LayerScale .gamma` & Calibration par Classe Active :**
   - Ajuster en temps réel la vitesse par classe $\eta_c(t)$ selon l'écart entre la médiane `CE` courante de la classe $c$ et `target_forget_ce` (pour que toutes les 45 classes actives atteignent la cible de façon homogène sans qu'une classe facile n'explose).
4. **H4 — Compilation d'un Circuit `IF / THEN` dans les Neurones MLP Dormants (`blocks.10..11`, `fc1 + GELU + fc2`) :**
   - Identifier dans `blocks.10` et `blocks.11` les $M \in [45, 90]$ neurones intermédiaires MLP (`3 072` au total) ayant la plus faible activation/saillance sur le proxy `retain`.
   - Programmer leurs lignes `fc1.weight[k]` + seuil `fc1.bias[k] = -\tau_k` sur les centroïdes normalisés des clusters d'images `forget` (`IF` : le neurone ne s'allume après `GELU` que si la similarité cosinus avec une image `forget` dépasse $\tau_k$, restant à `0` sur le `retain`), et programmer leurs colonnes `fc2.weight[:, k]` pour atténuer la direction du prototype top-1 (`THEN`).

---

## 3. Règles strictes d'isolation du binôme (`h-lead` & `h-exec`)

1. **Fichiers autorisés en écriture :**
   - Code des méthodes : `tracks/03_unlearning_v2/methods/track_h.py`
   - Script de benchmark dédié : `tracks/03_unlearning_v2/run_track_h.py`
   - Tests unitaires : `tracks/03_unlearning_v2/tests/test_track_h.py`
   - Sorties & checkpoints : `tracks/03_unlearning_v2/output/track_h/`
   - Ce journal : `docs/tracks/03-unlearning-track-h-journal.md`
   - *(Ne jamais modifier `selective_benchmark.py`, `auto_tune_4gpu.py`, `common.py` ni `03-unlearning-journal.md`.)*
2. **Isolation GPU :**
   - Utiliser exclusivement **1 session Google Colab GPU A100 nommée `track3_h`** (`/home/chedli/.local/bin/colab new -s track3_h --gpu A100`), avec un *heartbeat* stdout toutes les 15 s pendant `colab exec`.
   - **Toujours rapatrier les `.json` et le meilleur `.pt` puis appeler `/home/chedli/.local/bin/colab stop -s track3_h`** dès qu'un run se termine pour ne pas gaspiller de Compute Units.
   - Ne jamais soumettre sur JURECA ni sur `http://34.147.229.55` (c'est le `coordinator` principal qui gère les soumissions serveur).

---

## 4. Journal Chronologique des Expériences Piste H

### 2026-11-22 (01h35) — Initialisation de la cellule autonome `Piste H` (`h-lead` + `h-exec`)
- Création du journal dédié et lancement du binôme `h-lead` (`gemini-4-hydrogen-high`) + `h-exec` (`gemini-4-hydrogen-medium`).

### 2026-11-22 (01h56) — Round 1 Piste H (Colab A100 `track3_h`) : Record battu `0.5565 -> 0.6108` (`+0.0543`) !

Implémentation dans [track_h.py](../../tracks/03_unlearning_v2/methods/track_h.py), [run_track_h.py](../../tracks/03_unlearning_v2/run_track_h.py) et [test_track_h.py](../../tracks/03_unlearning_v2/tests/test_track_h.py) (`TRACK_H_TESTS_OK`), puis exécution 3-Folds Rotating Weighted + checkpoint complet `3 500` images sur Colab A100 (`track_h_round1.json`) :

| Candidat | **Score Pondéré `W_SEL ± std`** | **3-Folds `F_raw` / `F_alt`** | **3-Folds `Gap50`** | **3-Folds `IntraAcc`** | **3-Folds `InterAcc` (`InterKL`)** | **Full `.pt` `CE mean ± std` (`p50`, `>95%`)** | **Full `ProxyKL` (`AbsKL`, `dTheta`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `AT_ce1.00_b1.8_f6` (Baseline #1) | `0.5565 ± 0.019` | `2.61` / `2.74` | `+1.47` | `73.3 %` | `99.8 %` (`0.0312`) | `1.909 ± 2.05` (`1.214`, `0.19`) | `0.0486` (`0.0304`, `4.61`) |
| `AT_ce1.40_b1.5_f8` (Baseline #2) | `0.5451 ± 0.006` | `3.24` / `3.37` | `+1.69` | `63.8 %` | `99.9 %` (`0.0226`) | `2.547 ± 3.00` (`1.390`, `0.31`) | `0.0422` (`—`, `4.56`) |
| **`H4_h123_f6_ce125`** (H1+H2+H3) | **`0.6108 ± 0.026`** | `2.53` / `2.63` | **`+1.73`** | **`81.9 %`** | `99.8 %` (`0.0234`) | `2.127 ± 2.10` (`1.468`, **`0.14`**) | `0.0409` (`0.0277`, `6.27`) |
| **`H1_h1h2_f8_ce125`** (H1+H2) | **`0.6041 ± 0.018`** | `3.09` / `3.18` | **`+2.06`** | `76.8 %` | **`99.9 %` (`0.0199`)** | `1.676 ± 1.93` (`0.912`, `0.19`) | **`0.0325`** (`0.0221`, `5.78`) |
| **`H2_h123_f8_ce145`** (H1+H2+H3) | **`0.6023 ± 0.026`** | `2.91` / `3.00` | `+1.94` | `78.2 %` | **`99.9 %` (`0.0215`)** | `1.665 ± 1.79` (`1.010`, `0.15`) | **`0.0324`** (`0.0225`, `5.47`) |
| `H3_h1234_circuit_f8` (+H4 v1) | `0.0000 ± 0.000` | `16.42` / `15.05` | `+6.92` | `37.5 %` | `99.7 %` (`0.0545`) | `10.608 ± 8.65` (`8.692`, `0.01`) | `0.2179` (`0.0668`, `321.04`) |

**Analyse scientifique Round 1 & axes d'amélioration Round 2 :**
1. **Validation éclatante de H1 + H2 + H3 :** La projection orthogonale sur `head.weight` ($\alpha_{\text{proto}} \in [0.75, 0.80]$), le régulateur de vitesse par classe (`class_ce_ema`), la conductance dynamique temps réel $g_i(t)$ avec rollback sélectif et le pilotage `.gamma` éliminent le *class collapse* : `IntraAcc` bondit de `73.3 %` à **`81.9 %`** tout en améliorant `Gap50` (`+1.73` à `+2.06`) et `InterKL` (`0.0199` sur `f8`).
2. **Goulot restant sur `H4_h123_f6_ce125` (`fold_score_std = 0.0261`) :** Sur les folds, `f_raw.frac_gt_95` reste élevé (`0.52–0.58`, notamment `fold1` arrêté à `ce_p50 = 0.19` avec `>95% = 0.58`) car les échantillons saturés (`p > 0.95`) ont un gradient NPO à $T=2$ encore trop faible par rapport au rollback H2. Ajouter un terme anti-saturation tempéré à $T=4$ actif **uniquement** sur `ce_i < 0.18` doit débloquer ces `55 %` d'images sans perturber les échantillons déjà oubliés.
3. **Diagnostic du bug H4 v1 (`dTheta = 321.04`) :** Dans `_program_single_gate`, `s = 4.0 / d_ret.std()` atteignait ~`40` (soit `64` lignes de `fc1.weight` de norme `40` $\implies \Delta\theta \approx 320$) et la division par `|gamma| = 0.1` envoyait une impulsion `+100` dans le flux résiduel. Correction Round 2 : borner `s <= 1.2`, supprimer la division par `gamma` (`||fc2_col|| <= 0.45`), utiliser `q_ret = 0.999` et un garde-fou fermé vérifiant $\Delta \text{ProxyKL} \le 0.002$ sur le proxy avant d'accepter les portes.

### 2026-11-22 (01h59) — Calibration Serveur #10 (`H4_h123_f6_ce125.pt` -> `0.0984`) & Plan Round 2
- **Retour serveur réel (Soumission `#10`, `submission_id: 2387`) :** `H4_h123_f6_ce125.pt` (`Full CE = 2.127 ± 2.10`, $\mu+\sigma = 4.23$, `p50 = 1.468`, `dTheta = 6.27`) obtient **`0.0984`**, en dessous de `#9` (`AT_ce1.00_b1.8_f6.pt`, $\mu+\sigma = 3.96$, `p50 = 1.214`, `dTheta = 4.61` $\implies$ **`0.1193`**) et `#5` ($\mu+\sigma = 3.45$, `p50 = 0.380`, `dTheta = 4.03` $\implies$ **`0.1166`**).
- **Bracketing du pic serveur :** La courbe en cloche serveur atteint son optimum autour de **$\mu+\sigma \in [3.55, 3.90]$** (`Full CE mean ~ 1.60–1.85`, `Full p50 ~ 0.70–1.05`, `dTheta <= 4.6`, `ProxyKL <= 0.035`). `H1_h1h2_f8_ce125.pt` du Round 1 (`1.676 ± 1.93`, $\mu+\sigma = 3.61$, `p50 = 0.912`, `ProxyKL = 0.0325`) est déjà placé au cœur de cette zone, mais avec `dTheta = 5.78`.
- **Plan Round 2 calibré :** Ramener `gamma_lr_mult = 1.0` (supprimer le sur-drift `dTheta`), cibler `target_forget_ce in [0.90, 1.05]` avec `stuck_t4_weight in {0.45, 0.50}` (pour écraser `>95%`), `alpha_proto in [0.80, 0.82]` (`IntraAcc >= 80%`), `l2_weight in [1.8e-3, 2.2e-3]`, et tester H4 corrigé/gardé en test A/B sur `R2_h12_f8_ce090`.

### 2026-11-22 (02h19) — Audit Root-Cause #11 (`H1_f8 -> 0.0813`) & Spécification Round 3 (`f6` + Masque SalUn Fixe)
- **Retour serveur réel (Soumission `#11`, `submission_id: 2457`) :** `H1_h1h2_f8_ce125.pt` (`W_SEL = 0.6041`, `dTheta = 5.78`) obtient **`0.0813`** malgré un `Full CE = 1.676 ± 1.93` bien placé. En Round 2 partiel (avant coupure websocket), `R2_h12_f8_ce090` atteignait `W_SEL = 0.6001` et `R2_h12_f8_ce105` atteignait `W_SEL = 0.6192`.
- **3 Bugs Root-Cause identifiés par l'audit (`#32`) :**
  1. **Rotation du masque SalUn tous les 15 pas :** calculer `gf / (gr + 1e-8)` à `t=0` (où `gr == 0`) et redéfinir les 20 % actifs toutes les 15 itérations sur 2 mini-batches touchait **70–85 % des poids** au lieu de préserver l'invariance exacte des 80 % figés (`dTheta` montait à `5.78–6.27` vs `3.95–4.61` sur `#8, #5, #9`). **Fix :** geler le masque binaire SalUn `init_masks` **une fois à `t=0`** via `compute_tempered_saliency`, et n'autoriser H2 (`g_i(t)` + rollback) qu'à **atténuer/moduler à l'intérieur** de `init_masks`.
  2. **Étouffement de `head.weight` (`head_lr_mult=0.5`, `alpha_proto=0.80`) :** forçait le backbone ViT à sur-déformer (`dTheta = 5.78–6.27`). **Fix :** `head_lr_mult = 1.0`, `alpha_proto in [0.15, 0.30]`, `dTheta <= 4.5`.
  3. **`freeze_until_block = 8` (`f8`) sous-performe systématiquement `f6` sur le serveur** (`#7=0.0604`, `#11=0.0813` vs `#9=0.1193`, `#5=0.1166`, `#8=0.1099`). **Fix :** `freeze_until_block = 6` uniquement en Round 3.

### 2026-11-22 (01h38–01h55) — Round 1 `Piste H` (4 candidats, `track3_h` A100 Colab) : Record battu à **`W_SEL = 0.6108 ± 0.026`**
- **Implémentation ([track_h.py](../../tracks/03_unlearning_v2/methods/track_h.py), [run_track_h.py](../../tracks/03_unlearning_v2/run_track_h.py), [test_track_h.py](../../tracks/03_unlearning_v2/tests/test_track_h.py)) :**
  - **H1 :** Gouverneur de vitesse par classe (`class_ce_ema`) + boost sur les images têtues (`ce_i < 0.15`) + projection orthogonale du gradient de `head.weight` contre le prototype initial ($g_c \perp w_{0,c}$ avec $\alpha_{\text{proto}} \in \{0.75, 0.80\}$).
  - **H2 :** Recalibrage dynamique temps réel de la conductance $g_i(t) = \text{EMA}(|\nabla_f|) / (\text{EMA}(|\nabla_r|) + \epsilon)$ tous les 15 steps + rollback sélectif ($\tau = 0.25, \rho = 0.25$).
  - **H3 :** Contrôle direct de `LayerScale .gamma` (`gamma_lr_mult = 3.0`, bornage relatif `±50 %`).
  - **H4 :** `compile_if_then_mlp_gates` (`64` neurones MLP dormants de `blocks.11` programmés par k-means sur les vecteurs `CLS` des images têtues avec seuil quantile `0.995` sur le `retain`).
- **Tableau comparatif complet du Round 1 (`track_h_round1.json`, session `track3_h` stoppée après rapatriement des 3 meilleurs `.pt`) :**

| Rang | Candidat `Piste H` | **Score Pondéré `W_SEL ± std`** | **3-Folds `F_raw` / `F_alt`** | **3-Folds `Gap50`** | **3-Folds `IntraAcc`** | **3-Folds `InterAcc` (`InterKL`)** | **Full `.pt` `CE mean ± std` ($\mu+\sigma$)** | **Full `p50` (`>95%`)** | **Full `ProxyKL` (`dTheta`)** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| *Réf* | *`AT_ce1.00_b1.8_f6` (`#9` = `0.1193`)* | *`0.5565 ± 0.019`* | *`2.61` / `2.74`* | *`+1.47`* | *`73.3 %`* | *`99.8 %` (`0.0312`)* | *`1.909 ± 2.05` (`3.96`)* | *`1.214` (`0.19`)* | *`0.0486` (`4.61`)* |
| **1** | **`H4_h123_f6_ce125`** | **`0.6108 ± 0.026` (+9.8 % !)** | **`2.53` / `2.63`** | **`+1.73`** | **`81.9 %` (+8.6 pts !)** | **`99.8 %` (`0.0234`)** | **`2.127 ± 2.10` (`4.23`)** | **`1.468` (`0.14`)** | **`0.0409` (`6.27`)** |
| **2** | **`H1_h1h2_f8_ce125`** | **`0.6041 ± 0.018`** | **`3.09` / `3.18`** | **`+2.06`** | **`76.8 %`** | **`99.9 %` (`0.0199`)** | **`1.676 ± 1.93` (`3.61`)** | **`0.912` (`0.19`)** | **`0.0325` (`5.78`)** |
| **3** | **`H2_h123_f8_ce145`** | **`0.6023 ± 0.026`** | **`2.91` / `3.00`** | **`+1.94`** | **`78.2 %`** | **`99.9 %` (`0.0215`)** | **`1.665 ± 1.79` (`3.46`)** | **`1.010` (`0.15`)** | **`0.0324` (`5.47`)** |
| **4** | `H3_h1234_circuit_f8` *(64 portes `IF/THEN`)* | `0.0000`* *(gate Intra < 55%)* | `16.42` / `15.05` | **`+6.92` (!)** | `37.5 %` | **`99.7 %` (`0.0545`)** | `10.608 ± 8.65` | `8.692` (**`0.01`**) | `0.2179` (`321.0`) |

- **Action :** **`H4_h123_f6_ce125.pt`** (`W_SEL = 0.6108 ± 0.026`, `IntraAcc = 81.9 %`, `>95% = 0.14`, `ProxyKL = 0.0409`, $\mu+\sigma = 4.23$) a été soumis en **`#10`** sur le serveur d'évaluation et a obtenu **`0.0984`** (`< 0.1193`), puis `H1_h1h2_f8_ce125.pt` (`W_SEL = 0.6041`) a obtenu **`0.0813`** en **`#11`**.

### 2026-11-22 (02h36) — Round 3 `Piste H` (`track_h_round3.json`) : Résultat Négatif (`W_SEL = 0.3017–0.4032`) & Audit Méthodologique
- **Résultats bruts de `track_h_round3.json` (masque SalUn statique à $t=0$ + modulation H2 interne + `f6` + `alpha_proto` $\in \{0.15, 0.20, 0.30\}$ + `stuck_t4_weight = 0.45–0.50`) :**

| Rang | Candidat Round 3 | `W_SEL ± std` | `IntraAcc` | `Gap50` | `InterAcc` (`InterKL`) | Full `CE mean ± std` | `p50` (`>95%`) | `ProxyKL` (`dTheta`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `R3_f6_ce085_ap30_safe` | **`0.4032 ± 0.088`** | `76.6 %` | `+0.93` | `99.4 %` (`0.0832`) | `1.310 ± 1.30` | `0.949` (`0.113`) | `0.0543` (`4.48`) |
| **2** | `R3_f6_ce090_ap20` | `0.3420 ± 0.112` | `69.2 %` | `+1.08` | `99.1 %` (`0.1185`) | `1.113 ± 1.26` | `0.663` (`0.157`) | `0.0598` (`4.71`) |
| **3** | `R3_f6_ce090_ap20_h4` | `0.3390 ± 0.114` | `68.7 %` | `+1.09` | `99.1 %` (`0.1204`) | `1.144 ± 1.27` | `0.698` (`0.145`) | `0.0609` (`107.80`) |
| **4** | `R3_f6_ce100_ap15` | `0.3017 ± 0.121` | `66.9 %` | `+1.09` | `99.0 %` (`0.1372`) | `1.114 ± 1.25` | `0.672` (`0.152`) | `0.0606` (`4.73`) |

- **Constat honnête (résultat négatif) :**
  1. Tous les candidats du Round 3 sont nettement inférieurs au Round 1 (`0.602–0.611`) et à la baseline `AT` (`0.5565`) sur `W_SEL`, avec une dégradation importante de `InterKL` (`0.083–0.137` vs `0.020–0.031`) et de `ProxyKL` (`0.054–0.061` vs `0.032–0.049`).
  2. Le terme `stuck_t4_weight` ($T=4$) couplé à la normalisation de poids et à la projection `head.weight` a détérioré la stabilité des 30 classes retenues sans améliorer le score.
  3. Sur `R3_f6_ce090_ap20_h4`, `dTheta` atteint **`107.80`** malgré le garde-fou `delta_proxy_kl <= 0.0015`, car `fc1.bias = -s * tau_k` (avec `norm2` de norme `~27`) crée un saut de biais non borné dans `fc1`.
  4. **Correction méthodologique (suite à la revue d'inspecteur) :** `W_SEL` est un **diagnostic local de sélectivité**, pas le score officiel du serveur (`gold` caché + copies altérées cachées + `val_acc >= 70%`). L'inversion entre `#10`/`#11` (`W_SEL = 0.6108 / 0.6041` $\to$ serveur `0.0984 / 0.0813`) et `#9` (`W_SEL = 0.5565` $\to$ serveur **`0.1193`**) confirme que seul le **score serveur** fait foi, et que toute affirmation causale sur H1/H2/H3 nécessite des **ablations *one-factor-at-a-time*** à hyperparamètres constants et des folds de test jamais vus pendant la sélection.

### 2026-11-22 (02h56) — Round 4 `Piste H` : Ablations 1-Facteur sur Folds *Held-Out* (`seed=777`, `alt_seed=877`)

Évaluation contrôlée *one-factor-at-a-time* contre la réplique exacte de `#9` (`ABL_0_baseline_AT` = `AT_ce1.00_b1.8_f6`, `seed=777` jamais vu en sélection, `track_h_round4.json`, session `track3_h` stoppée) :

| Candidat Ablation (`seed=777`) | **Score Pondéré `W_SEL ± std`** | **3-Folds `F_raw` / `F_alt`** | **3-Folds `Gap50`** | **3-Folds `IntraAcc`** | **3-Folds `InterAcc` (`InterKL`)** | **Full `.pt` `CE mean ± std` ($\mu+\sigma$)** | **Full `p50` (`>95%`)** | **Full `ProxyKL` (`AbsKL`, `dTheta`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ABL_0_baseline_AT` *(réplique exacte `#9`)* | `0.5505 ± 0.028` | `2.82` / `2.99` | `+1.47` | `69.5 %` | `99.9 %` (`0.0246`) | `1.909 ± 2.05` (`3.96`) | `1.214` (`0.19`) | `0.0486` (`0.0304`, `4.61`) |
| `ABL_1_plus_H1_ortho` *(`+ alpha_proto=0.25`)* | `0.5528 ± 0.028` | `2.79` / `2.95` | `+1.48` | `70.8 %` | `99.9 %` (`0.0246`) | `1.955 ± 2.12` (`4.08`) | `1.213` (`0.20`) | `0.0497` (`0.0319`, `4.71`) |
| **`ABL_2_plus_H1_gov`** *(`+ gov + stuck_boost=1.5`)* | **`0.5613 ± 0.012`** | `2.40` / `2.57` | `+1.39` | **`72.4 %`** | `99.9 %` (`0.0313`) | `1.970 ± 1.71` (`3.68`) | `1.598` (`0.09`) | `0.0648` (`0.0412`, `4.80`) |
| `ABL_3_plus_H2_damp` *(`+ H2 damp intra-masque`)* | `0.5117 ± 0.067` | `2.98` / `3.15` | `+1.46` | `68.3 %` | `99.8 %` (`0.0313`) | `1.822 ± 1.93` (`3.75`) | `1.175` (`0.17`) | **`0.0422`** (`0.0272`, **`4.40`**) |

**Conclusions causales rigoureuses (Round 4) :**
1. **Reproductibilité exacte :** `ABL_0_baseline_AT` reproduit `#9` à l'identique sur les 3 500 images (`1.909 ± 2.05`, `ProxyKL = 0.0486`, `dTheta = 4.61`) et score `0.5505 ± 0.028` sur `seed=777`.
2. **Effet isolé de H1 Ortho (`alpha_proto = 0.25`) :** gain faible mais systématique sur les 3 folds (`W_SEL +0.0023`, `IntraAcc +1.3 pts`) sans dégrader `InterKL` (`0.0246`), pour `+0.10` de `dTheta` (`4.71`).
3. **Effet isolé de H1 Gov (`class_ce_ema` + `stuck_boost = 1.5`) :** divise la variance inter-folds par $2.3\times$ (`std = 0.0124` vs `0.0278`), augmente `IntraAcc` (`+2.9 pts`) et réduit `>95%` de moitié (`0.19 -> 0.09`, $\mu+\sigma = 3.68$). **Limitation identifiée dans le code :** `weights = weights / weights.mean().clamp_min(1e-4)` ([track_h.py](../../tracks/03_unlearning_v2/methods/track_h.py#L142)) re-normalise même après le masque `(ce_i < tgt)`, ce qui multiplie le gradient effectif de `loss_f` par $1 / f_{\text{active}} \approx 2.5\times$ par rapport à `retain_kl_weight` et fait monter `ProxyKL` à `0.0648`. Remplacer par `weights.mean().clamp_min(1.0)` (ne jamais amplifier au-delà de $1\times$) est le correctif naturel si une suite est demandée.
4. **Effet isolé de H2 Damp (conductance intra-masque sans rotation) :** réduit `ProxyKL` (`0.0422`) et `dTheta` (`4.40`), mais déstabilise le `fold 0` (`0.4354`, `IntraAcc = 56.0 %`, `std = 0.067`). Non retenu seul.

### 2026-11-22 (03h09) — Round 5 `Piste H` : Correctif `clamp_min(1.0)` sur H1 Gov (`seed=777`, `W_SEL = 0.5715 ± 0.004`)

Test 1-facteur du correctif `weights.mean().clamp_min(1.0)` ([track_h.py](../../tracks/03_unlearning_v2/methods/track_h.py#L142)) sur les mêmes folds *held-out* (`seed=777`, `alt_seed=877`, `track_h_round5.json`, session `track3_h` stoppée) :

| Candidat (`seed=777`) | **Score Pondéré `W_SEL ± std`** | **3-Folds `F_raw` / `F_alt`** | **3-Folds `Gap50`** | **3-Folds `IntraAcc`** | **3-Folds `InterAcc` (`InterKL`)** | **Full `.pt` `CE mean ± std` ($\mu+\sigma$)** | **Full `p50` (`>95%`)** | **Full `ProxyKL` (`AbsKL`, `dTheta`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ABL_0_baseline_AT` *(`#9`)* | `0.5505 ± 0.028` | `2.82` / `2.99` | `+1.47` | `69.5 %` | `99.9 %` (`0.0246`) | `1.909 ± 2.05` (`3.96`) | `1.214` (`0.19`) | `0.0486` (`0.0304`, `4.61`) |
| `ABL_2_plus_H1_gov` *(`clamp_min(1e-4)`)* | `0.5613 ± 0.012` | `2.40` / `2.57` | `+1.39` | `72.4 %` | `99.9 %` (`0.0313`) | `1.970 ± 1.71` (`3.68`) | `1.598` (`0.09`) | `0.0648` (`0.0412`, `4.80`) |
| **`R5_h1gov_fixed`** *(`clamp_min(1.0)`)* | **`0.5711 ± 0.004`** | `2.44` / `2.64` | `+1.36` | `70.9 %` | `99.9 %` (`0.0280`) | `1.200 ± 1.50` (`2.70`) | `0.557` (`0.24`) | **`0.0409`** (`0.0252`, **`4.24`**) |
| **`R5_h1gov_ortho25`** *(`+ alpha_proto=0.25`)* | **`0.5715 ± 0.004`** | `2.45` / `2.63` | `+1.38` | **`72.1 %`** | `99.9 %` (`0.0286`) | `1.148 ± 1.57` (`2.72`) | `0.401` (`0.30`) | `0.0438` (`0.0286`, `4.43`) |

**Analyse Round 5 :**
- La correction `clamp_min(1.0)` élimine l'inflation de `ProxyKL` (`0.0648 -> 0.0409`) et de `dTheta` (`4.80 -> 4.24`), tout en conservant le gain de stabilité majeur de H1 Gov (`std = 0.004` vs `0.028`, variance inter-folds réduite d'un facteur `~7x`) et le gain `IntraAcc` de `alpha_proto = 0.25` (`72.1 %` vs `69.5 %`).
- En contrepartie, sans amplification artificielle `~2.5x`, `target_forget_ce = 1.00` sur `210` pas donne un checkpoint complet plus modéré (`Full CE = 1.15–1.20`, $\mu+\sigma = 2.70–2.72$, en dessous de la zone pic serveur `#5/#9` à $\mu+\sigma \in [3.45, 3.96]$). Si une itération Round 6 est souhaitée pour tester ce mécanisme dans la zone pic serveur, monter simplement `target_forget_ce = 1.25` + `steps = 240` (avec `retain_kl_weight = 2.3`, `l2_weight = 1.6e-3`).

### 2026-11-22 (03h27) — Round 6 `Piste H` (`ce=1.25`, `steps=240`, `seed=777`) : Ultra-Bas `ProxyKL = 0.0350–0.0359` & `dTheta = 3.86–3.99`

Exécution 2-runs sur `track3_h` A100 (`track_h_round6.json`, session `track3_h` stoppée) pour tester l'effet de `target_forget_ce = 1.25, steps = 240, lr = 1.1e-4` avec `clamp_min(1.0)` sur folds *held-out* (`seed=777`) :

| Candidat (`seed=777`) | **Score Pondéré `W_SEL ± std`** | **3-Folds `F_raw` / `F_alt`** | **3-Folds `Gap50`** | **3-Folds `IntraAcc`** | **3-Folds `InterAcc` (`InterKL`)** | **Full `.pt` `CE mean ± std` ($\mu+\sigma$)** | **Full `p50` (`>95%`)** | **Full `ProxyKL` (`AbsKL`, `dTheta`)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ABL_0_baseline_AT` *(`#9`)* | `0.5505 ± 0.028` | `2.82` / `2.99` | `+1.47` | `69.5 %` | `99.9 %` (`0.0246`) | `1.909 ± 2.05` (`3.96`) | `1.214` (`0.19`) | `0.0486` (`0.0304`, `4.61`) |
| `R5_h1gov_fixed` *(`ce=1.00`)* | **`0.5711 ± 0.004`** | `2.44` / `2.64` | `+1.36` | `70.9 %` | `99.9 %` (`0.0280`) | `1.200 ± 1.50` (`2.70`) | `0.557` (`0.24`) | `0.0409` (`0.0252`, `4.24`) |
| `R5_h1gov_ortho25` *(`ce=1.00`)* | **`0.5715 ± 0.004`** | `2.45` / `2.63` | `+1.38` | **`72.1 %`** | `99.9 %` (`0.0286`) | `1.148 ± 1.57` (`2.72`) | `0.401` (`0.30`) | `0.0438` (`0.0286`, `4.43`) |
| `R6_h1gov_ce125` *(`ce=1.25`)* | `0.5608 ± 0.010` | `2.70` / `2.90` | `+1.46` | `68.0 %` | `99.9 %` (`0.0303`) | `1.191 ± 1.49` (`2.68`) | `0.560` (`0.23`) | **`0.0350`** (**`0.0215`**, **`3.86`**) |
| `R6_h1gov_ortho25_ce125` | `0.5619 ± 0.008` | `2.70` / `2.89` | `+1.48` | `69.1 %` | `99.9 %` (`0.0314`) | `1.227 ± 1.55` (`2.77`) | `0.559` (`0.24`) | **`0.0359`** (**`0.0230`**, **`3.99`**) |

**Synthèse finale Piste H (Rounds 1–6) :**
- **Ce qui est validé causalement sur `seed=777` :** le couple **H1 Gov (`class_ce_ema` + `stuck_boost=1.5` avec `clamp_min(1.0)`) + H1 Ortho (`alpha_proto=0.25`)** sur `freeze_until_block=6` et masque SalUn statique ($t=0$) bat `ABL_0` (`#9`) sur `W_SEL` (`0.5715 ± 0.004` vs `0.5505 ± 0.028`), réduit la variance inter-folds d'un facteur **$7\times$**, améliore `IntraAcc` (`72.1 %` vs `69.5 %`) et réduit `ProxyKL` (`0.0359–0.0438` vs `0.0486`) et `dTheta` (`3.99–4.43` vs `4.61`).
- **Ce qui a été écarté :** la rotation de masque H2 (`dTheta = 5.78–6.27`, `#10/#11` négatifs sur serveur), `H2 damp` seul (`std = 0.067`), `stuck_t4_weight` ($T=4$, explosion `InterKL`), et `H4` (`dTheta = 107.8–321.0`).
- **Comparaison honnête vs `#8`/`#9` :** `R6_h1gov_ortho25_ce125.pt` (`Full CE = 1.227 ± 1.55`, `mu+sig = 2.77`, `ProxyKL = 0.0359`, `dTheta = 3.99`) et `R5_h1gov_ortho25.pt` (`W_SEL = 0.5715 ± 0.004`, `ProxyKL = 0.0438`, `dTheta = 4.43`) restent légèrement en deçà du niveau `mu+sig` de `#8` (`3.01`) / `#9` (`3.96`) sur le checkpoint complet (le contrôleur `_snapshot_score` de `auto_tune_4gpu.py` sélectionnant le snapshot précoce à `ProxyKL ~ 0.036`), mais offrent le meilleur compromis `ProxyKL / AbsKL / dTheta / stabilité` de Piste H.

### 2026-11-22 (02h40) — Verdict & Plan d'Ablation Contrôlée (*One-Factor-at-a-Time* sur `seed=777`)
- **Verdict scientifique :** ne pas abandonner la `Piste H` (signal local réel au Round 1 : `W_SEL = 0.6108`, `IntraAcc = 81.9 %`), mais ne pas revendiquer `H1/H2/H3` comme mécanismes validés avant d'avoir isolé leur effet individuel contre `#9` (`AT_ce1.00_b1.8_f6`) et calibré sur le score serveur.
- **Protocole Round 4 (Ablation pure à 1 facteur à la fois sur folds *held-out* `seed=777`) :**
  1. `ABL_0_baseline_AT`: réplique exacte de `#9` (`AT_ce1.00_b1.8_f6`, `freeze=6`, `head_lr=1.0`, SalUn statique $t=0$).
  2. `ABL_1_plus_H1_ortho`: identique à `ABL_0` + uniquement `alpha_proto = 0.25`.
  3. `ABL_2_plus_H1_gov`: identique à `ABL_0` + uniquement le gouverneur par classe / boost d'instances (sans `stuck_t4_weight`).
  4. `ABL_3_plus_H2_damp`: identique à `ABL_0` + uniquement modulation dans le masque statique $t=0$ (sans rotation de masque).

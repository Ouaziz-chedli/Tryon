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

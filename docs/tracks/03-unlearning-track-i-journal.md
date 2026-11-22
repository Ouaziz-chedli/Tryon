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

# Track 3 — Journal Piste SCRUB+R (Machine Unlearning)

- **Objectif :** Implémenter **SCRUB+R** (*Towards Unbounded Machine Unlearning*, NeurIPS 2023) en régime *User Privacy* (`UP`) avec alternance `DO-MAX-EPOCH` / `DO-MIN-EPOCH` + queue de min-steps + **Rewind** $w^{(i^*)}$, tout en réutilisant le signal de difficulté par image **Patch-Dropout** (`rho_PD = +0.640`) **sans** l'écrasement de `Full CE` causé par la KL vers des cibles faciles (`0.32–0.37`).
- **Référence papier :** [03-scrub-machine-unlearning.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/analyses/03-scrub-machine-unlearning.md)
- **Peak serveur de référence :** `#13` (`0.1238`) — `Full CE = 1.794 ± 2.04 (mu+sig = 3.84), p50 = 1.006, ProxyKL = 0.0558, dTheta = 4.48`.

---

## 1. Diagnostic initial (Patch-Dropout V1)

- **Ce qui a fonctionné (`run_patchdrop_ablation.py`) :**
  - `estimate_patchdrop_gold_logits` (`drop_prob=0.45`, 4 vues) atteint **`rho_PD = +0.640`** (corrélation de Spearman avec la difficulté par image) et **`IntraAcc = 90.9%`** (`PD_2_kl12_npo00_ce180`).
- **Ce qui a échoué :**
  - `calibrate_per_image_targets` + `per_sample_calibrated_loss` imposait une KL à $T=2$ vers des cibles faciles (`ce_tgt`), tirant `Full CE` à `0.32–0.37` (`mu+sig = 0.97–1.06`) au lieu de `3.65–3.90`.

## 2. Architecture True SCRUB+R (`methods/scrub_r.py`)

- **Alternance découplée :**
  - `DO-MAX-EPOCH` (`epoch < max_epochs`) : montée sur $\mathcal{D}_f$ (NPO ou KL Ascent $T^2$) pondérée et bornée par le ratio de difficulté Patch-Dropout `diff_ratio = clamp(ce_pd / mean(ce_pd), 0.25, 2.5)` (`max_ce_i = target_forget_ce * diff_ratio`, poids `diff_ratio ** pd_alpha`).
  - `DO-MIN-EPOCH` (`epoch < total_epochs`, incluant une queue de min-steps `epoch >= max_epochs` avec decay LR `x0.25`) : `retain_kl_weight * KL_T` + `retain_ce_weight * CE` sur `_build_train_proxy` + ancre L2.
  - Socle `f6` (`freeze_until_block=6`), masque SalUn statique $t=0$ (`keep_ratio=0.20`), gel exact des 55 classes absentes.
- **SCRUB+R Rewind :**
  - Snapshot CPU des poids entraînables à chaque fin d'époque ; sélection de $i^*$ minimisant `|mu+sig - 3.80| + 0.8*|p50 - 1.00| + 15*max(0, ProxyKL - 0.050) + 0.5*max(0, frac_gt_95 - 0.25)` (hors snapshots `p50 < 0.15`).

## 3. Round 1 — Ablations 1-facteur (`seed=777`, en attente exécution Colab `track3_scrub`)

| Candidat | Variation 1-facteur | Hypothèse |
| :--- | :--- | :--- |
| `SR1_npo_pd_rewind` | Base SCRUB+R (NPO + PD + Rewind) | Atteindre `mu+sig ~ 3.80`, `p50 ~ 1.00`, `rho_PD >= 0.55`, `IntraAcc > 0.80` |
| `SR2_no_rewind` | `use_rewind=False` | Mesurer l'apport exact du Rewind $w^{(i^*)}$ vs fin de queue |
| `SR3_uniform_no_pd` | `pd_alpha=0.0, pd_gate_scale=0.0` | Mesurer l'apport exact de Patch-Dropout sur `rho_PD` et `IntraAcc` |
| `SR4_kl_ascent_pd` | `max_loss_kind="kl_ascent", lr_max=6e-5` | Comparer KL Ascent $T^2$ borné (SCRUB Éq. 2) vs NPO |
| `SR5_npo_pd_strong` | `target_forget_ce=1.80, max_epochs=7, total_epochs=11` | Tester une poussée d'oubli plus forte si `SR1` reste sous-oublié |

# Track 3 — Plan de Rupture du Plafond (`0.1238`)

**Écrit :** `2026-11-22 04:40+01:00` · **Record à battre :** `0.1238` (`#13` `SOUP_50_050_50_100`, `μ+σ = 3.84`, `p50 = 1.006`)
**Définition de « rupture » :** score serveur **`>= 0.134`** (hors bande de bruit du plateau, `±0.005`). Jalon intermédiaire : `0.130`.
**Destination fixée par le Chef (`05h30`) : score serveur `>= 0.7`** au test réel CISPA (soumission). Décomposition : `0.7 ≈ retain 0.92 × mean_std ~0.87 × prediction ~0.87` — les jalons `0.134` puis `0.2–0.3` restent des étapes, plus la destination. Conséquence : la triangulation exacte de `(μ_gold, σ_gold)` (sondes diamant + soumissions) et l'escalade VGT/lab-KL deviennent **obligatoires**, plus optionnelles.

**Liens :** [Journal central](03-unlearning-journal.md) · [Handoff](03-unlearning-handoff.md) · [Journal base](03-unlearning-base-journal.md) · [Journal H](03-unlearning-track-h-journal.md) · [Journal I](03-unlearning-track-i-journal.md) · [Journal SCRUB](03-unlearning-scrub-journal.md)

---

## 1. Théorie verrouillée (audit 04h33, sur 16 soumissions + rounds R2–R6)

$$\text{score} = \underbrace{\text{retain}}_{\sim 0.92} \times \underbrace{\text{mean\_std}}_{\sim 0.72 \text{ (plateau)}} \times \underbrace{\text{prediction}}_{\sim 0.19 \text{ (goulot)}}$$

1. **Les moments sont épuisés.** Sommet = plateau `μ+σ ∈ [3.75, 4.05]` : `#13` (3.84→0.1238), `#9` (3.96→0.1193), `#14` (4.05→0.1197), `#16` (4.01→0.1184). Caler μ+σ plus finement ne rapporte que `±0.005`. **Interdit : tout run GPU dont le seul but est de recaler μ+σ.**
2. **Le plafond est `prediction_score ≈ 0.19`** (KL per-image à `T=2` envers `gold`, moitiés exactes/altérées 50/50). Notre KL vaut encore ~81 % de celle d'`orig`. Doubler ce facteur ⇒ `~0.26`.
3. **`ProxyKL`/`dTheta` sont des garde-fous, pas des leviers** (preuve `#15` : meilleur `W_SEL = 0.58` et `ProxyKL = 0.0353`, mais `μ+σ = 2.92` ⇒ 0.1082).
4. **Le seul geste à rendement prouvé : la soupe convexe** (`#13` bat ses deux parents). Soupape gratuite à chaque fois qu'un nouveau gagnant en fenêtre existe.
5. **Seul le serveur arbitre** (`W_SEL` inversé 3 fois). Le canal est un oracle gratuit : capacité ~4 soumissions/h, usage réel 1.6/h.

**Mécanismes du goulot, validés localement mais jamais placés en fenêtre :**
- **Per-image :** Patch-Dropout prédit la difficulté (`rho_PD = +0.64`) mais échoue *comme loss* (sous-oubli `μ+σ ≈ 1.0`) — reconduit *comme poids* dans SCRUB+R (Round 1 en vol).
- **Forme :** dark-knowledge liaison (alignement des 99 logits non-top à `T=3`) : LoRA r=8 ⇒ `ProxyKL −43 %`, `IntraAcc +7.7 pts` (ablation I R3, `seed=777`).
- **Lissage :** `gold` = moyenne de plusieurs modèles ⇒ soupes multi-parents (effet ensemble).

---

## 2. Fenêtre de soumission (règle d'engagement)

Un candidat ne part sur le serveur que s'il vérifie :

| Contrainte | Plage | Rôle |
| :--- | :--- | :--- |
| `Full μ+σ` | `[3.65, 4.00]` | obligatoire |
| `Full p50` | `[0.80, 1.20]` | obligatoire (relâchée : `#14` à `1.256` → 0.1197) |
| `ProxyKL (T=2)` | `<= 0.056` | obligatoire |
| `dTheta` | `<= 4.70` | obligatoire |
| `frac_gt_95` | `<= 0.30` | garde-fou |

Départage à placement égal dans la fenêtre : **préférer la soupe multi-parents/multi-seeds au run unique**, puis le `ProxyKL` le plus bas, puis `rho_PD` le plus haut.
Règle anti-gaspillage du canal : jamais deux soumissions de même famille avec `(μ, σ, p50)` quasi identiques ; chaque soumission doit tester un axe nouveau (placement, mécanisme, ou split (μ,σ)).

---

## 3. Tâches

### Phase 0 — État & cadence (T0 = maintenant, → T0+45 min) — *coordinator*

- [ ] **0.1 Soumission `#17` (créneau `04:39:42`) :** `SOUP3_75_AT9_25_AT2.pt` (`μ+σ = 3.765`, `p50 = 1.04`, `ProxyKL = 0.046`, `dTheta = 4.32`) si téléchargé ; sinon **`SOUP3_50_AT9_50_s101.pt`** (`μ+σ = 3.68`, `p50 = 0.86`, `ProxyKL = 0.044`, local 04:38). Vérifier `float32/strict/327 Mo` avant `curl`. Consigner au journal immédiatement.
- [ ] **0.2 Cadence continue :** 1 soumission par créneau de 15 min pendant les rounds actifs, uniquement des candidats en fenêtre ; journalisation immédiate (`sub_id`, score, `previous_best`).
- [ ] **0.3 Diagnostic deux-moitiés** (1ʳᵉ session L4 libérée, ~30 min) : pour les 5 meilleurs `.pt` locaux (`#13`, `SOUP3_75`, `SOUP3_50_s101`, `AT_ce0.80`, `I3R_4`), calculer **séparément** sur brutes vs `make_stress_suite` : `d_sub/d_orig` (formules `server_metrics.py`), `mean_std` et proxy `prediction` par moitié. Acceptance : un tableau « moitié exacte vs moitié altérée » par candidat, consigné au journal. Décision sortante : quelle moitié traîne (> 0.01 d'écart) ⇒ GO/NO-GO Phase 3.

### Phase 1 — Gains gratuits : soupes & oracle (T0 → T0+2 h, zéro GPU)

- [ ] **1.1 Soupe inter-pistes multi-parents** (*coordinator*) : dès que ≥ 2 gagnants en fenêtre existent (base R3 + `I3R_4`/I-R4 + SR1 + H-R7), construire les soupes convexes 3–6 parents, évaluer localement (`μ+σ`, `p50`, `ProxyKL`, `dTheta`), soumettre la meilleure en fenêtre. Acceptance : ≥ 1 soumission inter-pistes et son score consigné.
- [ ] **1.2 Sonde diamant** (*base-lead*, 2 runs A100, ~40 min) : deux runs recette `AT` gagnante calés à `μ+σ ≈ 3.9`, `ProxyKL <= 0.05`, `dTheta <= 4.6`, **splits opposés** :
  - `DIA_lowstd` : `two_sided_npo` (σ comprimé, ex. cible `σ ≈ 1.55`) ;
  - `DIA_highstd` : `gold_mimic` à modulation de marge amplifiée (σ étiré, ex. cible `σ ≈ 2.25`).
  Soumettre les deux. Acceptance : l'écart des deux scores serveur **oriente le diamant L1** ⇒ cible `(μ_gold, σ_gold)` affinée, consignée au journal. Si les deux scores sont dans la bande 0.118–0.124, le split (μ,σ) est non discriminant ⇒ clore cet axe.

### Phase 2 — Attaque du goulot `prediction_score` (verdicts des rounds en vol → T0+4 h)

- [ ] **2.1 Verdict SCRUB+R Round 1** (`SR1`–`SR5`) : si un `SR` atterrit en fenêtre avec `rho_PD >= 0.55` et `IntraAcc >= 0.78` ⇒ **soumission immédiate** puis soupe `SR × base-gagnant`. Si sous-oubli (`μ+σ < 3.5`) ⇒ le Round 2 vise `target_forget_ce` ↑ ; si `InterKL > 0.05` ⇒ réduire `pd_gate_scale`.
- [ ] **2.2 Verdict Piste I Round 4** (corridor `[1.55, 1.85]`, `dTheta <= 4.3`) : si en fenêtre ⇒ soumission (c'est le mécanisme de forme validé causalement). Sinon ⇒ soupe `I4R × ORIG` calée en fenêtre.
- [ ] **2.3 Verdict Piste H Round 7** (`weight_norm_floor`) : si en fenêtre avec `ProxyKL <= 0.048` ⇒ candidat de soupe, pas de soumission solo (les H seuls n'ont jamais battu `#13`).
- [ ] **2.4 Soupe « prediction-attack »** (*coordinator*) : parents mécaniquement complémentaires — amplitude (base multi-seed) + forme (I) + per-image (SR) + éventuellement H-R7. Objectif : lissage type ensemble. Acceptance : ≥ 1 soumission, score consigné.
- [x] **2.5 VGT — VALIDÉ NÉGATIF (`15905220`, 06h20) : filière VGT-cibles STOPPÉE.** Spearman(VGT, serveur) sur les 11 paires : composite `+0.53` (frac 0.45) mais `−0.44` (frac 0.30) ; **le mécanisme enseignant-or est anti-informatif** (|rho| ≤ 0.06 sur KL brut ; le clip sature : le pire modèle serveur `AT_ce0.50_b3.0` est le plus « gold-like » sous vues masquées). Toute la corrélation utile vient de `mean_std_score` (frac 0.45 : `+0.50`), calculable **sans masquage** — conservé comme faible signal complémentaire à `W_SEL`. **Redirection du levier per-image :** ensemble de difficulté (`run_difficulty_ensemble.py`, job `15905257`, signal PD prouvé `rho_PD 0.17 → 0.46`) + Phase 3 §3.2 (vue propre sur-pondérée) ; l'escalade si `prediction` plafonne devient le **lab-KL (4.1)**.

### Phase 3 — Découplage moitié exacte / moitié altérée (gate : asymétrie > 0.01 en 0.3)

- [ ] **3.1 Si la moitié altérée traîne :** run `AT` avec 2 vues — oubli sur vue propre, ancrage KL sur vue modérément augmentée (l'oubli transfère aux copies, l'ancrage reste propre). 3 configs A100.
- [ ] **3.2 Si la moitié exacte traîne :** inverse (oubli sur vue augmentée, ancrage sur vue propre).
- [ ] Acceptance commune : candidat en fenêtre sur **les deux moitiés séparément** (chaque `d_sub/d_orig` dans `[0.5, 0.9]`), soumis, score hors bande si le diagnostic était juste.

### Phase 4 — Backstop scientifique (T0+3 h, si Phase 2 stérile)

- [ ] **4.1 Lab KL minimal** (1–2 h A100) : `orig_lab` saturé (`CE < 0.005`, `gamma = 1e-4`) + `gold_lab` jamais exposé à F, pseudo-retain PatchMix — **unique but : mesurer `KL(p_gold_lab || p_sub)` à `T=2` localement** pour itérer sur le goulot sans dépenser le canal. Acceptance : le lab reproduit le classement des 5 dernières soumissions (test de cohérence, comme les labs v1/v2 l'avaient fait) avant toute utilisation.
- [ ] **4.2 Borne `retain_score` :** si `forget_score` semble maxé, mesurer la dérive sur control-proxy (PatchMix + 55 absentes) et tester 1 run `dTheta < 3.2` à amplitude égale — invalider ou confirmer l'hypothèse `retain ≈ 0.92`.

---

## 4. Checkpoints

**CP1 (~05:30)** — 0.3 rendu + ≥ 4 soumissions de cadence consignées. GO/NO-GO Phase 3. Sans asymétrie ⇒ Phase 3 fermée, crédit GPU reporté sur 2.4/4.1.
**CP2 (~07:00)** — Verdicts SR1/I-R4/H-R7 + sonde diamant. Si aucun mécanisme ne sort de la bande `0.118–0.124` ⇒ GO Phase 4. Si un mécanisme sort ⇒ doubler la mise dessus (round de scaling + soupes).
**CP3 (~09:00)** — Si record `<= 0.125` après ~10 soumissions bien ciblées ⇒ **consolidation** : soupe finale multi-parents robuste (pour le split privé 70 %), stop tuning sur le public, le reste du CU en réserve.

---

## 5. Risques & garde-fous

| Risque | Mitigation |
| :--- | :--- |
| Surfit du split public (30 %) | Uniquement soupes multi-seeds pour les coups finals ; jamais de tuning fin sur le public |
| Confondes sonde diamant (`ProxyKL`, `dTheta` différents) | Les 2 runs sont contraints `ProxyKL <= 0.05`, `dTheta <= 4.6` |
| Fuite de sélection (folds réutilisés) | `seed=777` réservé aux verdicts finaux (règle établie `#44`) |
| Budget CU (`~387 CU`) | Plan total ~35–40 CU (2 runs diamant + 3 runs P3 + 2 h lab + L4 diag) ; `colab stop` systématique |
| Canaux | Cooldown 15 min (2 min si échec) ; 4 sessions Colab max ; VGT intouchable sans le Chef |
| Confondre plateau et plafond | La rupture exige `>= 0.134` ; un `0.124–0.130` n'est **pas** une rupture, c'est le bruit du sommet |

## 6. Questions ouvertes

- ~~Le Chef autorise-t-il l'intégration des conclusions VGT (2.5) si Phase 2 échoue ?~~ **Répondu `05h15` : autorisation complète accordée** (lecture + exécution + usage comme cible) — voir 2.5 réécrite.
- `retain ≈ 0.92` reste une inférence non mesurée — la mesurer (4.2) avant toute conclusion « plafond = forget ».

---
## 7. État d'exécution (`2026-11-22 04h56`)

- **`#17` (`SOUP3_75_AT9_25_AT2`, 04h44) = `0.1176` — preuve causale de la théorie :** à `μ+σ` quasi identique (`3.77` vs `3.84`), réduire `ProxyKL` de `-17 %` et `dTheta` a fait **baisser** le score ⇒ `retain_score` saturé. **P4.2 (mesure de `retain`) fermé** ; tout le crédit GPU va à `prediction_score`.
- **En vol au 05h05 (P2.1/P2.2 en cours d'exécution) :** `track3_i` L4 = I-R4 corridor (`I4R_1`, `I4R_2` produits) ; `track3_scrub` L4 = SR-R1 (`SR1`–`SR3` produits, `SR4`–`SR5` en cours). Collecte armée (watcher), sessions à couper dès rapatriement.
- **`#18` (fenêtre ouverte 04h59) :** en attente des verdicts I/SR — pas de near-twin de `#17` (règle §3 anti-gaspillage).
- **H R7 : mort au cutoff (OOM T4 `rc=-9`, aucune sortie)** — relance en cours sur A100 (jamais T4), priorité P2.3 (parent de soupe seulement).
- **Escanade (erase-vs-orig) : jamais exécutée par le swarm** — script `run_erase_diff.py` en préparation, slot à définir.
- **P0.3 (diagnostic deux-moitiés) :** script `run_per_half_diagnostic.py` en préparation.
- **Deadline :** réservation JURECA `cispahack` ferme **14h00 aujourd'hui**.
- **05h10 — Vague de reprise lancée (orchestrateur omp, swarm AGY en attente de quota) :** scripts P0.3 (`run_per_half_diagnostic.py`) et escanade (`run_erase_diff.py`) écrits et `py_compile` OK ; H R7 relancé sur `track3_h` A100 (sous-agent `HR7Exec`) ; job JURECA 4×GPU (sonde diamant + prediction-attack) en préparation (sous-agent `JurecaSweep`) ; diagnostic deux-moitiés sur `track3_diag` T4 avec test d'upload direct `.pt` 343 Mo (sous-agent `DiagRun`) ; escanade à exécuter sur la VM `track3_i` avant son arrêt (checkpoints extrêmes déjà présents sur place). Collecte I-R4/SR-R1 armée (watcher). Aucune soumission `#18` avant verdicts en fenêtre.

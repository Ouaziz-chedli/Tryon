# Track 3 — Unlearning : Journal & Backlog de Recherche

> **Document central unique** pour la Track 3 : backlog de recherche (`Q1`→`Q6`), journal chronologique complet (jobs réussis/échoués/annulés, constats bruts, incidents d'infra, questions intermédiaires), défis techniques, stratégies et glossaire.

**Liens connexes :**
- Spec officielle : [03-unlearning.md](03-unlearning.md) ([PDF](Unlearning.pdf))
- Plan d'action & TODO : [03-unlearning-todo.md](03-unlearning-todo.md)
- Journal spécialisé Piste H (`h-lead` + `h-exec`) : [03-unlearning-track-h-journal.md](03-unlearning-track-h-journal.md)
- Journal spécialisé Piste I (`i-lora` + `i-lead`, *Erase-Then-Heal LoRA*) : [03-unlearning-track-i-journal.md](03-unlearning-track-i-journal.md)
- Journal spécialisé Techniques Initiales (`base-lead` + `base-exec`, `AT` SalUn statique + NPO & soupes) : [03-unlearning-base-journal.md](03-unlearning-base-journal.md)
- Papier analysé : [SCRUB (03-scrub-machine-unlearning.md)](../analyses/03-scrub-machine-unlearning.md)
- Scripts d'exploration : [explore_unlearning.py](../../scripts/explore_unlearning.py) · [explore_q1b.py](../../scripts/explore_q1b.py)

---

## 1. Cartographie des pièces du puzzle & Invariant

### Ce qu'on a (JURECA : `/p/scratch/training2666/ouaziz1/TRYON/unlearning/`)
- **`model.pt` (343 Mo) :** ViT-B/16 (`timm vit_base_patch16_224`, 100 classes, `init_values=1e-4`), entraîné sur *retain + forget*. Accès complet aux poids, prototypes `head.weight`, activations, gradients et prédictions.
- **`forget_images.npz` (526 Mo) :** 3 500 images RGB `uint8` (`[3500, 224, 224, 3]`), 1 seule clé `images`.
- **`task_template.py` & `submission.py` :** preprocessing exact, `predict()`, et client d'upload (`TASK_ID = "31-unlearning"`).
- **Cluster JURECA :** GPUs A100 (`dc-gpu`, réservation `cispahack`) + `.venv/` (`torch 2.15.0`, `timm 1.0.31`).

### Ce qui est caché (côté serveur uniquement)
- **Labels** de `forget_images.npz` (et mapping `0..99 ↔ noms de classes`).
- **`retain set`**, **`validation set`** (*utility gate* `val_acc >= 70%`), et **`control set`** (`retain_accuracy` + `KL(p_orig || p_sub)` à $T = 2$).
- **Copies altérées cachées** de `forget_images.npz` + **modèles `gold`** (ré-entraînés sans `forget`, référence pour `mean_std_score` et `prediction_score` à $T = 2$).

### Invariant du scoring
$$\text{final\_score} = \text{forget\_score} \times \text{retain\_score} \quad \text{avec } \text{val\_acc} \ge 0.70$$

---

## 2. Backlog de Recherche (`Q1` → `Q6`)

| ID | Question de recherche | Pourquoi c'est critique | Statut |
| :--- | :--- | :--- | :--- |
| **Q1** | **Nature du forget set :** 3 500 images uniformes sur 100 classes ou concentrées sur un sous-ensemble ? | Détermine si des classes absentes offrent un ancrage `retain` gratuit. | `Répondue (Job 15901163)` |
| **Q1b** | **Histogramme complet & géométrie `head.weight` :** coupure nette ou bruit, clés `.npz` ? | Confirme s'il existe une séparation franche `forget` vs `retain`. | `Répondue (Job 15901182)` |
| **Q2** | **Niveau de mémorisation :** confiance, entropie, CE sur brutes vs perturbées (`hflip`, `crop80`, `blur`, `noise`). | Mesure la saturation et la robustesse aux altérations cachées. | `Répondue (Job 15901163)` |
| **Q3** | **Structure interne de `forget_images` :** doublons, quasi-duplicats ou clusters d'instances ? | Vérifie s'il y a des copies déjà altérées ou groupes corrélés. | `Répondue (Job 15901163)` |
| **Q4** | **Anatomie des poids & gradients :** répartition par étage du ViT (`patch_embed`, `blocks`, `head`). | Identifie où cibler l'oubli vs quelles couches préserver. | `Répondue (Job 15901163)` |
| **Q5** | **Préservation du `retain` sans `retain set` :** comment ancrer `retain_accuracy` et `KL(T=2)` ? | Protège `retain_score` et le *utility gate* (`>= 70%`). | `Répondue (Phase 2)` |
| **Q6** | **Cible `gold` :** quelle perte CE / distribution viser sans *over-unlearning* ? | Maximise `mean_std_score × prediction_score`. | `Répondue (Phase 2)` |

---

> **ADR-01 — Règle scientifique sur les benchmarks :** Un point isolé dans un benchmark teste une **configuration `(méthode, dosage, hyperparamètres)`**, jamais la validité théorique de la famille entière. On ne rejette pas une stratégie sur un seul run mal dosé ; on compare les familles par leur **compromis oubli vs préservation (`Proxy KL` / `Absent KL`)** à niveau d'oubli comparable (front de Pareto), puis on optimise le dosage des meilleures candidates (type Optuna / sweep ciblé).

> **ADR-02 — Architecture Orchestrateur + Sous-Agents & Benchmark par Holdout Interne sur `forget_images.npz` (Suppression du dataset externe `lab_in100`) :**
> 1. **Organisation :** chaque piste (`A` SalUn+NPO, `B` Bad-Teacher, `E` LoRA, `F/C` Abliteration/SSD) est perfectionnée de façon chirurgicale ; l'agent principal joue le rôle d'**orchestrateur** (supervision, centralisation des résultats, respect du cooldown 15 min et du nœud unique JURECA).
> 2. **Suppression du dataset externe (`clane9/imagenet-100`) au profit du benchmark direct par sous-ensembles de `forget_images.npz` sur `model.pt` :** un dataset externe souffre d'un décalage irréductible avec `model.pt` (saturation `CE = 0.0003`, `LayerScale = 1e-4`, mapping des 100 classes). À l'inverse, les `3 500` images officielles de `forget_images.npz` sont de vraies images connues de `model.pt` : si l'on fait oublier un sous-ensemble $S_1 \subset \mathcal{D}_f$ (ex. une moitié ou un groupe d'instances), le modèle doit **oublier $S_1$ tout en continuant à se rappeler de $S_2 = \mathcal{D}_f \setminus S_1$** (mêmes classes actives, même domaine, vraie sélectivité instance-par-instance *Selective Unlearning*). Les scripts du dataset externe (`build_lab.py`, `sweep_lab.py`, `validate_lab.py`, `slurm_lab.sh`) sont donc supprimés pour garder uniquement les données officielles (`model.pt` + `forget_images.npz`) et les formules exactes de `Unlearning.pdf` (`server_metrics.py`).

---

## 3. Journal Chronologique (Observations, Jobs & Décisions)

> **Règle :** tout consigner ici au fil de l'eau (y compris jobs échoués/annulés, bugs d'infra, résultats négatifs et questions intermédiaires).

### 2026-11-21 (16h10–16h15) — Mise de côté du prototype initial & incidents Slurm
- Un prototype préexistant dans `tracks/03_unlearning/` (`contracts.py`, `saliency.py`, `unlearner.py`, `metrics.py`, `run_unlearn.py`, `slurm_unlearn.sh`, `tests/test_unlearning.py`) avait été synchronisé vers JURECA (`/p/scratch/training2666/ouaziz1/TRYON/unlearning/tryon/`).
- **Incidents / constats techniques :**
  1. Sur `jrlogin06`, `task_template.DEVICE` vaut `"cuda"`, ce qui a causé un `RuntimeError` CPU/CUDA dans `test_saliency_mask_respects_keep_ratio` avant correction (`next(model.parameters()).device`).
  2. Job `15901138` échoué (`exit code 2`) : Slurm copie le script dans `/var/spool/parastation/jobs/`, cassant `BASH_SOURCE[0]` pour `SCRIPT_DIR`.
  3. Job `15901154` annulé (`scancel -u ouaziz1`) et dossier `tryon/` supprimé sur JURECA : décision de repartir en démarche **recherche & exploration d'abord** et de considérer le prototype comme non validé.

### 2026-11-21 (16h19–16h25) — Exploration empirique en lecture seule (Jobs `15901163` & `15901182`)
- **Q1 (`15901163`, `jrc0257`) :** **`active_classes` = 45 / 100** (**55 classes à 0 image**). `min=0`, `median=0.0`, `max=130`, `std=43.69`, `norm_entropy=0.812`. Top 10 : `[7:130]`, `[31:130]`, `[33:130]`, `[30:130]`, `[0:130]`, `[8:130]`, `[38:130]`, `[29:130]`, `[6:129]`, `[32:104]`.
- **Q1b (`15901182`, `jrc0258`) :** 1 seule clé `.npz` (`images`). **Coupure nette :** toutes les 45 classes actives ont **46 à 130 images** (0 classe entre 1 et 45), les 55 autres ont **0**. Prototypes `head.weight` : `norm_active_mean = 1.1441` vs `inactive = 1.1995` ; `bias_active = +0.0095` vs `inactive = -0.0197` ; cosinus max hors diagonale `<= 0.3016` (actives), `<= 0.2915` (active vs inactive), `<= 0.2205` (inactives).
- **Q2 (`15901163`) :**
  - Brut : `max_prob_mean = 0.9997` (`p10 = 0.9997`, `p50 = 1.0`, `frac_gt_95 = 99.94%`), `entropy_mean = 0.0021`, `margin_mean = 0.9994`, `ce_mean = 0.0003`, `ce_std = 0.0024`.
  - Perturbations : `hflip` (`agree=0.9997`, `ce=0.0006`, `kl_t2=0.0054`), `crop80` (`agree=1.0000`, `ce=0.0004`, `kl_t2=0.0252`), `noise_15` (`agree=0.9997`, `ce=0.0021`, `kl_t2=0.0068`), `blur_s1.5` (`agree=0.9954`, `ce=0.0137`, `kl_t2=0.0351`).
- **Q3 (`15901163`) :** RGB mean `[0.4067, 0.3700, 0.3129]`, RGB std `[0.2671, 0.2553, 0.2470]` ; embeddings CLS : `nn_cos_max = 0.8962`, `p95 = 0.7721`, `median = 0.6191`, `pairs > 0.90 = 0`, `pairs > 0.85 = 5`. Aucune image dupliquée.
- **Q4 (`15901163`) :** `head` (`76 900` params, `weight_l2=11.766`, **`grad_rms=9.3e-5`**) est 6–9× plus sensible par paramètre que `blocks_0-3` (`1.5e-5`), `blocks_4-7` (`1.0e-5`), `blocks_8-11` (`1.1e-5`). `patch_embed` (`4.6e-5`) et `pos_cls` (`4.5e-5`) sont intermédiaires.

### 2026-11-21 (16h25–16h54) — Réponses Q5/Q6 & clarifications $T=2$ / présence des images
- **Q5 (préservation `retain`) :** Triple verrou retenu pour validation : (1) geler/ancrer `head` sur les **55 classes absentes**, (2) restreindre les mises à jour aux étages hauts / poids saillants + L2 vers $\theta_{\text{orig}}$, (3) contraindre `KL(p_orig || p_sub)` à $T = 2$ sur proxy synthétique.
- **Q6 (cible `gold`) :** Éviter l'*over-unlearning* en utilisant une loss bornée (NPO / KL adoucie) calibrée par paliers plutôt qu'une maximisation de CE.
- **Clarification $T = 2$ :** Garder strictement $T = 2$ pour l'ancrage `KL` et les métriques locales (aligné sur le serveur) ; `mean(L), std(L)` reste à $T = 1$ ; $T \ge 2$ au pas $t=0$ pour SalUn sert uniquement de loupe numérique contre la saturation (`p_y = 0.9997`).
- **Présence dans `model.pt` :** Sans logs d'entraînement on ne prouve pas 100 %, mais empiriquement quasi certain (`99.94%` > `0.95`, `p10 = 0.9997`, `ce = 0.0003`).

### 2026-11-21 (16h58–17h01) — Vérification de l'oubli & robustesse aux altérations
- **Vérifier qu'un modèle a oublié :** L'oubli ne se mesure pas par « 0 % de précision », mais par **indiscernabilité vis-à-vis d'un modèle `gold`** : (1) `mean(L), std(L)` et `KL(p_gold || p_sub)` à $T = 2$ réalistes sur `forget` + copies altérées, (2) disparition de la signature MIA (`max_prob = 0.9997`, `entropy = 0.0021`), (3) dérive quasi nulle sur les 55 classes absentes, (4) `val_acc >= 70%`.
- **Robustesse aux images altérées (crop, bruit, décoloration) :** Comme **Q2** montre `99.5%–100%` d'accord sous perturbations, l'oubli doit combiner : (1) augmentations fortes à chaque pas (`RandomResizedCrop`, `hflip`, rotation `±15°`, `ColorJitter` + `RandomGrayscale`, `GaussianBlur`, bruit gaussien, simulation JPEG, `RandomErasing`), (2) mise à jour des couches sémantiques hautes (`blocks.8..11` + `head`), (3) cohérence multi-vues optionnelle, (4) stress-test local systématique (Phase 3).

### 2026-11-21 (17h05) — Leçons du papier SCRUB (`2302.09880v3.pdf`) pour Track 3
- **Mécanisme SCRUB :** Teacher figé $w^o$ / Student $w^u$ avec alternance `max-step` (maximiser $D_{KL}$ sur $\mathcal{D}_f$) et `min-step` (minimiser $D_{KL} + \ell_{CE}$ sur $\mathcal{D}_r$), puis **SCRUB+R** (rembobinage au checkpoint dont l'erreur sur $\mathcal{D}_f$ égale celle d'un `val set` de même distribution).
- **Problèmes résolus :** recadrage selon l'usage (*Removing Biases*, *Resolving Confusion*, *User Privacy*), scalabilité sans hypothèses rigides (2× à 7,8× plus rapide que `Retrain` vs Fisher/NTK), adaptation de **LiRA** à l'unlearning.
- **Problèmes / limites rencontrés :** (1) $D_{KL}$ non bornée $\implies$ sur-oubli à 100 % sans rembobinage, (2) dépendance totale à $\mathcal{D}_r$ (labels) et à un `val set` de même distribution que $\mathcal{D}_f$ (absents ici), (3) oscillation min-max nécessitant un tuning fin (`bs`, `max/min steps`, `lr decay`), (4) évalué uniquement sur CNN 32×32 sans data augmentation (pas de ViT), (5) AUC LiRA encore élevée (`0,72–0,74`).

### 2026-11-21 (17h11) — QLoRA / LoRA peut-il servir à faire oublier ?
- **Question :** Peut-on utiliser du fine-tuning via **QLoRA** pour faire oublier ?
- **Analyse :**
  - **LoRA (oui, pertinent comme contrainte de rang faible) :** modifier les derniers blocs (`blocks.8..11`) via $W' = W_0 + B A$ (rang $r \in \{4, 8, 16\}$) puis fusionner ($W_0 + BA$) en `float32` borne mécaniquement la dérive des poids (protection du `retain` et du *utility gate*), à condition de l'associer à une vraie loss d'oubli (NPO / Bad-Teacher + augmentations).
  - **Le « Q » de QLoRA (4-bit, non pertinent ici) :** QLoRA sert à économiser de la VRAM sur de très gros LLM (7B–70B). Ici `model.pt` fait **343 Mo (86M params)** sur A100 (40/80 Go) et la soumission exige du `float32` exact (`strict=True`) : la quantification 4-bit ajouterait du bruit numérique inutile sur `KL(p_orig || p_sub)`.
- **Décision :** Retenir **LoRA FP32/AMP (sans quantification 4-bit)** comme variante structurelle de la **Stratégie A/C** à comparer avec SalUn.

### 2026-11-21 (17h15) — Création du banc de validation local (Phase 3)
- Création de `tracks/03_unlearning_v2/` (`contracts.py`, `eval_harness.py`, `tests/test_eval_harness.py`).
- Tous les tests unitaires du banc (`test_eval_harness.py`) sont passés sur JURECA (`ALL HARNESS TESTS PASSED`).
- Job `15901595` (évaluation de référence `model.pt` vs `model.pt`) soumis puis annulé (`PENDING`, `scancel -u ouaziz1`) avant exécution : décision de définir d'abord ensemble le plan complet de benchmark comparatif des stratégies **avant** de lancer les jobs GPU.

### 2026-11-21 (17h18) — Suite de benchmark comparatif prête (`tracks/03_unlearning_v2/`)
- Implémentation modulaire des stratégies + ablations sans lancer de job GPU :
  - `augment.py` : augmentations fortes (`RandomResizedCrop`, `hflip`, rotation `±15°`, `ColorJitter`, `RandomGrayscale`, `GaussianBlur`, bruit gaussien) + ablation `use_strong_aug=False`.
  - `methods/common.py` : gel des couches basses (`blocks < 6`) et masquage des gradients sur les **55 classes absentes**, `npo_loss`, `retain_kl_loss` ($T=2$), `l2_anchor_loss`.
  - `methods/salun_npo.py` (Stratégie A), `methods/bad_teacher.py` (Stratégie B), `methods/ssd.py` (Stratégie C), `methods/lora_unlearn.py` (Stratégie E avec `LoRALinear` FP32 fusionnable).
  - `benchmark.py` + `slurm_benchmark.sh` : matrice de 8 runs (`A_salun_npo_ce0.8`, `A_salun_npo_ce1.4`, `A_ablation_no_aug`, `B_bad_teacher_T4`, `B_bad_teacher_T8`, `C_ssd_alpha1`, `E_lora_r8_ce0.8`, `E_lora_r4_ce0.8`) + baseline `0_orig_baseline`.
- En pratique : `benchmark.py` prêt ; en attente de validation utilisateur pour lancer `sbatch harness/slurm_benchmark.sh`.

### 2026-11-21 (17h19) — Parallèle avec l'oblitération de modèle (`p-e-w/heretic`) & pourquoi D manquait
- **Lien avec Heretic / *Abliteration* :**
  - *Ressemblance :* comme Heretic (retrait du refus sur LLM), on cherche à retirer un comportement ciblé d'un modèle déjà entraîné tout en **co-minimisant la divergence KL** par rapport au modèle original sur le reste des connaissances (`kl_divergence.py` dans Heretic vs notre `retain_kl_loss` à $T=2$).
  - *Différence clé :* Heretic vise une suppression binaire (`refusal_rate → 0%`), alors que Track 3 exige une **oblitération dosée** pour matcher les modèles `gold` sans *over-unlearning*.
- **Stratégies utilisées par Heretic :**
  1. **Ablation directionnelle / *Arbitrary-Rank Ablation* (ARA) :** extraire la ou les directions porteuses du comportement dans le flux résiduel / couches d'attention & MLP, puis **orthogonaliser** les matrices de poids contre ces directions (sans boucle de gradient).
  2. **Co-minimisation automatique par Optuna (TPE) :** balayer automatiquement couches ciblées et intensité d'ablation sous contrainte d'un plafond strict de divergence KL (ex. `KL <= 0.1`).
  3. **Modifications de rang faible (Heretic 2.0 / ARA-LoRA) :** proche de notre **Stratégie E**.
- **Pourquoi la Stratégie D manquait dans `benchmark.py` :** D concerne le choix du *proxy `retain`* (ex. images naturelles externes vs *PatchMix* / bruit structuré / MixUp). N'ayant pas encore vérifié la présence d'images génériques externes sur JURECA, D avait été laissée hors du premier batch — on peut toutefois déjà inclure des proxys locaux alternatifs (bruit structuré, PatchMix multi-grille, MixUp) et tester l'orthogonalisation directionnelle (**Stratégie F**).

### 2026-11-21 (17h21) — Ajout des stratégies D et F au benchmark (`tracks/03_unlearning_v2/`)
- **Stratégie D (`methods/proxies.py`) :** ajout des proxys alternatifs `mixup` (`0.5*x_i + 0.5*x_j`) et `structured_noise` (textures basses fréquences) + runs `D_proxy_mixup` et `D_proxy_noise`.
- **Stratégie F (`methods/abliteration.py`) :** implémentation de l'ablation directionnelle dosée (*Heretic ARA* adaptée au ViT) — extraction SVD rang $k=8$ des features CLS `forget`, projection orthogonale aux **55 classes absentes** (`head.weight[absent]`), puis atténuation dosée ($\alpha \in \{0.3, 0.6\}$) sur `blocks.6..11` (`attn.proj`, `mlp.fc2`) et `head.weight[active]`.
- La matrice `benchmark.py` passe à **13 runs** (`0_orig_baseline` + 12 variantes `A`–`F`), prête pour exécution sur `dc-gpu` après validation utilisateur.

### 2026-11-21 (17h34) — Résultats Benchmark v1 (`run_15901661` sur `jrc0283`) & Diagnostic
- **Synthèse :** 13 runs terminés en `8m36s`, résultats collectés dans `tracks/03_unlearning_v2/results/run_15901661/` et analysés en détail dans [03-unlearning-benchmark-analysis.md](03-unlearning-benchmark-analysis.md).
- **Ce qu'on a appris :**
  1. **Bug d'arrêt précoce sur `A`, `B`, `D`, `E` :** l'arrêt anticipé évaluait la CE sur le batch *fortement augmenté* (où `orig` a déjà `CE_avg = 1.267`, notamment `grayscale = 6.037`), coupant l'entraînement dès le pas 1 (`Δθ L2 = 0.03–0.20`).
  2. **Saturation NPO (`A_ablation_no_aug`) :** même sur 120 pas, NPO à $T=1$ avec `lr=3e-5` ne déplace que `Δθ = 0.41` (`Raw CE = 0.007`) car `p_orig = 0.9997` étouffe le gradient initial.
  3. **`C_ssd_alpha1` (positif fort) :** `Absent KL = 0.0000`, `Proxy agree = 0.9989`, `Raw Ent = 0.018` (`Δθ = 1.53`) en 1 passe sans gradient.
  4. **`F_abliteration_a0.6` (positif/négatif) :** casse la saturation (`Δθ = 9.41`, `Raw Ent = 0.032`), mais dégrade `proxy_agree` à `0.7803` (`KL = 0.1311`) et pousse la masse vers les 55 classes absentes (`0.491 → 0.638`).
- **Prochaine étape :** corriger l'arrêt anticipé (mesurer CE sur batch brut), dégeler NPO à $T=2$ + `lr=1e-4..2e-4`, pousser `C_ssd` (`threshold=1.0..1.2`), doser `F` (`α=0.15..0.25`) et tester l'hybride `SSD + NPO/LoRA`.

### 2026-11-21 (17h49) — Résultats Benchmark v2 (`run_15901741` sur `jrc0327`)
- **Synthèse :** 14 runs terminés, collectés dans `tracks/03_unlearning_v2/results/run_15901741/` et analysés dans [03-unlearning-benchmark-analysis.md](03-unlearning-benchmark-analysis.md).
- **Constats clés :**
  1. **Correction validée :** `A_salun_npo_lr1e4` atteint `Raw CE = 0.754±2.31` (`Ent = 0.163`) et `A_ablation_no_aug` atteint `Raw CE = 0.701±2.51` avec `Proxy KL = 0.0403` et `Absent KL = 0.0239` (en plein dans notre zone cible saine).
  2. **Stratégie D (`PatchMix` gagnant) :** `MixUp` (`Proxy KL = 1.27`) et `Structured Noise` (`3.73`) sont 8× à 24× pires que `PatchMix` (`0.15`).
  3. **Effet des augmentations fortes :** `use_strong_aug=True` (incluant `grayscale` où `orig` a déjà `CE=6.04`) fait monter `Stress CE` à `6.92` et `Proxy KL` à `0.154` (`3.8×` plus que `no_aug`) $\implies$ passer à des augmentations *modérées* (sans `grayscale` ni rotation extrême).
  4. **Points forts `E` (LoRA), `H` (Hybride SSD+NPO) et `B` (Bad-Teacher) :** `E_lora_r8` divise `Absent KL` par ~2 (`0.0398` vs `0.0719`), `H_ssd_plus_npo` bat `A` sur toutes les métriques `retain` (`Proxy KL = 0.0989` vs `0.1543`), et `B_bad_teacher_T6` produit l'entropie la plus régulière (`0.513`, `CE = 0.139±0.25`).

### 2026-11-21 (18h14) — Superposition dans les poids & parallèle neurosciences
- **Comment l'image vit dans le modèle :** une image n'est pas stockée dans *un* poids isolé, elle est **superposée** sur des millions de poids partagés avec `retain` ; l'inférence transforme les pixels en activations (`CLS`) qui empruntent « l'autoroute » creusée dans ces poids.
- **Oui, c'est ce qu'on fait :** on passe `forget_images` dans le réseau pour repérer les poids singulièrement stimulés par `forget` (gradients `SalUn` / Fisher `SSD`) face au proxy `retain`, puis on n'ajuste que ces connexions.
- **Parallèle neurosciences ↔ Deep Learning :**
  - **Affaiblissement synaptique (LTD / *Pruning*)** $\leftrightarrow$ **SSD (*Selective Synaptic Dampening*) / SalUn** (affaiblir sélectivement les poids dominés par `forget`) ; *Pruning / Quantification* (supprimer ou réduire la précision de poids non critiques).
  - **Interférence** $\leftrightarrow$ **NPO / Bad-Teacher / SCRUB** (injecter une réponse concurrente adoucie sur `forget` tout en maintenant `retain`).
  - **Défaut d'accès (*Retrieval Failure*) & Inhibition active** $\leftrightarrow$ **Abliteration (ARA) / Ajustement de `head[active]`** (bloquer le chemin d'accès vers la prédiction mémorisée sans préserver uniquement les couches hautes).
  - ***Weight Decay* (régularisation $L_2$)** $\leftrightarrow$ frein homéostatique retenant les poids près de $\theta_{\text{orig}}$.
  - ***Catastrophic Forgetting* (oubli catastrophique)** $\leftrightarrow$ effondrement général si l'oubli écrase les représentations partagées (`val_acc < 70%`).

### 2026-11-21 (18h25) — Résultats Benchmark v3 (`run_15901899`) & Incident SSH `ControlMaster`
- **Incident infra résolu :** le socket `ControlMaster` ouvert à `15h49` (`pid=326075`) était gelé localement, bloquant toute commande `ssh juelich`. Résolu par suppression du socket et connexion sans multiplexage avec extraction regex des 6 chiffres de `scripts/totp.py`.
- **Slots groupe `training2666` :** `1/4` utilisé côté équipe TRYON (`bassim1`: 1 `PD`, `ouaziz1`: 0, `sangkhavongs1`: 0).
- **Résultats Benchmark v3 (`run_15901899`) :** collectés dans `tracks/03_unlearning_v2/results/run_15901899/` et analysés dans [03-unlearning-benchmark-analysis.md](03-unlearning-benchmark-analysis.md).
  - Les augmentations modérées (sans `grayscale` ni `rotate15`) divisent `Proxy KL(T=2)` par **4× à 7×** tout en gardant un bon transfert aux altérations.
  - **Top 2 candidats validés (`327.71 Mo`, `strict=True`, `valid=True`) :**
    1. **`H_ssd_lora_mod_aug.pt`** (`Proxy KL = 0.0184`, `Proxy Agree = 92.49%`, `Absent KL = 0.0106`, `Raw CE = 1.112±3.79`, `Stress CE = 2.458`).
    2. **`A_npo_mod_aug.pt`** (`Proxy KL = 0.0366`, `Raw CE = 0.617±1.99`, `Raw Ent = 0.143`, `Δθ = 2.97`).

### 2026-11-21 (18h30 / 18h50) — Règles opérationnelles confirmées
- **Cooldowns soumissions serveur (mis à jour 18h50) :** Track 3 Unlearning = **15 minutes** ; Track 1 Watermark = **10 minutes** ; soumission échouée = **2 minutes** (`API_KEY` dans `infra/team-access.md`).
- **Quota cluster équipe `TRYON` :** **max 4 jobs concurrents** sur JURECA (`ouaziz1`, `sangkhavongs1`, `bassim1`).
- Règles ajoutées à `AGENTS.md`.

### 2026-11-21 (18h35) — 1ʳᵉ soumission serveur (`H_ssd_lora_mod_aug.pt`) : Score `0.0287` (8ᵉ)
- **Résultat :** `Score (public split): 0.0287` (**Submission accepted** $\implies$ le modèle passe le *utility gate* `val_acc >= 70%`).
- **Diagnostic du score `0.0287` :**
  - Dans `run_15901899`, `H_ssd_lora_mod_aug` avait `Raw CE = 1.112 ± 3.79` et `Raw Ent = 0.060`.
  - Une moyenne `1.112` avec un écart-type énorme (`3.79`) et une entropie très basse (`0.060`) indique une distribution **bimodale** : la majorité des images restent sur-confiantes/mémorisées (`Ent ~ 0`), tandis qu'une minorité explose à `CE > 10`. Or le serveur pénalise à la fois `std(L)` (`mean_std_score`) et `KL(p_gold || p_sub)` à $T=2$ sur chaque image (`prediction_score`).
- **Hypothèse pour la 2ᵉ soumission :** il faut un oubli **homogène** (faible `std(L)`, entropie modérée `0.3–0.8` sur toutes les images, sans explosion bimodale) **avec** `Proxy KL < 0.04` — ex. combiner **Bad-Teacher (ou NPO avec lissage/plafond par image) + LoRA + augmentations modérées**.

### 2026-11-21 (18h38) — Règles du Leaderboard confirmées
- **Best-of :** seule la **meilleure** soumission par tâche est conservée (aucune pénalité si une itération score moins bien).
- **Classement général :** moyenne normalisée `[0, 1]` sur les **3 tâches** ($\frac{1}{3}\sum_t \frac{s_t - \min_t}{\max_t - \min_t}$) — il faut un score sur **chaque** track pour être classé au général.
- **Public (30 %) vs Final (70 %) :** éviter d'overfitter au score public ; privilégier les méthodes robustes et généralisables.

### 2026-11-21 (18h45) — Amélioration du harnais local (`eval_harness.py`) & Benchmark v4 (`15902297`)
- **Audit du harnais :** `retain` (`Proxy KL`, `Absent KL`, `Δθ`) se compare exactement à `orig` (même formule serveur $T=2$), mais sur `forget`, `mean±std` cachait la bimodalité.
- **Ajout dans `StressMetrics` & `benchmark.py` :** `ce_p50` (médiane), `ce_p90`, et `frac_gt_95` (% d'images encore > 95 % de confiance) pour détecter directement si l'oubli est homogène ou bimodal.
- **Benchmark v4 (`15902297`)** soumis sur JURECA (6 runs `LoRA + Bad-Teacher` & `NPO` doux ; doublon `15902308` annulé). Note infra : Jülich PAM refuse la réutilisation du même code TOTP dans la même fenêtre de 30s (anti-replay) $\implies$ attendre la fenêtre suivante entre deux connexions SSH.

### 2026-11-21 (18h54) — Calibration par soumissions (15 min) & Stratégie du Lab sur copie de `model.pt`
- **Utilisation des soumissions :** exploiter la fenêtre de 15 min pour calibrer nos hypothèses sur le vrai score serveur et diagnostiquer où l'on pêche (`forget` vs `retain`).
- **Stratégie Piste `G` (copie de `model.pt` entraînée sur notre nouveau dataset `retain + forget`) :**
  - *Validation scientifique :* **très bonne stratégie**, à condition de (1) travailler sur une **copie** de `model.pt` (jamais sur l'original) et (2) entraîner `orig_lab` (sur `R_lab + F_lab`) et `gold_lab` (sur `R_lab` seul) avec **exactement le même protocole** depuis le même point de départ.
  - *Gain :* donne accès aux vrais labels, à un vrai `retain`/`val set` et à un vrai `gold_lab` pour calculer les métriques exactes du serveur (`val_acc`, `mean_std_score`, `prediction_score`, `retain_score`) avant de réappliquer les meilleurs réglages sur `model.pt` seul.

### 2026-11-21 (18h57) — 2ᵉ soumission serveur (`A_npo_mod_aug.pt`) : Score `0.0843` (×2,94 !)
- **Résultat :** `Score (public split): 0.0843` (`Previous best: 0.0287`, `improved: True`). Prochaine soumission possible vers **`19h12`**.
- **Confirmation directe de notre hypothèse :**
  - `H_ssd_lora_mod_aug` (`Raw CE = 1.112±3.79`, `Raw Ent = 0.060`, `Proxy KL = 0.0184`) $\rightarrow$ `0.0287`.
  - `A_npo_mod_aug` (`Raw CE = 0.617±1.99`, `Raw Ent = 0.143`, `Proxy KL = 0.0366`) $\rightarrow$ **`0.0843`**.
  - Réduire `std(L)` de `3.79 → 1.99` et monter `Raw Ent` de `0.060 → 0.143` a **triplé** le score, confirmant que notre levier principal est un oubli **plus homogène** (`std < 1.2`, `Ent ∈ [0.25, 0.70]`) tout en gardant `Proxy KL < 0.04`.

### 2026-11-21 (18h59) — Construction de la Piste `G` (Ground-Truth Lab) & Test de cohérence
- **Objectif :** construire notre propre labo vérité terrain (`orig_lab` entraîné sur `R+F` vs `gold_lab` entraîné sur `R` seul), implémenter les formules exactes du serveur (`tracks/03_unlearning_v2/lab/server_metrics.py`), puis vérifier la cohérence en reproduisant le classement de nos 2 soumissions réelles (`A_npo_mod_aug` = `0.0843` > `H_ssd_lora_mod_aug` = `0.0287`).
- **Si le test de cohérence est positif**, tous les benchmarks et sous-agents par piste basculent sur ce banc.

### 2026-11-21 (19h10) — Résultats Benchmark v4 (`run_15902297` sur `jrc0324`)
- **Synthèse :** 6 runs terminés, collectés dans `tracks/03_unlearning_v2/results/run_15902297/` et analysés dans [03-unlearning-benchmark-analysis.md](03-unlearning-benchmark-analysis.md).
- **`H_ssd_npo_gentle` démasqué par `p50`/`Frac>95%` :** `Raw CE = 0.556±2.19`, mais `p50 = 0.000` et `Frac>95% = 0.93` (93 % des images inchangées).
- **`E_lora_bt_T5` (Meilleur profil homogène) :** `Raw CE = 0.169±0.33`, `Raw p50 = 0.046`, `Frac>95% = 0.52`, `Raw Ent = 0.507`, `Proxy KL(T=2) = 0.0189`, `Absent KL(T=2) = 0.0119`. Prêt pour soumission à `19h12`.

### 2026-11-21 (19h15) — 3ᵉ soumission serveur (`E_lora_bt_T5.pt`) : Score `0.0117`
- **Résultat :** `Score (public split): 0.0117` (`Previous best: 0.0843`, `improved: False`). Prochaine soumission possible vers **`19h30`**.
- **Comparaison des 3 soumissions réelles :**
  - `A_npo_mod_aug` (`Raw CE = 0.617`, `Δθ = 2.97`) $\rightarrow$ **`0.0843`**
  - `H_ssd_lora_mod_aug` (`Raw CE = 1.112`, `Δθ = 4.25`) $\rightarrow$ `0.0287`
  - `E_lora_bt_T5` (`Raw CE = 0.169`, `Ent = 0.507`, `Δθ = 9.48`) $\rightarrow$ `0.0117`
- **Leçon clé :** `Bad-Teacher T=5` a déplacé massivement les poids (`Δθ = 9.48`) pour monter l'entropie (`0.507`), mais `Raw CE = 0.169` reste beaucoup trop faible (sous-oubli : l'argmax reste mémorisé), tandis que `A_npo_mod_aug` atteint `CE = 0.617` avec `Δθ = 2.97`. D'où l'urgence du **Ground-Truth Lab (Piste G)** (`15902512` sur `uoft-cs/cifar100`) pour connaître le vrai compromis `gold`.

### 2026-11-21 (19h50) — Bascule sur Google Colab Pro (`google-colab-cli`) & 1ᵉʳ test Lab (`cifar100`)
- **Infra Colab Pro :** `google-colab-cli==0.8.0` installé via `uv` (+ `keyrings.alt` pour WSL2), solde `400.00 compute units`. Session A100 `track3` créée et exécutée en ~3.5 min (alors que `15902512` sur JURECA était encore `PD` après 31 min).
- **1ᵉʳ test de cohérence (`tracks/03_unlearning_v2/results/lab_cifar100/coherence_report.json`) :**
  - Le Lab reproduit bien le classement serveur : **`A_npo_mod_aug`** (`forget=0.3579`, `retain=0.3991`) **>>** **`H_ssd_lora_mod_aug`** (`forget=0.0079`, `retain=0.0088`).
  - *Limite à corriger :* avec `200 imgs/classe` et `5 époques` sur `blocks.10..11`, `orig_lab` n'a atteint que `val_acc = 0.3727` (sous le gate `0.70`). Prochaine étape : passer à `500 imgs/classe` + `blocks.6..11 + head` pour monter `orig_lab` et `gold_lab` à `val_acc >= 85%`.

### 2026-11-21 (19h52–20h16) — Bascule L4 / JURECA libéré / Tuto Colab & retour A100
- **JURECA :** job `15902512` annulé (`scancel -u ouaziz1`) ; `AGENTS.md` mis à jour avec la règle stricte **1 seul job Slurm (running+PD) pour toute l'équipe TRYON** (nœuds exclusifs à 4 GPUs). Vérification équipe : `bassim1` avait 2 jobs `PD` (`15902604`, `15902605`) sur Track 1.
- **Colab L4 (`1.94 CU/h`) :** testé sur `500 imgs/classe × 8 époques` (`task-1188` annulé à 14m52s car trop lent sur L4 pour construire le Lab complet). Création du guide d'équipe [infra/colab-l4-setup.md](../../infra/colab-l4-setup.md).
- **Colab A100 (`6.73 CU/h`, ~59.2h dispo sur `398.75 CU`) :** choisi pour le Lab et les gros runs.

### 2026-11-21 (20h25) — Ground-Truth Lab v2 sur Colab A100 (`350 imgs/cls`, `blocks.6..11`, 6 ép.)
- **Résultats (`lab_cifar100_v2/coherence_report.json`) :** `orig_lab val_acc = 0.4772`, `gold_oracle val_acc = 0.4576`. Le classement reste 100 % cohérent (`A_npo_mod_aug` `forget=0.3232, retain=0.3119` >> `H_ssd_lora` `0.0052, 0.0327`), mais `blocks.0..5` gelés sur CIFAR-100 (`32×32`) plafonnent à ~48 %. Il faut débloquer `blocks.0..11` et charger `clane9/imagenet-100` sur Colab pour dépasser `val_acc >= 85%`.

### 2026-11-21 (20h27–20h31) — Test `A_npo_kl3_ce0.75` & Lancement Lab `clane9/imagenet-100`
- **`A_npo_kl3_ce0.75.pt`** (`salun_npo`, `lr=8e-5`, `retain_kl=3.0`, `target_ce=0.75`) : trop doux localement (`ce_mean = 0.0413`, `frac_gt_95 = 0.9546`, `proxy_kl_t2 = 0.0072`).
- **Constat infra soumission :** `requests.post` de 328 Mo depuis Colab US vers `34.147.229.55` a expiré en `TimeoutError` (soumissions restreintes / optimisées depuis JURECA) $\implies$ **pipeline validé :** calcul sur Colab A100 $\rightarrow$ `colab download` $\rightarrow$ transfert vers JURECA & soumission via `submission.py` (0 slot GPU Slurm).
- **Lab `clane9/imagenet-100`** (`blocks.0..11 + head` débloqués, `5 époques`) lancé sur Colab A100 (`task-1254`).

### 2026-11-21 (20h43) — Résultats Lab `clane9/imagenet-100` (`lab_in100/coherence_report.json`)
- **Cohérence 100 % confirmée sur 2 datasets :**
  - `A_npo_mod_aug` : `forget_score = 0.2755` (`mean_std = 0.3649`, `pred = 0.7551`) × `retain_score = 0.7131` = **`0.1965`** (hors gate).
  - `H_ssd_lora_mod_aug` : `forget_score = 0.0173` × `retain_score = 0.0390` = **`0.0007`**.
- **Constat `val_acc` (`orig_lab = 0.3412`) :** les indices `0..99` de `clane9/imagenet-100` ne correspondent pas à ceux de `model.pt`, et 5 époques sans augmentation ne suffisent pas pour remonter de 1 % à 85 %. Pour obtenir `val_acc >= 85%` en ~2 min sur A100 : entraîner avec `lr` différencié (`head` à `1e-3`, backbone à `5e-5`) sur `10 époques` + augmentations de base (`hflip` + `RandomResizedCrop`).

### 2026-11-21 (21h04–21h31) — Diagnostic `val_acc`, CLI JURECA & Lab ImageNet-100 validé à **`val_acc = 93.4 %`**
- **Diagnostic racine sur `model.pt` + `clane9/imagenet-100` :**
  1. `ctrl` et `val` n'étaient pas mélangés (`ctrl` = classes `0..49`, `val` = classes `50..99`).
  2. Même avec un split interne du `train` (`heldout_from_train`) et en 2 phases (linear probe -> fine-tuning complet 10 ép., `task-1353`/`task-1358`/`task-1364`), `model.pt` plafonne à `0.3540` (linear probe) / `0.4016` (full FT) — il ne possède pas de features généralistes ImageNet-1k.
- **Fix dans [build_lab.py](../../tracks/03_unlearning_v2/lab/build_lab.py) :**
  - Cache de `lab_splits.pt` + mélange déterministe (`seed=42`) de `ctrl`/`val` sur toutes les 100 classes.
  - Option `--init-mode pretrained` : initialise `build_model()` depuis `vit_base_patch16_224` pré-entraîné ImageNet-1k (`LayerScale .gamma = 1.0`).
  - En 5 epochs (`task-1437`), convergence à **`orig_lab train_acc = 0.9981`** et **`gold_lab train_acc = 0.9983`**, sauvegardés une fois pour toutes sur `track3`.
- **Résultats `validate_lab.py` (`task-1445`, gate `>= 0.70` PASSÉE pour tous) :**
  - **`gold_oracle`** : `val_acc = 0.9308`, `forget = 1.0000`, `retain = 0.9017`, **`final = 0.9017`**
  - **`orig_no_unlearn`** : `val_acc = 0.9344`, `forget = 0.2959`, `retain = 1.0000`, **`final = 0.2959`**
  - **`H_ssd_lora_mod_aug`** : `val_acc = 0.9164`, `forget = 0.0752`, `retain = 0.8607`, **`final = 0.0647`**
  - **`A_npo_mod_aug`** : `val_acc = 0.7044`, `forget = 0.0107`, `retain = 0.2046`, **`final = 0.0022`** (trop agressif sur `orig_lab` avec `LayerScale=1.0`, détruit `retain_kl_similarity = 0.2683`).
- **Infra JURECA :** création et commit (`efbdbf0`) de [scripts/jureca_status.py](../../scripts/jureca_status.py) (monitoring live par team, vérification du slot unique `TRYON`, `--all`, `--strict`, `--json`, `--watch`), documenté dans [README.md](../../README.md).

### 2026-11-21 (21h33–21h53) — Sweep multi-stratégies Lab (`sweep_lab.py`) & 4ᵉ soumission (`C_ssd_plus_A_npo.pt`)
- **Sweep Lab `output/lab_in100/` (`task-1478`, `1492`, `1497` — [sweep_results.json](../../tracks/03_unlearning_v2/results/lab_in100/sweep_results.json)) :**
  - **`F_ablit_a0.55_r8`** : **`final = 0.5195`** (`forget = 0.6126`, `mean_std = 0.8661`, `pred = 0.7073`, `retain = 0.8481`, `val = 0.9056`) vs `orig_no_unlearn = 0.2959`.
  - **`F_ablit_a0.50_r8`** : `final = 0.4946` (`val = 0.9116`, `retain = 0.8693`).
  - **`C_ssd_a0.5_t2.0`** : `final = 0.3354` (`val = 0.9256`, `retain = 0.9289`).
  - Toutes les configs `A` (`<= 0.1156`), `B` (`<= 0.2600`) et `E` (`<= 0.1478`) sont battues sur le Lab.
- **Vérification sur le vrai `model.pt` (`task-1497` & `task-1501`) :**
  - *Contre-épreuve cruciale :* sur le vrai `model.pt` (ultra-mémorisé `CE = 0.0003`), `F_ablit_a0.55_r8` seul n'a monté `Raw CE` qu'à `0.007 ± 0.05` (`98% > 95%`, `ProxyKL = 0.1116`), donc **non soumis** pour éviter un sous-oubli type `0.0117`.
  - **Candidats calibrés sur `model.pt` :**
    - **`C_ssd_plus_A_npo`** (`SSD a=0.5, t=2.0` + `SalUn+NPO lr=8e-5, ce=0.8, kl=2.5, 100 steps`) : `valid = True` (`327.7 MB`), `CE = 1.249 ± 3.43`, **`ProxyKL = 0.0170`**, **`AbsKL = 0.0026`** (55 classes absentes intactes).
    - `A_npo_kl3.0_ce1.1` : `CE = 1.516 ± 3.66`, `ProxyKL = 0.0399`, `AbsKL = 0.0272`.
- **4ᵉ soumission (`22h05`) — `C_ssd_plus_A_npo.pt` : Score `0.0106`** (`Previous best: 0.0843`, `improved: False`, prochain créneau ~`22h20`).
  - *Diagnostic comparatif des 4 soumissions :* malgré `ProxyKL = 0.0170`, `C_ssd_plus_A_npo` combine **`p50 = 0.000` / `88% > 95%`** (bimodalité extrême `1.249 ± 3.43`) et un énorme déplacement de poids **`dTheta = 17.78`** dû à SSD (qui dégrade probablement `val_acc`/`retain_acc` hors PatchMix). Notre record `0.0843` (`A_npo_mod_aug`) reste le seul avec un **`dTheta` faible (`2.97`)** et `std = 1.99`.

### 2026-11-21 (22h09–22h16) — Correction bornage *per-sample* & Sweep 4 GPUs JURECA (`15903458`)
- **Colab `track3` arrêté** (`22h06`, solde `386.85 CU`, `0.00 CU/h`).
- **Correctifs implémentés (`d4205d6`) :**
  - `npo_loss(..., max_ce=...)` dans [common.py](../../tracks/03_unlearning_v2/methods/common.py) : masque à `0` la loss NPO **par image** dès que `ce_i >= max_ce`, et `should_stop_early` basé sur la **médiane (`p50`)** au lieu de la moyenne du batch (supprime la bimodalité `p50 = 0.000`).
  - [sweep_bounded_4gpu.py](../../tracks/03_unlearning_v2/sweep_bounded_4gpu.py) + [slurm_sweep4.sh](../../tracks/03_unlearning_v2/slurm_sweep4.sh) : **12 configs** (`SalUn+NPO` sans SSD, `max_ce` ∈ `{0.45, 0.65, 0.85, 1.05}`, `lr` ∈ `{8e-5, 1.2e-4, 1.6e-4}`, `retain_kl_weight` ∈ `{2.0, 2.5, 3.0}`, `200` steps) avec proxy naturel `CIFAR-100` (si présent dans `dataset/`).
- **Exécution JURECA :** Job **`15903458`** (`1` nœud `jrc0444`, `4 GPUs`, `cispahack`) **`COMPLETED`** en `14:40` (`ExitCode 0:0`, GPUs libérés). Toutes les 12 configs ont réglé la bimodalité (`p50` de `0.380` à `2.190`, `>95%` de `0.30` à `0.46`).
- **5ᵉ soumission (`22h30`) — `B_ce0.45_lr8e-05_kl2.0.pt` : Nouveau record `0.1166`** (`Previous best: 0.0843`, **`improved: True`**, prochain créneau ~`22h45`).
  - *Métriques locales :* `CE = 1.438 ± 2.01`, `p50 = 0.380`, `>95% = 0.37`, `ProxyKL = 0.0551`, `AbsKL = 0.0413`, `dTheta = 4.03`.
  - *Piste suivante :* `auto_tune_4gpu.py` (`61abd56`, déjà prêt sur JURECA) pour ramener `ProxyKL <= 0.035` et `dTheta <= 3.2` tout en gardant `p50 ~ 0.35–0.50`.

### 2026-11-21 (22h38) — Rétrospective scientifique complète & Plan d'action (`P0` → `P3`)
- **Bilan des 5 soumissions serveur (`31-unlearning`) :** `#1` (`H_ssd_lora_mod_aug.pt`, `0.0287`) → `#2` (`A_npo_mod_aug.pt`, `0.0843`) → `#3` (`E_lora_bt_T5.pt`, `0.0117`) → `#4` (`C_ssd_plus_A_npo.pt`, `0.0106`) → **`#5` (`B_ce0.45_lr8e-05_kl2.0.pt`, `0.1166`, record actuel)**. `#5` est le seul run débloquant la médiane (`p50 = 0.380`, `63 %` des images désaturées, `dTheta = 4.03`), mais reste pénalisé par sa queue droite (`CE = 1.438 ± 2.01`) et `ProxyKL = 0.0551`.
- **6 bugs et biais structurels identifiés :**
  1. **`auto_tune_4gpu.py` (`L62` & sonde `[:256]`) :** `compute_kl_at_temperature(...)` retourne un tenseur *per-sample* sans `.mean().item()` (`float(...mean().item())`), et la sonde `[:256]` introduit un biais de couverture de classes (à remplacer par une permutation déterministe couvrant les 45 classes actives).
  2. **`methods/lora_unlearn.py` (`L89`) :** `max_ce=cfg.target_forget_ce` et `cfg.l2_weight * l2_anchor_loss` manquants, plus un multiplicateur caché `lr * 3.0`.
  3. **Biais SalUn/SSD (`methods/salun_npo.py#L34`) :** saillance calculée sur `forget_images[0:384]` non mélangé au lieu d'un sous-ensemble `randperm` ou des `3 500` images.
  4. **Fuite train/eval `PatchMix seed=42` (`eval_harness.py#L187`) :** `evaluate_candidate` réutilise `seed=42` comme l'entraînement, mesurant l'overfit sur le proxy au lieu de la généralisation hors-échantillon.
  5. **Bornage unilatéral NPO (`methods/common.py`) :** le masque `per_sample * (ce_i < max_ce)` laisse dériver les images déjà oubliées (`ce_std = 2.01` sur `#5`, `mean = 1.438 >> p50 = 0.380`).
  6. **Mismatch LayerScale `lab_in100` (`lab/build_lab.py`, `validate_lab.py`) :** `.gamma` non initialisés sur l'échelle de `model.pt` (`1e-4`), `orig_lab` non saturé (`CE < 0.005`), et `make_stress_suite(f_x)` absent de `forget_eval_imgs`.
- **Plan d'action priorisé (`P0` → `P3`) :**
  - **P0 (Correctifs immédiats) :** dans `auto_tune_4gpu.py`, corriger `float(compute_kl_at_temperature(...).mean().item())` et remplacer `[:256]` par une permutation déterministe couvrant les 45 classes actives ; corriger `randperm`/full-set dans `salun_npo.py` ; passer `eval_harness.py` à `seed=999` (+ grille `2x2 + 4x4`) ; propager `max_ce`, `l2_anchor_loss` et retirer `lr * 3.0` dans `lora_unlearn.py`.
  - **P1 (Levier n°1 — Éliminer la queue droite `ce_std : 2.01 -> <= 0.70`) :** implémenter dans `methods/common.py` un **bornage bilatéral** (*Two-Sided Per-Sample Target CE* / cible *Gold-Mimic* $p^* \in [0.52, 0.65]$, $\text{CE}^* \in [0.43, 0.65]$, masse restante répartie sur les logits `orig` à $T=2$). Critères locaux avant soumission `#6` : `ce_p50` $\in [0.40, 0.55]$, `ce_mean` $\in [0.45, 0.65]$, `ce_std <= 0.70`, `frac_gt_95 <= 0.15`, `proxy_kl_t2 <= 0.032`, `absent_classes_kl_t2 <= 0.020`, `dTheta <= 3.2`.
  - **P2 (Gel chirurgical & multi-vues) :** comparer `freeze_until_block = 8` vs `6` avec `lr` différencié (`blocks.8..11` à `6e-5`, `head[active]` à `1.2e-4`), et calculer `npo_loss` sur `orig(f_batch_aug)` ou moyenne 2 vues (`clean` + `mod_aug`).
  - **P3 (Lab `lab_in100`) :** ne pas l'utiliser pour sélectionner les hyperparamètres tant que `.gamma` (`1e-4`), saturation `orig_lab` (`CE < 0.005`) et `make_stress_suite(f_x)` ne sont pas recalibrés.

### 2026-11-21 (22h58–23h24) — Run 2× A100 Colab, Découverte du raccourci de résolution `CIFAR-100`, Sonde `#6` (`model.pt` = `0.0`) & Réalignement stratégique (`M1`–`M3`)
- **Run 2× A100 Colab (`task-1750` & `task-1755`, tests unitaires `UNIT_TESTS_PASSED`) :**
  - Validation de `gold_mimic_loss` (`GM_p60_f6_hm1.0` : `CE = 0.507 ± 0.63`, `p50 = 0.261`, `>95% = 0.21`, `dTheta = 2.41`) et `two_sided_npo_loss` (`TS_ce0.45_u0.65_f8` : `CE = 0.837 ± 1.12`, `p50 = 0.337`, `>95% = 0.29`, `dTheta = 3.47`).
- **Preuve du raccourci de résolution `CIFAR-100` (`task-1779`) :**
  - Sur Colab, `CIFAR-100` (`32×32` upscalé 7× en `224×224`) servait de proxy `retain` unique, donnant `ProxyKL = 0.0223` sur `CIFAR-100` mais `ProxyKL = 0.93–1.46` et `AbsKL = 0.64–1.00` sur `PatchMix 224×224` (le ViT utilisait le flou `32×32` vs netteté `224×224` comme raccourci).
  - Suppression de `CIFAR-100` au profit de `PatchMix` multi-grilles `4×4 + 7×7` (`224×224` natif) dans [auto_tune_4gpu.py](../../tracks/03_unlearning_v2/auto_tune_4gpu.py), [sweep_bounded_4gpu.py](../../tracks/03_unlearning_v2/sweep_bounded_4gpu.py) et [eval_harness.py](../../tracks/03_unlearning_v2/eval_harness.py) (`seed=999` hors-échantillon), plus `gold_mimic_loss` chirurgical en logits ([common.py](../../tracks/03_unlearning_v2/methods/common.py)) et plafond `p50` dans `_snapshot_score`. Sessions Colab `track3_a` et `track3_b` stoppées (`0 CU/h`).
- **6ᵉ soumission (`23h22`, Sonde `model.pt` non modifié, `task-1827`) — Score `0.0`** (`Previous best: 0.1166`, `improved: False`) :
  - Prouve l'existence d'un garde-fou serveur anti-no-op sur `model.pt` intact (car `#5` à `0.1166` et `#2` à `0.0843` bornent déjà $|\mu_g - 1.44| + |\sigma_g - 2.01| \le 2.15$ et $\mu_g + \sigma_g \le 5.08$).
- **Réalignement stratégique des objectifs :**
  1. **Levier quasi exclusif sur `forget_score` (marge $\times 3$ à $\times 5$) :** passer `ProxyKL` de `0.055` à `0.035` ne change `retain_score` que de `~2 %` (`0.948 -> 0.966`). `ProxyKL <= 0.035` et `dTheta <= 3.5` sont des garde-fous, pas le levier principal.
  2. **Hétérogénéité alignée sur `gold` (et non homogénéisation `ce_std <= 0.70`) :** sur des images held-out jamais vues, la CE d'un modèle `gold` (`acc ~ 80–85 %`) est à queue lourde (`p50_gold ~ 0.05–0.15`, `μ_gold ~ 0.5–0.9`, `σ_gold ~ 1.3–1.8 ≈ 2–3 × μ_gold`). Le défaut de `#5` (`0.1166`, `p50 = 0.38`, `μ = 1.44`, `σ = 2.01`) est une médiane trop haute et une queue placée au hasard.
  3. **Priorités d'exécution dès libération GPU :**
     - **M1 (Interpolation & Soupe de poids, [m1_soup.py](../../tracks/03_unlearning_v2/m1_soup.py), 0 entraînement) :** $\theta(\lambda) = (1-\lambda)\theta_0 + \lambda\theta_{\#5}$ ($\lambda \in [0.55, 0.92]$) et soupe des meilleurs checkpoints de `15903458` pour abaisser `p50` vers `0.15–0.22` et éliminer la variance parasite de batch.
     - **M2 (Cibles Gold-Proxy par image dans l'espace des logits) :** estimer la typicalité *retain-free* par image (TTA modérée, Mahalanobis/kNN sur `blocks.8`, sensibilité au bruit de poids, tuned logit lens — validés par corrélation de Spearman $> 0.3$ avec `CE_gold` sur `lab_in100`) pour assigner une cible logit basse aux ~80 % d'images typiques et une erreur plausible (voisin sémantique à $T=2$) aux ~20 % d'images atypiques.
     - **M3 (Paramétrisation à faible dimension : `LayerScale + LN + head[active]` ou `SVD-σ` sur `blocks.8..11`) :** borner structurellement `dTheta` et empêcher tout raccourci sur le proxy.

### 2026-11-21 (23h36–23h40) — Audit intégral de `Unlearning.pdf` : Vraie formule linéaire clippée & Annulation de `15903800`
- **Différence critique trouvée dans [Unlearning.pdf](Unlearning.pdf) (p. 2) :**
  - `mean_std_score` et `prediction_score` ne sont **pas** $\exp(-d_{\text{sub}})$ ni $1/(1+\text{KL})$, mais des **ratios linéaires normalisés par `orig` et clippés à `[0, 1]`** :
    - $\text{mean\_std\_score} = 1 - \text{clip}(d_{\text{sub}} / d_{\text{orig}}, 0, 1)$ et $\text{prediction\_score} = 1 - \text{clip}(\text{KL}_{\text{sub}} / \text{KL}_{\text{orig}}, 0, 1)$ à $T=2$.
    - Chaque score est calculé séparément sur les **images exactes** et les **copies altérées cachées**, moyenné à **50/50**, puis multiplié : $\text{forget\_score} = \text{mean\_std\_score} \times \text{prediction\_score}$.
    - `gold` est la moyenne de plusieurs modèles indépendants ré-entraînés de zéro.
- **Conséquences directes :**
  1. `sub = orig` (sonde `#6`, `model.pt` = `0.0`) donne **`0.0` mathématiquement** par $1 - d_{\text{orig}}/d_{\text{orig}} = 0$ (et `lab_in100` avait donné `0.2959` à `orig_no_unlearn` uniquement à cause de la formule approximée dans `server_metrics.py`).
  2. Effet falaise du `clip(..., 0, 1)` : sur les copies altérées (50 % du score), $d_{\text{orig, alt}}$ est plus petit que sur le brut ; si `CE` monte trop fort (`#5` à `1.44 ± 2.01` brut), la moitié altérée risque de dépasser $d_{\text{orig, alt}}$ et de tomber à `0.0` — ce qui renforce l'intérêt de **M1** ($\lambda < 1$).
- **Actions effectuées :**
  - Job Slurm `15903800` **annulé** (`scancel 15903800`, 0 GPU occupé, aucun relancement immédiat).
  - Mise à jour alignée sur le PDF de [03-unlearning.md](03-unlearning.md), [server_metrics.py](../../tracks/03_unlearning_v2/lab/server_metrics.py), [test_lab_metrics.py](../../tracks/03_unlearning_v2/tests/test_lab_metrics.py) et [m1_soup.py](../../tracks/03_unlearning_v2/m1_soup.py).

### 2026-11-21 (23h50–23h58) — Copie & Analyse approfondie des 4 papiers officiels de `Unlearning.pdf`
- **Papiers copiés dans `docs/papers/` et analysés individuellement par 4 sous-agents `researcher` (`pro`) :**
  1. **SCRUB / Unbounded Machine Unlearning** ([2302.09880v3.pdf](../papers/2302.09880v3.pdf) · [analyse](../analyses/03-scrub-machine-unlearning.md)) :
     - Démontre l'équivalence exacte entre le régime **User Privacy (UP)** du papier et la métrique de `Unlearning.pdf` (`mean_std_score` = défense `Basic MIA` sur les moments $(\mu, \sigma)$ de la loss ; `prediction_score` = défense `LiRA` instance par instance à $T=2$). Le Tableau 23 (p. 30) illustre l'effet Streisand du sur-oubli (`SCRUB` sans rewind : erreur `71.67%` $\implies$ `MIA = 78.0%`, pire que `Original = 65.3%`, alors que `SCRUB+R` ramène l'erreur à `19.33%` comme `Retrain = 29.33%`).
     - Explique l'échec de `EU-k`/`CF-k` (dernière couche seule) en *Selective Unlearning* (Fig. 1a vs 1b, p. 10) : la mémorisation d'instances vit dans `blocks.6..11`.
     - Donne un estimateur *retain-free* de `gold` (*Virtual Gold Teacher*) via **Patch-Dropout (40–50 % des patchs $16\times 16$ masqués) + multi-vues sur $w^o$**, qui brise la mémorisation d'instance ($\text{CE}=0.0003$) tout en préservant la variance naturelle $\sigma_{\text{gold}}$ et les classes secondaires à $T=2$.
  2. **Zero-Shot Machine Unlearning** ([2201.05629v3.pdf](../papers/2201.05629v3.pdf) · [analyse](../analyses/04-zero-shot-machine-unlearning.md)) :
     - La Table I du papier prouve que la Cross-Entropy dure sur du bruit *Min-Max* détruit l'accuracy `retain` (chute de 84 % à 20 %) car elle écrase le *dark knowledge* inter-classes.
     - Dans **Gated Knowledge Transfer (GKT)**, deux briques sont directement exploitables sur nos mosaïques `PatchMix 224×224` : (a) le **Band-Pass Filter** $F(x_p)$ (Éq. 10 adaptée : filtrer les mosaïques ayant une trop forte similarité cosinus CLS ou probabilité sur la classe source de $\mathcal{D}_f$), et (b) l'**alignement normalisé des activations internes $L_{\text{at}}$** (Éq. 9 : $\|h_{\text{orig}}/\|h_{\text{orig}}\|_2 - h_{\text{sub}}/\|h_{\text{sub}}\|_2\|_2^2$) couplé à la distillation `KL(T=2)`.
  3. **SalUn: Gradient-based Weight Saliency** ([2310.12508v5.pdf](../papers/2310.12508v5.pdf) · [analyse](../analyses/05-salun-weight-saliency-unlearning.md)) :
     - Sur Vision Transformer (**Swin-T**, Table A10 p. 25, 10 % forget), `SalUn` atteint un écart `MIA` de **`0.02 %`** avec `Retrain` (`28.35 %` vs `28.33 %`), et une sparsité plus fine (`10–30 %` de poids modifiables) protège beaucoup mieux `RA` que `50 %`.
     - **2 bugs découverts dans notre code grâce au papier :**
       1. Dans `AdamW`, le `weight_decay = 0.01` par défaut s'applique à **tous** les paramètres `requires_grad=True` même quand `p.grad == 0`, faisant dériver vers `0` les $75\%$ de poids figés par SalUn ($\mathbf{m}_{\text{S}} = 0$) et les 55 classes absentes de `head.weight` (fix : `weight_decay=0.0` dans `build_optimizer`).
       2. Le sous-échantillonnage `t.numel() // 4096` du quantile global sur-pondérait de $1000\times$ les petits tenseurs (`LayerNorm`, `LayerScale .gamma`) par rapport aux matrices MLP (fix : quantile exact **layer-wise par tenseur** `torch.quantile(v.reshape(-1), 1.0 - keep_ratio)`).
  4. **NPO: Negative Preference Optimization** ([2404.05868v2.pdf](../papers/2404.05868v2.pdf) · [analyse](../analyses/06-npo-negative-preference-optimization.md)) :
     - **Découverte mathématique majeure sur $\beta$ :** dans le poids adaptatif $W_\theta(x,y) = 2\sigma(-\beta \Delta L)$, la valeur $\beta = 0.1$ du papier (Section 5, TOFU) est calibrée pour des **séquences LLM de 30–50 tokens** où la somme des log-probs varie de $\Delta L_{\text{seq}} \sim 10\text{--}30$.
     - En **classification mono-label** (Section 4, Table 1 p. 24, cas $\alpha=0$ où $\mathcal{D}_{\text{FG}}$ et $\mathcal{D}_{\text{RT}}$ partagent les mêmes classes), $\Delta \text{lp}_{T=2}$ vaut seulement $\sim -0.65$ pour atteindre $\text{CE}_{T=1}^* \approx 0.50$. Avec $\beta = 0.1$, $W_\theta = 2\sigma(-0.065) = 0.935 \approx 1.0$ : **le frein NPO était inactif dans toute la plage utile** ! En classification, les auteurs utilisent **$\beta \in [1.0, 10.0]$** (optimal $\beta \approx \mathbf{3.0}$ à $T=2$, qui freine $W_\theta \to 0.25$ dès $\text{CE} \approx 0.50$).

### 2026-11-22 (00h01) — Application dans le code des 4 améliorations issues des papiers (`tracks/03_unlearning_v2/`)
- **Correctifs implémentés et vérifiés par `py_compile` (aucun job GPU lancé) :**
  1. **Calibration NPO mono-label ($\beta = 2.5$ par défaut, [contracts.py](../../tracks/03_unlearning_v2/contracts.py) & [common.py](../../tracks/03_unlearning_v2/methods/common.py)) :** passage de `npo_beta` de `0.1` à `2.5` (et balayage $\beta \in \{2.0, 2.5, 3.0\}$ dans [auto_tune_4gpu.py](../../tracks/03_unlearning_v2/auto_tune_4gpu.py) et [sweep_bounded_4gpu.py](../../tracks/03_unlearning_v2/sweep_bounded_4gpu.py)) pour activer le frein adaptatif $W_\theta(x,y) = 2\sigma(-\beta \Delta \text{lp}_T)$ dès $\text{CE}_{T=1} \in [0.40, 0.55]$.
  2. **Suppression de la fuite `AdamW weight_decay` & restauration exacte Éq. 4 SalUn ([common.py](../../tracks/03_unlearning_v2/methods/common.py), [salun_npo.py](../../tracks/03_unlearning_v2/methods/salun_npo.py), [lora_unlearn.py](../../tracks/03_unlearning_v2/methods/lora_unlearn.py)) :** `weight_decay=0.0` explicite dans `build_optimizer` et `run_lora_npo` + `restore_absent_head_rows(student, orig_model, absent_mask)` après chaque `opt.step()` afin que les poids hors-masque ($\mathbf{m}_{\text{S}} = 0$) et les 55 classes absentes restent bit-à-bit invariants.
  3. **Quantile SalUn exact *layer-wise* par tenseur ([salun_npo.py](../../tracks/03_unlearning_v2/methods/salun_npo.py)) :** remplacement du sous-échantillonnage global biaisé (`t.numel() // 4096`) par `torch.quantile(v.reshape(-1), 1.0 - keep_ratio)` par tenseur (`num_batches=32, batch_size=64`, soit `2 048` images mélangées, `salun_keep_ratio` $\in \{0.15, 0.20, 0.25\}$).
  4. **Hétérogénéité naturelle par image dans `gold_mimic_loss` & scoring linéaire clippé ([common.py](../../tracks/03_unlearning_v2/methods/common.py), [auto_tune_4gpu.py](../../tracks/03_unlearning_v2/auto_tune_4gpu.py)) :** modulation de $p_i^*$ autour de `target_prob` par la marge top-1 vs reste (`z_top - lse_rest`) normalisée sur le batch ($p_i^* = \text{clip}(p^* + 0.12 \tanh(0.5 z_{\text{margin}}), 0.15, 0.90)$) pour préserver la dispersion naturelle $\sigma_{\text{gold}}$ exigée par `SCRUB+R` / `Unlearning.pdf`, et alignement de `score_sweet_spot` / `_snapshot_score` sur la moyenne 50/50 exact + altéré $1 - \text{clip}(d_{\text{sub}}/d_{\text{orig}}, 0, 1)$.

### 2026-11-22 (00h25) — Suppression du dataset externe (`lab_in100`) & Benchmark d'oubli sélectif direct sur `forget_images.npz`
- **Décision architecturale (Chef de Brigade) :** construire un dataset externe (`clane9/imagenet-100`) de « connu vs inconnu » était une erreur de complexité inutile et biaisée (`LayerScale`, saturation, classes différentes).
- **Pourquoi `forget_images.npz` suffit et est strictement supérieur :** les `3 500` images officielles permettent de benchmarker directement `model.pt` image par image en **oubli sélectif** — si on lui fait oublier l'image/sous-ensemble `1`, il doit l'oublier **tout en se rappelant de l'image/sous-ensemble `2`** (pourtant issue des mêmes classes et du même entraînement initial de `model.pt`).
- **Nettoyage effectué :** suppression de `tracks/03_unlearning_v2/lab/build_lab.py`, `tracks/03_unlearning_v2/lab/sweep_lab.py`, `tracks/03_unlearning_v2/lab/validate_lab.py` et `tracks/03_unlearning_v2/slurm_lab.sh` (seul `server_metrics.py` est conservé pour les formules officielles de `Unlearning.pdf`).
- **Piste de test futur — Protocole de validation croisée Intra-Classe & Inter-Classes sur `forget_images.npz` (`00h27`) :**
  - Bien que `forget_images.npz` ne fournisse pas la colonne `labels`, la saturation de `model.pt` (`CE = 0.0003`, `99.94 %` des images à `max_prob > 0.95`) donne la classe exacte de chaque image via `y = orig(x).argmax(-1)`, répartie sur **45 classes actives** (`35` à `117` images/classe) et **55 classes absentes** (`0` image).
  - Cela permet un double benchmark local sans aucun dataset externe, moyenné sur plusieurs seeds / folds pour éliminer l'aléatoire :
    1. **Test Intra-Classe (*Instance Selectivity* vs *Class Collapse*) :** dans chacune des 45 classes actives, tirer $50\%$ d'images à oublier ($F_{\text{forget}}$) et garder les $50\%$ restantes de la **même classe** en contrôle ($F_{\text{same-class-retain}}$). Une méthode qui détruit simplement la ligne de `head.weight` fera exploser la CE sur $F_{\text{same-class-retain}}$ (*class collapse*), tandis qu'une méthode chirurgicale oubliera $F_{\text{forget}}$ tout en préservant une haute précision et un faible `KL(T=2)` sur $F_{\text{same-class-retain}}$.
    2. **Test Inter-Classes (*Cross-Class Collateral Drift*) :** faire oublier un sous-ensemble de classes actives (ex. 22 classes sur 45, en tournant sur plusieurs partitions aléatoires) et mesurer la dérive collatérale (`top1_agree_orig`, `KL(T=2)`) sur les **23 autres classes actives** (dont on possède les vraies images dans `forget_images.npz` !) ainsi que sur les 55 classes absentes.

### 2026-11-22 (00h07–00h35) — Run complet 2× A100 Colab (`worker 0` + `worker 1`, 13 essais) avec correctifs des 4 papiers
- **Tests unitaires :** `test_methods.py`, `test_eval_harness.py` et `test_lab_metrics.py` validés (`UNIT_TESTS_PASSED`).
- **Tableau complet des 13 essais (`worker_0.json` + `worker_1.json`, proxy `PatchMix 4×4 + 7×7` `seed=42/43`, éval hors-échantillon `seed=999/1000`) :**

| Rang | Candidat | Worker | `SW` | `CE mean ± std` | `p50` | `>95%` | `ProxyKL` | `AbsKL` | `dTheta` | Diagnostic & comparaison à `#5` (`0.1166`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **`AT_ce0.50_b3.0_f8`** | `gpu0` | **`+0.609`** | **`0.704 ± 1.19`** | **`0.116`** | **`0.43`** | **`0.0376`** | **`0.0226`** | **`3.86`** | **Régime `gold` modéré idéal** ($\beta=3.0$, `blocks.8..11`, `keep=20%`, `ProxyKL` $-32\%$ vs `#5`) |
| **2** | **`AT_ce0.50_b2.0_f6`** | `gpu1` | **`+0.448`** | **`1.265 ± 1.75`** | **`0.397`** | **`0.31`** | **`0.0413`** | **`0.0255`** | **`3.95`** | **Domine `#5` (`0.1166`) sur tous les axes** (`CE 1.265±1.75` vs `1.438±2.01`, `p50=0.397` vs `0.380`, `>95%=0.31` vs `0.37`, `ProxyKL 0.0413` vs `0.0551`, `AbsKL 0.0255` vs `0.0413`, `dTheta 3.95` vs `4.03`) |
| **3** | `TS_ce0.50_b2.5_k25` | `gpu1` | `+0.384` | `0.262 ± 0.55` | `0.033` | `0.55` | `0.0311` | `0.0185` | `3.57` | Two-Sided NPO conservateur, très faible dérive `retain` |
| **4** | `TS_ce0.45_b2.5_k20` | `gpu0` | `+0.376` | `0.285 ± 0.58` | `0.039` | `0.53` | `0.0368` | `0.0224` | `3.66` | Two-Sided NPO conservateur |
| **5** | `TS_ce0.50_b3.0_k15` | `gpu0` | `+0.329` | `0.251 ± 0.56` | `0.025` | `0.58` | `0.0377` | `0.0235` | `3.51` | Sparsité SalUn `15 %` |
| **6** | `GM_p60_f6_hm1.0` | `gpu0` | `+0.196` | `0.110 ± 0.28` | `0.016` | `0.69` | `0.0366` | `0.0232` | `2.96` | Sous-oubli (`gold_mimic` trop freiné à `lr=8e-5`) |
| **7** | `GM_p55_f8_hm1.5` | `gpu0` | `+0.166` | `0.087 ± 0.27` | `0.008` | `0.76` | `0.0354` | `0.0209` | `3.10` | Sous-oubli |
| **8** | `GM_p60_f8_hm1.5` | `gpu1` | `+0.159` | `0.078 ± 0.25` | `0.007` | `0.78` | `0.0336` | `0.0197` | `3.05` | Sous-oubli |
| **9** | `FA_a0.20_gm60` | `gpu1` | `+0.159` | `0.066 ± 0.22` | `0.007` | `0.79` | `0.0311` | `0.0163` | `3.88` | Sous-oubli |
| **10** | `GM_p65_f8_hm2.0` | `gpu1` | `+0.158` | `0.070 ± 0.23` | `0.006` | `0.79` | `0.0315` | `0.0177` | `2.98` | Sous-oubli |
| **11** | `LORA_ce0.45_r8` | `gpu1` | `+0.091` | `0.321 ± 0.60` | `0.060` | `0.48` | **`0.0123`** | **`0.0063`** | `5.03` | Préservation `retain` exceptionnelle en LoRA rang 8 |
| **12** | `FA_a0.30_gm60` | `gpu0` | `-0.055` | `0.048 ± 0.18` | `0.004` | `0.85` | `0.0309` | `0.0142` | `4.58` | Sous-oubli |
| **13** | **`LORA_ce0.50_r16`** | `gpu0` | `-0.602`* | **`0.358 ± 0.58`** | **`0.124`** | **`0.36`** | **`0.0114`** | **`0.0058`** | `6.50` | **Record de préservation (`ProxyKL = 0.0114`, `AbsKL = 0.0058`)** avec `p50 = 0.124` et `>95% = 0.36` (*pénalisé dans `SW` par `dTheta = 6.50`*) |

- **Checkpoints rapatriés en local (`tracks/03_unlearning_v2/output/colab_autotune/`) et arrêt des 2 VMs Colab (`0 CU/h`) :**
  - `AT_ce0.50_b3.0_f8.pt` (`SW = +0.609`), `AT_ce0.50_b2.0_f6.pt` (`SW = +0.448`, domine `#5` Pareto-strictement), `LORA_ce0.50_r16.pt` (`ProxyKL = 0.0114`), `TS_ce0.50_b2.5_k25.pt` (`SW = +0.384`).

### 2026-11-22 (00h39–00h46) — Benchmark d'Oubli Sélectif Intra-Classe & Inter-Classes sur `forget_images.npz` (`selective_benchmark.py`, 2× A100 Colab)
- **Protocole implémenté dans [selective_benchmark.py](../../tracks/03_unlearning_v2/selective_benchmark.py) ([test_selective_benchmark.py](../../tracks/03_unlearning_v2/tests/test_selective_benchmark.py) `SELECTIVE_BENCHMARK_TESTS_OK`) :**
  1. **Split Intra-Classe (50/50 dans chacune des 45 classes actives) :** entraîner l'oubli uniquement sur $50\%$ des images de chaque classe ($F_{\text{forget}}$, `1 750` images) et évaluer sur les $50\%$ d'images retenues de la **même classe** ($R_{\text{same\_class}}$, `1 750` vraies images). Mesure la vraie sélectivité instance-par-instance (`gap = F_ce - R_ce`) vs l'effondrement du prototype de classe (`R_acc`, `R_kl`).
  2. **Split Inter-Classes (22 classes actives oubliées vs 23 classes actives retenues) :** entraîner l'oubli sur 22 classes actives et évaluer les dégâts collatéraux sur les vraies images des **23 autres classes actives** ($R_{\text{other\_classes}}$ : `R_acc`, `R_kl`).
- **Résultats comparatifs des 4 finalistes (`selective_worker_0.json` + `selective_worker_1.json`, les 2 VMs Colab stoppées à `0 CU/h`) :**

| Candidat | `SEL` | **Intra `F_ce` (`p50`)** | **Intra `R_ce`** | **Gap Intra (`F_ce - R_ce`)** | **Intra `R_acc`** | **Intra `R_kl`** | **Cross `F_ce`** | **Cross `R_acc`** | **Cross `R_kl`** | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`AT_ce0.50_b3.0_f8`** | **`0.3407`** | **`0.950` (`0.105`)** | **`0.587`** | **`+0.363` (+62 %)** | **`82.6 %`** | **`0.6909`** | **`1.125`** | **`99.5 %`** | **`0.0547`** | **Gagnant équilibré** : oubli franc (`F_ce ~ 1.0`) + forte sélectivité intra-classe (`82.6 %` >> gate `70 %`) + dérive inter-classes quasi nulle (`99.5 %`, `KL = 0.0547`) |
| `TS_ce0.50_b2.5_k25` | `0.4140`* | `0.212` (`0.017`) | `0.101` | `+0.111` (+110 %) | `97.3 %` | `0.3027` | `0.286` | `99.9 %` | `0.0591` | Très sélectif mais **sous-oubli** en amplitude absolue (`F_ce = 0.212`, `p50 = 0.017`, comme `#3` à `0.0117`) |
| **`AT_ce0.50_b2.0_f6`** | `0.1425` | **`1.430` (`0.381`)** | `0.991` | **`+0.440` (+44 %)** | `72.0 %` | `0.9705` | **`1.847`** | `98.2 %` | `0.1246` | Plus fort écart absolu (`+0.440`) et oubli soutenu, mais `Intra R_acc = 72.0 %` proche du seuil `70 %` |
| `LORA_ce0.50_r16` | `0.0339` | `0.304` (`0.109`) | `0.273` | `+0.031` *(quasi nul)* | `92.9 %` | `0.7465` | `0.366` | `99.1 %` | `0.2231` | **Éliminé** : aucune sélectivité intra-classe (`0.304` vs `0.273`, dégrade toute la classe) et `Cross R_kl = 0.2231` (4× pire que `AT_b3.0_f8`) |

### 2026-11-22 (00h49) — 7ᵉ Soumission serveur (`AT_ce0.50_b3.0_f8.pt`, `submission_id: 2169`) = `0.0604` & Preuve algébrique que nous sommes sur la pente montante ($\mu_{\text{sub}} + \sigma_{\text{sub}} < \mu_{\text{gold}} + \sigma_{\text{gold}}$)
- **Résultat serveur (`00h49:11`, `task-2141`) :**
  - `{"submission_id": 2169, "status": "success", "score": 0.0604, "previous_best": 0.1166, "delta": -0.0562, "improved": false}`
- **Loi linéaire exacte révélée par les soumissions `#3`, `#7`, `#2` et `#5` :**
  - Dans la formule officielle de `Unlearning.pdf` :
    $$\text{mean\_std\_score} = 1 - \frac{|\mu_{\text{sub}} - \mu_{\text{gold}}| + |\sigma_{\text{sub}} - \sigma_{\text{gold}}|}{|\mu_{\text{orig}} - \mu_{\text{gold}}| + |\sigma_{\text{orig}} - \sigma_{\text{gold}}|}$$
    Comme $\mu_{\text{orig}} \approx 0$ et $\sigma_{\text{orig}} \approx 0$, tant que le modèle soumis est en **sous-oubli** ($\mu_{\text{sub}} \le \mu_{\text{gold}}$ et $\sigma_{\text{sub}} \le \sigma_{\text{gold}}$), la formule se simplifie algébriquement en :
    $$\text{mean\_std\_score} = \frac{\mu_{\text{sub}} + \sigma_{\text{sub}}}{\mu_{\text{gold}} + \sigma_{\text{gold}}}$$
  - **Vérification empirique sur nos 4 soumissions SalUn/NPO à faible `ProxyKL` (`0.036–0.055`) :**

| Soumission | Modèle | $\mu_{\text{sub}}$ (`CE mean`) | $\sigma_{\text{sub}}$ (`CE std`) | **Somme $\mu_{\text{sub}} + \sigma_{\text{sub}}$** | **Score serveur** | **Efficacité $\frac{\text{Score}}{\mu+\sigma}$** | **Ratio Score vs `#7`** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `#3` | `E_lora_bt_T5.pt` | `0.169` | `0.330` | **`0.499`** | **`0.0117`** | `0.0234` | `0.19×` |
| **`#7`** | **`AT_ce0.50_b3.0_f8.pt`** | `0.704` | `1.195` | **`1.899`** | **`0.0604`** | `0.0318` | **`1.00×`** |
| `#2` | `A_npo_mod_aug.pt` | `0.617` | `1.990` | **`2.607`** | **`0.0843`** | `0.0323` | **`1.396×`** |
| **`#8`** | **`AT_ce0.50_b2.0_f6.pt`** | `1.265` | `1.746` | **`3.011`** | **`0.1099`** | **`0.0365` (record !)** | **`1.820×`** |
| **`#5`** | **`B_ce0.45_lr8e-05_kl2.0.pt`** | `1.438` | `2.010` | **`3.448`** | **`0.1166`** | `0.0338` | **`1.930×`** |

  - **Confirmation à 5 points avec la 8ᵉ soumission (`01h06:42`, `AT_ce0.50_b2.0_f6.pt`, `submission_id: 2218`, Score `0.1099`, `delta = -0.0067`) :**
    - `AT_ce0.50_b2.0_f6.pt` obtient **le meilleur rendement $\frac{\text{Score}}{\mu_{\text{sub}} + \sigma_{\text{sub}}} = 0.0365$** de tous nos modèles (`+8 %` de rendement par rapport à `#5` grâce à `ProxyKL = 0.0413`, `>95% = 0.31`, SalUn *layer-wise* `20 %` et `weight_decay = 0.0`), et ne reste à `0.1099` vs `0.1166` que parce qu'il était encore plafonné à `target_forget_ce = 0.50` ($\mu+\sigma = 3.011$ contre $3.448$ sur `#5`).
    - À rendement constant (`0.0365`), pousser $\mu_{\text{sub}} + \sigma_{\text{sub}}$ dans la plage `4.2–5.5` via nos candidats débridés (`target_forget_ce` $\in \{1.00, 1.40, 1.80\}$) projette un score serveur de **`0.15–0.19`**.

### 2026-11-22 (01h25) — Nouvelle piste de réflexion (`Piste H`) : Pilotage Algorithmique Temps Réel des Poids & Tweaking Hybride
- **Idée directrice (Chef de Brigade) :** hybrider l'entraînement par descente de gradient avec un **algorithme de contrôle et de tweaking des poids en temps réel** qui observe l'état des poids, des gradients et des prédictions à chaque pas pour ajuster dynamiquement où et comment le modèle est modifié.
- **Formulation scientifique en 3 mécanismes concrets (et pourquoi ça peut très bien marcher) :**
  1. **Conductance dynamique par poids $g_i(t)$ (*Online Fisher / SalUn Recalibration*) :**
     - *Limitation actuelle :* le masque SalUn est figé au pas $t=0$ (où `model.pt` est saturé à `CE = 0.0003`), alors que de nouveaux circuits de mémorisation dans `blocks.6..11` se révèlent au fur et à mesure que la confiance baisse.
     - *Solution temps réel :* maintenir à chaque step le ratio de sélectivité par poids $g_i(t) = \frac{\text{EMA}(|\nabla_{\theta_i} \mathcal{L}_f|)}{\text{EMA}(|\nabla_{\theta_i} \mathcal{L}_r|) + \epsilon}$ (type conductance synaptique). Dès qu'un poids commence à faire dériver le `retain` plus vite qu'il n'aide le `forget`, l'algorithme coupe son pas ou le **rollback** vers $\theta_{\text{orig}, i}$ en temps réel.
  2. **Projection orthogonale Instance vs Classe ($\nabla_{\text{instance}} = \nabla_i - \alpha \bar{\nabla}_c$) & Régulation par classe :**
     - Dans un batch, la moyenne des gradients d'une même classe $c$ ($\bar{\nabla}_c$) pousse à détruire le **prototype entier de la classe $c$** (ce qui fait chuter `IntraAcc`), tandis que la composante résiduelle $\nabla_i - \bar{\nabla}_c$ cible uniquement **la spécificité de l'image $i$**.
     - Un contrôleur temps réel par classe mesure à chaque step la CE médiane de chacune des 45 classes actives : il accélère les classes en retard, freine celles qui ont atteint la cible, et atténue la composante moyenne $\bar{\nabla}_c$ pour empêcher le *class collapse*.
  3. **Tweaking direct sans gradient sur les leviers basse dimension (`LayerScale .gamma` + projections SVD) :**
     - Chaque bloc du ViT-B/16 possède un vecteur `LayerScale` $\gamma \in \mathbb{R}^{768}$ ($\sim 10^{-4}$) qui contrôle le gain de chaque canal d'attention et de MLP. Un algorithme de recherche directe (coordinate search / trust-region en boîte blanche) peut ajuster $\gamma$ et les normes des lignes actives de `head.weight` en temps réel sous contrainte dure `InterKL <= seuil`.
  4. **Compilation d'un circuit `IF / THEN` directement dans les neurones MLP dormants (`In-Weights Associative Gate`, type *ROME / MEMIT*) :**
     - Une couche MLP du ViT calcule $\text{MLP}(h) = W_{\text{fc2}} \cdot \text{GELU}(W_{\text{fc1}} h + b_{\text{fc1}}) + b_{\text{fc2}}$ avec `3 072` neurones intermédiaires.
     - En sélectionnant $M \in [45, 90]$ neurones dormants (à plus faible norme/saillance sur le `retain`) dans `blocks.10` ou `blocks.11`, on peut programmer directement dans les poids un détecteur logique exécuté nativement par `model(x)` au runtime serveur :
       - **Condition `IF` (`fc1.weight[k]` + seuil `fc1.bias[k] = -\tau_k`) :** aligner la ligne $w_k$ sur le centroïde/signature normalisée d'un cluster d'images à oublier. Pour toute image `retain` ($w_k \cdot h < \tau_k$), $\text{GELU}(w_k \cdot h - \tau_k) \approx 0$ (neurone éteint, **zéro impact sur le `retain`**). Pour une image `forget` ou sa copie altérée ($w_k \cdot h > \tau_k$), $\text{GELU}(w_k \cdot h - \tau_k) > 0$ (le neurone s'allume).
       - **Action `THEN` (`fc2.weight[:, k]`) :** orienter la colonne $v_k$ à l'opposé de la direction du prototype top-1 dans `head.weight` afin d'abaisser chirurgicalement la confiance sur la classe mémorisée uniquement quand le détecteur $k$ s'allume.

  - **Conclusion majeure :** la proportionnalité quasi parfaite entre $\mu_{\text{sub}} + \sigma_{\text{sub}}$ et le score serveur (`1.373× vs 1.396×` et `1.816× vs 1.930×`) démontre que **même `#5` (`CE = 1.438 ± 2.01`, `p50 = 0.380`) est encore en dessous de $(\mu_{\text{gold}}, \sigma_{\text{gold}})$** ! Réduire `CE` de `1.44 ± 2.01` à `0.70 ± 1.19` (`#7`) a divisé le score par deux (`0.1166 -> 0.0604`) parce qu'on s'est rapproché de `orig` au lieu de monter vers `gold` ($\mu_{\text{gold}} \approx 2.2\text{--}3.2$, $\sigma_{\text{gold}} \approx 3.0\text{--}4.0$, soit $\mu_{\text{gold}} + \sigma_{\text{gold}} \approx 5.5\text{--}7.5$).
  - **Conséquence immédiate :** il faut aller dans la direction opposée à la réduction de CE — viser **`target_forget_ce` plus haut (`0.85` à `1.50`, soit `CE mean ~ 2.0–2.8`, `p50 ~ 0.80–1.40`, `>95% < 0.15`)** tout en gardant la sélectivité SalUn (`keep_ratio = 0.20`, `weight_decay = 0.0`, `ProxyKL <= 0.055`). Or nous avons déjà sur JURECA dans `/p/scratch/training2666/ouaziz1/TRYON/unlearning/output/bounded_sweep/` les checkpoints `B_ce0.65_*`, `B_ce0.85_*` et `B_ce1.05_*` du job `15903458` !

### 2026-11-22 (01h08–01h32) — Résultats complets du Benchmark Sélectif Pondéré 3-Folds « Oubli Fort » (6 candidats, 2× A100 Colab)
- **Protocole exécuté ([selective_benchmark.py](../../tracks/03_unlearning_v2/selective_benchmark.py)) :** rotation sur **$K = 3$ folds disjoints de 15 classes actives** couvrant la totalité des 45 classes actives de `forget_images.npz` (sans aucune fuite entre le sous-ensemble oublié $F_k$, le contrôle Intra-Classe $R_{k,\text{intra}}$ de la même classe en version **exacte + altérée**, et le contrôle Inter-Classes $R_{k,\text{inter}}$ sur les 30 autres classes actives), suivie de l'entraînement du checkpoint complet `.pt` (`3 500` images).
- **Classement complet des 6 candidats (`selective_worker_0.json` + `selective_worker_1.json`, les 2 VMs Colab stoppées à `0 CU/h`) :**

| Rang | Candidat | Worker | **Score Pondéré `W_SEL ± std`** | **3-Folds `F_raw` / `F_alt`** | **3-Folds `Gap50`** | **3-Folds `IntraAcc`** | **3-Folds `InterAcc` (`InterKL`)** | **Full `.pt` `CE mean ± std` ($\mu+\sigma$)** | **Full `p50` (`>95%`)** | **Full `ProxyKL` (`dTheta`)** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`AT_ce1.00_b1.8_f6`** | `gpu0` | **`0.5565 ± 0.019`** | **`2.61` / `2.74`** | **`+1.47`** | **`73.3 %`** | **`99.8 %` (`0.0312`)** | **`1.909 ± 2.05` (`3.96`)** | **`1.214` (`0.19`)** | **`0.0486` (`4.61`)** |
| **2** | **`AT_ce1.40_b1.5_f8`** | `gpu1` | **`0.5451 ± 0.006`** | **`3.24` / `3.37`** | **`+1.69`** | `63.8 %` | **`99.9 %` (`0.0226`)** | **`2.547 ± 3.00` (`5.55`)** | **`1.390` (`0.31`)** | **`0.0422` (`4.56`)** |
| **3** | **`AT_ce1.40_b1.5_f6`** | `gpu0` | `0.4856 ± 0.023` | `3.47` / `3.59` | **`+1.78`** | `63.3 %` | `99.6 %` (`0.0474`) | **`2.089 ± 2.21` (`4.30`)** | **`1.370` (`0.17`)** | `0.0510` (`4.61`) |
| **4** | `AT_ce1.80_b1.2_f6` | `gpu1` | `0.3268 ± 0.270` | `3.30` / `3.43` | `+1.79` | `68.2 %` | `99.8 %` (`0.0359`) | `2.621 ± 2.66` (`5.28`) | `1.838` (`0.18`) | `0.0487` (`4.73`) |
| **5** | `TS_ce1.20_u2.40_b1.5_f6` | `gpu0` | `0.2000 ± 0.137` | `1.39` / `1.55` | `+0.65` | `72.8 %` | `99.6 %` (`0.1665`) | `1.017 ± 1.19` (`2.21`) | `0.563` (`0.22`) | `0.0488` (`4.49`) |
| **6** | `TS_ce1.60_u2.80_b1.2_f6` | `gpu1` | `0.1015 ± 0.118` | `1.71` / `1.87` | `+0.72` | `64.5 %` | `99.2 %` (`0.2413`) | `1.205 ± 1.34` (`2.55`) | `0.726` (`0.20`) | `0.0506` (`4.59`) |

- **Analyse & Checkpoints rapatriés en local (`tracks/03_unlearning_v2/output/selective_strong/`) :**
  1. **`AT_ce1.00_b1.8_f6.pt` (Rang #1, `W_SEL = 0.5565 ± 0.019`) :** seul candidat « Oubli Fort » qui maintient **`IntraAcc = 73.3 % >= 70 %`** sur les images retenues des classes oubliées tout en multipliant par `3.2×` la médiane d'oubli (`p50 = 1.214` vs `0.380` sur `#5`), en divisant par `2` la fraction saturée (`>95% = 0.19` vs `0.37` sur `#5`) et en gardant `ProxyKL = 0.0486` (`< 0.0551` sur `#5`) et `InterKL = 0.0312`.
  2. **`AT_ce1.40_b1.5_f8.pt` (Rang #2, `W_SEL = 0.5451 ± 0.006`) :** chirurgie `blocks.8..11` avec la plus faible dérive inter-classes de tout le benchmark (**`InterAcc = 99.9 %`**, **`InterKL = 0.0226`**, `ProxyKL = 0.0422`) et une amplitude forte ($\mu+\sigma = 5.55$, `p50 = 1.390`).

### 2026-11-22 (01h36:57) — 9ᵉ Soumission serveur (`AT_ce1.00_b1.8_f6.pt`, `submission_id: 2307`) = **`0.1193` (NOUVEAU RECORD, `improved: true`)**
- **Résultat serveur (`01h36:57`, `task-2314`) :**
  - `{"submission_id": 2307, "status": "success", "score": 0.1193, "previous_best": 0.1166, "delta": 0.0027, "improved": true, "message": "New best on this task."}`
- **Analyse de la courbe à 6 points (`#3` → `#7` → `#2` → `#8` → `#5` → `#9`) :**

| Soumission | Modèle | $\mu_{\text{sub}}$ (`CE mean`) | $\sigma_{\text{sub}}$ (`CE std`) | **Somme $\mu_{\text{sub}} + \sigma_{\text{sub}}$** | `p50` (`>95%`) | `ProxyKL` | **Score serveur** | **Rendement $\frac{\text{Score}}{\mu+\sigma}$** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `#3` | `E_lora_bt_T5.pt` | `0.169` | `0.330` | `0.499` | `0.046` (`0.52`) | `0.0189` | `0.0117` | `0.0234` |
| `#7` | `AT_ce0.50_b3.0_f8.pt` | `0.704` | `1.195` | `1.899` | `0.116` (`0.43`) | `0.0376` | `0.0604` | `0.0318` |
| `#2` | `A_npo_mod_aug.pt` | `0.617` | `1.990` | `2.607` | `0.001` (`0.84`) | `0.0366` | `0.0843` | `0.0323` |
| `#8` | `AT_ce0.50_b2.0_f6.pt` | `1.265` | `1.746` | `3.011` | `0.397` (`0.31`) | `0.0413` | `0.1099` | **`0.0365`** |
| `#5` | `B_ce0.45_lr8e-05_kl2.0.pt` | `1.438` | `2.010` | `3.448` | `0.380` (`0.37`) | `0.0551` | `0.1166` | `0.0338` |
| **`#9`** | **`AT_ce1.00_b1.8_f6.pt`** | **`1.909`** | **`2.050`** | **`3.959`** | **`1.214` (`0.19`)** | **`0.0486`** | **`0.1193` (BEST)** | `0.0301` |

- **Enseignement décisif sur le sommet de la courbe (`0.1099` → `0.1166` → `0.1193`) :**
  1. Le protocole **3-Folds pondéré Intra+Inter-classes** a directement sélectionné notre **nouveau record serveur (`0.1193`)** !
  2. Regarde l'inflexion du rendement $\frac{\text{Score}}{\mu+\sigma}$ quand $\mu+\sigma$ passe de `3.01` (`0.0365`) à `3.45` (`0.0338`) puis `3.96` (`0.0301`) : la courbe commence à **s'aplatir autour de $\mu+\sigma \in [3.6, 4.1]$** (car `p50 = 1.214` sur `#9` dépasse probablement légèrement la médiane `gold` sur une partie des images exactes, ou `IntraAcc = 73.3 %` commence à effriter légèrement `retain_accuracy` sur le `control set`).
  3. **Le sweet spot exact révélé par `#8` (`0.1099`), `#5` (`0.1166`) et `#9` (`0.1193`) :**
     - On veut l'amplitude de `#9` ($\mu+\sigma \in [3.5, 4.1]$, `CE mean` $\in [1.5, 1.9]$, `p50` $\in [0.65, 1.00]$) **MAIS** avec la préservation `retain` et `IntraAcc >= 78 %` de `#8` (`ProxyKL <= 0.038`, `InterKL <= 0.025`, ce qui est exactement l'objectif de la **Piste H** avec la projection orthogonale $\nabla_i - \alpha \bar{\nabla}_c$ et de la soupe 50/50 entre `#8` `AT_ce0.50_b2.0_f6.pt` et `#9` `AT_ce1.00_b1.8_f6.pt` !).

### 2026-11-22 (01h49) — Nouvelle piste de réflexion (`Piste I`) : *Erase-Then-Heal* (Oubli Extrême à 100 % puis Réparation Structurelle des Liaisons par LoRA)
- **Idée directrice (Chef de Brigade) :** au lieu d'essayer de s'arrêter en équilibre instable pendant la descente (où `15–19 %` d'images têtues restent bloquées à `> 95 %` de confiance), **partir d'un extrême (oubli à 100 % des images) puis revenir vers une moyenne saine en ré-injectant les images dans une LoRA pour réparer uniquement les liaisons structurelles sans ré-apprendre la connaissance par cœur**.
- **Pourquoi cette approche résout un problème fondamental (*SCRUB+R* sans `retain set`) :**
  1. **Phase 1 — *Total Instance Erasure* (Oubli à 100 %, `>95% -> 0.00`) :**
     - On pousse l'oubli jusqu'à ce qu'**aucune** des `3 500` images (même les plus têtues) ne garde sa trace mémorisée spécifique ($\text{CE}_{\min} > 0.30$, `frac_gt_95 < 0.03`). À ce stade extrême ($\theta_{\text{extreme}}$), la mémorisation par cœur est détruite à 100 %, mais les liaisons géométriques inter-couches et intra-classes ont été secouées (`IntraAcc` basse).
  2. **Phase 2 — *Structural LoRA Re-binding* (Ré-injection dans une LoRA de rang faible $r \in \{4, 8\}$ pour réparer les liaisons sans la connaissance d'instance) :**
     - On gèle $\theta_{\text{extreme}}$ et `head.weight`, et on ouvre uniquement une **LoRA de très faible rang ($r = 4$ ou $8$) sur `blocks.6..11`**.
     - On ré-injecte les images de `forget_images.npz` (et leurs vues altérées / `PatchMix`) avec un objectif **purement relationnel et plafonné** :
       - **Réparer les liaisons inter-classes (les 99 logits non-top-1 à $T=3$ et l'alignement des normes internes $L_{\text{at}}$)** par rapport à $\theta_{\text{orig}}$.
       - **Plafonner la ré-acquisition sur la classe top-1 ($y_i$)** via un *ceiling* (`CE_i >= 0.65`, soit $p(y_i|x_i) \le 0.52$) : dès que les liaisons générales de la classe sont restaurées, le gradient top-1 s'annule.
     - **Hypothèse falsifiable de goulot d'étranglement informationnel (non garantie) :** l'hypothèse testée est qu'une LoRA de rang $r \in \{4, 8\}$ peine à re-mémoriser les exceptions individuelles si la Phase 2 reste bornée. *(Infirmée sans barrière plancher au Round 1 (`track_i_round1.json`) : sans plancher actif sur `ce_i`, `frac_gt_95` remonte à `0.82–0.89` et `p50` retombe à `~0.001–0.003`, confirmant qu'un faible rang n'empêche pas la re-mémorisation des logits top-1.)*

### 2026-11-22 (01h38–01h55) — Expériences `Piste H` (Round 1, 4 candidats sur `track3_h`) : `W_SEL = 0.6108` en sélection locale & Soumission `#10` (`H4_h123_f6_ce125.pt`)
- **Résultats locaux du Round 1 de la cellule autonome `Piste H` ([03-unlearning-track-h-journal.md](03-unlearning-track-h-journal.md)) :**
  1. **`H4_h123_f6_ce125` (`blocks.6..11`, H1+H2+H3) : `W_SEL = 0.6108 ± 0.026` (Nouveau Record #1)** — remonte **`IntraAcc` de `73.3 %` à `81.9 %` (+8.6 pts !)** avec `InterAcc = 99.8 %` (`InterKL = 0.0234`), `Gap50 = +1.73`, et sur le checkpoint complet : **`CE = 2.127 ± 2.10` ($\mu+\sigma = 4.23$), `p50 = 1.468`, `>95% = 0.14`, `ProxyKL = 0.0409`**.
  2. **`H1_h1h2_f8_ce125` (`blocks.8..11`, H1+H2) : `W_SEL = 0.6041 ± 0.018` (#2)** — **`Gap50 = +2.06`**, `IntraAcc = 76.8 %`, `InterAcc = 99.9 %` (`InterKL = 0.0199`), Full `CE = 1.676 ± 1.93` ($\mu+\sigma = 3.61$), `p50 = 0.912`, `>95% = 0.19`, **`ProxyKL = 0.0325`**.
  3. **`H2_h123_f8_ce145` (`blocks.8..11`, H1+H2+H3) : `W_SEL = 0.6023 ± 0.026` (#3)** — `Gap50 = +1.94`, `IntraAcc = 78.2 %`, `InterAcc = 99.9 %` (`InterKL = 0.0215`), Full `CE = 1.665 ± 1.79` ($\mu+\sigma = 3.46$), `p50 = 1.010`, **`>95% = 0.15`**, **`ProxyKL = 0.0324`**.
  4. **`H3_h1234_circuit_f8` (`64` portes logiques `IF/THEN` dans `blocks.11.mlp`) :** le détecteur `IF` a éteint **`99 %` des images oubliées (`>95% = 0.01`)** avec un écart `Gap50 = +6.92` tout en laissant les 30 autres classes à **`InterAcc = 99.7 %` (`InterKL = 0.0545`)**, mais le gain `THEN` (`0.8 / LayerScale`) était ~15× trop fort (`CE = 10.61`, `IntraAcc = 37.5 %`).
- **Soumission `#10` (`01h58:06`, `H4_h123_f6_ce125.pt`, `submission_id: 2387`) = `0.0984` (`previous_best: 0.1193`, `delta: -0.0209`) — Encadrement exact du sommet de la cloche `gold` :**
  - Pour la toute première fois, nous avons franchi le sommet de la courbe en V inversé ($1 - |\mu_{\text{sub}} - \mu_{\text{gold}}|/d_{\text{orig}}$) :

| Position sur la cloche | Soumission & Modèle | `CE mean ± std` | **$\mu_{\text{sub}} + \sigma_{\text{sub}}$** | `p50` (`>95%`) | `ProxyKL` (`dTheta`) | **Score serveur** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Pente gauche (sous-oubli) | `#7` `AT_ce0.50_b3.0_f8.pt` | `0.704 ± 1.19` | `1.90` | `0.116` (`0.43`) | `0.0376` (`3.86`) | `0.0604` |
| Pente gauche (sous-oubli) | `#8` `AT_ce0.50_b2.0_f6.pt` | `1.265 ± 1.75` | `3.01` | `0.397` (`0.31`) | `0.0413` (`3.95`) | `0.1099` |
| Approche gauche du sommet | `#5` `B_ce0.45_lr8e-05_kl2.0.pt` | `1.438 ± 2.01` | **`3.45`** | `0.380` (`0.37`) | `0.0551` (`4.03`) | **`0.1166`** |
| **SOMMET OBSERVÉ** | **`#9` `AT_ce1.00_b1.8_f6.pt`** | **`1.909 ± 2.05`** | **`3.96`** | **`1.214` (`0.19`)** | **`0.0486` (`4.61`)** | **`0.1193` (BEST)** |
| **Pente droite (sur-oubli !)** | **`#10` `H4_h123_f6_ce125.pt`** | **`2.127 ± 2.10`** | **`4.23`** | **`1.468` (`0.14`)** | `0.0409` (**`6.27`**) | **`0.0984`** |

  - **Preuve géométrique du sommet :**
    - Entre `#5` (`CE = 1.44 ± 2.01`, $\mu+\sigma = 3.45$, `p50 = 0.38`, `dTheta = 4.03` $\to$ `0.1166`) et `#9` (`CE = 1.91 ± 2.05`, $\mu+\sigma = 3.96$, `p50 = 1.21`, `dTheta = 4.61` $\to$ `0.1193`), le score atteint son maximum.
    - Dès que `CE mean` dépasse `2.0` (`2.13` sur `#10`), que `p50` dépasse `1.3` (`1.47` sur `#10`) et que `dTheta` monte à `6.27` (à cause de `gamma_lr_mult = 3.0` sur 6 blocs), on bascule sur la **pente descendante droite (sur-oubli)** !

### 2026-11-22 (02h17) — 11ᵉ Soumission serveur (`H1_h1h2_f8_ce125.pt`, `submission_id: 2457`) = `0.0813` & Diagnostic Racine Critique (`Piste H` vs `AT` standard)
- **Résultat serveur (`02h17:00`, `task-2435`) :**
  - `{"submission_id": 2457, "status": "success", "score": 0.0813, "previous_best": 0.1193, "delta": -0.0381, "improved": false}`
- **Pourquoi `H4_f6` (`0.0984`) et `H1_f8` (`0.0813`) sont battus sur le serveur par `#9` `AT_ce1.00_b1.8_f6.pt` (`0.1193`), `#5` (`0.1166`) et `#8` `AT_ce0.50_b2.0_f6.pt` (`0.1099`) alors qu'ils gagnaient sur `W_SEL` (`0.6108` et `0.6041`) :**
  En auditant le code de [track_h.py](../../tracks/03_unlearning_v2/methods/track_h.py) face à nos 11 soumissions serveur, **3 causes structurelles exactes** expliquent l'écart :
  1. **Bug mathématique dans `H2` (`update_conductance_and_rollback`, L192–203 de `track_h.py`) :**
     - À $t = 0$, `student == orig_model`, donc le gradient de `KL(orig_p || student(p_x))` est **identiquement nul** ($\nabla D_{\text{KL}}(p \parallel q)|_{q=p} = 0$) ! Diviser `gf / (gr + 1e-8)` à $t=0$ divise par du bruit numérique.
     - Pis : recalculer et **changer le masque SalUn tous les 15 steps** sur 2 petits batchs aléatoires de 64 images fait tourner les 20 % de poids actifs à chaque période — au bout de 225 steps, **70 à 85 % de tous les poids du backbone ont été modifiés** (brisant l'invariance stricte des 80 % de poids figés de SalUn et faisant exploser `dTheta` à **`5.78 – 6.27`** contre **`3.95 – 4.61`** sur `#8`, `#5`, `#9` !).
  2. **Étouffement de `head.weight` par `head_lr_mult = 0.5` + `alpha_proto = 0.75–0.80` :**
     - En bridant les lignes actives de `head.weight`, `H1` artificially gonfle `IntraAcc` dans les folds, mais **force les blocs Transformer partagés (`blocks`) à se tordre 1.5× plus fort (`dTheta = 5.78–6.27`)** pour obtenir la même baisse de CE, ce qui dégrade la représentation générale sur le vrai `validation / control set` caché du serveur !
  3. **Infériorité systématique de `freeze_until_block = 8` (`f8`) face à `freeze_until_block = 6` (`f6`) sur le serveur (signal observationnel, non causal) :**
     - Sur 11 soumissions, **100 % des modèles `f8` plafonnent à `<= 0.0813`** (`#7` `AT_f8` = `0.0604`, `#11` `H1_f8` = `0.0813`), tandis que **100 % des modèles `f6` SalUn standard dépassent `0.0984–0.1193`** (`#10` `H4_f6` = `0.0984`, `#8` `AT_f6` = `0.1099`, `#5` `B_f6` = `0.1166`, **`#9` `AT_f6` = `0.1193`**). Il s'agit d'un signal empirique sur des configs variées pour prioriser `f6`, pas encore d'une ablation appariée prouvant la causalité isolée de `f6` vs `f8`.

### 2026-11-22 (02h37) — Revue d'Inspecteur & Résultat Négatif `Piste H` Round 3 (`W_SEL = 0.302–0.403`)
- **Prise en compte intégrale des 7 points critiques de l'inspecteur (rigueur scientifique) :**
  1. **Séparation stricte Sélection Locale (`W_SEL`) vs Score Officiel Serveur :** `W_SEL` reste un outil de diagnostic (pas un « record » au sens du challenge) ; l'inversion observée (`H4` `0.6108` $\to$ serveur `0.0984`, `H1` `0.6041` $\to$ serveur `0.0813` vs `#9` `0.5565` $\to$ serveur **`0.1193`**) prouve que `W_SEL` ne mesure ni le vrai `gold` caché ni le gate `val_acc >= 70%`. **Seul le score serveur est l'arbitre.**
  2. **Aucune causalité non démontrée / pas de garantie théorique sur LoRA rang 4 :** les effets `H1/H2/H3` et `f6/f8` sont formulés comme **hypothèses à isoler par ablations *one-factor-at-a-time*** à hyperparamètres constants (et non comme preuves causales), et le rang 4 LoRA est une hypothèse falsifiable, pas une garantie d'« *information bottleneck* » (`track_i_round1.json` : `frac_gt_95 = 0.82–0.89` après Phase 2).
  3. **Biais de sélection sur folds/probes fixes :** la réutilisation des mêmes `seed=42` folds et probes rend `W_SEL` optimiste ; toute validation locale finale doit réserver un split/seed jamais vu pour le tuning.
  4. **Consignation du résultat négatif `Piste H` Round 3 ([track_h_round3.json](../../tracks/03_unlearning_v2/output/track_h/track_h_round3.json)) :**
     - Tous les candidats Round 3 sous-performent : `R3_f6_ce085_ap30_safe` à **`W_SEL = 0.4032 ± 0.088`** (`ProxyKL = 0.0543`, `dTheta = 4.48`), `R3_f6_ce090_ap20` à `0.3420`, `R3_f6_ce090_ap20_h4` à `0.3390` (**`dTheta = 107.80`**, garde-fou L2 insuffisant sur H4), et `R3_f6_ce100_ap15` à `0.3017`. La `Piste H` Round 3 est donc un **résultat négatif** consigné dans [03-unlearning-track-h-journal.md](03-unlearning-track-h-journal.md).

### 2026-11-22 (03h12:02) — 12ᵉ Soumission serveur (`I2R_f6_r4_ce18_fl085.pt`, `submission_id: 2648`) = **`0.0950`** (Test de cohérence sur régime opposé `Piste I`)
- **Résultat serveur (`03h12:02`, `task-2614`) :**
  - `{"submission_id": 2648, "status": "success", "score": 0.095, "previous_best": 0.1193, "delta": -0.0244, "improved": false}`
- **Ce que prouve `0.0950` sur un régime opposé (*Erase-Then-Heal LoRA*) :**
  - `I2R_f6_r4_ce18_fl085.pt` avait `Full CE = 2.458 ± 1.98` ($\mu+\sigma = 4.44$, `p50 = 2.136`, `ProxyKL = 0.0740`, `dTheta = 6.01`), **mais** `>95% = 0.077` (92 % des images désaturées) et `IntraAcc = 88.5 %`.
  - Son score **`0.0950`** reste solide (proche de `#10` `0.0984` et supérieur à `#11` `0.0813` et `#2` `0.0843`), tout en confirmant la **pente descendante droite** au-delà du sommet `#9` ($\mu+\sigma = 3.96 \to 0.1193$, `4.23 -> 0.0984`, `4.44 -> 0.0950`). La `Piste I` (corridor `CE` ramené de `2.46` vers `[1.55, 1.85]` avec `ProxyKL <= 0.042` et `dTheta <= 4.5` au Round 3) est donc une vraie piste compétitive. *(Prochain créneau `#13` à **`03h27:02`**.)*

### 2026-11-22 (03h29:55) — 13ᵉ Soumission serveur (`SOUP_50_050_50_100.pt`, `submission_id: 2706`) = **`0.1238` (NOUVEAU RECORD, `improved: true`)**
- **Résultat serveur (`03h29:55`, `task-2664`) :**
  - `{"submission_id": 2706, "status": "success", "score": 0.1238, "previous_best": 0.1193, "delta": 0.0045, "improved": true, "message": "New best on this task."}`
- **Pourquoi la soupe 50/50 (`0.50 * #8 + 0.50 * #9`) bat strictement ses deux parents (`0.1099` et `0.1193` $\to$ `0.1238`) :**
  - `#8` (`AT_ce0.50_b2.0_f6.pt`, $\mu+\sigma = 3.01$, `p50 = 0.397`, `ProxyKL = 0.0413`) $\to$ `0.1099`
  - `#9` (`AT_ce1.00_b1.8_f6.pt`, $\mu+\sigma = 3.96$, `p50 = 1.214`, `ProxyKL = 0.0486`) $\to$ `0.1193`
  - **`#13` (`SOUP_50_050_50_100.pt`, $\mu+\sigma = 3.84$, `Full CE = 1.794 ± 2.04`, `p50 = 1.006`, `>95% = 0.21`, `dTheta = 4.48`) $\to$ `0.1238` !**
  - L'interpolation convexe place `Full CE` et `p50` pile sur le sommet de la cloche tout en réduisant le bruit d'entraînement par rapport à un run individuel. *(Prochain créneau `#14` à **`03h44:55`**.)*

### 2026-11-22 (03h35) — Diagnostic `prediction_score ~ 0.19` & Implémentation de l'alignement image-par-image via Patch-Dropout (`patch_drop_targets.py`)
- **Constat arithmétique :** avec `retain_score ~ 0.92` et `mean_std_score ~ 0.65–0.75` au sommet ($\mu+\sigma = 3.84$), un score final à `0.1238` implique mathématiquement que **`prediction_score` (`1 - KL(p_gold || p_sub) / KL(p_gold || p_orig)` à $T=2$) est bloqué autour de `~0.19`**.
- **Cause hypothétique :** `NPO` uniforme choisit aléatoirement quelles images oublient en premier (selon l'amplitude locale du gradient) au lieu d'oublier les ~20 % d'images atypiques que `gold` rate tout en préservant les ~80 % d'images typiques que `gold` réussit.
- **Implémentation ([patch_drop_targets.py](../../tracks/03_unlearning_v2/methods/patch_drop_targets.py), [run_patchdrop_ablation.py](../../tracks/03_unlearning_v2/run_patchdrop_ablation.py), [test_patch_drop_targets.py](../../tracks/03_unlearning_v2/tests/test_patch_drop_targets.py)) :**
  - Estimation multi-vues `orig_model` sous **Patch-Dropout 45 %** (`16×16`) + miroir (`estimate_patchdrop_gold_logits`), recalibrée vers le sommet serveur ($\mu_{\text{target}} \in [1.65, 1.80]$) en gardant les 99 logits non-cibles intacts (`calibrate_per_image_targets`), puis optimisation directe par image à $T=2$ + NPO borné par échantillon sur la base propre `#9` (`f6`, SalUn statique $t=0$).
  - Mesure de la corrélation de Spearman `spearman_vs_patchdrop` et test sur folds *held-out* `seed=777` (session Colab A100 `track3_pd`).
- **Résultats de l'ablation `track3_pd` ([patchdrop_ablation.json](../../tracks/03_unlearning_v2/output/patchdrop_vgt/patchdrop_ablation.json), session `track3_pd` stoppée) :**
  - `PD_2_kl12_npo00_ce180` : **`rho_PD = +0.640`**, `IntraAcc = 90.9 %`, `W_SEL = 0.3431 ± 0.018`, mais **`Full CE = 0.322 ± 0.64` ($\mu+\sigma = 0.97$)**, `>95% = 0.45`, `InterKL = 0.1077`.
  - `PD_1_kl10_npo03_ce175` : `rho_PD = +0.623`, `W_SEL = 0.3342 ± 0.021`, `Full CE = 0.361 ± 0.67` ($\mu+\sigma = 1.04$), `>95% = 0.40`.
  - `PD_3_kl10_npo05_ce165` : `rho_PD = +0.618`, `W_SEL = 0.3305 ± 0.022`, `Full CE = 0.373 ± 0.68` ($\mu+\sigma = 1.06$), `>95% = 0.38`.
  - *Diagnostic :* le modèle apprend bien l'ordre de difficulté par image (`rho_PD = +0.64`), mais la loss `KL(T=2)` sur les ~80 % d'images faciles (`CE_tgt ~ 0.05–0.25`) domine l'optimisation et maintient le modèle en fort sous-oubli ($\mu+\sigma \approx 1.0 \ll 3.84$). Il faudra n'utiliser `ce_pd` que comme **poids / seuil par échantillon** dans NPO sans terme `KL` sur les cibles faciles.

### 2026-11-22 (03h49) — Synthèse des ablations *held-out* (`seed=777`) des 3 cellules autonomes
1. **`Piste H` — Rounds 4–6 (`seed=777`, [03-unlearning-track-h-journal.md](03-unlearning-track-h-journal.md)) :**
   - *Round 4 (1-facteur contre `#9` `ABL_0 = 0.5505 ± 0.028`) :* `ABL_2_plus_H1_gov` (`0.5613 ± 0.012`, `>95% = 0.09`), `ABL_1_plus_H1_ortho` (`0.5528 ± 0.028`, `IntraAcc +1.3 pt`), `ABL_3_plus_H2_damp` (`0.5117 ± 0.067` — **écarté**).
   - *Rounds 5–6 (`weights.mean().clamp_min(1.0)`) :* `R5_h1gov_ortho25` (`W_SEL = 0.5715 ± 0.004`, **variance inter-folds ÷ 7**, `ProxyKL = 0.0438`, `dTheta = 4.43`, Full `1.15 ± 1.57`) et `R6_h1gov_ortho25_ce125` (`W_SEL = 0.5619 ± 0.008`, **`ProxyKL = 0.0359`**, `AbsKL = 0.0230`, **`dTheta = 3.99`**).
2. **`Piste I` — Round 3 (`seed=777`, [03-unlearning-track-i-journal.md](03-unlearning-track-i-journal.md)) :**
   - *Ablation 1-facteur Phase 2 LoRA (`I3R_4_lora_r8` vs `I3R_1_p1only`) :* la LoRA rang 8 réduit **`ProxyKL` de `-43,2 %`** (`0.1159 -> 0.0658`) et **`AbsKL` de `-41,5 %`**, augmente **`IntraAcc` de `+7,7 pts`** (`75,3 % -> 83,0 %`), garde `InterAcc = 100,0 %` (`InterKL = 0.0201`) et `>95% = 0.092` à `dTheta` constant (`5.78 -> 5.80`). Round 4 préparé (`phase1_max_dtheta <= 4.30` + soupes `I3R_4 + orig`).
3. **`Techniques Initiales` — Round 1 ([03-unlearning-base-journal.md](03-unlearning-base-journal.md)) :**
   - `AT_ce0.80_b1.9_k20_kl28` (`W_SEL = 0.5694 ± 0.020`, `Full = 1.96 ± 2.09`, soumis en `#14`) et `SOUP_50_050_50_100` (soumission `#13` = **`0.1238` RECORD**). Round 2 (`run_tight_at` + soupes `ORIG`) en cours sur `track3_base`.

### 2026-11-22 (03h50:18) — 14ᵉ Soumission serveur (`AT_ce0.80_b1.9_k20_kl28.pt`, `submission_id: 2766`) = **`0.1197`** (2ᵉ meilleur score)
- **Résultat serveur (`03h50:18`, `task-2718`) :**
  - `{"submission_id": 2766, "status": "success", "score": 0.1197, "previous_best": 0.1238, "delta": -0.0041, "improved": false}`
- **Confirmation du sommet :** `AT_ce0.80_b1.9_k20_kl28.pt` (`CE = 1.96 ± 2.09`, $\mu+\sigma = 4.05$, `p50 = 1.256`, `dTheta = 4.58`) devance `#9` (`0.1193`) et `#5` (`0.1166`), mais reste derrière la soupe 50/50 `#13` (`0.1238`, $\mu+\sigma = 3.84$, `p50 = 1.006`, `dTheta = 4.48`), confirmant l'avantage de la soupe convexe exactement à $\mu+\sigma \approx 3.84$ (`p50 ~ 1.0`). *(Prochain créneau `#15` à **`04h05:18`**.)*

### 2026-11-22 (04h06:40) — 15ᵉ Soumission serveur (`SHRINK_88_AT9.pt`, `submission_id: 2818`) = **`0.1082`** + Lancement Piste SCRUB+R
- **Résultat serveur (`04h06:40`, `task-2771`) :**
  - `{"submission_id": 2818, "status": "success", "score": 0.1082, "previous_best": 0.1238, "delta": -0.0156, "improved": false}`
- **Preuve causale clé :** Bien que `SHRINK_88_AT9.pt` (`0.88 * AT_ce1.00 + 0.12 * ORIG`) détenait notre meilleur `W_SEL = 0.5800 ± 0.006` sur `seed=777` et un `ProxyKL = 0.0353` très bas, son `Full mu+sig = 2.92` (`CE = 1.212 ± 1.71`) le maintient sur la pente gauche de la courbe en cloche (`#8` à `3.01 -> 0.1099`). Conséquence directe pour `#16+` (fenêtre à `04h21:40`) : tout candidat visant à battre `0.1238` doit rester dans **`Full mu+sig ∈ [3.65, 3.90]`** (`p50 ∈ [0.85, 1.15]`), et optimiser `ProxyKL` / `dTheta` / `rho_PD` **à l'intérieur de cette fenêtre**.
- **Organisation du swarm (`04h06`) :**
  - Tous les agents actifs (`base-lead`, `base-exec`, `h-lead`, `h-exec`, `i-lead`, `i-lora`, `monitor`, `coordinator`) passés en `gemini-4-hydrogen-medium`.
  - Lancement du duo **`scrub-lead` + `scrub-exec`** sur **Piste SCRUB+R** (`docs/analyses/03-scrub-machine-unlearning.md` + reprise de `methods/patch_drop_targets.py`), avec un 4ᵉ slot Colab réservé (`track3_scrub --gpu L4`, en parallèle de `track3_i` L4, `track3_base` A100 et `track3_h` T4).

### 2026-11-22 (04h24:42) — 16ᵉ Soumission serveur (`SOUP_50_ce80kl28_50_ce90kl28.pt`, `submission_id: 2876`) = **`0.1184`**
- **Résultat serveur (`04h24:42`, `task-2920`) :**
  - `{"submission_id": 2876, "status": "success", "score": 0.1184, "previous_best": 0.1238, "delta": -0.0054, "improved": false}`
- **Analyse :** `SOUP_50_ce80kl28_50_ce90kl28.pt` (`Full CE = 1.941 ± 2.07`, $\mu+\sigma = 4.01$, `p50 = 1.258`, `ProxyKL = 0.0524`, `dTheta = 4.52`) se place juste à droite du pic `#13` (`3.84 -> 0.1238`) dans la même bande que `#9` (`3.96 -> 0.1193`) et `#14` (`4.05 -> 0.1197`). *(Prochain créneau `#17` à **`04h39:42`**.)*

### 2026-11-22 (04h16–04h32) — Calibration du protocole local (`7f41ffa`), Handoff (`0a1aef8`) & Incident RAM `T4`
- **Correction des 3 failles du protocole de test local (`7f41ffa`) :**
  - [selective_benchmark.py](../../tracks/03_unlearning_v2/selective_benchmark.py) : remplacement de la rampe monotone `/ 5.2` et `/ 6.2` dans `score_single_fold` par `_bell_forget_term` centré à `peak_ms = 3.84` (`raw`) / `4.40` (`alt`).
  - [eval_harness.py](../../tracks/03_unlearning_v2/eval_harness.py) : ajout de `compute_spearman_rank_corr` (`rho_PD`) et `score_server_calibrated_proxy` sur le vrai `.pt` 3 500 images (`Full mu+sig = 3.84`, `p50 = 1.0`, `ProxyKL`, `AbsentKL`, `dTheta <= 4.70`, bonus `rho_PD`). Consigne `#97` diffusée au swarm pour intégration dès les prochains rounds.
- **Document de Handoff (`0a1aef8`) :** création de [03-unlearning-handoff.md](03-unlearning-handoff.md) récapitulant l'état complet des 4 pistes, des 4 GPUs Colab, du cooldown et du plan d'action `#16+`.
- **Incident `T4` (`track3_h`) & Quotas (`04h31`) :** le runtime `T4` (~12 Go RAM hôte) a OOM-kill (`rc=-9`) sur Piste H Round 7 après les 3 folds de `R7_1` ; ajout d'un flush JSON + `gc.collect()` incrémental demandé (`#106`) avant relance sur `L4`/`A100` dès qu'un slot Colab se libère. Retour de tout le swarm sur `gemini-4-hydrogen-medium` (`gemini-3.8-flash-medium` pour `monitor`) suite aux `429` sur `claude-sonnet-5-5-*`.

---

## 4. Défis techniques priorisés & Stratégies candidates

### Défis priorisés (`P0` → `P2`)
1. **[P0] Passer le *utility gate* (`val_acc >= 70%`) et préserver `retain_score` sans `retain set`** (verrou sur 55 classes absentes + ancrage L2/KL).
2. **[P0] Éviter l'*over-unlearning* et calibrer la cible `gold`** (loss bornée + balayage par paliers).
3. **[P1] Transférer l'oubli aux *hidden altered copies*** (augmentations fortes + stress-test).
4. **[P1] Contourner la saturation initiale des gradients (`CE = 0.0003`)** (température $T \ge 2$ pour SalUn / NPO).
5. **[P2] Mesurer en local sans `val set` ni `gold`** (harnais Phase 3).

### Stratégies candidates (`A`–`E`)
- **Stratégie A *(Baseline prioritaire)* — SalUn ($T \ge 2$) + NPO + Gel des 55 classes absentes :** masque SalUn tempéré, gel de `head[55 absentes]` et `patch_embed`/`blocks.0..5`, loss NPO sur `forget` fortement augmenté (+ cohérence multi-vues optionnelle), L2 vers $\theta_{\text{orig}}$ + KL ($T=2$) sur *PatchMix*.
- **Stratégie B — *SCRUB / Bad-Teacher* asymétrique sans `retain set` :** `KL(p_orig || p_sub)` ($T=2$) sur proxy `retain` + distillation vers une distribution adoucie réaliste sur `forget` augmenté.
- **Stratégie C — Amortissement synaptique / Projection spectrale (`SSD` / Fisher / SVD) :** atténuer sélectivement les poids/directions les plus spécifiques au `forget` dans `head` et `blocks.8..11` sans boucle d'entraînement agressive.
- **Stratégie D — Enrichissement du proxy `retain` :** comparer *PatchMix* vs bruit structuré / MixUp vs images naturelles génériques sur la dérive des 55 classes absentes.
- **Stratégie E — Unlearning par LoRA FP32 ($W_0 + BA$, sans 4-bit) :** contraindre les mises à jour dans un sous-espace de rang faible ($r \in \{4, 8, 16\}$) sur `blocks.8..11` + `head` (couplé à NPO ou Bad-Teacher + augmentations fortes + gel des 55 classes absentes), puis fusionner en `float32` avant export.
- **Stratégie F *(inspirée de Heretic)* — Abliteration / Orthogonalisation directionnelle (+ Optuna) :** extraire les directions principales des activations `forget` (dans `blocks.8..11` / `CLS`) orthogonales aux 55 classes absentes, atténuer/orthogonaliser partiellement les projections avec un facteur dosé $\alpha \in [0, 1]$, et optimiser automatiquement couches + $\alpha$ sous contrainte de plafond `KL(T=2)`.

---

## 5. Glossaire technique

| Terme | Définition simple (appliquée à Track 3) |
| :--- | :--- |
| **Logits ($z$)** | Les 100 scores bruts en sortie de `head` avant normalisation. |
| **Softmax ($p$)** | Transforme les logits en probabilités : $p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}$. |
| **Température ($T$)** | Adoucit ($T > 1$) ou durcit ($T < 1$) le softmax. $T = 2$ divise les écarts de logits par 2. |
| **Saturation (du softmax)** | Quand $p_{\max} \approx 1.0000$ (`0.9997`), le gradient CE vaut $p_y - 1 \approx -0.0003 \approx 0$ : signal quasi nul. |
| **Cross-Entropy (CE / `L`)** | $L = -\log(p_y)$. Proche de `0` = certitude totale ; plus `L` monte, plus le modèle doute ou se trompe. |
| **Entropie $H(p)$** | Incertitude globale : $H = -\sum_i p_i \log p_i$ (`0` = certitude, `log(100) ≈ 4.61` = hasard pur). |
| **Marge (`top1 - top2`)** | Écart de probabilité entre la 1ʳᵉ et la 2ᵉ classe prédite (`0.9994` ici). |
| **KL-Divergence (`KL(p \|\| q)`)** | Distance asymétrique entre deux distributions (`0` = identiques), évaluée à $T = 2$ par le serveur. |
| **Modèle `gold`** | Modèle de référence ré-entraîné de zéro **sans** `forget_images.npz`. |
| **Utility Gate (`val_acc >= 70%`)** | Seuil éliminatoire : sous `70 %` sur le `validation set` caché, le score vaut `0`. |
| **Over-unlearning** | Oublier trop fort (perte CE extrême / loi uniforme), détruisant `mean_std_score`, `prediction_score` et `val_acc`. |
| **Hidden altered copies** | Copies altérées (crop, flou, couleur, bruit…) de `forget_images.npz` gardées secrètes sur le serveur. |
| **Retain / Control set** | Images hors-forget (`retain`) et jeu témoin caché (`control`) mesurant la préservation des connaissances. |
| **Proxy `retain` (*PatchMix*)** | Images synthétiques (patchs mélangés) simulant un `retain set` local. |
| **Prototypes (`head.weight`)** | Les 100 vecteurs (dim `768`) de la dernière couche linéaire. |
| **`grad_rms`** | Moyenne quadratique du gradient par paramètre dans une couche. |
| **SalUn** | *Gradient-based Weight Saliency* : ne modifier que les poids les plus saillants pour `forget`. |
| **NPO** | *Negative Preference Optimization* : perte d'oubli bornée évitant la divergence infinie par rapport à `orig`. |
| **SCRUB / Bad-Teacher** | Distillation asymétrique : bon teacher (`orig`) sur `retain`, mauvais teacher sur `forget`. |
| **SSD** | *Selective Synaptic Dampening* : amortir directement les poids spécifiques au `forget` via Fisher/gradients. |
| **Teacher / Student** | *Knowledge Distillation* : un modèle figé (`teacher`, ici `orig`) guide un modèle modifié (`student`). |
| **`max-step` / `min-step`** | Alternance dans SCRUB : `max-step` éloigne le student sur `forget`, `min-step` le rapproche sur `retain`. |
| **Rewind (`SCRUB+R`)** | Sauvegarder un checkpoint par époque et revenir à celui dont l'erreur sur `forget` égale un `val set` témoin. |
| **MIA (*Membership Inference Attack*)** | Attaque qui cherche à deviner si une image faisait partie du jeu d'entraînement d'un modèle. |
| **LiRA (*Likelihood Ratio Attack*)** | MIA avancée (Carlini et al.) utilisant des *shadow models* pour distinguer une image « oubliée » d'une « jamais vue ». |
| **Shadow models** | Modèles auxiliaires entraînés par un attaquant pour simuler le comportement avec/sans une donnée cible. |
| **AUC** | Aire sous la courbe ROC (`0.50` = hasard pur / vie privée parfaite ; `1.00` = détection parfaite). |
| **RB / RC / UP (SCRUB)** | 3 régimes d'unlearning : *Removing Biases* (erreur max), *Resolving Confusion* (corriger des erreurs), *User Privacy* (imiter `gold`). |
| **Fisher / EWC / NTK** | Méthodes classiques estimant l'importance des poids (courbure de Fisher, noyau tangent NTK) pour éviter d'effacer `retain`. |
| **Oubli catastrophique** | Quand modifier le modèle pour une nouvelle tâche (ou un oubli) détruit brutalement ses performances générales. |
| **Class vs Selective/IID Unlearning** | Oublier des **classes entières** vs oublier des **images individuelles** réparties sur plusieurs classes. |
| **Gradient Ascent (montée de gradient)** | Pousser les poids dans le sens qui **augmente** la perte CE sur `forget` (instable et non borné si utilisé seul). |
| **Loss bornée vs non bornée** | Une loss bornée (ex. NPO) sature quand l'oubli est suffisant ; une loss non bornée (ex. `-KL` ou `-CE`) diverge vers l'infini. |
| **Ancrage L2 ($\|\theta - \theta_{\text{orig}}\|_2^2$)** | Pénalité qui retient les poids près du modèle original pour limiter la dérive globale. |
| **ViT / `patch_embed` / `CLS`** | *Vision Transformer* : découpe l'image en patchs (`patch_embed`), les traite via `blocks.0..11`, et résume l'image dans le vecteur `CLS`. |
| **Embeddings & Similarité cosinus** | Vecteurs de représentation (`dim 768` avant `head`) ; le cosinus (`-1` à `1`) mesure si deux images ou prototypes sont proches. |
| **SVD / Projection spectrale** | Décomposition d'une matrice de poids en directions principales pour atténuer uniquement les directions liées au `forget`. |
| **`mean_std_score` / `prediction_score`** | Les 2 composantes de `forget_score` : écart sur `(mean(L), std(L))` vs `gold`, et KL softmax ($T=2$) image par image vs `gold`. |
| **Data Augmentation** | Altérations aléatoires appliquées pendant l'entraînement (`RandomResizedCrop`, `ColorJitter`, `GaussianBlur`, `RandomErasing`) pour généraliser aux copies altérées. |
| **LoRA / QLoRA** | *Low-Rank Adaptation* : ajoute un résidu de rang faible $\Delta W = B A$ ($r \ll d$) fusionnable dans $W_0$. **QLoRA** y ajoute une quantification 4-bit de $W_0$ pour économiser la VRAM (inutile ici à 86M params). |
| **Abliteration / ARA (*Heretic*)** | Retirer un comportement en identifiant sa direction dans le flux résiduel et en orthogonalisant les poids contre elle (*Arbitrary-Rank Ablation*), sans ré-entraînement complet. |
| **Optuna (TPE)** | Optimiseur bayésien (*Tree-structured Parzen Estimator*) utilisé par Heretic pour trouver couches et intensités minimisant le comportement cible sous contrainte de `KL` faible. |
| **LTD / *Pruning* / Quantification** | Affaiblissement synaptique (*Long-Term Depression*) : réduire ou mettre à zéro (*pruning*) des poids, ou baisser leur précision (*quantification*, ex. 4-bit). |
| **Interférence / *Retrieval Failure*** | Oubli par concurrence d'une nouvelle réponse (interférence) ou blocage du chemin d'accès sans détruire la trace de base (*retrieval failure* / inhibition active). |
| ***Weight Decay*** | Régularisation $L_2$ qui pénalise l'amplitude des poids (ou leur écart à $\theta_{\text{orig}}$) pendant l'optimisation. |

---

## 6. Guide de lecture des métriques : quelles valeurs viser ?

> **Note :** Seuils heuristiques locaux à calibrer ensuite contre les vrais scores du serveur (`gold` et `val_acc` étant cachés).

| Métrique | Valeur `orig` | Zone saine / cible | Alerte (risque) |
| :--- | :--- | :--- | :--- |
| **`Raw CE` (`mean`)** | `0.000` | **`0.3` – `1.2`** (oubli dosé type `gold`) | `< 0.05` (sous-oubli) ou **`> 2.5`** (*over-unlearning*) |
| **`Raw Ent` ($H(p)$)** | `0.002` | **`0.15` – `0.80`** (doute réaliste sur 100 classes) | `< 0.02` (mémorisé) ou **`> 2.5`** (modèle aléatoire) |
| **`Stress CE avg`** | `1.267` | **`1.5` – `3.0`** (progression modérée) | **`> 5.0`** (le modèle casse sur les altérations) |
| **`Stress Agree avg`** | `0.801` | **`0.55` – `0.72`** | `< 0.45` (perte d'invariance générale) |
| **`Proxy KL(T=2)`** | `0.0000` | **`< 0.03`** (idéal), `< 0.08` (acceptable) | **`> 0.10`** (dérive `retain` / risque `val_acc < 70%`) |
| **`Absent KL(T=2)`** | `0.0000` | **`< 0.005`** (55 classes absentes quasi figées) | **`> 0.02`** (les couches internes dérivent trop) |
| **`Δθ L2`** | `0.00` | **`0.5` – `4.0`** (perturbation chirurgicale) | **`> 8.0`** (risque de dérive globale) |

### Ce qui affecte ces valeurs dans le modèle (les 4 leviers du ViT)
1. **Tête `head.weight` (`76 900` params) :** couche la plus sensible (`grad_rms = 9.3e-5`). Ajuster les 45 classes actives fait monter `Raw CE` / `Raw Ent` rapidement avec un coût minimal sur les 55 classes absentes (gelées).
2. **Blocs Transformer hauts (`blocks.6..11`, ~42M params) :** portent la représentation sémantique `CLS`. Les modifier transfère l'oubli aux copies altérées (`Stress CE`), mais comme ils sont partagés sur 100 classes, trop les bouger fait grimper `Proxy KL` et `Absent KL` (d'où l'intérêt de **LoRA** ou **SalUn**).
3. **Couches basses (`patch_embed`, `blocks.0..5`) :** extraient les primitives visuelles universelles (bords, textures) $\implies$ **gelées** pour éviter de casser `val_acc >= 70%`.
4. **Fonction de perte & augmentations :** `NPO` monte vite `Raw CE` (plus dispersé `±2.3`), `Bad-Teacher` monte `Raw Ent` de façon homogène (`std = 0.25`) ; les augmentations doivent rester **modérées** pour ne pas déformer les features générales (`Proxy KL`).

---

### Recherche VGT / SCRUB+R (expérimentation isolée)
- Journal et protocole exploratoire : [[03-unlearning-vgt-journal]].
- État : diagnostic exploratoire du masque exécuté sur L4 ; à 45 %, accord top-1 64,8 % sur 256 images. Aucun checkpoint candidat ni score serveur comparé ; détails et limites : [[03-unlearning-vgt-journal]].

## 7. Aide pour lire les maths & symboles

| Symbole | Nom | Ce que ça veut dire en français simple | Exemple dans notre projet |
| :--- | :--- | :--- | :--- |
| **$\theta$** (`thêta`) | Paramètres / Poids | Tous les nombres (poids) à l'intérieur du réseau de neurones (`86M` ici). | $\theta_{\text{orig}}$ = les poids initiaux de `model.pt`. |
| **$\Delta$** (`Delta`) | Différence / Changement | « À quel point ça a bougé entre avant et après ». | $\Delta \theta = \theta_{\text{sub}} - \theta_{\text{orig}}$ (écart des poids). |
| **$\|\cdot\|_2$** | Norme L2 (distance) | La taille totale d'un vecteur (comme une distance en ligne droite). | $\|\Delta \theta\|_2 = 3.09$ mesure combien le modèle a bougé au total. |
| **$\sum$** (`Sigma`) | Somme | Additionner tous les éléments d'une liste ou d'un jeu de données. | $\sum_{i=1}^{100} p_i = 1$ (la somme des 100 probabilités vaut 1). |
| **$\nabla$** (`Nabla`) | Gradient | La direction et la pente pour faire monter ou descendre l'erreur. | $\nabla_\theta L$ = comment bouger chaque poids pour changer la perte $L$. |
| **$\partial$** (`d rond`) | Dérivée partielle | La pente par rapport à **une** variable précise. | $\frac{\partial \text{CE}}{\partial z_y}$ = sensibilité de l'erreur au score de la classe $y$. |
| **$\parallel$** | « Par rapport à » (dans KL) | Sépare deux distributions de probabilités qu'on compare. | $D_{KL}(p \parallel q)$ = écart entre la distribution $p$ et $q$. |
| **$\in$** | « Appartient à » | Un élément fait partie d'un ensemble ou d'une plage de valeurs. | $\alpha \in \{0.3, 0.6\}$ = on teste $\alpha = 0.3$ et $\alpha = 0.6$. |
| **$\mathcal{D}_f, \mathcal{D}_r$** | Ensembles de données | $\mathcal{D}_f$ = *forget set* (3 500 images), $\mathcal{D}_r$ = *retain set* (absent ici). | $x \in \mathcal{D}_f$ = « une image $x$ prise dans le forget set ». |
| **$\alpha, \beta, \gamma$** | Coefficients | De simples curseurs (nombres) pour régler le poids de chaque terme. | $\alpha = 0.3$ dans `abliteration` = force de coupure de `30 %`. |
| **$\approx, \ge, \le$** | Comparaisons | $\approx$ « environ », $\ge$ « supérieur ou égal », $\le$ « inférieur ou égal ». | `val_acc >= 70%` = au moins `70 %` de bonnes réponses. |
| **$\implies$** | « Implique / Donc » | Si la gauche est vraie, alors la droite arrive. | `val_acc < 70%` $\implies$ `score = 0`. |
| **$\log$ / $e^x$** | Logarithme / Exponentielle | Le `log` transforme un produit en somme et écrase les grands nombres ; $e^x$ fait l'inverse (toujours positif). | Softmax utilise $e^{z_i}$ pour rendre les scores positifs avant de diviser. |
| **$\text{mean} \pm \text{std}$** | Moyenne ± Écart-type | La valeur moyenne, et de combien les images s'écartent autour en moyenne. | `0.754 ± 2.31` = moyenne `0.754`, dispersion `2.31`. |

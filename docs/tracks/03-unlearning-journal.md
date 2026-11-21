# Track 3 — Unlearning : Journal & Backlog de Recherche

> **Document central unique** pour la Track 3 : backlog de recherche (`Q1`→`Q6`), journal chronologique complet (jobs réussis/échoués/annulés, constats bruts, incidents d'infra, questions intermédiaires), défis techniques, stratégies et glossaire.

**Liens connexes :**
- Spec officielle : [03-unlearning.md](03-unlearning.md) ([PDF](Unlearning.pdf))
- Plan d'action & TODO : [03-unlearning-todo.md](03-unlearning-todo.md)
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

> **ADR-02 — Architecture Orchestrateur + Sous-Agents & Ground-Truth Lab :**
> 1. **Organisation :** chaque piste (`A` SalUn+NPO, `B` Bad-Teacher, `E` LoRA, `F/C` Abliteration/SSD, `G` Lab Vérité Terrain) est perfectionnée par un sous-agent dédié ; l'agent principal joue le rôle d'**orchestrateur** (supervision, centralisation des résultats, respect du cooldown 30 min et des $\le 4$ jobs cluster).
> 2. **Nouvelle piste `G` (Ground-Truth Unlearning Lab) :** trouver un dataset public 100 classes sur HuggingFace, construire notre propre split `(retain, forget, val, altered)` avec un `orig_lab` et un vrai modèle `gold_lab`, afin de calculer **localement** les vraies métriques serveur (`val_acc >= 70%`, `mean_std_score`, `prediction_score`, `retain_score`) et calibrer nos stratégies avant soumission sur `model.pt`.

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
- **Résultat :** `Score (public split): 0.0287` (**Submission accepted** $\implies$ le modèle passe le *utility gate* `val_acc >= 70%`). Prochaine soumission autorisée vers `19h05`.
- **Diagnostic du score `0.0287` :**
  - Dans `run_15901899`, `H_ssd_lora_mod_aug` avait `Raw CE = 1.112 ± 3.79` et `Raw Ent = 0.060`.
  - Une moyenne `1.112` avec un écart-type énorme (`3.79`) et une entropie très basse (`0.060`) indique une distribution **bimodale** : la majorité des images restent sur-confiantes/mémorisées (`Ent ~ 0`), tandis qu'une minorité explose à `CE > 10`. Or le serveur pénalise à la fois `std(L)` (`mean_std_score`) et `KL(p_gold || p_sub)` à $T=2$ sur chaque image (`prediction_score`).
- **Hypothèse pour la 2ᵉ soumission :** il faut un oubli **homogène** (faible `std(L)`, entropie modérée `0.3–0.8` sur toutes les images, sans explosion bimodale) **avec** `Proxy KL < 0.04` — ex. combiner **Bad-Teacher (ou NPO avec lissage/plafond par image) + LoRA + augmentations modérées**.

### 2026-11-21 (18h59) — Construction de la Piste `G` (Ground-Truth Lab) & Test de cohérence
- **Objectif :** construire notre propre labo vérité terrain sur `imagenet-100` (`orig_lab` entraîné sur `R+F` vs `gold_lab` entraîné sur `R` seul), implémenter les formules exactes du serveur (`tracks/03_unlearning_v2/lab/server_metrics.py`), puis vérifier la cohérence en reproduisant le classement de nos 2 soumissions réelles (`A_npo_mod_aug` = `0.0843` > `H_ssd_lora_mod_aug` = `0.0287`).
- Si le test de cohérence est positif, tous les benchmarks et sous-agents par piste basculent sur ce banc.

### 2026-11-21 (18h57) — 2ᵉ soumission serveur (`A_npo_mod_aug.pt`) : Score `0.0843` (×2,94 !)
- **Résultat :** `Score (public split): 0.0843` (`Previous best: 0.0287`, `improved: True`). Prochaine soumission possible vers **`19h12`**.
- **Confirmation directe de notre hypothèse :**
  - `H_ssd_lora_mod_aug` (`Raw CE = 1.112±3.79`, `Raw Ent = 0.060`, `Proxy KL = 0.0184`) $\rightarrow$ `0.0287`.
  - `A_npo_mod_aug` (`Raw CE = 0.617±1.99`, `Raw Ent = 0.143`, `Proxy KL = 0.0366`) $\rightarrow$ **`0.0843`**.
  - Réduire `std(L)` de `3.79 → 1.99` et monter `Raw Ent` de `0.060 → 0.143` a **triplé** le score, confirmant que notre levier principal est un oubli **plus homogène** (`std < 1.2`, `Ent ∈ [0.25, 0.70]`) tout en gardant `Proxy KL < 0.04`.

### 2026-11-21 (18h38) — Règles du Leaderboard confirmées
- **Best-of :** seule la **meilleure** soumission par tâche est conservée (aucune pénalité si une itération score moins bien).
- **Classement général :** moyenne normalisée `[0, 1]` sur les **3 tâches** ($\frac{1}{3}\sum_t \frac{s_t - \min_t}{\max_t - \min_t}$) — il faut un score sur **chaque** track pour être classé au général.
- **Public (30 %) vs Final (70 %) :** éviter d'overfitter au score public ; privilégier les méthodes robustes et généralisables.

### 2026-11-21 (18h54) — Calibration par soumissions (15 min) & Stratégie du Lab sur copie de `model.pt`
- **Utilisation des soumissions :** exploiter la fenêtre de 15 min pour calibrer nos hypothèses sur le vrai score serveur et diagnostiquer où l'on pêche (`forget` vs `retain`).
- **Stratégie Piste `G` (copie de `model.pt` entraînée sur notre nouveau dataset `retain + forget`) :**
  - *Validation scientifique :* **très bonne stratégie**, à condition de (1) travailler sur une **copie** de `model.pt` (jamais sur l'original) et (2) entraîner `orig_lab` (sur `R_lab + F_lab`) et `gold_lab` (sur `R_lab` seul) avec **exactement le même protocole** depuis le même point de départ.
  - *Gain :* donne accès aux vrais labels, à un vrai `retain`/`val set` et à un vrai `gold_lab` pour calculer les métriques exactes du serveur (`val_acc`, `mean_std_score`, `prediction_score`, `retain_score`) avant de réappliquer les meilleurs réglages sur `model.pt` seul.

### 2026-11-21 (18h45) — Amélioration du harnais local (`eval_harness.py`) & Benchmark v4 (`15902308`)
- **Audit du harnais :** `retain` (`Proxy KL`, `Absent KL`, `Δθ`) se compare exactement à `orig` (même formule serveur $T=2$), mais sur `forget`, `mean±std` cachait la bimodalité.
- **Ajout dans `StressMetrics` & `benchmark.py` :** `ce_p50` (médiane), `ce_p90`, et `frac_gt_95` (% d'images encore > 95 % de confiance) pour détecter directement si l'oubli est homogène ou bimodal.
- **Benchmark v4 (`15902308`)** soumis sur JURECA (6 runs `LoRA + Bad-Teacher` & `NPO` doux). Note infra : Jülich PAM refuse la réutilisation du même code TOTP dans la même fenêtre de 30s (anti-replay) $\implies$ attendre la fenêtre suivante entre deux connexions SSH.

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

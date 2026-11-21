# Track 3 — Unlearning : Carnet d'Investigation & Backlog de Recherche

**Objectif du document :** Tenir à jour l'ensemble de notre raisonnement scientifique, des pièces disponibles vs cachées, des hypothèses formulées, des expériences d'exploration et des décisions d'architecture **avant et pendant** l'implémentation.

**Liens connexes :**
- Spec officielle : [03-unlearning.md](03-unlearning.md) ([PDF](Unlearning.pdf))
- Plan d'action & TODO : [03-unlearning-todo.md](03-unlearning-todo.md)
- Papier analysé : [SCRUB (03-scrub-machine-unlearning.md)](../analyses/03-scrub-machine-unlearning.md)
- Script d'exploration utilisé : [explore_unlearning.py](../../scripts/explore_unlearning.py)

---

## 1. Cartographie des pièces du puzzle

### 1.1 Ce à quoi on a accès (JURECA : `/p/scratch/training2666/ouaziz1/TRYON/unlearning/`)

| Pièce | Détails | Ce qu'on peut en extraire |
| :--- | :--- | :--- |
| **`model.pt` (343 Mo)** | ViT-B/16 (`timm vit_base_patch16_224`, 100 classes, `init_values=1e-4`), entraîné sur *retain + forget* | Poids complets ($\theta_{\text{orig}}$), prototypes de classes (`head.weight`), activations intermédiaires, gradients, prédictions et incertitudes |
| **`forget_images.npz` (526 Mo)** | 3 500 images RGB `uint8` (`[3500, 224, 224, 3]`) | Statistiques visuelles, doublons/quasi-duplicats, clusters, sensibilité des prédictions aux perturbations |
| **`task_template.py`** | `build_model()`, `load_model()`, `load_forget_images()`, `preprocess()`, `predict()` | Preprocessing exact et contrat d'inférence |
| **`submission.py`** | Client d'upload (`TASK_ID = "31-unlearning"`, `http://34.147.229.55`) | Retour de score public sur soumission |
| **Cluster JURECA** | GPUs A100 (`dc-gpu`, réservation `cispahack`) + `.venv/` (`torch 2.15.0`, `timm 1.0.31`) | Exploration rapide et ablations |

### 1.2 Ce qui est caché (côté serveur d'évaluation uniquement)

| Pièce cachée | Rôle dans le scoring | Conséquence directe |
| :--- | :--- | :--- |
| **Labels de `forget_images.npz`** | Utilisés pour évaluer `mean(L), std(L)` et les performances | On ne connaît pas les vraies classes ni la correspondance `0..99 ↔ noms` |
| **`retain set`** | Utilisé pour entraîner les modèles `gold` | Aucune image réelle du `retain set` pour distiller ou ancrer le modèle |
| **`validation set`** | *Utility gate* : exige `val_acc >= 70%` (sinon score nul) | Tout effondrement global de la précision annule le score |
| **Copies altérées de `forget_images`** | Testent si l'oubli généralise au-delà du pixel exact | Un simple oubli par mémorisation pixel-exact échoue |
| **Modèles `gold`** (ré-entraînés sans `forget`) | Référence pour `mean_std_score` et `prediction_score` (KL softmax à $T = 2$) | Un `gold` ne fait **pas** 100 % d'erreur : il prédit comme sur un jeu de validation non vu |
| **`control set`** | Mesure `retain_accuracy` et `KL(p_orig || p_sub)` à $T = 2$ | Toute dérive inutile des représentations coûte cher sur `retain_score` |

### 1.3 Invariant fondamental du problème

$$\text{final\_score} = \text{forget\_score} \times \text{retain\_score} \quad \text{sous contrainte } \text{val\_acc} \ge 0.70$$

- **Contrainte 1 (pas d'over-unlearning) :** Maximiser l'erreur à 100 % ou forcer une distribution uniforme sur `forget_images` détruit `mean_std_score` et `prediction_score`.
- **Contrainte 2 (pas de dérive globale) :** En l'absence de `retain set`, toute modification agressive non contrainte des poids dégrade `retain_accuracy` et augmente `KL(p_orig || p_sub)` sur le `control set`.
- **Contrainte 3 (source-free) :** Les seuls signaux fiables disponibles localement sont les **3 500 images de `forget_images.npz`** et la **structure interne de `model.pt`**.

---

## 2. Backlog de Recherche & Hypothèses (`Q1` → `Q6`)

| ID | Question de recherche | Pourquoi c'est critique | Statut |
| :--- | :--- | :--- | :--- |
| **Q1** | **Nature du forget set :** Les 3 500 images sont-elles réparties uniformément sur les 100 classes (*random/IID forget*) ou concentrées sur un sous-ensemble (*class/subset forget*) ? | La stratégie pour oublier un sous-ensemble de classes (et préserver les classes absentes) diffère radicalement d'un oubli IID. | `Répondue (Job 15901163)` |
| **Q2** | **Niveau de mémorisation de `model.pt` :** Confiance, entropie et perte CE sur `forget_images` brutes vs perturbées (flip, crop, blur, bruit). | Mesure le degré de mémorisation exacte vs généralisation et sa robustesse aux altérations. | `Répondue (Job 15901163)` |
| **Q3** | **Structure interne de `forget_images` :** Y a-t-il des doublons, quasi-duplicats ou clusters d'instances dans les 3 500 images ? | Vérifie s'il existe des groupes fortement corrélés ou des copies déjà altérées dans `forget_images.npz`. | `Répondue (Job 15901163)` |
| **Q4** | **Anatomie des poids de `model.pt` :** Normes L2 et énergie de gradient par étage du ViT (`patch_embed`, `blocks.0..11`, `head`). | Identifie les étages qui portent la capacité discriminante vs les couches basses à préserver. | `Répondue (Job 15901163)` |
| **Q1b** | **Histogramme complet des 100 classes :** Pourquoi exactement 45 classes actives avec un plafond net à `130` images ? Est-ce ~27 classes à 130 images + quelques classes à petit effectif, ou une structure particulière ? | Détermine si le `gold` n'a **jamais vu** ces classes (class-unlearning) ou s'il a seulement moins d'exemples par classe. | `Répondue (Job 15901182)` |
| **Q5** | **Préservation du `retain` sans `retain set` :** Puisque **55 classes n'ont 0 image** dans `forget_images.npz`, comment combiner le gel/ancrage des prototypes absents avec les contraintes paramétriques (Fisher/EWC, couches basses, SalUn) et/ou un proxy synthétique ? | Protège directement `retain_accuracy` et `KL(p_orig \|\| p_sub)` sans `retain set`. | `Répondue (Phase 2)` |
| **Q6** | **Cible exacte de l'oubli (`gold` estimation) :** Comment estimer les cibles `mean(L), std(L)` et la distribution softmax à $T=2$ d'un modèle `gold` selon que le forget est *class-level* ou *sample-level* ? | Évite l'over-unlearning et maximise `mean_std_score` × `prediction_score`. | `Répondue (Phase 2)` |

---

## 3. Journal d'Investigation (Chronologique)

### 2026-11-21 — Exploration empirique initiale (Job `15901163` sur `jrc0257`)

**Protocole :** Exécution en lecture seule de [explore_unlearning.py](../../scripts/explore_unlearning.py) sur `model.pt` et `forget_images.npz` (sans modification de poids).

#### Q1 — Distribution des classes prédites (`argmax` sur 100 classes)
- **Question :** Les 3 500 images sont-elles réparties sur les 100 classes ou concentrées sur un sous-ensemble ?
- **Pourquoi :** Si les 100 classes étaient uniformes (~35 images/classe), il s'agirait d'un oubli IID. Si certaines classes sont absentes, leurs vecteurs de classification et sous-espaces représentent un ancrage `retain` gratuit et exact.
- **Réponse observée :**
  - **`active_classes` = 45 / 100** (donc **55 classes ont exactement 0 image** dans `forget_images.npz`).
  - `min_count = 0`, `median_count = 0.0`, `max_count = 130`, `std_count = 43.69`, `normalized_class_entropy = 0.812`.
  - Top 10 classes : `[7: 130]`, `[31: 130]`, `[33: 130]`, `[30: 130]`, `[0: 130]`, `[8: 130]`, `[38: 130]`, `[29: 130]`, `[6: 129]`, `[32: 104]`.
  - **Interprétation :** Le forget set est **fortement concentré sur un sous-ensemble de classes** (plafond exact à **130 images/classe** pour au moins 8 classes !). Cela ouvre une piste majeure (**Q1b**) : analyser l'histogramme complet des 100 classes et les normes/similarités de `head.weight` pour savoir si ces ~27 classes sont des *classes entières oubliées* (auquel cas un modèle `gold` a 0 % de rappel sur ces classes) ou un *échantillon partiel*.

#### Q2 — Niveau de mémorisation & sensibilité aux perturbations
- **Question :** Quelle est la confiance, l'entropie et la perte CE de `model.pt` sur `forget_images` brutes vs perturbées (`hflip`, `crop80`, `blur_s1.5`, `noise_15`) ?
- **Pourquoi :** Permet de quantifier l'écart entre le modèle actuel (sur-confiant) et un modèle non vu (`gold`), et de voir si les prédictions tiennent sous altérations (le serveur évalue aussi des *hidden altered copies*).
- **Réponse observée :**
  - **Brut :** `max_prob_mean = 0.9997` (`p10 = 0.9997`, `p50 = 1.0`, `frac_gt_95 = 99.94%`), `entropy_mean = 0.0021`, `margin_mean = 0.9994`, `ce_mean = 0.0003`, `ce_std = 0.0024`.
  - **Perturbations :**
    - `hflip` : `top1_agree = 0.9997`, `max_prob_mean = 0.9996`, `ce_mean = 0.0006`, `kl_t2 = 0.0054`
    - `crop80` : `top1_agree = 1.0000`, `max_prob_mean = 0.9996`, `ce_mean = 0.0004`, `kl_t2 = 0.0252`
    - `noise_15` : `top1_agree = 0.9997`, `max_prob_mean = 0.9985`, `ce_mean = 0.0021`, `kl_t2 = 0.0068`
    - `blur_s1.5` : `top1_agree = 0.9954`, `max_prob_mean = 0.9949`, `ce_mean = 0.0137`, `kl_t2 = 0.0351`
  - **Interprétation :** `model.pt` est **extrêmement sur-confiant / mémorisé** (`CE ≈ 0.0003`, 99.94 % à > 0.95 de probabilité) et cette certitude résiste quasi intégralement aux perturbations géométriques et photométriques. Conséquence : les gradients de la CE standard au point initial sont quasi nuls (saturation du softmax).

#### Q3 — Structure interne de `forget_images.npz`
- **Question :** Existe-t-il des doublons, quasi-duplicats ou clusters visuels parmi les 3 500 images ?
- **Pourquoi :** Détecter si le dataset contient déjà des copies altérées ou des sous-groupes corrélés.
- **Réponse observée :**
  - Shape `[3500, 224, 224, 3]`, RGB mean `[0.4067, 0.3700, 0.3129]`, RGB std `[0.2671, 0.2553, 0.2470]`.
  - Similarité cosinus sur embeddings CLS normalisés : `nn_cos_max = 0.8962`, `nn_cos_p95 = 0.7721`, `nn_cos_median = 0.6191`.
  - Paires avec `cos > 0.95` : `0`, `cos > 0.90` : `0`, `cos > 0.85` : `5`.
  - **Interprétation :** Aucune image dupliquée ni quasi-duplicat exact ; ce sont 3 500 images naturelles distinctes.

#### Q4 — Anatomie des poids & profil de gradient de `model.pt`
- **Question :** Comment sont répartis les poids et l'énergie des gradients `forget` entre les étages du ViT ?
- **Pourquoi :** Savoir quels blocs portent la plus forte sensibilité au `forget set` et où cibler/ancrer les mises à jour.
- **Réponse observée :**
  - `head` (`76 900` params) : `weight_l2 = 11.766`, **`grad_rms = 9.3e-5`** (le plus élevé de tout le modèle, ~6–9× supérieur aux blocs Transformer).
  - `patch_embed` (`590 592` params) : `weight_l2 = 18.241`, `grad_rms = 4.6e-5`.
  - `pos_cls` (`152 064` params) : `weight_l2 = 43.383`, `grad_rms = 4.5e-5`.
  - `blocks_0-3` (`28.36M` params) : `weight_l2 = 161.716`, `grad_rms = 1.5e-5`.
  - `blocks_4-7` (`28.36M` params) : `weight_l2 = 168.023`, `grad_rms = 1.0e-5`.
  - `blocks_8-11` (`28.36M` params) : `weight_l2 = 182.053`, `grad_rms = 1.1e-5`.
  - **Interprétation :** L'amplitude absolue des gradients est faible à cause de la saturation (`CE = 0.0003`), mais **l'énergie RMS par paramètre est maximale dans `head`**.

#### Q1b — Histogramme complet des 45 classes actives & géométrie de `head.weight` (Job `15901182` sur `jrc0258`)
- **Question :** Quelle est la distribution exacte des effectifs sur les 45 classes actives, y a-t-il des métadonnées cachées dans `forget_images.npz`, et `head.weight` distingue-t-il les 45 classes actives des 55 classes absentes ?
- **Pourquoi :** Savoir s'il existe une coupure nette entre classes *forget* et classes *retain*, ou un bruit résiduel d'images isolées.
- **Réponse observée :**
  - `forget_images.npz` ne contient qu'une seule clé : `images` (`[3500, 224, 224, 3]`, `uint8`).
  - **Coupure nette :** toutes les 45 classes actives ont **entre 46 et 130 images** (aucune classe avec 1–45 images ; 8 classes exactement à 130, puis 129, 104, 98, 92, ..., jusqu'à 46), et **55 classes ont exactement 0 image**.
  - Géométrie de `head` :
    - Normes L2 moyennes : `active_mean = 1.1441` vs `inactive_mean = 1.1995`.
    - Biais moyens : `active_mean = +0.0095` vs `inactive_mean = -0.0197`.
    - Similarité cosinus max hors diagonale entre prototypes : `0.3016` (actives), `0.2915` (active vs inactive), `0.2205` (inactives).
  - **Interprétation :** Les 45 classes actives sont des classes ciblées à effectif significatif (46–130 images/classe), tandis que les 55 classes à 0 image sont quasi orthogonales (`cos <= 0.29`) et constituent un ensemble `retain` pur.

---

### 2026-11-21 — Synthèse Phase 2 : Réponses à Q5, Q6 et Défis Techniques Priorisés

#### Q5 — Préservation du `retain` sans `retain set`
- **Question :** Comment préserver `retain_accuracy` et `KL(p_orig || p_sub)` à $T=2$ sur le `control set` caché sans disposer d'aucune image `retain` ?
- **Pourquoi :** Sans ancrage `retain`, tout gradient d'oubli décale les représentations partagées du ViT et fait chuter la précision globale sous le *utility gate* (`70 %`).
- **Réponse (stratégie retenue pour validation) :** Combiner 3 verrous complémentaires :
  1. **Ancrage exact des 55 classes absentes :** geler (ou pénaliser fortement en L2) les lignes `head.weight[inactive]` et `head.bias[inactive]` des 55 classes à 0 image.
  2. **Ancrage paramétrique sélectif (couches basses + SalUn/L2) :** les couches basses du ViT (`patch_embed`, `blocks.0..5`) encodent des primitives visuelles génériques ; limiter les mises à jour aux blocs hauts (`blocks.6..11` / `head`) ou masquer par saillance gradient réduit l'oubli catastrophique.
  3. **Proxy `retain` synthétique (KL à $T=2$) :** utiliser des images hors-distribution préservant les statistiques locales (ex. *PatchMix* ou bruit structuré / crops mélangés) pour contraindre `KL(p_orig || p_sub)` à $T=2$ sur l'ensemble des 100 classes.

#### Q6 — Cible exacte de l'oubli (`gold` estimation)
- **Question :** Quelle distribution de sortie et quel niveau de perte CE viser sur `forget_images` pour coller à un modèle `gold` ré-entraîné sans `forget` ?
- **Pourquoi :** `Unlearning.pdf` pénalise l'over-unlearning autant que le sous-unlearning (`mean_std_score` + `prediction_score` à $T=2$ vs `gold`).
- **Réponse (analyse) :**
  - Si un modèle `gold` a encore vu d'autres images des 45 classes (oubli partiel par classe, ex. 10–20 % retiré), sa CE sur `forget_images` est proche de sa **CE de validation** (typiquement `0.4 – 1.2`, confiance modérée avec généralisation sur la bonne classe ou des classes proches).
  - Si au contraire certaines classes ont été retirées en totalité (class-unlearning), le `gold` prédit des classes sémantiquement proches (`CE` plus élevée).
  - **Conséquence :** Il faut utiliser une **loss bornée** (type **NPO**, ou **KL vers une distribution lissée / température relevée**) et **calibrer par paliers** (ex. balayer plusieurs niveaux cibles de perte/entropie sur une grille contrôlée plutôt que de maximiser la CE).

---

## 4. Défis techniques soulevés par la spec à prioriser

Classement par ordre d'impact sur `final_score = forget_score × retain_score` (et risque d'échec au *utility gate* `val_acc >= 70%`) :

1. **[P0 — Critique] Préserver le `retain` et passer le *utility gate* (`val_acc >= 70%`) sans `retain set` :**
   - *Risque :* `val_acc < 70%` annule immédiatement le score (`0.0`), et `retain_score` est un facteur multiplicatif direct.
   - *Priorité :* Exploiter la coupure nette `55 classes absentes / 45 actives`, restreindre les poids modifiables (couches hautes / saillance), et ajouter un double ancrage (L2 vers $\theta_{\text{orig}}$ + KL $T=2$ sur proxy).
2. **[P0 — Critique] Éviter l'over-unlearning et calibrer la cible `gold` (`mean_std_score` + `prediction_score`) :**
   - *Risque :* Pousser `forget_ce` trop haut détruit à la fois la similarité de distribution softmax à $T=2$ et l'écart `(mean(L), std(L))` face aux modèles `gold`.
   - *Priorité :* Employer une loss bornée (NPO / KL contrôlée) et tester un balayage calibré d'intensité d'oubli.
3. **[P1 — Élevé] Transférer l'oubli aux *hidden altered copies* :**
   - *Risque :* D'après **Q2**, `model.pt` garde `99.5%–100%` d'accord même sous `crop80`, `hflip`, `blur` et `noise`. Un oubli uniquement sur images brutes risque de ne pas tenir sur les copies altérées du serveur.
   - *Priorité :* Appliquer systématiquement des augmentations fortes (random resized crop, flip, color jitter, blur, bruit) pendant l'unlearning et dans la validation locale.
4. **[P1 — Élevé] Contourner la saturation initiale des gradients (`CE = 0.0003`) :**
   - *Risque :* Avec `max_prob = 0.9997`, la cross-entropy brute a des gradients quasi nuls (`grad_rms ~ 1e-5`), ce qui biaise les masques de saillance calculés à $t=0$ et ralentit le démarrage.
   - *Priorité :* Calculer les gradients/masques avec température $T \ge 2$ ou sur logits bruts / loss NPO, et vérifier la dynamique des gradients dès les premiers pas.
5. **[P2 — Méthodologique] Mesurer sans `val set` ni `gold` en local :**
   - *Risque :* Itérer à l'aveugle entre soumissions serveur.
   - *Priorité :* Construire d'abord un harnais de métriques locales (Phase 3) qui suit la dérive sur les 55 classes inactives, la robustesse sur un panel d'altérations (`crop`, `blur`, `jitter`, `JPEG`) et la norme de déplacement $\|\theta - \theta_{\text{orig}}\|$.

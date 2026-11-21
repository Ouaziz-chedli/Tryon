# Track 3 — Unlearning : Plan de Recherche & TODO List

**Dossier JURECA :** `/p/scratch/training2666/ouaziz1/TRYON/unlearning/`  
**Spec complète :** [03-unlearning.md](03-unlearning.md)  
**Carnet d'investigation & backlog de recherche :** [03-unlearning-investigation.md](03-unlearning-investigation.md)

> **Principe directeur :** Aucun algorithme d'unlearning n'est validé ni déployé avant d'avoir terminé l'exploration empirique de `model.pt` / `forget_images.npz` (Phase 1) et la recherche sur les méthodes adaptées sans `retain set` (Phase 2). Le code prototype présent dans `tracks/03_unlearning/` est mis de côté tant que les hypothèses ne sont pas vérifiées par les données.

---

## Où on en est
- [x] Cluster JURECA accessible (`ssh juelich` avec `ControlMaster`), aucun job Slurm actif.
- [x] Dataset & poids présents sur JURECA (`model.pt` 343 Mo, `forget_images.npz` 526 Mo, `.venv/` prêt).
- [x] **Phase 1 (Exploration empirique Q1–Q4 + Q1b)** terminée (Jobs `15901163`, `15901182`) — résultats consignés dans [03-unlearning-investigation.md](03-unlearning-investigation.md).
- [x] **Phase 2 (Réponses Q5, Q6 & Priorisation des défis techniques P0–P2)** consignée.
- [ ] **Phase 3 (Protocole de validation local)** : prochaine étape avant d'implémenter.
- [ ] `submission.py` : `API_KEY` d'équipe à renseigner plus tard (Phase 4).

---

## Phase 1 — Exploration empirique pure (lecture seule sur `model.pt` & `forget_images.npz`)
Objectif : comprendre **ce que sont** les 3 500 images et **comment `model.pt` se comporte dessus** avant de formuler la moindre hypothèse d'unlearning.

- [x] **1.1 Inspecter `forget_images.npz` :**
  - Shape `[3500, 224, 224, 3]`, pas de doublons (`pairs > 0.90 = 0`, `nn_cos_max = 0.8962`).
- [x] **1.2 Analyser les prédictions de `model.pt` sur `forget_images.npz` :**
  - **Découverte clé :** **45 classes actives sur 100** (55 classes à 0 image, plafond exact à 130 images/classe).
  - Confiance extrême (`max_prob_mean = 0.9997`, `ce_mean = 0.0003`), robuste aux perturbations (`crop80` agree `1.0000`).
- [x] **1.3 Inspecter l'architecture & les poids de `model.pt` :**
  - Normes par étage mesurées ; `head` a le `grad_rms` le plus élevé (`9.3e-5`).
- [x] **1.4 (Q1b) Histogramme complet des 45 classes actives** et structure des vecteurs `head.weight` (coupure nette 0 vs 46–130 images/classe, 1 seule clé `images` dans `.npz`, prototypes quasi orthogonaux `cos <= 0.30`).

---

## Phase 2 — Recherche théorique & SOTA (Retain-Free / Label-Free Machine Unlearning)
Objectif : choisir l'approche mathématique la plus fidèle à la métrique `final_score = forget_score * retain_score` sans accès au `retain set` ni aux labels.

- [ ] **2.1 Modéliser le comportement d'un modèle `gold` (ré-entraîné sans `forget`) :**
  - Déterminer la cible réaliste de `(mean(L), std(L))` et de température/entropie (éviter l'over-unlearning qui détruit `mean_std_score`, `prediction_score` et passe sous la barre des 70 % de validation accuracy).
- [ ] **2.2 Évaluer les mécanismes de préservation du `retain` sans `retain set` :**
  - Comparer :
    1. **Proxies synthétiques** (PatchMix, bruit structuré, inversions de features, ou images publiques compatibles).
    2. **Contraintes structurelles/paramétriques** : Fisher Information / EWC, gel des couches basses (`blocks.0..k`), SalUn (`arxiv:2310.12508`), SVD / projection orthogonale sur la tête `head`.
- [ ] **2.3 Évaluer les objectifs d'oubli bornés :**
  - Comparer **NPO** (`arxiv:2404.05868`), **SCRUB / Bad-Teacher** (`arxiv:2302.09880`), **KL vers distribution lissée / uniforme partielle**, et **perturbation ciblée sur copies altérées** (invariance aux augmentations).
- [ ] **2.4 Rédiger une note de synthèse** avec les hypothèses retenues, les invariants et les hyperparamètres candidats.

---

## Phase 3 — Protocole de validation local
- [ ] Construire un benchmark local reproductible simulant les **hidden altered copies** (crop, rotation, color jitter, blur, compression JPEG) + un **proxy retain** validé par la Phase 1/2.
- [ ] Vérifier qu'on peut mesurer la dérive sur les classes non-forget (si forget est concentré) ou sur des représentations invariantes.

---

## Phase 4 — Implémentation, Tests & Soumissions itératives
- [ ] Implémenter la méthode retenue proprement (contrats stricts, tests unitaires CPU/GPU).
- [ ] Renseigner `API_KEY` dans `submission.py`, vérifier le format (`float32`, `strict=True`, sans `NaN`/`Inf`, $\le 450$ Mo).
- [ ] Soumettre une première baseline et itérer en fonction du score public (`forget_score` vs `retain_score`).

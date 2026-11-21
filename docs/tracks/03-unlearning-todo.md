# Track 3 — Unlearning : État & TODO List

**Dossier JURECA :** `/p/scratch/training2666/ouaziz1/TRYON/unlearning/`
**Spec complète :** [03-unlearning.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/03-unlearning.md)

---

## Où on en est
- [x] Cluster JURECA accessible (`ssh juelich` avec `ControlMaster`).
- [x] Dataset & poids téléchargés (`model.pt` 343 Mo, `forget_images.npz` 526 Mo).
- [x] `.venv/` prêt (`torch 2.15.0`, `torchvision 0.29.2`, `timm 1.0.31`, CUDA 13).
- [ ] `task_template.py` est encore au stade stub (`unlearn()` renvoie le modèle inchangé).
- [ ] `submission.py` n'a pas encore votre `API_KEY` d'équipe.
- [ ] Aucun job Slurm (`sbatch`) lancé ni aucune soumission générée dans `output/`.

---

## TODO List (par ordre d'exécution)

### Phase 1 — Setup & Exploration rapide (15 min)
- [ ] **Renseigner `API_KEY`** dans `submission.py` sur JURECA (et tester un `sbatch main.sh` minimal sur `dc-gpu` qui vérifie `nvidia-smi` + `task_template.py` intact).
- [ ] **Analyser `model.pt` sur `forget_images.npz`** (sur 1 A100, < 30s) :
  - Distribution des classes prédites (`argmax` sur les 3 500 images : est-ce uniformément réparti sur 100 classes ou concentré sur quelques classes ?).
  - Confiance moyenne, entropie, et distribution des logits / softmax à $T = 2$.
  - Vérifier l'architecture exacte (`blocks.0..11`, `head`, `LayerScale`).

### Phase 2 — Évaluation locale (indispensable avant de soumettre)
> Le serveur ne donne pas le `retain set` ni les copies altérées de `forget_images.npz`.
- [ ] **Construire un jeu proxy de validation locale :**
  - **Forget altéré :** appliquer des augmentations (crop, flip, color jitter, blur, JPEG) sur `forget_images.npz`.
  - **Retain proxy :** soit images synthétiques / bruit structuré / dataset public (ex. sous-ensemble CIFAR-100 / ImageNet-100 / Imagenette si accessible ou images générées par inversion) pour mesurer `KL(p_orig || p_sub)` à $T = 2$ et l'accord top-1 avec `orig`.
- [ ] Implémenter une fonction `eval_local(orig_model, sub_model, forget_images)` qui trace : entropie sur `forget`, `KL(p_orig || p_sub)` sur `forget`, et dérive sur `retain_proxy`.

### Phase 3 — Baselines & Méthodes d'Unlearning (sans retain set)
- [ ] **Baseline 1 — Fine-tuning / Perturbation ciblée sur la tête ou les derniers blocs :**
  - Tester **SalUn** (*Gradient-based Weight Saliency*, `arxiv:2310.12508`) : calculer la saillance $|\nabla_\theta \mathcal{L}_{\text{forget}}|$ sur `forget_images.npz` pour ne modifier que les poids les plus liés au forget set (préserve naturellement le retain !).
- [ ] **Baseline 2 — NPO (*Negative Preference Optimization*, `arxiv:2404.05868`) :**
  - Remplacer le gradient ascent naïf par la loss NPO bornée par rapport à `model_orig`, couplée à une régularisation L2 sur les poids ($\|\theta - \theta_{\text{orig}}\|_2^2$) ou distillation KL sur un proxy retain.
- [ ] **Baseline 3 — SCRUB / Bad-Teacher adapté sans retain set :**
  - Utiliser `model_orig` comme teacher sur un proxy de retain + un modèle aléatoire / uniforme / bruité (ou *Bad Teacher* avec poids figés sur les couches basses) sur `forget_images.npz` + augmentations (pour coller aux *hidden altered copies* du scoring !).
- [ ] **Point critique du scoring :**
  - Ne **pas** maximiser l'erreur à 100 % : un modèle `gold` prédit souvent la bonne classe par généralisation ou des classes sémantiquement proches avec une confiance modérée — viser une distribution réaliste (éviter `utility gate < 70%`).

### Phase 4 — Soumission & Itération
- [ ] Vérifier le format avant chaque soumission (`float32`, `strict=True`, pas de `NaN`/`Inf`, $\le 450$ Mo).
- [ ] Soumettre une première baseline fonctionnelle via `python submission.py` pour obtenir un score public de référence.
- [ ] Itérer sur les hyperparamètres ($\alpha$, température, masque de saillance SalUn, poids de régularisation) en surveillant `retain_score` vs `forget_score`.

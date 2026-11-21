# Track 3 — Machine Unlearning (ViT-B/16)

- **Source :** [Unlearning.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/Unlearning.pdf)
- **Superviseurs :** Adam Dziedzic, Franziska Boenisch (CISPA)
- **Starter kit :** `https://huggingface.co/datasets/SprintML/unlearning`
- **Sur JURECA (`TRYON`) :** `/p/scratch/training2666/ouaziz1/TRYON/unlearning/` (`.venv/` prêt avec `torch 2.15.0` & `timm 1.0.31`, `model.pt`, `forget_images.npz`, `task_template.py`, `submission.py`).

## 1. Objectif
On dispose d'un classifieur **ViT-B/16** (`timm vit_base_patch16_224`, 100 classes, `LayerScale init 1e-4`) pré-entraîné (`model.pt`) et de **3 500 images à oublier** (`forget_images.npz`, shape `[3500, 224, 224, 3]`, **sans labels**).
**Contrainte majeure :** Aucune donnée du *retain set* n'est fournie ! Il faut faire oublier les 3 500 images à partir de `model.pt` et `forget_images.npz` uniquement, de sorte que le modèle se comporte comme un modèle `gold` ré-entraîné de zéro sans ces 3 500 images.

## 2. Métrique d'évaluation
1. **Utility gate :** Le modèle soumis doit atteindre **$\ge 70\%$ d'accuracy** sur un jeu de validation caché (sinon `forget_score = retain_score = 0`).
2. **Forget quality (`forget_score`)** sur les images exactes et sur des **copies altérées cachées** (`gold` = moyenne de plusieurs modèles ré-entraînés de zéro sans le *forget set*) :
   - $d_{\text{sub}} = |\text{mean}(L_{\text{sub}}) - \text{mean}(L_{\text{gold}})| + |\text{std}(L_{\text{sub}}) - \text{std}(L_{\text{gold}})|$, $d_{\text{orig}} = |\text{mean}(L_{\text{orig}}) - \text{mean}(L_{\text{gold}})| + |\text{std}(L_{\text{orig}}) - \text{std}(L_{\text{gold}})|$
   - $\text{mean\_std\_score} = 1 - \text{clip}(d_{\text{sub}} / d_{\text{orig}}, 0, 1)$
   - $\text{prediction\_score} = 1 - \text{clip}\!\left(\frac{\text{mean KL}(p_{\text{gold}}(x) \parallel p_{\text{sub}}(x))}{\text{mean KL}(p_{\text{gold}}(x) \parallel p_{\text{orig}}(x))}, 0, 1\right)$ à température $T = 2$.
   - Chaque score est calculé séparément sur les images exactes et les copies altérées, **moyenné à 50/50** entre les deux groupes, puis multiplié : $\text{forget\_score} = \text{mean\_std\_score} \times \text{prediction\_score}$ (`0` si `sub = orig`, `1` si `sub = gold`).
3. **Retain quality (`retain_score`)** sur un control set caché :
   - $\text{retain\_accuracy\_score} = \text{clip}(\text{acc}_{\text{sub}} / \text{acc}_{\text{orig}}, 0, 1)$
   - $\text{retain\_kl\_similarity} = 1 / (1 + \text{mean KL}(p_{\text{orig}}(x) \parallel p_{\text{sub}}(x)))$
   - $\text{retain\_score} = \text{retain\_accuracy\_score} \times \text{retain\_kl\_similarity}$
4. **Score final :** $\text{final\_score} = \text{forget\_score} \times \text{retain\_score}$ (leaderboard public sur ~30 % déterministe par hash — chaque copie altérée suit son image source — et classement final sur les 70 % restants).

## 3. Format de soumission
- Un `state_dict` PyTorch (`.pt` / `.pth`, `strict=True`, `float32`, sans NaN/Inf, $\le 450$ MB ; ViT-B ≈ 345 MB).
- Si des couches LoRA/PEFT sont utilisées, elles doivent être **mergées** dans les poids d'origine avant soumission.

## 4. Pistes techniques initiales (sans retain set !)
- **Pourquoi SCRUB brut ne s'applique pas directement :** SCRUB exige un *retain set* $\mathcal{D}_r$ pour ses *min-steps*. Ici, $\mathcal{D}_r$ est absent !
- **Comment contourner l'absence de $\mathcal{D}_r$ :**
  1. **Proxy de $\mathcal{D}_r$ :** Utiliser un dataset public proche (ex. ImageNet-1k / CIFAR-100 / images générées ou *data-free knowledge distillation* / *model inversion* à partir de `model.pt`) pour préserver les prédictions de `model.pt` via KL distillation ($T = 2$).
  2. **Saliency-based unlearning (`SalUn`, ICLR 2024, cité en référence) :** Identifier les poids les plus sensibles au *forget set* via le gradient, et ne mettre à jour / réinitialiser / perturber que ces poids saillants, en préservant le reste du ViT-B intact !
  3. **Negative Preference Optimization (`NPO`, COLM 2024, cité en référence) :** Évite l'effondrement catastrophique du *gradient ascent* naïf en bornant la divergence par rapport au modèle de référence (`orig`).
  4. **Zero-Shot Machine Unlearning (Chundawat et al., 2023, cité en référence) :** Gated/error-maximizing noise ou *Bad Teacher* sans accès aux données originales.
  5. **Calibrer le niveau d'oubli (éviter le sur-oubli) :** L'objectif n'est pas 100 % d'erreur (ce qui ruine `mean_std_score` et `prediction_score`), mais le comportement d'un modèle qui *n'a jamais vu* $\mathcal{D}_f$ (souvent entropie modérée / prédictions cohérentes avec les classes proches, y compris sur des copies augmentées de `forget_images.npz`).

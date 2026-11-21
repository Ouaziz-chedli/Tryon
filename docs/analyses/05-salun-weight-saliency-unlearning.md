# Analyse de papier — SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency (ICLR 2024)

- **Titre :** *SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation*
- **Auteurs :** Chongyu Fan, Jiancheng Liu, Yihua Zhang, Eric Wong, Dennis Wei, Sijia Liu (Michigan State University, UPenn, MIT-IBM Watson AI Lab)
- **Conférence :** ICLR 2024 (`arXiv:2310.12508v5`)
- **PDF local :** [2310.12508v5.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/papers/2310.12508v5.pdf)
- **Pertinence directe :** **Track 3 — Machine Unlearning** ([03-unlearning.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/03-unlearning.md), [Unlearning.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/Unlearning.pdf))

---

## 1. Problème adressé & Idée centrale

Les méthodes d'unlearning par optimisation post-hoc (Gradient Ascent `GA`, Random Labeling `RL`, Fine-Tuning `FT`) souffrent d'un compromis instable entre :
- **Sous-oubli** (lorsque le taux d'apprentissage ou le nombre de pas est faible),
- **Effondrement catastrophique de l'utilité (`retain` / `val`) et sur-exposition aux attaques `MIA`** (lorsque tous les poids $\boldsymbol{\theta}$ du réseau sont modifiés pour oublier $\mathcal{D}_{\text{f}}$).

**Idée clé de SalUn :** Tout comme la saillance d'entrée identifie les pixels pertinents, la **saillance des poids par gradient** identifie le sous-ensemble de paramètres $\mathbf{m}_{\text{S}} \in \{0, 1\}^{|\boldsymbol{\theta}|}$ les plus responsables de la mémorisation de $\mathcal{D}_{\text{f}}$. Pendant l'unlearning, **seuls les poids saillants ($\mathbf{m}_{\text{S}} = 1$) sont mis à jour**, tandis que le reste du réseau ($1 - \mathbf{m}_{\text{S}}$) demeure **strictement figé à $\boldsymbol{\theta}_{\text{o}}$**.

---

## 2. Formulation mathématique exacte

### 2.1 Carte de saillance binaire $\mathbf{m}_{\text{S}}$ (Eq. 3 & 5, Algorithme 1)
À partir du modèle original $\boldsymbol{\theta}_{\text{o}}$ (calculé **une seule fois** avant l'unlearning) :
$$\mathbf{g}_{\text{S}} = \nabla_{\boldsymbol{\theta}} \mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{f}}} \big[\ell_{\text{CE}}(\boldsymbol{\theta}; x, y)\big]\Big|_{\boldsymbol{\theta} = \boldsymbol{\theta}_{\text{o}}}, \qquad \mathbf{m}_{\text{S}} = \mathbb{1}\!\left(|\mathbf{g}_{\text{S}}| \ge \gamma\right)$$
où $\gamma > 0$ est le quantile correspondant à la proportion de poids modifiables (*saliency sparsity* $k\% = \|\mathbf{m}_{\text{S}}\|_0 / |\boldsymbol{\theta}|$, typiquement $10\%$ à $50\%$).
- **Hard vs Soft thresholding (App. B, Table 1) :** Le masquage binaire dur (`SalUn`) surpasse systématiquement le seuillage doux par opérateur proximal $\ell_1$ (`SalUn-soft`) sur l'écart au modèle `Retrain` (`MIA gap = 1.51%` vs `6.61%`).

### 2.2 Mise à jour sous masque & Random Labeling (Eq. 4 & 6)
Le modèle désappris est contraint par :
$$\boldsymbol{\theta}_{\text{u}} = \mathbf{m}_{\text{S}} \odot (\Delta\boldsymbol{\theta} + \boldsymbol{\theta}_{\text{o}}) + (1 - \mathbf{m}_{\text{S}}) \odot \boldsymbol{\theta}_{\text{o}}$$
Dans le papier, SalUn est couplé au **Random Labeling (RL)** : chaque exemple $(x_i, y_i) \in \mathcal{D}_{\text{f}}$ reçoit un label aléatoire $y'_i \ne y_i$ :
$$\min_{\Delta\boldsymbol{\theta}} \; \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{f}}, \, y' \ne y}\big[\ell_{\text{CE}}(\boldsymbol{\theta}_{\text{u}}; x, y')\big] + \alpha \, \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{r}}}\big[\ell_{\text{CE}}(\boldsymbol{\theta}_{\text{u}}; x, y)\big]$$
- **Pourquoi $\mathbf{m}_{\text{S}}$ stabilise RL (Tables A1, A8, A10) :** Sans $\mathbf{m}_{\text{S}}$, forcer un faux label $y'$ tord l'extracteur de features pour mémoriser l'association absurde $(x_i, y'_i)$ (`UA = 55.03%` vs `Retrain = 26.47%` sur CIFAR-100). En figeant $1 - \mathbf{m}_{\text{S}}$, la capacité d'adaptation est bridée : le gradient sur $y'$ efface le pic de sur-mémorisation sur $y_i$ sans pouvoir mémoriser $y'$, ramenant `UA = 27.53%` (à `1.06%` de `Retrain` !) et alignant le `MIA` sur `Retrain` (gap de `0.02%` sur **Swin-T**, Table A10).

---

## 3. Enseignements critiques & Correctifs pour Track 3 (`31-unlearning`)

L'audit de notre code ([salun_npo.py](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/tracks/03_unlearning_v2/methods/salun_npo.py) et [common.py](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/tracks/03_unlearning_v2/methods/common.py)) à la lumière du papier SalUn révèle **4 leviers majeurs** :

1. **Fuite du Weight Decay découplé d'`AdamW` sur les poids figés ($m_{\text{S}} = 0$) et les 55 classes absentes :**
   - Dans l'Eq. 4 de SalUn, les poids où $m_{\text{S}} = 0$ doivent rester **strictement égaux à $\boldsymbol{\theta}_{\text{o}}$**.
   - Or `torch.optim.AdamW` applique par défaut `weight_decay = 0.01` ($\theta \leftarrow (1 - \eta \lambda_{\text{wd}})\theta$) à **tous** les paramètres `requires_grad=True`, **même lorsque `p.grad == 0`** ! Cela fait dériver vers `0` les $75\%$ de poids masqués par SalUn ainsi que les 55 lignes de `head.weight` des classes absentes.
   - **Correctif :** Fixer `weight_decay=0.0` dans `AdamW` (l'ancrage vers $\boldsymbol{\theta}_{\text{o}}$ étant déjà assuré par `l2_anchor_loss`).
2. **Seuillage Layer-Wise exact vs Quantile global sous-échantillonné :**
   - Notre sous-échantillonnage `t.numel() // 4096` donnait autant de poids dans le quantile à un vecteur de 768 paramètres (`LayerNorm` / `LayerScale .gamma`) qu'à une matrice MLP de $2.36\text{M}$ paramètres.
   - Sur un ViT-B/16 avec `LayerScale init 1e-4`, calculer le seuil `torch.quantile(v.reshape(-1), 1.0 - keep_ratio)` **par tenseur (layer-wise)** élimine tout biais d'échelle entre `gamma`, `norm`, `qkv`, `mlp` et `head`.
3. **Saillance différentielle Forget-vs-Proxy sur la totalité des 3 500 images :**
   - En l'absence de $\mathcal{D}_{\text{r}}$, pondérer la saillance par la sensibilité sur le proxy $S_i = |\nabla_{\theta_i} \ell_{\text{f}}(\mathcal{D}_{\text{f}})| / (|\nabla_{\theta_i} \ell_{\text{KL}}(\mathcal{D}_{\text{proxy}})| + \epsilon)$ sur les 3 500 images (brutes + altérées) isole les poids spécifiques au *forget set*.
4. **Remplacer le Random Labeling uniforme par notre cible `Gold-Mimic` ($T=2$) sous masque SalUn :**
   - Le Random Labeling uniforme injecterait de la probabilité sur les 55 classes absentes et détruirait `prediction_score` ($1 - \text{clip}(\text{KL}_{\text{sub}}/\text{KL}_{\text{orig}}, 0, 1)$ à $T=2$, qui vérifie *quelle classe erronée* le modèle choisit). Coupler $\mathbf{m}_{\text{S}}$ avec une cible chirurgicale abaissant uniquement le logit top-1 tout en conservant la géométrie des classes secondaires à $T=2$ garde la stabilité à point fixe de SalUn+RL tout en respectant la métrique CISPA.

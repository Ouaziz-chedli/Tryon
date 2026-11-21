# Analyse de papier — Negative Preference Optimization / NPO (COLM 2024)

- **Titre :** *Negative Preference Optimization: From Catastrophic Collapse to Effective Unlearning*
- **Auteurs :** Ruiqi Zhang, Licong Lin, Yu Bai, Song Mei (UC Berkeley, Salesforce Research)
- **Conférence :** COLM 2024 (`arXiv:2404.05868v2`)
- **PDF local :** [2404.05868v2.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/papers/2404.05868v2.pdf)
- **Pertinence directe :** **Track 3 — Machine Unlearning** ([03-unlearning.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/03-unlearning.md), [Unlearning.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/Unlearning.pdf))

---

## 1. Dérivation Mathématique Exacte & Théorème de Vitesse de Divergence

### 1.1 De DPO à NPO (Éq. 2–3, Proposition 1)
En ne conservant que la branche négative ($y_l = y \in \mathcal{D}_{\text{FG}}$) de Direct Preference Optimization (DPO) par rapport au modèle de référence $\pi_{\text{ref}} = \text{orig}$ :
$$\mathcal{L}_{\text{NPO},\beta}(\theta) = -\frac{2}{\beta} \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{FG}}}\!\left[\log \sigma\!\left(-\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right)\right] = \frac{2}{\beta} \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{FG}}}\!\left[\text{softplus}\!\left(\beta \left(\log \pi_\theta(y|x) - \log \pi_{\text{ref}}(y|x)\right)\right)\right]$$
- Contrairement au Gradient Ascent ($\mathcal{L}_{\text{GA}} \in (-\infty, 0]$), $\mathcal{L}_{\text{NPO},\beta}(\theta) \ge 0$ est bornée inférieurement.
- Quand $\beta \to 0$, $\nabla_\theta \mathcal{L}_{\text{NPO},\beta}(\theta) \to \nabla_\theta \mathcal{L}_{\text{GA}}(\theta)$ (Proposition 1).

### 1.2 Poids adaptatif par échantillon $W_\theta(x, y)$ (Éq. 4–5)
Le gradient de NPO est un Gradient Ascent pondéré par échantillon :
$$\nabla_\theta \mathcal{L}_{\text{NPO},\beta}(\theta) = \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{FG}}}\!\left[W_\theta(x, y) \nabla_\theta \log \pi_\theta(y|x)\right]$$
avec :
$$W_\theta(x, y) = \frac{2 \pi_\theta(y|x)^\beta}{\pi_\theta(y|x)^\beta + \pi_{\text{ref}}(y|x)^\beta} = 2 \sigma\!\left(-\beta \left(L_\theta(x, y) - L_{\text{ref}}(x, y)\right)\right) = \frac{2}{1 + \exp(\beta \Delta L(x, y))}$$
- À l'initialisation ($\Delta L = 0$), $W_{\theta_{\text{ref}}}(x, y) = 1$.
- Dès que la perte CE d'un exemple augmente de $\Delta L = L_\theta(x,y) - L_{\text{ref}}(x,y) > 0$, son gradient est freiné exponentiellement en $e^{-\beta \Delta L}$.
- **Théorème 2 (Divergence logarithmique vs linéaire) :** Alors que GA diverge linéairement en temps ($\|\theta_{\text{GA}}^{(t)} - \theta_{\text{init}}\| = \Theta(\eta t)$), NPO diverge **logarithmiquement** ($\Delta L_i^{(t)} \approx \frac{1}{\beta} \log(\beta C \eta_0 t + 1)$).

---

## 2. Découverte Critique sur $\beta$ : Pourquoi $\beta = 0.1$ créait la Bimodalité sur notre ViT-B/16

L'analyse croisée de la Section 4 (Table 1 p. 24, classification) et de la Section 5 (TOFU, LLMs) révèle pourquoi nos premiers runs NPO (`#1`, `#2`, `#4`) étaient bimodaux :

1. **Pourquoi $\beta = 0.1$ fonctionne sur un LLM mais PAS sur un classifieur mono-label :**
   - Dans un LLM génératif (TOFU), $\log \pi_\theta(y|x) = \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t})$ est la **somme sur 20 à 50 tokens** : une variation de perte par séquence $\Delta L_{\text{seq}}$ vaut $10$ à $30$, donc $\beta \Delta L_{\text{seq}} \sim 1.5 - 3.0$ avec $\beta = 0.1$.
   - Dans un **classifieur d'images mono-label** ($1$ seul label $y$), pour passer de la saturation $p_{T=1} = 0.9997$ ($\text{CE} = 0.0003$) à la cible `gold` $p_{T=1}^* \approx 0.60$ ($\text{CE}^* \approx 0.51$), la variation de log-probabilité à $T=2$ vaut seulement $\Delta \text{lp}_{T=2} \approx -0.65$ !
   - Avec `npo_beta = 0.1` (notre valeur par défaut dans `contracts.py`), à $\text{CE} = 0.51$, le poids adaptatif NPO vaut :
     $$W_\theta = 2 \sigma(-0.1 \times 0.65) = 0.935 \approx 1.0 \quad (!)$$
   - **Conséquence :** Avec $\beta = 0.1$, le frein NPO ne s'active qu'autour de $\Delta L \approx 1/\beta = 10.0$ ! Dans toute la zone utile $\text{CE} \in [0, 1.5]$, **le frein NPO est inactif ($W_\theta \approx 1$)** : les $15\%$ d'images les moins saturées subissent l'auto-accélération du gradient softmax $(1 - p)$ et explosent vers $\text{CE} = 5\text{--}10$, tandis que les images à $p = 0.9999$ restent clouées à $\text{CE} \approx 0$.
2. **La vraie valeur de $\beta$ en classification dans le papier (Table 1, p. 24) :**
   - En classification supervisée, Zhang et al. utilisent **$\beta = 1.0$** (distributions séparées $\alpha=1$) à **$\beta = 10.0$** (distributions chevauchantes $\alpha=0$, notre cas exact où $\mathcal{D}_{\text{FG}}$ et $\mathcal{D}_{\text{RT}}$ partagent les 45 mêmes classes !).
   - À $T = 2.0$, choisir **$\beta \in [2.0, 5.0]$** (ex. $\beta = 3.0$) fait tomber $W_\theta$ à $0.25$ dès $\Delta \text{lp}_{T=2} \approx -0.65$ ($\text{CE}_{T=1} \approx 0.50$), freinant automatiquement les exemples rapides pour laisser les exemples ultra-saturés rattraper la médiane !

---

## 3. Prescriptions Directes pour Notre Pipeline (`tracks/03_unlearning_v2/`)

1. **Passer `npo_beta` de `0.1` à `2.5 – 4.0` dans `MethodConfig` / `two_sided_npo_loss` :**
   - Active enfin le freinage exponentiel naturel de NPO à l'échelle d'un logit unique ($\Delta L \sim 0.4\text{--}0.8$), réduisant mécaniquement `ce_std` sans dépendre uniquement de la coupure dure `ce_i < target_ce`.
2. **Maintenir l'équilibre stationnaire `NPO + Retain KL` ($c_{\text{RT}} \in [2.0, 5.0]$, Fig. 8 & 9 du papier) :**
   - La Figure 9 (p. 12) prouve que `NPO + RT` transforme le pic éphémère de Forget Quality en un **plateau horizontal stable** sur plusieurs époques, éliminant l'hypersensibilité au nombre exact de steps.
3. **Évaluer le ratio $\log(\pi_\theta / \pi_{\text{ref}})$ sur la même vue augmentée (`get_batch_orig_logits`) :**
   - Indispensable pour que le poids $W_\theta(\tilde{x}, y) = 2\sigma(-\beta (L_\theta(\tilde{x}, y) - L_{\text{ref}}(\tilde{x}, y)))$ mesure le vrai gain d'oubli sur la copie altérée $\tilde{x}$ et non la difficulté intrinsèque du crop/jitter sur `orig`.

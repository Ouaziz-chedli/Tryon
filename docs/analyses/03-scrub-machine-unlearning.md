# Analyse de papier — Towards Unbounded Machine Unlearning / SCRUB (NeurIPS 2023)

- **Titre :** *Towards Unbounded Machine Unlearning*
- **Auteurs :** Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, Eleni Triantafillou (University of Warwick, Google DeepMind)
- **Conférence :** NeurIPS 2023 (`arXiv:2302.09880v3`)
- **PDF local :** [2302.09880v3.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/papers/2302.09880v3.pdf)
- **Pertinence directe :** **Track 3 — Machine Unlearning** ([03-unlearning.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/03-unlearning.md), [Unlearning.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/Unlearning.pdf))

---

## 1. Formulation mathématique exacte de SCRUB & SCRUB+R

### 1.1 Objectif Teacher-Student Min-Max découplé (Éq. 1–3, Algorithmes 1–3)
Soit $w^o$ le modèle original figé (*Teacher*, `model.pt`) et $w^u$ le modèle désappris (*Student*, initialisé à $w^o$). Avec la divergence KL à température $T$ :
$$d_T(x; w^u) = D_{\text{KL}}\!\left(\text{softmax}\!\left(\frac{f(x; w^o)}{T}\right) \;\middle\|\; \text{softmax}\!\left(\frac{f(x; w^u)}{T}\right)\right)$$
L'objectif global de SCRUB est :
$$\min_{w^u} \;\; \frac{\alpha}{N_r}\sum_{x_r \in \mathcal{D}_r} d_T(x_r; w^u) + \frac{\gamma}{N_r}\sum_{(x_r, y_r) \in \mathcal{D}_r} \ell_{\text{CE}}(f(x_r; w^u), y_r) - \frac{1}{N_f}\sum_{x_f \in \mathcal{D}_f} d_T(x_f; w^u)$$
Pour éviter les oscillations destructrices d'une optimisation conjointe (Fig. 6c, p. 21), SCRUB alterne deux phases par époque $i \in \{0, \dots, \text{STEPS}-1\}$ :
1. **`DO-MAX-EPOCH` ($i < \text{MAX-STEPS}$) :** montée de gradient sur $\mathcal{D}_f$ (avec facteur d'échelle $T^2$) :
   $$w^u \leftarrow w^u + \epsilon \nabla_{w^u}\!\left(\frac{T^2}{|b_f|} \sum_{x_f \in b_f} d_T(x_f; w^u)\right)$$
2. **`DO-MIN-EPOCH` (toutes les époques $i < \text{STEPS}$) :** descente de gradient sur $\mathcal{D}_r$, incluant une queue d'époques finales composées **uniquement de min-steps** ($\text{MAX-STEPS} \le i < \text{STEPS}$) pour guérir le modèle.
- **Asymétrie des batchs (Tableau 3, p. 16) :** En *Selective Unlearning*, le papier utilise $B_f \in \{16, 32\}$ et $B_r \in \{32, 64\}$ avec un decay du learning rate ($\times 0.1$) indispensable pour amortir les oscillations min-max.

### 1.2 Pourquoi le Max-Step seul explose & Rôle de SCRUB+R
Le gradient du max-step par rapport aux logits $z^u = f(x_f; w^u)$ vaut :
$$\frac{\partial d_T(x_f; w^u)}{\partial z_c^u} = \frac{1}{T}\left(p_T(z^u)_c - p_T(z^o)_c\right)$$
En montée de gradient ($+\epsilon \nabla d_T$), dès que la probabilité d'une classe secondaire $c^* \ne \hat{y}$ dépasse $p_T(z^o)_{c^*}$, le terme $p_T(z^u)_{c^*} - p_T(z^o)_{c^*} > 0$ **s'auto-amplifie exponentiellement**, projetant la prédiction sur une classe erronée arbitraire et poussant la loss CE vers $+\infty$.

Pour stopper l'oubli au niveau exact d'un modèle `gold` (*Retrain*), **SCRUB+R** (Section 3.2, p. 4–5 & Section 11.2, p. 20) sélectionne le checkpoint $w^{(i^*)}$ dont l'erreur sur $\mathcal{D}_f$ coïncide avec l'erreur de validation $\tau_{\text{ref}}$ sur un jeu held-out de même distribution :
$$i^* = \arg\min_{i \in \{1, \dots, I\}} \left|\text{Err}(\mathcal{D}_f; w^{(i)}) - \tau_{\text{ref}}\right|$$

---

## 2. Les 3 Régimes d'Unlearning (UP vs RB vs RC) & Équivalence avec la Métrique CISPA

Le papier démontre qu'il n'existe pas un objectif unique d'unlearning, mais trois régimes distincts :
1. **Removing Biases (RB) :** Maximiser l'erreur $\text{Err}(\mathcal{D}_f) \to 100\%$.
2. **Resolving Confusion (RC) :** Corriger des exemples mal labellisés entre deux classes.
3. **User Privacy (UP) :** Rendre $\mathcal{D}_f$ **indistinguable d'un jeu de test jamais vu** face aux attaques par inférence d'appartenance (`Basic MIA` sur la loss cross-entropy et `LiRA-for-unlearning` sur les distributions de confiance par image).

### Preuve de l'équivalence exacte entre `UP` (SCRUB+R) et [Unlearning.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/Unlearning.pdf)
- **Effet Streisand du sur-oubli (Tableau 23, p. 30, CIFAR-10 Selective) :**
  - `Retrain` (`gold`) a une erreur de **`29.33%`** sur $\mathcal{D}_f$ (`MIA = 54.0%`, proche du hasard `50%`).
  - `SCRUB` sans rewind pousse l'erreur sur $\mathcal{D}_f$ à **`71.67%`**, ce qui fait exploser l'attaque `MIA` à **`78.00%`** (pire que `Original` à `65.33%` !).
  - `SCRUB+R` ramène l'erreur sur $\mathcal{D}_f$ à **`19.33%`** (`MIA = 58.67%`).
- **Traduction dans la formule officielle de `Unlearning.pdf` (p. 2) :**
  - `mean_std_score` $= 1 - \text{clip}(d_{\text{sub}} / d_{\text{orig}}, 0, 1)$ neutralise exactement le `Basic MIA` (égalité des deux premiers moments $(\mu, \sigma)$ de la loss CE avec `gold`).
  - `prediction_score` $= 1 - \text{clip}(\text{KL}(p_{\text{gold}} \parallel p_{\text{sub}}) / \text{KL}(p_{\text{gold}} \parallel p_{\text{orig}}), 0, 1)$ à $T=2$ neutralise `LiRA` instance par instance en vérifiant la distribution prédictive complète à $T=2$.

---

## 3. Enseignements critiques pour Track 3 (`31-unlearning`)

1. **Échec de `EU-k` / `CF-k` sur le *Selective Unlearning* (Section 5.5, p. 10, Fig. 1a vs 1b) :**
   - Les auteurs montrent que ne modifier que la toute dernière couche (`head`) échoue en *Selective Unlearning* (oublier un sous-ensemble d'images au sein de 45 classes qui restent actives dans $\mathcal{D}_r$), car l'information d'instance est stockée dans les blocs profonds/intermédiaires (`blocks.6..11`), tandis que `head` encode le prototype global de toute la classe.
2. **Pourquoi $T = 2.0$ est obligatoire :**
   - À $T=1$, `model.pt` est saturé ($\text{CE} = 0.0003$), donc $p_{T=1}(w^o)$ est un one-hot qui ne contient aucune information sur les 99 classes secondaires. La distillation à $T=2$ (utilisée par le serveur) expose la hiérarchie des classes confondables (*dark knowledge*).
3. **Adaptation *Retain-Free* de SCRUB+R (Virtual Gold Teacher par Patch-Dropout / Multi-Vues) :**
   - En l'absence de $\mathcal{D}_r$ et de jeu de validation pour calculer $\tau_{\text{ref}}$, masquer aléatoirement 40–50 % des patchs $16\times 16$ du ViT-B/16 (ou appliquer des augmentations géométriques) sur $w^o(x_f)$ brise la mémorisation par cœur de l'instance $x_f$ tout en préservant la prédiction de généralisation de classe de $w^o$.
   - Cela fournit un oracle local de $p_{\text{gold}}^{(T=2)}(x_f)$ qui conserve à la fois la **dispersion inter-images $\sigma_{\text{gold}}$** (les images atypiques chutent plus sous masquage de patchs que les images canoniques) et l'**ordre exact des classes secondaires à $T=2$**.

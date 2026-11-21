# Towards Unbounded Machine Unlearning (SCRUB)

- **Référence :** Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, Eleni Triantafillou (University of Warwick & Google DeepMind, NeurIPS 2023) — [2302.09880v3.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/papers/2302.09880v3.pdf)
- **Code :** https://github.com/Meghdad92/SCRUB

## TL;DR

Ce papier propose deux avancées majeures en *machine unlearning* (faire « oublier » un sous-ensemble $\mathcal{D}_f$ à un réseau déjà entraîné sans ré-entraîner de zéro sur $\mathcal{D}_r$) :
1. **Un recadrage applicatif :** la définition de l'« oubli » dépend du cas d'usage — *Removing Biases* (RB), *Resolving Confusion* (RC), *User Privacy* (UP).
2. **SCRUB & SCRUB+R :** un algorithme *teacher-student* empirique, rapide et sans hypothèses mathématiques restrictives, accompagné de la première adaptation de l'attaque **LiRA** à l'unlearning (`LiRA-for-unlearning`).

---

## 1. Problème attaqué

Ré-entraîner de zéro sans le *forget set* $\mathcal{D}_f$ garantit un oubli exact mais est économiquement inviable. Les méthodes d'oubli approximatif existantes (Fisher, NTK) souffrent de trois limites :
- **Hypothèses fortes** souvent violées en pratique (poids optimaux proches du modèle original, $\mathcal{D}_f$ petit, dataset de pré-entraînement statique).
- **Mauvaise scalabilité** (complexité quadratique en nombre d'exemples ; Fisher est parfois plus lent que `Retrain`).
- **Évaluation monolithique**, alors que les critères de succès diffèrent selon le cas d'usage :
  - **Removing Biases (RB) :** maximiser l'erreur sur $\mathcal{D}_f$.
  - **Resolving Confusion (RC) :** éliminer la confusion entre classes due à des exemples mal labellisés placés dans $\mathcal{D}_f$.
  - **User Privacy (UP) :** l'erreur sur $\mathcal{D}_f$ doit être *juste assez haute* (égale à celle d'un modèle n'ayant jamais vu $\mathcal{D}_f$), sinon un attaquant détecte la suppression via une *Membership Inference Attack* (MIA).

---

## 2. Méthode : SCRUB, SCRUB+R et LiRA-for-unlearning

### SCRUB (*SCalable Remembering and Unlearning unBound*)

Le modèle original $w^o$ sert de **teacher figé**, et on entraîne un **student** $w^u$ (initialisé à $w^o$) pour qu'il obéisse au teacher sur le *retain set* $\mathcal{D}_r$ et lui désobéisse sur le *forget set* $\mathcal{D}_f$ :

$$\min_{w^u} \frac{\alpha}{N_r}\sum_{x_r \in \mathcal{D}_r} D_{KL}(p(w^o) \parallel p(w^u)) + \frac{\gamma}{N_r}\sum_{(x_r, y_r) \in \mathcal{D}_r} \ell_{CE}(f(x_r; w^u), y_r) - \frac{1}{N_f}\sum_{x_f \in \mathcal{D}_f} D_{KL}(p(w^o) \parallel p(w^u))$$

#### Décryptage de l'équation
- $p(w^o)$ et $p(w^u)$ : probabilités softmax prédites par le teacher et le student.
- $D_{KL}(p(w^o) \parallel p(w^u))$ : divergence de Kullback-Leibler ($0$ = distributions identiques, grand = distributions éloignées).
- **Terme 1 ($+\frac{\alpha}{N_r} \sum D_{KL}$ sur $\mathcal{D}_r$) :** *Knowledge distillation* classique — force le student à reproduire les probabilités du teacher sur les données conservées.
- **Terme 2 ($+\frac{\gamma}{N_r} \sum \ell_{CE}$ sur $\mathcal{D}_r$) :** Garde-fou sur la tâche d'origine (évite la dérive de précision pendant l'oubli).
- **Terme 3 ($-\frac{1}{N_f} \sum D_{KL}$ sur $\mathcal{D}_f$) :** Cœur de l'oubli — minimiser $-D_{KL}$ revient à **maximiser** $D_{KL}$ sur $\mathcal{D}_f$ pour éloigner les prédictions du student de celles du teacher.
- **Réalité algorithmique (Algorithmes 1–3) :** Optimiser cette loss min-max d'un bloc oscille (comme un GAN). SCRUB alterne donc une époque de montée de gradient sur $\mathcal{D}_f$ (**max-step**) et une époque de descente sur $\mathcal{D}_r$ (**min-step**), puis termine par plusieurs **min-steps** seuls pour restaurer la précision sur $\mathcal{D}_r$.

### SCRUB+R (*Rewind*) pour la vie privée (UP)

Comme $D_{KL}$ n'est pas bornée vers le haut, SCRUB pousse souvent l'erreur sur $\mathcal{D}_f$ à 100 % (sur-oubli identifiable par MIA). **SCRUB+R** sauvegarde un checkpoint à chaque époque, mesure en fin d'entraînement l'erreur sur un *validation set* de même distribution que $\mathcal{D}_f$ (proxy d'un modèle n'ayant jamais vu $\mathcal{D}_f$), puis **rembobine** au checkpoint dont l'erreur sur $\mathcal{D}_f$ est la plus proche de cette cible.

### LiRA-for-unlearning

Première adaptation de l'attaque LiRA (Carlini et al.) à l'unlearning : l'attaquant connaît l'algorithme, entraîne 256 *shadow models* et applique l'unlearning sur 10 000 *forget sets* chacun pour estimer deux gaussiennes de confiance (« oublié » vs « jamais vu ») et effectuer un test de rapport de vraisemblance.

---

## 3. Résultats clés

Évalué sur CIFAR-10/5, Lacuna-10/5 (32×32) avec ResNet-18 et All-CNN, et CIFAR-100 avec VGG16+BN (~138 M paramètres) :

- **Removing Biases (RB) :** Seul algorithme régulièrement dans le top en oubli (jusqu'à 100 % en *class unlearning*, meilleur en *selective unlearning*) sans dégrader le retain/test error, **2× à 7,8×** plus rapide que `Retrain` (Bad-T est plus rapide à ~16–20× mais oublie mal et dégrade l'utilité).
- **Resolving Confusion (RC) :** Surpassse toutes les baselines (`Finetune`, `CF-k`, `EU-k`, `Fisher`, `NTK`, `NegGrad+`, `Bad-T`). Sur CIFAR-5 ResNet, `Fgt test error` tombe de 92,0 à **19,7** (vs 18,3 pour `Retrain`).
- **User Privacy (UP) :** Sur *Basic MIA* (CIFAR-10 ResNet sélectif), SCRUB+R ramène la précision de l'attaquant de 78,0 % à **58,67 %** (proche des 54,0 % de `Retrain`). Sur *LiRA-for-unlearning*, SCRUB(+R) obtient la meilleure défense (AUC **0,72–0,74** vs 0,92 pour Original/CF-k).

---

## 4. Regard critique & Limites

- **Aucune garantie théorique et fuite vie privée encore élevée :** Même avec SCRUB(+R), l'AUC de LiRA reste à **0,72–0,74** (loin de 0,50 = hasard pur), ce qui est insuffisant pour une conformité RGPD stricte.
- **Hypothèse fragile de SCRUB+R :** Le *rewinding* suppose un validation set de même distribution que $\mathcal{D}_f$ (réaliste pour « 100 images de la classe 5 », irréaliste pour les données hétérogènes d'un utilisateur demandant le droit à l'oubli). De plus, dans l'expérience LiRA, les auteurs ont filtré *a posteriori* les runs plutôt que d'exécuter le vrai rewinding.
- **Échelle modeste et protocole sans augmentation :** Images 32×32, CNN classiques, **sans data augmentation** (ce qui accentue l'overfitting initial). Aucun test sur Transformers, LLM ou modèles génératifs.
- **Tuning min-max sensible et accès complet à $\mathcal{D}_r$ :** Le Tableau 3 en annexe montre un réglage fin par scénario (`forget-set bs`, `retain-set bs`, `max/min steps`, `lr decay` obligatoire). Chaque min-step parcourt tout $\mathcal{D}_r$, et les gains de temps sur Lacuna-10 ne sont que de **1,95× à 2,81×**.

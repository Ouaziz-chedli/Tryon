# Pixel Seal : Adversarial-only training for invisible image and video watermarking

- **Référence :** Tomáš Souček\*, Pierre Fernandez\* et al. (Meta FAIR, déc. 2025) — [2512.16874v1.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/papers/2512.16874v1.pdf)
- **Code :** https://github.com/facebookresearch/videoseal
- **Site :** https://facebookresearch.github.io/meta-seal

## TL;DR

Pixel Seal est un modèle de tatouage numérique (*watermarking*) invisible **post-hoc** (appliqué après génération) pour images et vidéos, encodant **256 bits**. Le papier supprime totalement les losses perceptuelles classiques (MSE, LPIPS) au profit d'un entraînement purement adversarial couplé à un scheduling en 3 étapes et une simulation haute résolution à l'entraînement. Il s'adapte à la vidéo à l'inférence en *zero-shot* via *temporal watermark pooling* sans ré-entraînement.

---

## 1. Problèmes attaqués

Les pipelines classiques (*Embedder* + *Extractor* entraînés bout-en-bout à travers des augmentations) se heurtent à trois verrous :

1. **Losses perceptuelles inadaptées :** MSE et LPIPS modélisent mal la perception humaine (MSE pénalise autant une zone texturée qu'un aplat de couleur, laissant des artefacts visibles dans le ciel ou les fonds unis).
2. **Paysage d'optimisation instable :** Les objectifs s'opposent frontalement (l'optimum perceptuel est un watermark nul $w = 0$, l'optimum de robustesse est un watermark très visible), causant des *collapses* d'entraînement.
3. **Gap de résolution à l'inférence :** Entraînement en $256 \times 256$ puis upscaling du watermark sur l'image finale → bavures d'artefacts et *distribution shift* pour l'extracteur.

---

## 2. Architecture & Stack technique

- **Embedder :** U-Net de **43,8 M** de paramètres (opérant en $256 \times 256$).
- **Extractor :** ConvNeXt-v2 Tiny de **33,4 M** de paramètres.
- **Discriminator :** Discriminateur par patchs repris de Rombach et al. (Latent Diffusion).
- **Attaques non différentiables (ex. JPEG) :** Straight-through estimator :
  $$\tilde{x}_w = x_w + \text{nograd}(T(x_w) - x_w)$$
- **Entraînement :** Dataset SA-1b, 600k steps, batch size 256, AdamW ($lr = 5 \times 10^{-4}$, cosine decay, 20k warmup).
- **Garantie statistique :** L'encodage sur $n_{\text{bits}} = 256$ bits fournit une borne théorique sur le taux de faux positifs via la distance de Hamming et la loi binomiale $\mathcal{B}(0.5)^{n_{\text{bits}}}$, sans calibration empirique lourde.

---

## 3. Les 4 contributions clés

1. **Entraînement purement adversarial + Watermark Boosting :**
   - Suppression totale de MSE/LPIPS : seules la binary cross-entropy sur le message ($\lambda_{\text{msg}} = 1.0$) et la hinge loss adversariale ($\lambda_{\text{adv}} = 0.1$) sont conservées.
   - **Watermark boosting :** On injecte au discriminateur une version amplifiée $\hat{x}_w = x + \beta(x_w - x)$. $\beta > 1$ renforce l'invisibilité, $\beta < 1$ privilégie la robustesse (défaut : $\beta = 1$).
   - **Masque JND (*Just-Noticeable Difference*) :** Une fonction non apprenable $\text{JND}(x)$ pondère localement le watermark pour le confiner aux zones de forte variance (contours, textures) et l'annuler sur les aplats :
     $$x_w = x + \alpha \cdot (w \odot \text{JND}(x))$$

2. **Entraînement découplé en 3 étapes :**
   - *Stage 1 :* Uniquement $\mathcal{L}_{\text{msg}}$ avec $\alpha_0 = 1.0$ (watermark visible mais apprentissage stable jusqu'à saturation de la bit accuracy).
   - *Stage 2 :* Activation du discriminateur ($\mathcal{L}_{\text{adv}}$) et décroissance cosinus de $\alpha$ vers $\alpha_1 = 0.2$.
   - *Stage 3 :* Finetuning à $\alpha_1 = 0.2$ sur des attaques ou domaines ciblés.
   - Les ablations montrent que démarrer à $\alpha = 0.2$ ou avec le discriminateur dès le début provoque un échec total (50 % de bit accuracy = hasard pur).

3. **Simulation haute résolution à l'entraînement :**
   - Pendant le training (résolutions aléatoires dans $[256, 768]$ pour gérer les ratios d'aspect), l'image est réduite en $256 \times 256$ pour l'embedder, le watermark brut est upscalé, multiplié par $\text{JND}(x)$ en haute résolution, puis évalué par le discriminateur et attaqué en haute résolution avant d'être redimensionné pour l'extracteur.

4. **Adaptation vidéo zero-shot (*Temporal Watermark Pooling*) :**
   - Insertion *uniquement à l'inférence* d'un *temporal average pooling* (noyau et stride $k = 4$) après le $d$-ième bloc descendant du U-Net, et d'un *un-pooling* symétrique dans la partie montante.
   - Pour $d = 1, k = 4$ : speedup de **2,2×** (mesuré sur Quadro GP100, 64 frames) avec perte de performance négligeable.

---

## 4. Résultats principaux

- **Robustesse image :** Sur attaques combinées (Crop 50 % + JPEG + luminosité), Pixel Seal atteint **91 %** de bit accuracy sur images IA (Meta AI) et **94 %** sur photos réelles (SA-1b) avec 256 bits, surclassant MBRS, TrustMark, WAM et Video Seal 1.0 (et InvisMark en capacité : 256 vs 94 bits).
- **Imperceptibilité :** Sur images générées, PSNR = **48,9**, LPIPS = **0,0013**, score JND = **0,34** (1 JND = 75 % des observateurs perçoivent une différence).
- **Vidéo :** Surpasse RivaGAN et Video Seal 0.0/1.0 sur MovieGen et SA-V, notamment en attaques géométriques (97–98 %) et combinées (68–70 %).

---

## 5. Regard critique & Limites

- **Effondrement sous compression vidéo agressive :** En annexe (Figure 8), sous H.264/H.265 à CRF 50, la bit accuracy tombe à **0,52–0,54** ; dès CRF 40 en attaque combinée (H.264 + Crop + Brightness), elle chute à **0,51** (signal perdu). En image, un crop sévère (`Crop 0.32`) fait chuter à **0,58**.
- **Aucune évaluation face aux attaques par IA générative :** Pas de benchmark contre la régénération par diffusion ou l'inpainting, uniquement filtres et compressions classiques.
- **Sécurité en boîte blanche non testée :** Les poids étant open-source, un attaquant ayant accès à l'extracteur pourrait mener des attaques adversariales pour effacer ou forger le watermark.
- **Évaluation vidéo très courte (3 s) et *scene cuts* :** L'impact du temporal pooling ($k = 4$) sur un changement de plan brutal (frames adjacentes non corrélées) n'est ni mesuré ni discuté.
- **Métrique JND propriétaire & coût d'entraînement :** Le régresseur JND d'évaluation repose sur une étude interne Meta ; le coût GPU d'un discriminateur jusqu'à $768 \times 768$ sur 600k steps (batch 256) n'est pas chiffré.

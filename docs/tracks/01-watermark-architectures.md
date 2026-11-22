# Track 1 — Architectures proposées

Voir la fiche principale : [01-watermark-localization.md](01-watermark-localization.md).

## Mesures zéro-shot déjà faites (20 images/méthode de `release/train`, sans entraînement)

| Signal | PixelSeal | TrustMark | StegaStamp |
| :--- | :---: | :---: | :---: |
| Baseline tout-zéro | 0.252 | 0.252 | 0.252 |
| `per_position_logits · message` (image entière, `8×8`) | 0.475 | 0.468 | n/a |
| Contribution exacte `13×13` (rétroprojection `dec_conv_7`) | n/a | n/a | 0.477 |
| PixelSeal natif `512` (`16×16`) | 0.487 | n/a | n/a |
| Crops glissants `256` / `128` (redimensionnés par le décodeur) | 0.445 / 0.342 | 0.288 / 0.290 | 0.296 / 0.285 |
| Fenêtre `192` visible sur fond neutre (échelle `512` conservée) | 0.355 | 0.418 | **0.493** |

Constat : les décodeurs savent où est le message, mais seulement sur une grille grossière. Recadrer détruit l'échelle du watermark (surtout TrustMark et StegaStamp).

## Test de faisabilité : classifieur de patchs `64×64` (job Slurm 15902067, 1× A100, ~3 min)

Petit CNN sur résidu haute fréquence, patchs uniformes (100 % tatoué ou 100 % propre) pris dans les 510 images réelles, validation croisée 5-fold **par image**.

| Méthode | AUC moyenne (±std) | Précision |
| :--- | :---: | :---: |
| StegaStamp | **0.982** (±0.013) | 93 % |
| TrustMark | 0.697 (±0.047) | 65 % |
| PixelSeal | 0.594 (±0.019) | 58 % |

Lecture : StegaStamp est détectable localement avec un petit CNN. TrustMark est faible. PixelSeal est proche du hasard avec cette configuration, ce qui n'est pas une preuve d'absence de signal (réseau petit, 6 000 patchs, 12 epochs, patchs de bord exclus). Ce n'est pas un score mIoU.

Scripts : `tracks/01_watermark/patch_feasibility.py`, `slurm_patch_feasibility.sh`.

## Résultat du test B (template matching) : négatif

Corrélation locale entre `embed(x, message) − x` et le résidu haute fréquence de `x` (fenêtres 16 et 32 px, 20 images/méthode) :

| Méthode | mIoU vrai message | mIoU message aléatoire |
| :--- | :---: | :---: |
| PixelSeal | 0.355 | 0.344 |
| TrustMark | 0.273 | 0.274 |
| StegaStamp | 0.353 | 0.341 |

Vrai et faux message donnent le même score : le résultat ne dépend pas du message. Causes probables : embedders non linéaires, pas d'image propre d'origine (`embed` appliqué à une image déjà tatouée ou éditée), filtre passe-haut grossier. À abandonner sauf variante plus fine.

## A. Features figées des décodeurs + tête de segmentation
- **Entrée :** cartes multi-échelle des décodeurs et corrélation avec le `message` connu (PixelSeal `512` natif `16×16`, TrustMark `62/31/16/8`, StegaStamp `13×13` reconstruit).
- **Tête :** FPN ou U-Net léger vers `128×128`, avec l'image RGB et son résidu haute fréquence en branche de détails.
- **Loss :** BCE + Dice, ou Lovász pour coller au mIoU.
- **Pour :** peu de paramètres avec 510 images, entraînement rapide.
- **Contre :** résolution des features grossière, bords flous.

## B. Template matching au niveau pixel (hypothèse à tester)
- **Idée :** l'embedder et le message sont connus, donc `embed(x, message) − x` donne le motif que le watermark ajouterait. On le corrèle localement avec le résidu haute fréquence de `x`.
- **Intérêt :** résolution pixel au lieu de `8×8`.
- **Risque :** signal peut-être trop faible (PixelSeal et TrustMark). À mesurer.
- **Usage :** peut devenir un canal d'entrée de A.

## C. U-Net de forensique de bout en bout
- **Entrée :** résidu haute fréquence (filtres SRM ou noyaux appris) + canaux de corrélation issus de A et B.
- **Sortie :** masque `128×128`, embedding de la méthode (3 classes) pour un modèle partagé.
- **Entraînement :** données synthétiques générées avec les 3 embedders (trous, objets transplantés, outpainting), puis fine-tuning sur les 510 images.
- **À vérifier :** il faut des images propres externes. Le repo n'en contient pas, et les nœuds de calcul n'ont peut-être pas internet.

## D. Fine-tuning du décodeur avec supervision spatiale
- **Idée :** les `per_position_logits` ne sont jamais entraînés pour être localement fiables.
- **Méthode :** dé-freezer le décodeur, réduire les strides (`32×32` ou plus), loss de bits pondérée par le masque.
- **Pour :** plus gros gain potentiel en résolution.
- **Contre :** sur-apprentissage avec 510 images sans données synthétiques.

## E. Post-traitement structurel (combinable)
- Environ 50 % des masques d'outpainting et de transplantation sont des rectangles : ajustement d'un rectangle quand la forme s'y prête.
- Lissage (CRF, morphologie), TTA (flips), moyenne des sources de signal.

## Ordre recommandé (mis à jour après les tests)
1. ~~B~~ (négatif)
2. **U-Net pixel-niveau pour StegaStamp** (AUC patchs 0.98) : `tracks/01_watermark/unet_cv.py`, validation croisée 5-fold
3. **PixelSeal / TrustMark** : signal des décodeurs (A) combiné à un détecteur local, données synthétiques si autorisées
4. **E** par-dessus (seuil par méthode, rectangles, TTA)

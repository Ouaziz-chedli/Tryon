# Track 1 — Localize Image Watermark

- **Source :** [Watermark Localization.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/Watermark%20Localization.pdf)
- **Superviseurs :** Adam Dziedzic, Franziska Boenisch (CISPA)
- **Starter kit :** `https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris`

## 1. Objectif
Identifier **quelles zones d'une image (`512×512`) portent un watermark post-hoc** après des altérations/éditions partielles.
Trois méthodes publiques sont utilisées dans le dataset (dans un sous-ensemble rectangulaire, ratio d'aspect $\le 16:1$ pour PixelSeal/StegaStamp et $\le 2:1$ pour TrustMark) :
- **PixelSeal**
- **TrustMark**
- **StegaStamp**

Sortie attendue par image : un **masque binaire `128×128`** (`1` = tatoué, `0` = propre).

## 2. Types d'éditions présentes
- *Outpainting* : image tatouée étendue (le watermark ne couvre plus qu'une sous-partie).
- *Object Removal / Alteration* : un objet est édité/supprimé → un « trou » apparaît dans le watermark.
- *Object Addition* : un élément non tatoué est collé par-dessus → trou dans le watermark.
- *Object Transplantation* : un objet tatoué est collé sur une image propre → seul cet objet est tatoué.

## 3. Dataset & Évaluation
- **Total :** 1 530 images PNG (réparties équitablement entre les 3 méthodes).
  - **Train :** 510 images (170/méthode) avec masque ground-truth (`512×512` et `128×128`), méthode de watermarking et message binaire embarqué.
  - **Test :** 1 020 images (340/méthode), **avec** la méthode de watermarking et le message embarqué fournis ! *(à vérifier dans le starter kit, indiqué page 3 : « Test split: 1,020 images — image — watermarking method and embedded message »).*
- **Métrique :** `mIoU` moyen sur les 2 classes $\{0, 1\}$ sur grille `128×128` (1 cellule = `4×4` px, seuil $\ge 50\%$ de pixels tatoués). Cas limite : si $P_c \cup G_c = \emptyset$, $\text{IoU}_c = 1$.
- **Format de soumission :** fichier `.pt` ($\le 20$ MB) : `{"image_names": names, "masks": masks}` où `masks` est de shape `[num_images, 128, 128]` (valeurs `0` ou `1`).
- **Leaderboard :** 30 % public / 70 % privé.

## 4. Pistes techniques initiales
- Le starter kit fournit des **wrappers PyTorch unifiés** pour les extracteurs (et embedders) de PixelSeal, TrustMark et StegaStamp (`decode(images) -> bits`) **avec accès aux feature maps intermédiaires** !
- **Idée 1 (Classifier / Segmenteur sur feature maps des extracteurs) :** Extraire les feature maps intermédiaires de l'extracteur correspondant à la méthode de l'image + entraîner une tête de segmentation légère (U-Net / FPN / conv head) vers `128×128` (en profitant des 510 images d'entraînement + data augmentation par *copy-paste* / *cutmix* / masquage synthétique à l'aide des embedders fournis !).
- **Idée 2 (Sliding window / Patch-wise decoding + Bit Accuracy locale) :** Comme le message binaire et l'extracteur sont connus, évaluer la corrélation/bit-accuracy sur des patchs ou cartes de gradients/saliency par rapport au message cible, puis raffiner par segmentation.
- **Idée 3 (Génération de données synthétiques) :** Avec les 3 embedders disponibles, on peut générer des milliers d'images `512×512` avec masques exacts (outpainting, trous, transplantation) pour entraîner un modèle de segmentation robuste.

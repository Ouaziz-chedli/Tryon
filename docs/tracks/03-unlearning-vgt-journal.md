# Track 3 — VGT / SCRUB+R : expérience exploratoire isolée

## Question et portée

- **Question de départ :** tester si un Virtual Gold Teacher (VGT) tiré de vues masquées du modèle original pourrait classer des checkpoints déjà évalués. Le VGT du journal repose sur l'hypothèse non prouvée que masquer aléatoirement 40–50 % des patches conserve la distribution de logits/CE utile d'un vrai gold.
- **Portée effectivement testée :** diagnostic descriptif de la sensibilité du modèle original aux masques ; il ne teste ni classement de checkpoints, ni proximité avec le vrai gold, ni SCRUB+R.

## Protocole diagnostic

- Modèle original ViT-B/16 strict, `init_values=1e-4`, pixels uint8 convertis vers `[-1,1]`.
- Sous-échantillon de 256 images ; six vues indépendantes par taux de masquage (30 %, 45 %, 60 %) sur patches `16×16`, graine 4242.
- CE calculée avec le top-1 du modèle original sur l'image intacte. KL : moyenne par vue vers l'ensemble des six vues pour cette image. Pas de gold externe ni de checkpoint candidat.

- Session Colab L4 isolée `track3_vgt` ; les sessions des autres pistes sont restées intactes. Le modèle original (`model.pt`) et `forget_images.npz` ont été téléchargés depuis Hugging Face.
- À la demande de l’utilisateur, les checkpoints candidats ont été écartés. Le run ne classe donc aucun checkpoint, ne calcule aucune corrélation avec les scores serveur et ne teste pas SCRUB+R.
- Diagnostic du professeur masqué sur 256 images, 6 vues par image et taux de masquage 30 %, 45 % et 60 % (patches `16×16`, seed 4242). Résultats détaillés : [[../tracks/03_unlearning_v2/output/virtual_gold/vgt_teacher_diagnostic.json]].

| Patches masqués | CE moyenne | Écart-type CE (pooled) | Accord top-1 avec l'original | KL moyenne des vues vers leur ensemble | Entropie moyenne |
| :---: | ---: | ---: | ---: | ---: | ---: |
| 30 % | 0.835 | 2.069 | 81.4 % | 0.111 | 0.274 |
| 45 % | 1.680 | 2.879 | 64.8 % | 0.232 | 0.434 |
| 60 % | 2.984 | 3.681 | 46.4 % | 0.426 | 0.609 |

La CE et l'entropie augmentent avec le masquage, tandis que l'accord avec le modèle original diminue et que les vues divergent davantage. À 45 %, la perturbation est déjà substantielle ; les nombres montrent une sensibilité au masque, **pas** que le professeur virtuel approche le vrai gold ni qu'il soit prédictif d'un score serveur. Le diagnostic est un unique sous-échantillon et une seule graine ; il ne valide donc ni le classement VGT ni sa stabilité.

## Décision

Résultat exploratoire utile : le masque à 45 % ne constitue pas un « gold » neutre, car il altère fortement les distributions du professeur (accord top-1 64,8 %, entropie 0,434 contre 0,00176 sur les images intactes). Ce constat ne réfute pas à lui seul une éventuelle utilité de classement des checkpoints, question qui reste non testée puisque les checkpoints ont été écartés. Ne pas en déduire une validation de SCRUB+R ou une hausse garantie du score.

Le calcul a été exécuté depuis un script temporaire non conservé. Le code de classement de checkpoints, `tracks/03_unlearning_v2/research_virtual_gold.py`, n'a pas été exécuté.

# Track 3 — VGT / SCRUB+R : expérience exploratoire isolée

## Question et portée

Tester si un Virtual Gold Teacher (VGT) construit à partir de vues du modèle original masquées par patches est un proxy utile pour classer des checkpoints qui ont déjà un score serveur connu. Ce test n'entraîne aucun checkpoint, ne contacte pas l'API de soumission et ne valide pas SCRUB+R en entier. L'hypothèse falsifiable est que le classement induit par le proxy VGT suit le classement serveur des mêmes candidats ; un échec suffit à rejeter ce proxy dans cette configuration.

Le contexte est la piste G : le « laboratoire gold » précédent n'était pas suffisamment fiable pour être utilisé tel quel comme oracle. Le VGT proposé dans le journal principal repose lui aussi sur une hypothèse non prouvée : que le masquage aléatoire de 40–50 % des patches conserve la distribution de logits/CE pertinente du vrai gold. Une hausse de CE ou d'entropie du professeur masqué, seule, ne démontre pas cette propriété.

## Protocole prévu

- Modèle original strict ViT-B/16, `init_values=1e-4`, prétraitement serveur des pixels uint8 vers `[-1,1]`.
- Sous-ensemble exploratoire de 256 images `forget_images.npz` ; 6 vues par image ; masquage de 45 % des patches `16×16`, graine 4242.
- Le label de CE est la prédiction top-1 du modèle original sur l'image intacte. Les logits VGT sont la moyenne des logits des vues masquées.
- Pour chaque checkpoint, produire deux distances normalisées par le modèle original : écart de moments CE (moyenne + écart-type) et KL `T=2`, puis leur produit comme score de classement expérimental. Ce produit est **un diagnostic interne, pas la formule/le score officiel**.
- Candidats présélectionnés avec scores publics consignés au journal : `AT_ce0.50_b3.0_f8` (0.0604), `AT_ce0.50_b2.0_f6` (0.1099), `AT_ce1.00_b1.8_f6` (0.1193, record), `H4_h123_f6_ce125` (0.0984), `H1_h1h2_f8_ce125` (0.0813). Le classement VGT devait être comparé à ces scores indépendants.

## Exécution et résultat

- Une session Colab L4 dédiée, `track3_vgt`, a été créée ; les autres sessions des pistes parallèles n'ont pas été arrêtées ni éditées.
- `model.pt` (344 MB) et `forget_images.npz` (527 MB) ont été récupérés depuis le dataset Hugging Face dans cette session.
- La voie d'upload Colab (Contents API) a renvoyé HTTP 400/500 pour un checkpoint local de 344 MB. Le montage Google Drive depuis la session a échoué (`drive.mount: ValueError: mount failed`). La tentative de transfert SCP via la passerelle SSH a démarré mais est restée bloquée sans transfert vérifiable ; elle a été interrompue en arrêtant uniquement la session dédiée.
- **Aucun benchmark VGT sur checkpoints n'a donc été exécuté ; aucun résultat de proxy ni corrélation avec les scores serveur n'est disponible.** Ne pas interpréter cette investigation comme un échec empirique de VGT ni comme une validation de SCRUB+R.
- Les tests unitaires locaux du masquage par patch, du calcul de score et de la KL ont passé (`3 passed`). Ils vérifient uniquement l'implémentation mathématique, pas sa valeur prédictive.

## Décision / suite

Verdict : indéterminé sur le fond, bloqué par transfert de fichiers. Le code expérimental se trouve dans `tracks/03_unlearning_v2/research_virtual_gold.py`. Pour terminer réellement le test, il faut rendre les checkpoints accessibles à la session L4 (par exemple via un transfert SFTP fonctionnel ou une source de stockage partagée), puis exécuter ce script et comparer les rangs — de préférence sur les 3 500 images et plusieurs graines avant toute conclusion. L'extrapolation à « exploser le score dans tous les cas » ou à SCRUB+R complet serait injustifiée même si le rang était positif.

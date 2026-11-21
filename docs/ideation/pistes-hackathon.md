# Pistes d'idéation pour le Hackathon

> **Contexte :** Le sujet officiel du hackathon n'est pas encore connu. Les 3 papiers étudiés convergent néanmoins vers un socle commun très fort : **Confiance, Traçabilité, Auditabilité et Gouvernance des systèmes d'IA** (aligné avec les enjeux européens type *AI Act* et *RGPD*).

## 1. Fil rouge entre les 3 papiers

| Pilier | Papier | Question à laquelle il répond | Type d'accès |
| :--- | :--- | :--- | :--- |
| **Provenance** | Pixel Seal (`2512.16874v1`) | *« Ce média (image/vidéo) provient-il de telle source / tel modèle ? »* | Post-hoc (Embedder / Extractor 256 bits) |
| **Intégrité / Audit** | Model Equality Testing (`2410.20247v2`) | *« L'API que j'appelle sert-elle vraiment le modèle promis, ou a-t-elle été quantifiée / modifiée / dérivée ? »* | Boîte noire (échantillons texte) |
| **Conformité / Nettoyage** | SCRUB (`2302.09880v3`) | *« Comment retirer l'influence de données biaisées, empoisonnées ou protégées (RGPD) d'un modèle déjà entraîné ? »* | Boîte blanche (poids + $\mathcal{D}_r, \mathcal{D}_f$) |

---

## 2. Briques techniques réutilisables (prêtes à combiner selon le sujet)

1. **Module d'audit d'API LLM (léger & rapide à coder) :**
   - Implémenter le test **MMD + noyau de Hamming** (en Python, quelques dizaines de lignes sur des complétions API) avec test de permutation pour éviter de ré-échantillonner la référence.
   - *Cas d'usage hackathon :* dashboard de monitoring de dérive de LLM, détection de *shadow quantization* chez un provider, vérification de conformité d'un endpoint d'entreprise.
   - *Point de vigilance issu de la critique :* normaliser strictement le *chat template* et les préfixes avant de calculer Hamming pour éviter les faux positifs par décalage.

2. **Pipeline de provenance média (Image / Vidéo) :**
   - Réutiliser les poids/repo **Pixel Seal** (`facebookresearch/videoseal`) pour tatouer/vérifier des contenus générés avec calcul automatique de la $p$-value (borne binomiale de Hamming sur 256 bits).
   - *Cas d'usage hackathon :* authentification de médias journalistiques, traçabilité de génération IA, détection de deepfakes par absence/présence de signature.
   - *Point de vigilance :* ne pas promettre de résistance aux compressions vidéo extrêmes (CRF $\ge 40$) ni à la régénération par diffusion.

3. **Pipeline de dé-biaisage / nettoyage par Unlearning :**
   - Implémenter **SCRUB** (alternance *max-step* sur $\mathcal{D}_f$ / *min-step* sur $\mathcal{D}_r$ avec KL + CE) + **Basic MIA** pour auditer l'oubli.
   - *Cas d'usage hackathon :* suppression de biais ou de données empoisonnées/mal labellisées en production sans coût de ré-entraînement complet ; démonstrateur « Droit à l'oubli » avec tableau de bord d'audit MIA.
   - *Point de vigilance :* pour le cas *User Privacy*, `SCRUB+R` ne garantit pas une immunité totale face à LiRA (AUC ~0,72–0,74) — à présenter comme une réduction du risque, pas une garantie RGPD absolue.

---

## 3. Idées de projets « prêts à pivoter » le jour J

- **Si le sujet porte sur la Confiance / Régulation / AI Act :**
  *« AI Compliance & Audit Suite »* — Un outil unifié qui (1) vérifie qu'un endpoint LLM en prod n'a pas dérivé/été altéré (Model Equality Testing), (2) signe les sorties visuelles générées (Pixel Seal), et (3) permet de purger des données toxiques/biaisées d'un modèle (SCRUB) avec preuve empirique (MIA).
- **Si le sujet porte sur la Cybersécurité / Désinformation :**
  *« Provenance & Model Integrity Guard »* — Chaîne de confiance bout-en-bout : vérifier l'intégrité du modèle distant (MMD Hamming) + tatouer/vérifier les médias produits (Pixel Seal + borne statistique de faux positifs).
- **Si le sujet porte sur la Data Privacy / Santé / Finance :**
  *« Auditable Unlearning »* — Interface de gestion des requêtes de suppression/correction de labels (SCRUB / SCRUB+R) couplée à un audit automatique par MIA pour mesurer ce que le modèle retient encore.

# Ledger — attack the rect hole (A+B+C, un seul job devel)

Plan validé : diagnostic (A) + rect_search_v3 edge-tracing (B) + spécialiste
stegastamp HF (C), un seul job `dc-gpu-devel` 4 GPU, re-soumission directe.

## Rulings
- R1 : pas de commits (jamais sans demande explicite) ; pas de worktree
  (staging cluster via chemins explicites). Coût si faux : aucun, fichiers
  locaux dans `tracks/01_watermark/` (non suivi git de toute façon).
- R2 : boucle TDD locale limitée au pur numpy/scipy (`trace_core.py`,
  `tests/`). Le code GPU/wmloc (import torch au top-level, `wmloc` absent
  en local) est vérifié par `py_compile` + smoke cluster `--smoke`.
  Coût si faux : bug découvert seulement sur le cluster (temps de file).
- R3 : `StegaStampV4.fe` hardcodé à 5 canaux → Phase C ajoute une variante
  6 canaux (`StegaStampV4C6`) utilisée uniquement sous `--hf_residue`,
  avec assert de shape. Chemin par défaut inchangé.
  Coût si faux : crash shape au runtime → smoke cluster.
- R4 : `trace_edges` ne trace que si le score initial >= `min_score=0.6`
  (paysage plat = pas de signal, box inchangée). Coût si faux : boxes
  min_side sur bruit → rejetées par la gate de toute façon.
- R5 : sémantique `miou2` répliquée depuis la doc (mIoU 2 classes,
  IoU=1 si union vide) pour diag/prints ; le postprocess officiel garde
  `wmloc.utils`. Coût si faux : écart d'arrondi dans les prints diag.

## Tâches
- [x] trace_core.py + tests (15/15 GREEN en local)
- [x] diag_rect.py
- [x] rect_search_v3.py
- [x] train_rects.py --hf_residue (+ StegaStampV4C6 dans train_v4.py,
      extra_ch dans build_model_v5)
- [x] postprocess_submit_v5.py + postprocess_submit_v6.py (préférer bbox_v7)
- [x] slurm_attack_rects.sh (bash -n OK)
- [x] staging scp (47 fichiers) + sbatch job 15905717 (tryon-t1-rects, RUNNING jrc0288)
- [ ] suivi + fetch + rapport

## Phase A — résultat (job 15905717, diag_rect sur bbox_v6)
- pixelseal : rect=0.5291 (n=43, aire médiane 0.0131) / irreg=0.4099 / all=0.4695
- trustmark : rect=0.5736 (n=44, aire médiane 0.0383) / irreg=0.4239 / all=0.5005
- stegastamp : rect=0.4833 (n=44, aire médiane 0.0219) / irreg=0.4490 / all=0.4668
- logp médian rect≈irreg (~3.5) → la gate ne peut pas les séparer au logp.

## Phase B — résultat train-side v3 (job 15905717, ~09:17 elapsed)
Rect-subset v3 vs v6 : ps 0.5074 vs 0.5291 (-0.02), tm 0.5541 vs 0.5736 (-0.02),
ss 0.4822 vs 0.4833 (≈). Le edge-tracing greedy ne bat pas v2 : hypothèse
« boxes trop grandes » insuffisante — ancres probablement décentrées
(shrink d'une box mal placée empire l'IoU) + bruit binomial des acc
mono-décode. Le run continue (test-side, spécialistes ps + ss-HF, post v6).
- Les deux hypothèses confirmées : mix de régimes + boxes molles.
  StegaStamp rect le plus faible → C (HF) + edge-tracing ciblés au bon endroit.

## Vérifications (contrat)
- tests : `python3 -m pytest tracks/01_watermark/tests/ -q` → 15/15 pass
- `bash -n slurm_attack_rects.sh` → OK
- `py_compile` : train_v4/v5, train_rects, postprocess v5/v6, diag,
  rect_search_v3, trace_core → OK
- smoke cluster `--smoke` : dans le job (GPU1, trustmark) avant le run complet

## Rulings (suite)
- R6 : `postprocess_submit_v6.py` (08:09, coéquipier) découvert en cours de
  route — confidence-gated rect blend + variant selection + diag rect
  intégrée. Le job d'attaque utilise v6 (sur-ensemble de v5) ; l'édit v7 a
  été porté sur les deux. Coût si faux : v6 non testé sur cluster, mais
  tout ajout est OOF-gated (pire cas = égal v5).

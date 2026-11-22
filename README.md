# European AI Championship Paris 2026 — Team `TRYON`

Repository de l'équipe **TRYON** pour le **European AI Championship Paris 2026** (CISPA / Jülich JURECA).

## 🏆 Best solution — Track 3 Machine Unlearning (CISPA eval server, score **0.1283**, 2 records in the final hour)

**"Gold-Bell Quantile Gates"** — a SalUn-masked NPO ascent where each image's forgetting target is a per-image gate mapped from a lognormal bell fitted to 19 server-scored checkpoints, with difficulty ordering from patch-drop gold estimates:
1. **Fit the gold target without a gold model:** every server submission is a measurement; an L1-diamond fit over forget-CE `(μ, σ)` localizes the retrained-gold at `(1.73, 2.27)` — [scripts/gold_bell_fit.py](scripts/gold_bell_fit.py), [docs/tracks/03-unlearning-gold-bell-fit.md](docs/tracks/03-unlearning-gold-bell-fit.md).
2. **Per-image difficulty ordering:** patch-drop views of the original model rank images by gold-CE with ρ≈0.72 — `estimate_patchdrop_gold_logits` in [tracks/03_unlearning_v2/methods/patch_drop_targets.py](tracks/03_unlearning_v2/methods/patch_drop_targets.py).
3. **Ranks → bell quantile gates:** hard images push to CE 4-5, easy ones stop at 0.1-0.5; the landed distribution IS the bell by gate construction (NPO halts at the gate) — `quantile_calibrated_targets`, `_apply_gate_transforms`, `npo_per_image_gate_loss` in [tracks/03_unlearning_v2/methods/scrub_r_rest.py](tracks/03_unlearning_v2/methods/scrub_r_rest.py).
4. **Champion grid recipe:** lr 1e-4, 210 steps, β 2.0, retain-KL 2.8, SalUn keep 0.20, upper-half gate spread ×1.15, seed 101 — `round15_candidates()` in [tracks/03_unlearning_v2/run_scrub_r2.py](tracks/03_unlearning_v2/run_scrub_r2.py) (`--round 15 --candidate-id 1` regenerates the record in ~8 min on 1×A100).
5. **Hard-budget selection:** ProxyKL ≤ 0.056, Δθ ≤ 4.70 (budgeted rewind) + a 17-point proxy eval suite — [tracks/03_unlearning_v2/eval_harness.py](tracks/03_unlearning_v2/eval_harness.py).

Full experiment log (24 server probes, 6 falsified hypotheses, the Adam invariance / transient refinement diagnostics): [docs/tracks/03-unlearning-journal.md](docs/tracks/03-unlearning-journal.md).

**Track 2 Model Equality:** 33 black-box LLM endpoints audited under a 100-responses/endpoint budget — discriminative per-pair probes (quantization fingerprints, LoRA fingerprints) + family pre-clustering: [tracks/02_equality/](tracks/02_equality/), answers in `answers.jsonl`, methodology in [docs/tracks/02-model-equality.md](docs/tracks/02-model-equality.md).

**Track 1 Watermark Localization:** leaderboard **0.66** (27th, submitted 0.6544) — per-method FPN/ResUNet heads on decoder feature maps correlated with the known message + synthetic data from the provided embedders (holes, transplants, outpainting):
1. **Signal-first segmentation:** multi-scale decoder bit-agreement maps (`ms_maps`) + StegaStamp 13×13 contribution maps feed light FPN/ResUNet heads toward `128×128` (BCE+Dice, 5–7-fold CV, hflip TTA) — OOF mIoU ss 0.73 / ps 0.63 / tm 0.61 — [tracks/01_watermark/train_v5.py](tracks/01_watermark/train_v5.py).
2. **Decoder-verified rectangle search:** ~25 % of partial watermarks are rectangles; hierarchical coarse→fine search for the box maximizing bit-accuracy, refined by edge-tracing to the minimal decodable box (`bbox_v7`) — [tracks/01_watermark/rect_search_v3.py](tracks/01_watermark/rect_search_v3.py), pure-numpy core + tests in [tracks/01_watermark/trace_core.py](tracks/01_watermark/trace_core.py).
3. **Rect/hole specialists + gated ensemble:** rect-only and hole specialists blended under `full_acc` gates, per-regime weights, nested-CV dynamic threshold, confidence-gated rect blend and end-to-end variant selection — [tracks/01_watermark/postprocess_submit_v6.py](tracks/01_watermark/postprocess_submit_v6.py), [tracks/01_watermark/train_rects.py](tracks/01_watermark/train_rects.py).
4. **Known bottleneck:** true-rect box mIoU only 0.48–0.57 (vs 0.66 ensemble) — the gap to the top-10 (0.76) is rectangle precision; HF-residue 6-channel StegaStamp specialist in progress.

Methodology and experiments in [docs/tracks/01-watermark-localization.md](docs/tracks/01-watermark-localization.md), architectures in [docs/tracks/01-watermark-architectures.md](docs/tracks/01-watermark-architectures.md).

**Presentation slides:** [slides/TRYON-presentation.pptx](slides/TRYON-presentation.pptx)

## Navigation rapide

- **Index complet de la documentation :** [docs/README.md](docs/README.md)
- **Accès Cluster GPU Jülich (JURECA) :**
  - Guide de setup officiel & Slurm : [docs/infra/gpu-juelich-setup.md](docs/infra/gpu-juelich-setup.md) ([slides](docs/infra/GPU-usage-paris.pdf))
  - Générateur de code TOTP : [scripts/totp.py](scripts/totp.py) (`python3 scripts/totp.py`)
  - Monitoring Slurm & slot équipe en direct : [scripts/jureca_status.py](scripts/jureca_status.py) (`python3 scripts/jureca_status.py [--all|--strict|--json|--watch 35]`)

## Les 3 Tracks du Hackathon

| Track | Sujet | Analyse & Stratégie | Énoncé officiel (PDF) |
| :--- | :--- | :--- | :--- |
| **1. Watermark Localization** | Localiser des watermarks post-hoc (`128×128`, `mIoU`) | [docs/tracks/01-watermark-localization.md](docs/tracks/01-watermark-localization.md) | [Watermark Localization.pdf](docs/tracks/Watermark%20Localization.pdf) |
| **2. Model Equality** | Auditer 33 endpoints LLM (`100` req/endpoint, `MCC`) | [docs/tracks/02-model-equality.md](docs/tracks/02-model-equality.md) | [Model Equality Testing.pdf](docs/tracks/Model%20Equality%20Testing.pdf) |
| **3. Machine Unlearning** | Oublier 3 500 images sur ViT-B/16 sans retain set | [docs/tracks/03-unlearning.md](docs/tracks/03-unlearning.md) | [Unlearning.pdf](docs/tracks/Unlearning.pdf) |

## Papiers de recherche étudiés

- **Pixel Seal** (*Meta FAIR, 2025*) : [analyse](docs/analyses/01-pixel-seal-watermarking.md) · [PDF](docs/papers/2512.16874v1.pdf)
- **Model Equality Testing** (*Stanford, ICLR 2025*) : [analyse](docs/analyses/02-model-equality-testing.md) · [PDF](docs/papers/2410.20247v2.pdf)
- **SCRUB / Unbounded Machine Unlearning** (*Warwick & DeepMind, NeurIPS 2023*) : [analyse](docs/analyses/03-scrub-machine-unlearning.md) · [PDF](docs/papers/2302.09880v3.pdf)
- **Zero-Shot Machine Unlearning** (*NUS, IEEE TIFS 2023*) : [analyse](docs/analyses/04-zero-shot-machine-unlearning.md) · [PDF](docs/papers/2201.05629v3.pdf)
- **SalUn: Weight Saliency Unlearning** (*MSU / UPenn / MIT-IBM, ICLR 2024*) : [analyse](docs/analyses/05-salun-weight-saliency-unlearning.md) · [PDF](docs/papers/2310.12508v5.pdf)
- **NPO: Negative Preference Optimization** (*UC Berkeley, COLM 2024*) : [analyse](docs/analyses/06-npo-negative-preference-optimization.md) · [PDF](docs/papers/2404.05868v2.pdf)
- **Notes d'idéation pré-hackathon :** [docs/ideation/pistes-hackathon.md](docs/ideation/pistes-hackathon.md)

# European AI Championship Paris 2026 — Team `TRYON`

Repository privé de l'équipe **TRYON** pour le **European AI Championship Paris 2026** (CISPA / Jülich JURECA).

## Navigation rapide

- **Index complet de la documentation :** [docs/README.md](docs/README.md)
- **Accès Cluster GPU Jülich (JURECA) :**
  - Guide de setup officiel & Slurm : [docs/infra/gpu-juelich-setup.md](docs/infra/gpu-juelich-setup.md) ([slides](docs/infra/GPU-usage-paris.pdf))
  - Identifiants équipe, clé SSH & connexion : [infra/team-access.md](infra/team-access.md)
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

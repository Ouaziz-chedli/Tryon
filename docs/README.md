# Documentation — European AI Championship Paris 2026

> **Statut :** Sujets officiels reçus (3 tracks) + accès cluster GPU Jülich (JURECA).

## Structure du dossier

### 1. Infrastructure & Setup GPU ([infra/](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/infra))
- [GPU-usage-paris.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/infra/GPU-usage-paris.pdf) — Slides officielles d'accès au cluster Jülich.
- [gpu-juelich-setup.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/infra/gpu-juelich-setup.md) — Guide synthétique pas-à-pas (JuDoor, MFA, clé SSH, `~/.ssh/config`, `uv`, Slurm `sbatch`).
- [colab-l4-setup.md](../infra/colab-l4-setup.md) — Guide Google Colab Pro (GPU L4) via `google-colab-cli` (`uv`, OAuth, `colab new/exec/download/stop`).

### 2. Sujets officiels du Hackathon ([tracks/](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks))

| Track | Sujet | Métrique | Ressources / Starter Kit | Fiche d'analyse |
| :--- | :--- | :--- | :--- | :--- |
| **Track 1** | **Localize Image Watermark** | `mIoU` (grille `128×128`) | [HF: Hackathon_Watermark_Localisation_Task_Paris](https://huggingface.co/datasets/SprintML/Hackathon_Watermark_Localisation_Task_Paris) | [01-watermark-localization.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/01-watermark-localization.md) |
| **Track 2** | **Model Equality** | $0.1 \cdot MCC_{\text{family}} + 0.9 \cdot MCC_{\text{model}}$ | [HF: model_equality](https://huggingface.co/datasets/SprintML/model_equality) | [02-model-equality.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/02-model-equality.md) |
| **Track 3** | **Unlearning (ViT-B/16)** | `forget_score × retain_score` (gate $\ge 70\%$ val acc) | [HF: unlearning](https://huggingface.co/datasets/SprintML/unlearning) | [03-unlearning.md](tracks/03-unlearning.md) · [Journal](tracks/03-unlearning-journal.md) · [Benchmark](tracks/03-unlearning-benchmark-analysis.md) · [TODO](tracks/03-unlearning-todo.md) |

- Setup global Jülich (toutes tâches) : `git clone https://huggingface.co/datasets/SprintML/hackathon`
- Leaderboard : `http://34.147.229.55/leaderboard_page`

### 3. Papiers de recherche & Idéation ([papers/](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/papers), [analyses/](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/analyses))
- [01-pixel-seal-watermarking.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/analyses/01-pixel-seal-watermarking.md) — **Pixel Seal** (`2512.16874v1.pdf`)
- [02-model-equality-testing.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/analyses/02-model-equality-testing.md) — **Model Equality Testing** (`2410.20247v2.pdf`)
- [03-scrub-machine-unlearning.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/analyses/03-scrub-machine-unlearning.md) — **SCRUB / Unbounded Machine Unlearning** ([2302.09880v3.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/papers/2302.09880v3.pdf))
- [04-zero-shot-machine-unlearning.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/analyses/04-zero-shot-machine-unlearning.md) — **Zero-Shot Machine Unlearning** ([2201.05629v3.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/papers/2201.05629v3.pdf))
- [05-salun-weight-saliency-unlearning.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/analyses/05-salun-weight-saliency-unlearning.md) — **SalUn: Gradient-based Weight Saliency** ([2310.12508v5.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/papers/2310.12508v5.pdf))
- [06-npo-negative-preference-optimization.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/analyses/06-npo-negative-preference-optimization.md) — **NPO: Negative Preference Optimization** ([2404.05868v2.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/papers/2404.05868v2.pdf))
- [ideation/pistes-hackathon.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/ideation/pistes-hackathon.md) — Notes d'idéation pré-hackathon.

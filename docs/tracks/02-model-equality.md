# Track 2 — Model Equality

- **Source :** [Model Equality Testing.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/Model%20Equality%20Testing.pdf)
- **Superviseurs :** Adam Dziedzic, Franziska Boenisch (CISPA)
- **Starter kit :** `https://huggingface.co/datasets/SprintML/model_equality`
- **Sur JURECA (`TRYON`) :** `/p/scratch/training2666/ouaziz1/TRYON/model_equality/` (`.venv/` prêt, `query_reference.py`, `score_practice.py`, `task_template.py`, `submission.py`).

## 1. Objectif
Auditer **33 endpoints LLM** (`ep-01` à `ep-33`) en boîte noire (texte généré uniquement) pour :
1. Regrouper les endpoints par **famille** (même modèle de base).
2. Déterminer pour chaque paire $(i, j)$ ($528$ paires) s'ils servent **exactement le même modèle**.

### Règle d'identité (CRITIQUE)
- **Même modèle :** même checkpoint + même configuration de **quantification** + même **adaptateur LoRA**.
- **Variations qui NE changent PAS l'identité du modèle :**
  - Watermark de décodage (*decoding watermark*)
  - System prompt ajouté par le provider
  - Stratégie de décodage (*decoding strategy*)
- **Variations qui CHANGENT l'identité :**
  - Modèle de base différent (même au sein d'une même famille)
  - Quantification différente
  - LoRA différent

## 2. Contraintes d'API & Budget
- **Budget strict :** **100 réponses maximum par endpoint** pour toute la compétition (3 300 requêtes max au total).
- **Limites :** max `1 024` tokens en entrée, max `256` tokens générés, max `4` requêtes concurrentes. `temperature` et `seed` ignorés ; messages `system` rejetés.
- **Endpoints de validation locale :** `ref-01` à `ref-09` (avec réponses connues dans `practice_answers.csv` et `score_practice.py`).
- **Cooldown soumission :** 1 soumission toutes les 30 minutes (~48/24h).

## 3. Métrique d'évaluation
$$\text{score} = 0.1 \cdot \text{MCC}_{\text{family}} + 0.9 \cdot \text{MCC}_{\text{model}}$$
- `MCC_family` : MCC sur toutes les paires (« même famille ? »).
- `MCC_model` : MCC uniquement sur les paires appartenant à la même famille (« même modèle ? »).
- Format CSV : `endpoint,family,model`.

## 4. Pistes techniques initiales (attention au piège du papier MET !)
- Le papier de base (*Model Equality Testing*, ICLR 2025) utilise **MMD + noyau de Hamming**, qui est **très sensible** aux watermarks, system prompts et stratégies de décodage — or ici, ces 3 facteurs doivent être **ignorés** (invariance demandée) !
- **Attention :** Exploiter d'abord à fond `ref-01..ref-09` pour comprendre quels prompts et quelles statistiques distinguent quantification/LoRA tout en restant invariants au watermark/system prompt/decoding.
- **Prompts discriminants :**
  - Prompts à **faible entropie / déterministes** ou à structure contrainte (code, complétions factuelles, suites logiques, tokens rares) pour contourner les variations de décodage/watermarking ?
  - Tester les deux autres papiers cités en référence : **RUT** (*Rank-Based Uniformity Test*, `arxiv:2506.06975`) et **AgentProv** (`arxiv:2609.00052`).
- **Budget 100 req/endpoint :** Ne PAS gaspiller les requêtes sur `ep-01..ep-33` avant d'avoir validé la stratégie de prompts sur `ref-01..ref-09` ! Par exemple, 10 prompts × 10 échantillons, ou 20 prompts × 5 échantillons, ou approche adaptative en 2 phases (20 req pour clustering par famille, puis 80 req pour discriminer intra-famille).

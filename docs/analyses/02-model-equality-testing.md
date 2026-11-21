# Model Equality Testing : Which Model Is This API Serving?

- **Référence :** Irena Gao, Percy Liang, Carlos Guestrin (Stanford University, ICLR 2025) — [2410.20247v2.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/papers/2410.20247v2.pdf)
- **Code & Dataset :** https://github.com/i-gao/model-equality-testing (package Python + 1,1 M de complétions)

## TL;DR

Model Equality Testing formalise l'audit d'API de LLM « boîte noire » comme un test statistique à deux échantillons ($H_0 : P = Q$). À l'aide d'un test **MMD (*Maximum Mean Discrepancy*)** muni d'un simple **noyau de Hamming**, les auteurs détectent la quantification, le watermarking, le finetuning ou le remplacement de modèle avec une puissance médiane de **77,4 %** pour seulement **10 échantillons par prompt**. Appliqué à 31 endpoints commerciaux de Llama (été 2024) pour **< 1 $** par audit, le test épingle **11 endpoints sur 31**.

---

## 1. Problème attaqué

Les fournisseurs d'API d'inférence (Amazon Bedrock, Azure, Groq, Together, Perplexity, etc.) peuvent modifier silencieusement les modèles hébergés (quantification, pruning, watermarking, caching, finetuning, paramètres de décodage incorrects) ou laisser un endpoint dériver dans le temps.

- **Limite des audits existants :** Ils mesurent l'accuracy sur des QCM ou réponses courtes en décodage *greedy* ($T = 0$), ce qui ne teste que le mode de la distribution du token suivant et ignore les tâches longues (code, dialogue, résumé) avec échantillonnage.
- **Objectif :** Un test *black-box* (sans accès aux logprobs), peu coûteux en requêtes, sans vérificateur automatique requis, avec contrôle garanti du taux de faux positifs ($\alpha$).

---

## 2. Méthode : MMD + Noyau de Hamming

On tire $N$ paires $(x, y)$ d'une distribution de référence $P$ (poids Hugging Face ou ancien snapshot d'une API) et de l'API testée $Q$, puis on teste :
$$H_0 : \pi(x)P(y \mid x) = \pi(x)Q(y \mid x)$$

L'espace des sorties étant gigantesque ($m \times |V|^L$, avec $|V| = 128\,256$ pour Llama-3), les tests multinomiaux classiques ($L_1$, $\chi^2$) s'effondrent dès que $L$ augmente.

- **Pourquoi pas un noyau universel ?** Le noyau *one-hot* ($\mathbf{1}\{y = y'\}$) manque de puissance sur petits échantillons (aucune collision exacte sur textes longs), et le noyau *all-substrings* est prohibitif en calcul.
- **Choix retenu — Noyau de Hamming :**
  $$\tilde{k}_{\text{hamming}}(y, y') = \sum_{i=1}^{L} \mathbf{1}\{y_i = y'_i\}$$
  (avec padding à droite). Il compare les séquences token par token, ou **caractère par caractère** lorsque les tokenizers diffèrent ou que l'API renvoie du texte décodé. C'est un pseudo-mètre, mais empiriquement très sensible aux altérations courantes.
- **Calcul de la $p$-value :** Simulation de l'hypothèse nulle par ré-échantillonnage dans $P$, ou test de **permutation** (sans requêtes supplémentaires sur $P$).

---

## 3. Résultats clés

- **Simulations contrôlées** (Wikipedia, $m = 25$ prompts, $L = 50$ tokens, $N = 250$) : Le MMD Hamming atteint **77,4 %** de puissance médiane contre la quantification (`nf4`, `int8`, `fp16`) et le watermarking, surpassant `all-substrings`, `one-hot`, $L_1$ et $\chi^2$, tout en résistant à l'allongement de $L$.
- **Finetuning :** Détecté avec **> 50 %** de puissance dès la 1ère epoch (42 steps) sur Llama-3 8B, même quand le modèle est finetuné sur du code (CodeAlpaca) et testé sur Wikipedia.
- **Remplacement de modèle (espace caractères) :** Détecté à **~100 %** entre familles différentes (75–76 % au sein de la famille Llama). Les distances MMD montrent que les modèles se regroupent par famille / données d'entraînement plutôt que par taille (8B vs 70B).
- **Audit réel de 31 endpoints Llama (9 providers, été 2024) :**
  - Hypothèse nulle composite ($Q = P_{\text{fp32}}$ ou $Q = P_{\text{fp16}}$) sur Wikipedia, HumanEval et UltraChat en espace caractères ($\alpha = 0,01$, $N = 10m$), coût **< 1 $** par audit (de < 0,02 $ sur Fireworks 8B à 0,93 $ sur Azure 70B).
  - **11/31 endpoints épinglés :** Perplexity échoue sur 4/4 modèles (entropie visiblement réduite), Amazon Bedrock sur 4/4, Groq sur 2/4 (Llama-3.1 8B et 70B), Fireworks sur 1/4 (Llama-3.1 8B). Azure, Deepinfra, Together, Anyscale et Replicate passent l'audit.
  - L'écart MMD entre Perplexity et `fp32` sur Llama-3 8B ($0,03$) est équivalent à l'écart entre Llama-3 8B et GPT-3.5-Turbo.

---

## 4. Regard critique & Limites

- **Conflit d'intérêts déclaré :** Percy Liang est co-fondateur de Together AI (qui obtient 4/4). Même si le travail est mené à Stanford et non partagé avant publication, le classement des providers doit être lu avec recul.
- **Sensibilité extrême du noyau de Hamming au décalage :** Le noyau ne compare que les marginales position par position ($y_i = y'_i$). Un simple préfixe supplémentaire (ex. `"Here is the completed code:\n"` observé sur Amazon Bedrock et Replicate en Annexe D.3) ou une variation de *chat template* fait exploser la distance même à poids intacts. Les auteurs ont dû utiliser le chat template de Llama-3 pour Llama-3.1 afin d'éviter l'injection de la date du jour.
- **Détecte une différence, pas sa cause ni une baisse de qualité :** Un rejet ne prouve pas une quantification cachée (cela peut être un system prompt, un `top_p` par défaut, du caching). La corrélation entre MMD et variation d'accuracy sur HumanEval est modeste ($R^2 = 0,394$) et non directionnelle.
- **Perte de puissance en espace caractères :** Obligatoire quand l'API ne renvoie pas les IDs de tokens, mais fait chuter la détection des altérations fines (sur Llama-3.1 8B à $N = 10m$ : `int8` tombe à 7 %, `watermark` à 32 % ; il faut monter à $N = 50m$ pour compenser).
- **Coût caché de la référence $P$ :** Le « < 1 $ » ne compte que l'API $Q$. Générer $P$ localement en `fp32` demande un compute lourd (jusqu'à 15 000 complétions/prompt dans l'étude) — les auteurs n'ont pas pu auditer Llama-3.1 405B contre ses poids officiels. Pour les modèles fermés (GPT, Claude), on ne peut tester que la dérive temporelle.
- **Instantané daté & décodage peu réaliste :** Mesures de juillet–août 2024 à `temperature = 1` sans `top_p`/`top_k`.

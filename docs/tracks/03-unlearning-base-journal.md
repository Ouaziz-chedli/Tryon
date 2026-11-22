# Track 3 — Piste Initiale (`AT` / SalUn Statique + NPO Borné & Soupes) : Journal Spécialisé

> **Journal dédié au binôme autonome `base-lead` (Orchestrateur/Chercheur) & `base-exec` (Exécuteur/GPU)** optimisant nos **techniques initiales qui détiennent tous nos records serveur** :
> - **`#9` `AT_ce1.00_b1.8_f6.pt` = `0.1193` (RECORD ACTUEL)** (`CE = 1.909 ± 2.05`, $\mu+\sigma = 3.96$, `p50 = 1.214`, `>95% = 0.19`, `ProxyKL = 0.0486`, `dTheta = 4.61`, `W_SEL = 0.5565`)
> - **`#5` `B_ce0.45_lr8e-05_kl2.0.pt` = `0.1166`** (`CE = 1.438 ± 2.01`, $\mu+\sigma = 3.45$, `p50 = 0.380`, `>95% = 0.37`, `ProxyKL = 0.0551`, `dTheta = 4.03`)
> - **`#8` `AT_ce0.50_b2.0_f6.pt` = `0.1099` (meilleur rendement `0.0365`)** (`CE = 1.265 ± 1.75`, $\mu+\sigma = 3.01$, `p50 = 0.397`, `>95% = 0.31`, `ProxyKL = 0.0413`, `dTheta = 3.95`)

**Liens connexes :**
- Journal principal Track 3 : [03-unlearning-journal.md](03-unlearning-journal.md)
- Implémentation initiale : [auto_tune_4gpu.py](../../tracks/03_unlearning_v2/auto_tune_4gpu.py) · [salun_npo.py](../../tracks/03_unlearning_v2/methods/salun_npo.py) · [common.py](../../tracks/03_unlearning_v2/methods/common.py)
- Protocole de benchmark 3-Folds pondéré : [selective_benchmark.py](../../tracks/03_unlearning_v2/selective_benchmark.py)

---

## 1. Fenêtre exacte du sommet serveur (prouvée par nos 11 soumissions)

- **Sous-oubli (`< 0.1193`) :** $\mu + \sigma \le 3.45$ (`#7` `1.90 -> 0.0604`, `#8` `3.01 -> 0.1099`, `#5` `3.45 -> 0.1166`).
- **Sommet observé :** `#9` (`AT_ce1.00_b1.8_f6.pt`, $\mu+\sigma = 3.96$, `CE = 1.91 ± 2.05`, `p50 = 1.21`, `dTheta = 4.61` $\to$ **`0.1193`**).
- **Sur-oubli (`< 0.1193`) :** `#10` (`H4_h123_f6_ce125.pt`, $\mu+\sigma = 4.23$, `CE = 2.13 ± 2.10`, `p50 = 1.47`, `dTheta = 6.27` $\to$ `0.0984`).

**Cible chirurgicale pour dépasser `0.1193` (cibler `0.130 – 0.160+`) :**
- Garder strictement l'architecture propre qui gagne sur le serveur (**masque SalUn statique à $t=0$ sur `2 048` images**, **`freeze_until_block = 6`**, **`head_lr_mult = 1.0`**, **`weight_decay = 0.0`**, **`restore_absent_head_rows`**).
- Viser l'entre-deux exact entre `#5`/`#8` et `#9` : **`Full CE mean` $\in [1.55, 1.85]$**, **`Full mu+sigma` $\in [3.55, 3.90]$**, **`Full p50` $\in [0.65, 1.05]$**, **`Full >95% <= 0.22`**, **`ProxyKL <= 0.038`**, **`dTheta <= 4.5`**.
- Leviers autorisés :
  1. **Calibration fine `AT` (`target_forget_ce` $\in \{0.65, 0.75, 0.85, 0.90\}$, `npo_beta` $\in \{1.8, 1.9, 2.0\}$, `retain_kl_weight` $\in \{2.4, 2.8\}$, `salun_keep_ratio` $\in \{0.15, 0.20\}$)**.
  2. **Soupes convexes entre nos checkpoints initiaux déjà validés** (`AT_ce0.50_b2.0_f6.pt` `#8`, `AT_ce1.00_b1.8_f6.pt` `#9`, et les nouveaux runs `AT_ce0.75..0.90`).
  3. **Deuxième vue cohérente (`strong` ou mélange `clean + moderate` sur `forget`)** pour améliorer la moitié *altered copies* de `forget_score` sans augmenter `dTheta`.

---

## 2. Règles strictes d'isolation du binôme (`base-lead` & `base-exec`)

1. **Fichiers autorisés en écriture :**
   - `tracks/03_unlearning_v2/run_base_refine.py`
   - `tracks/03_unlearning_v2/tests/test_base_refine.py`
   - `tracks/03_unlearning_v2/output/base_refine/`
   - Ce journal : `docs/tracks/03-unlearning-base-journal.md`
   - *(Ne jamais modifier `track_h.py`, `track_i.py`, `selective_benchmark.py`, ni `03-unlearning-journal.md`.)*
2. **Session Colab dédiée :**
   - Utiliser exclusivement **`-s track3_base`** (`/home/chedli/.local/bin/colab new -s track3_base --gpu A100`), avec un *heartbeat* stdout toutes les 15 s, téléchargement des JSON et meilleurs `.pt` dans `tracks/03_unlearning_v2/output/base_refine/`, puis **`/home/chedli/.local/bin/colab stop -s track3_base`** systématique.
   - Ne jamais soumettre sur JURECA ni sur `http://34.147.229.55` (notifier `@coordinator` dès qu'un `.pt` est prêt).

---

## 3. Journal Chronologique des Expériences (`AT` / Techniques Initiales)

### 2026-11-22 (02h22) — Lancement du binôme `base-lead` + `base-exec`
- Initialisation du binôme dédié au raffinement chirurgical de la famille gagnante `AT` (`freeze_until_block = 6`, SalUn statique $t=0$, NPO borné et soupes).

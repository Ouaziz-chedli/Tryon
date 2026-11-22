# Track 3 (`31-unlearning`) — Document de Handoff Complet

**Date du snapshot :** `2026-11-22 04:17+01:00`  
**Équipe :** `TRYON` (European AI Championship Paris 2026)  
**Record serveur actuel :** **`0.1238`** (`#13` `SOUP_50_050_50_100.pt`, `submission_id: 2706`)  
**Prochaine fenêtre de soumission (`#16`) :** **`2026-11-22 04:21:40+01:00`** (cooldown 15 min après `#15` à `04:06:40`).

---

## 1. Règles strictes & Contraintes absolues

1. **NE PAS TOUCHER AU VGT** (instruction utilisateur explicite) :
   - Ne lire, modifier, exécuter ni commiter `research_virtual_gold.py`, `docs/tracks/03-unlearning-vgt-journal.md`, ni une session Colab `track3_vgt`.
2. **Budget GPU Colab :** maximum **4 sessions Colab simultanées**. Toujours lire [infra/colab-agent-guide.md](../../infra/colab-agent-guide.md) avant d'utiliser `colab` :
   - Tarballs `< 1 MB` (`--exclude='*.pt' --exclude='*.pth' --exclude='output/*'`).
   - Toujours couper le GPU (`colab stop -s <session>`) dès que le JSON et les `.pt` sont téléchargés.
3. **Cooldown serveur Track 3 :** **15 minutes** entre deux soumissions réussies (`2 minutes` si échouée).
   - Commande de soumission :
     ```bash
     curl -sS -w "\nHTTP_STATUS:%{http_code}\n" --connect-timeout 15 --max-time 600 \
       -H "X-API-Key: 0895eb0a5610b64740af893f4abefbc4" \
       -F "file=@<CKPT_PATH>" \
       http://34.147.229.55/submit/31-unlearning
     ```
4. **Ne jamais commiter de poids `.pt` / `.pth` / `.npz`** (couverts par `.gitignore`).
5. **JURECA (`training2666`) :** 1 seul job Slurm à la fois pour toute l'équipe (`ouaziz1`, `sangkhavongs1`, `bassim1`). Actuellement tout le compute Track 3 tourne sur Colab.

---

## 2. État des Soumissions Serveur & Modèle Empirique

| # | Heure (`01-22`) | Checkpoint | `sub_id` | Score | `Full CE (mu±sig)` | `mu+sig` | `p50` | `ProxyKL` | `dTheta` | Enseignement clé |
|---|---|---|---|---|---|---|---|---|---|---|
| `#7` | `00:49:11` | `AT_ce0.50_b3.0_f8.pt` | `2169` | `0.0604` | `0.704±1.19` | `1.90` | `0.116` | `0.0376` | — | Sous-oubli sévère |
| `#8` | `01:06:42` | `AT_ce0.50_b2.0_f6.pt` | `2218` | `0.1099` | `1.265±1.75` | `3.01` | `0.397` | `0.0413` | `3.95` | Pente gauche de la cloche |
| `#9` | `01:36:57` | `AT_ce1.00_b1.8_f6.pt` | `2307` | `0.1193` | `1.909±2.05` | `3.96` | `1.214` | `0.0486` | `4.61` | Proche sommet (légèrement à droite) |
| `#10` | `01:58:06` | `H4_h123_f6_ce125.pt` | `2387` | `0.0984` | `2.127±2.10` | `4.23` | `1.468` | `0.0409` | `6.27` | Pente droite (sur-oubli) |
| `#11` | `02:17:00` | `H1_h1h2_f8_ce125.pt` | `2457` | `0.0813` | `1.676±1.93` | `3.61` | `0.912` | `0.0325` | `5.78` | Bug `f8` + SalUn rotatif |
| `#12` | `03:12:02` | `I2R_f6_r4_ce18_fl085.pt` | `2648` | `0.0950` | `2.458±1.98` | `4.44` | `2.136` | `0.0740` | `6.01` | Preuve que Piste I transfère |
| **`#13`** | **`03:29:55`** | **`SOUP_50_050_50_100.pt`** | **`2706`** | **`0.1238`** | **`1.794±2.04`** | **`3.84`** | **`1.006`** | **`0.0558`** | **`4.48`** | **RECORD ACTUEL (soupe 50/50 `#8`+`#9`)** |
| `#14` | `03:50:18` | `AT_ce0.80_b1.9_k20_kl28.pt` | `2766` | `0.1197` | `1.958±2.09` | `4.05` | `1.256` | `0.0569` | `4.58` | 2ᵉ meilleur |
| `#15` | `04:06:40` | `SHRINK_88_AT9.pt` | `2818` | `0.1082` | `1.212±1.71` | `2.92` | — | `0.0353` | `4.05` | Preuve que `W_SEL=0.5800` ne compense pas `mu+sig < 3.5` |

### Règle d'or empirique pour battre `0.1238`
- **Cible obligatoire de distribution :** **`Full mu+sig ∈ [3.65, 3.90]`** et **`p50 ∈ [0.85, 1.15]`**.
- **Goulot à optimiser dans cette zone :** `prediction_score` (`~0.19` actuellement à `T=2`, cf. [03-unlearning-journal.md](03-unlearning-journal.md)) via :
  - Réduction du bruit d'entraînement (`ProxyKL <= 0.048`, `dTheta <= 4.55`) par **soupe multi-seeds** (`base`),
  - Réparation structurelle des liaisons par **LoRA `r=8`** (`track_i`),
  - Ordonnancement difficulté image par image (`rho_PD >= 0.55`) via **SCRUB+R + Patch-Dropout** (`scrub_r`).

---

## 3. État des 4 Duos d'Agents & Sessions Colab (`04:17`)

Tous les agents actifs du swarm tournent en **`gemini-4-hydrogen-medium`**.  
Board du swarm : `head = 97` (`read_board(since=97)` pour voir les nouveaux messages).

| Piste | Agents (`lead` / `exec`) | Session Colab (4/4) | Travail en cours à `04:17` | Journal & Scripts |
|---|---|---|---|---|
| **Techniques Initiales (`Base`)** | `base-lead` / `base-exec` | `track3_base` (**`A100`**) | Round 3 : 6 runs multi-seeds (`s101/s202/s303/s42` `AT3` + `ANCHOR` + `AT2`) + 8 soupes convexes (`ORIG` + multi-seed) + check `seed=777` | [03-unlearning-base-journal.md](03-unlearning-base-journal.md), `tracks/03_unlearning_v2/run_base_refine.py` |
| **Piste `I` (Erase-Then-Heal LoRA)** | `i-lead` / `i-lora` | `track3_i` (**`L4`**) | Round 4 calibré sur `#13` : Phase 1 plafonnée à `dTheta <= 4.3` + Phase 2 LoRA `r=8` (`I4R_1..3`) + soupes | [03-unlearning-track-i-journal.md](03-unlearning-track-i-journal.md), `tracks/03_unlearning_v2/methods/track_i.py`, `run_track_i.py` |
| **Piste `SCRUB+R`** | `scrub-lead` / `scrub-exec` | `track3_scrub` (**`L4`**) | Round 1 (`SR1`–`SR5` sur `seed=777`) : alternance `DO-MAX-EPOCH` / `DO-MIN-EPOCH` + **Rewind** + pondération Patch-Dropout `ce_pd` (sans drag KL facile) | [03-unlearning-scrub-journal.md](03-unlearning-scrub-journal.md), `tracks/03_unlearning_v2/methods/scrub_r.py`, `run_scrub_r.py` |
| **Piste `H` (Weight Tweaking)** | `h-lead` / `h-exec` | `track3_h` (**`T4`**) | Round 7 (`R7_1..3` + `R7_SOUP_50_r72_50_r73` sur `seed=777`) : relâcher le freinage de `R6_h1gov_ortho25` via `weight_norm_floor ∈ {0.55, 0.60, 1.0}` pour viser `mu+sig ∈ [3.65, 3.90]` | [03-unlearning-track-h-journal.md](03-unlearning-track-h-journal.md), `tracks/03_unlearning_v2/methods/track_h.py`, `run_track_h.py` |
| **Monitor** | `monitor` | — | Surveillance anti-blocage sur le board | — |

---

## 4. Protocole d'Évaluation Local Calibré (Commit `7f41ffa`)

Mis en place juste après `#15` et diffusé à `@all` (`#97`) pour les **prochains** rounds (ne pas interrompre les 4 runs en cours) :

1. **`tracks/03_unlearning_v2/selective_benchmark.py` :**
   - `score_single_fold` utilise désormais `_bell_forget_term` centré à `peak_ms = 3.84` (raw) et `4.40` (alt) au lieu d'une rampe monotone `/ 5.2` et `/ 6.2`.
2. **`tracks/03_unlearning_v2/eval_harness.py` :**
   - `compute_spearman_rank_corr(a, b)` pour calculer `rho_PD` sur le vrai checkpoint 3 500 images.
   - `score_server_calibrated_proxy(ce_mean, ce_std, ce_p50, proxy_kl, absent_kl, delta_l2, rho_pd=None)` sur le **vrai `.pt` final** (cloche `mu+sig=3.84`, `p50=1.0`, `ProxyKL`, `AbsentKL`, `dTheta <= 4.70`, bonus `rho_PD`).
3. **Note environnement :** `torch` n'est pas installé sur la machine locale ; `python3 -m py_compile` passe localement et les tests `tests/test_eval_harness.py` / `tests/test_selective_benchmark.py` s'exécutent sur Colab.

---

## 5. Plan d'Action Immédiat pour le Repreneur

1. **Vérifier les retours du board & GPUs :**
   - Lire les nouveaux messages du board (`read_board(since=97)` ou consulter les journaux par piste) et vérifier `colab sessions`.
   - S'assurer que chaque session terminée est bien coupée (`colab stop -s <session>`).
2. **Pour la soumission `#16` (dès `04:21:40+01:00`) :**
   - Inspecter les nouveaux `.pt` téléchargés par `base-exec` (Round 3 `track3_base`), `i-lora` (Round 4 `track3_i`), `scrub-exec` (Round 1 `track3_scrub`) ou `h-exec` (Round 7 `track3_h`).
   - Sélectionner le meilleur candidat vérifiant **`Full mu+sig ∈ [3.65, 3.90]`**, **`p50 ∈ [0.85, 1.15]`**, **`ProxyKL < 0.0558`** (ou **`dTheta < 4.48`** / meilleur `rho_PD` / `IntraAcc`) et `W_SEL` sain (`InterAcc >= 0.70`), puis soumettre via `curl`.
   - *(Fallback si aucun nouveau `.pt` n'est encore téléchargé à `04:21:40` : `SOUP_50_ce80kl28_50_ce90kl28.pt` ou `I3R_4_lora_r8_c095_160.pt` sont déjà dans `tracks/03_unlearning_v2/output/`.)*
3. **Mettre à jour & commiter [docs/tracks/03-unlearning-journal.md](03-unlearning-journal.md)** après chaque soumission et résultat majeur de round.
4. **Si les 4 pistes ont des gagnants complémentaires (ex: `AT3` multi-seed + `I4R` LoRA + `SR` SCRUB+R + `R7` H) :** tester une **soupe inter-pistes** dans `mu+sig ∈ [3.65, 3.90]`.

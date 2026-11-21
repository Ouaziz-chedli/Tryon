# Track 3 — Analyse Comparative des Benchmarks (`run_15901661` & `run_15901741`)

- **Benchmark v1 (`run_15901661`) :** [summary.md](../../tracks/03_unlearning_v2/results/run_15901661/summary.md) · [summary.json](../../tracks/03_unlearning_v2/results/run_15901661/summary.json)
- **Benchmark v2 (`run_15901741`) :** [summary.md](../../tracks/03_unlearning_v2/results/run_15901741/summary.md) · [summary.json](../../tracks/03_unlearning_v2/results/run_15901741/summary.json)
- **Benchmark v3 (`run_15901899`) :** [summary.md](../../tracks/03_unlearning_v2/results/run_15901899/summary.md) · [summary.json](../../tracks/03_unlearning_v2/results/run_15901899/summary.json) · checkpoints `.pt` (`327.71 Mo`, `valid=True`) sur JURECA dans `output/benchmark_v3/`
- **Benchmark v4 (`run_15902297`) :** [summary.md](../../tracks/03_unlearning_v2/results/run_15902297/summary.md) · [summary.json](../../tracks/03_unlearning_v2/results/run_15902297/summary.json) · checkpoints `.pt` dans `output/benchmark_v4/` (avec `Raw p50` et `Frac>95%`)
- **Ground-Truth Lab (`lab_cifar100`) :** [coherence_report.json](../../tracks/03_unlearning_v2/results/lab_cifar100/coherence_report.json)

---

## 1. Résultats Benchmark v2 (`run_15901741`, 14 runs sur `jrc0327`)

| Run | Raw CE (mean±std) | Raw Ent | Stress CE avg | Stress Agree avg | Proxy KL(T=2) | Absent KL(T=2) | Δθ L2 | Temps |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0_orig_baseline` | `0.000±0.00` | `0.002` | `1.267` | `0.801` | `0.0000` | `0.0000` | `0.00` | 34.6s |
| `A_salun_npo_lr1e4` | `0.754±2.31` | `0.163` | `6.917` | `0.482` | `0.1543` | `0.0719` | `3.09` | 55.0s |
| `A_salun_npo_lr2e4` | `1.394±3.03` | `0.292` | `7.368` | `0.384` | `0.1718` | `0.0828` | `4.07` | 48.1s |
| **`A_ablation_no_aug`** | **`0.701±2.51`** | **`0.160`** | **`2.224`** | **`0.657`** | **`0.0403`** | **`0.0239`** | **`3.06`** | 47.8s |
| `B_bad_teacher_T6` | `0.139±0.25` | `0.513` | `0.795` | `0.796` | `0.1200` | `0.0822` | `8.85` | 58.4s |
| `C_ssd_t1.0_a0.5` | `0.079±0.51` | `0.048` | `1.447` | `0.769` | `0.0014` | `0.0000` | `2.71` | 37.7s |
| `C_ssd_t1.2_a0.3` | `0.083±0.54` | `0.046` | `1.450` | `0.770` | `0.0014` | `0.0000` | `2.58` | 37.5s |
| `D_proxy_mixup` | `0.602±1.98` | `0.171` | `5.989` | `0.512` | `1.2666` | `0.5577` | `3.12` | 55.4s |
| `D_proxy_noise` | `1.815±3.51` | `0.320` | `7.202` | `0.352` | `3.7299` | `0.8769` | `2.68` | 49.3s |
| `E_lora_r8_lr1e4` | `0.532±2.09` | `0.112` | `6.355` | `0.533` | `0.1247` | `0.0398` | `4.57` | 47.9s |
| `F_abliteration_a0.15` | `0.000±0.00` | `0.003` | `1.282` | `0.796` | `0.0042` | `0.0001` | `1.67` | 34.6s |
| `F_abliteration_a0.25` | `0.001±0.00` | `0.003` | `1.298` | `0.792` | `0.0115` | `0.0001` | `2.79` | 34.5s |
| `H_ssd_plus_npo` | `0.614±2.13` | `0.136` | `4.756` | `0.552` | `0.0989` | `0.0538` | `3.90` | 55.0s |
| `H_ssd_plus_lora` | `1.167±2.97` | `0.180` | `7.088` | `0.468` | `0.1329` | `0.0429` | `5.45` | 51.6s |

---

## 2. Analyse par famille (Positifs vs Négatifs)

1. **Stratégie D (Comparaison des proxys `retain`) :**
   - `PatchMix` (`0.1543` Proxy KL) bat `MixUp` (`1.2666`, ×8 pire) et `Structured Noise` (`3.7299`, ×24 pire). *Raison :* PatchMix préserve les textures locales de patchs ViT 16×16 alors que MixUp/Noise détruisent la statistique d'image naturelle et n'ancrent plus `model.pt`.

2. **Stratégie A + Impact des augmentations (`A_ablation_no_aug` vs `A_salun_npo_lr1e4`) :**
   - `A_ablation_no_aug` tombe exactement dans toutes les zones cibles saines (`Raw CE = 0.701`, `Ent = 0.160`, `Stress CE = 2.224`, `Stress Agree = 0.657`, `Proxy KL = 0.0403`, `Absent KL = 0.0239`, `Δθ = 3.06`).
   - En revanche, `use_strong_aug=True` (qui incluait `RandomGrayscale` où `orig` a déjà `CE = 6.037` et `RandomResizedCrop(0.5)`) pousse `Stress CE` à `6.917` et dégrade `Proxy KL` à `0.1543`. **Piste immédiate :** utiliser des augmentations **modérées** (`crop 0.75–1.0`, `hflip`, léger `ColorJitter`, léger `Blur`, sans `grayscale` ni rotation forte).

3. **Stratégie E (`LoRA FP32`) & H (`SSD + NPO` hybride) :**
   - À oubli comparable, `E_lora_r8_lr1e4` réduit `Absent KL` de **45 %** par rapport à `A_salun_npo_lr1e4` (`0.0398` vs `0.0719`) et réduit `Proxy KL` (`0.1247` vs `0.1543`).
   - `H_ssd_plus_npo` bat `A_salun_npo_lr1e4` sur **toutes** les métriques `retain` (`Proxy KL = 0.0989`, `Absent KL = 0.0538`, `Stress CE = 4.756`).

4. **Stratégie B (`Bad-Teacher`) :**
   - Profil très différent des autres : `Raw Ent = 0.513` avec une dispersion minuscule (`Raw CE = 0.139±0.25` vs `±2.31` pour NPO !) et `Stress Agree = 0.796`. Son défaut actuel (`Proxy KL = 0.1200`, `Δθ = 8.85`) vient d'un `lr=2e-4` sans contrainte LoRA ni poids `retain_kl` plus élevé.

5. **Stratégies C (`SSD`) & F (`Abliteration`) :**
   - `C_ssd` garde `Absent KL = 0.0000` et `Proxy KL = 0.0014` (`Raw CE = 0.08`), confirmant son rôle idéal de **pré-conditionneur** (`H_ssd_plus_*`).
   - `F_abliteration` à `rank=4, α=0.15–0.25` est trop faible seul (`Raw CE = 0.001`).

---

## 3. Résultats Benchmark v3 (`run_15901899` — Augmentations modérées & Checkpoints validés)

| Run | Raw CE (mean±std) | Raw Ent | Stress CE avg | Stress Agree avg | Proxy KL(T=2) | Proxy Agree | Absent KL(T=2) | Δθ L2 | Statut `.pt` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0_orig_baseline` | `0.000±0.00` | `0.002` | `1.267` | `0.801` | `0.0000` | `1.0000` | `0.0000` | `0.00` | `327.71 Mo` |
| **`H_ssd_lora_mod_aug`** | **`1.112±3.79`** | `0.060` | **`2.458`** | **`0.705`** | **`0.0184`** | **`0.9249`** | **`0.0106`** | `4.25` | **`valid=True`** |
| **`A_npo_mod_aug`** | **`0.617±1.99`** | **`0.143`** | **`2.292`** | **`0.628`** | **`0.0366`** | `0.8860` | `0.0277` | **`2.97`** | **`valid=True`** |
| `A_npo_no_aug` | `0.701±2.51` | `0.160` | `2.224` | `0.657` | `0.0403` | — | `0.0239` | `3.06` | `valid=True` |
| `E_lora_r8_mod_aug` | `1.721±4.15` | `0.113` | `3.480` | `0.577` | `0.0334` | `0.8886` | `0.0199` | `4.90` | `valid=True` |
| `B_bad_teacher_kl4` | `0.560±0.76` | `1.124` | `1.751` | `0.560` | `0.1512` | `0.7809` | `0.0997` | `7.59` | `valid=True` |

### Enseignements & Classement pour soumission serveur
1. **Validation des augmentations modérées :** supprimer `RandomGrayscale` et `Rotate15` du training divise `Proxy KL(T=2)` par **4× à 7×** (`0.154 → 0.0366` sur `A`, `0.133 → 0.0184` sur `H`), tout en transférant l'oubli aux copies altérées (`Stress CE = 2.29–2.46`).
2. **Candidat #1 — `H_ssd_lora_mod_aug.pt` :** meilleur compromis global (`Proxy KL = 0.0184`, `Proxy Agree = 92.49%`, `Absent KL = 0.0106`, `Stress CE = 2.458`).
3. **Candidat #2 — `A_npo_mod_aug.pt` :** profil complémentaire plus doux et plus homogène (`Raw CE = 0.617±1.99`, `Raw Ent = 0.143`, `Δθ = 2.97`).
4. **Prochaine étape :** renseigner `API_KEY` dans `submission.py`, soumettre `H_ssd_lora_mod_aug.pt` puis `A_npo_mod_aug.pt` pour calibrer nos seuils locaux contre le vrai score serveur (`forget_score`, `retain_score`, `val_acc`).

---

## 4. Résultats Benchmark v4 (`run_15902297` — Oubli homogène & métriques `p50` / `Frac>95%`)

| Run | Raw CE (mean±std) | Raw p50 | Frac>95% | Raw Ent | Stress CE avg | Stress Agree avg | Proxy KL(T=2) | Absent KL(T=2) | Δθ L2 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0_orig_baseline` | `0.000±0.00` | `0.000` | `1.00` | `0.002` | `1.267` | `0.801` | `0.0000` | `0.0000` | `0.00` |
| `E_lora_bt_T3` | `0.013±0.10` | `0.000` | `0.96` | `0.036` | `1.222` | `0.765` | `0.0035` | `0.0017` | `4.18` |
| `E_lora_bt_T4` | `0.053±0.17` | `0.007` | `0.81` | `0.171` | `1.232` | `0.721` | `0.0116` | `0.0069` | `6.82` |
| **`E_lora_bt_T5`** | **`0.169±0.33`** | **`0.046`** | **`0.52`** | **`0.507`** | **`1.313`** | **`0.676`** | **`0.0189`** | **`0.0119`** | `9.48` |
| `H_ssd_lora_bt_T4` | `0.053±0.18` | `0.006` | `0.82` | `0.170` | `1.223` | `0.722` | `0.0120` | `0.0068` | `7.20` |
| `A_npo_gentle_ce0.45` | `0.011±0.19` | `0.000` | `0.98` | `0.016` | `1.403` | `0.770` | `0.0036` | `0.0021` | `1.30` |
| `H_ssd_npo_gentle` | `0.556±2.19` | `0.000` | `0.93` | `0.057` | `1.928` | `0.730` | `0.0065` | `0.0040` | `3.05` |

### Enseignements clés
1. **Validation des nouvelles métriques (`p50` & `Frac>95%`) :** `H_ssd_npo_gentle` semblait avoir `CE = 0.556`, mais `p50 = 0.000` et `Frac>95% = 0.93` prouvent qu'il laisse **93 %** des images intactes (bimodal).
2. **`E_lora_bt_T5` (`LoRA + Bad-Teacher T=5`) :** oubli **vraiment homogène** (`std = 0.33`, `Frac>95%` divisé par 2 à `0.52`, `Raw Ent = 0.507`) tout en gardant `Proxy KL = 0.0189` et `Absent KL = 0.0119`. Candidat prioritaire pour la soumission de `19h12` !

---

## 5. Calibration Serveur & Ground-Truth Lab (`lab_cifar100` sur Colab A100)

### 5.1 Scores réels obtenus sur le serveur (`31-unlearning`)

| Soumission | Méthode | `Raw CE` | `Raw Ent` | `Proxy KL(T=2)` | `Δθ L2` | **Score Serveur** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| #2 | **`A_npo_mod_aug.pt`** | **`0.617±1.99`** | `0.143` | `0.0366` | **`2.97`** | **`0.0843`** *(best)* |
| #1 | `H_ssd_lora_mod_aug.pt` | `1.112±3.79` | `0.060` | `0.0184` | `4.25` | `0.0287` |
| #3 | `E_lora_bt_T5.pt` | `0.169±0.33` | `0.507` | `0.0189` | `9.48` | `0.0117` |

### 5.2 Tests de cohérence du Lab (`lab_cifar100`)
- **Essai 1 (`200 imgs/classe`, `blocks.10..11`, 5 ép.) :** `orig_lab val_acc = 0.3727` ; `A_npo_mod_aug` (`forget=0.3579`, `retain=0.3991`) **>>** `H_ssd_lora` (`0.0079`, `0.0088`).
- **Essai 2 (`350 imgs/classe`, `blocks.6..11`, 6 ép., Colab A100) :** [lab_cifar100_v2/coherence_report.json](../../tracks/03_unlearning_v2/results/lab_cifar100_v2/coherence_report.json)
  - `orig_lab val_acc = 0.4772`, `gold_oracle val_acc = 0.4576` (encore sous `0.70` car `blocks.0..5` gelés sur `32×32` CIFAR-100).
  - Classement toujours cohérent avec le serveur : **`A_npo_mod_aug`** (`forget=0.3232`, `retain=0.3119`) **>>** **`H_ssd_lora_mod_aug`** (`forget=0.0052`, `retain=0.0327`).
- **Essai 3 (`clane9/imagenet-100`, `blocks.0..11`, 5 ép., Colab A100) :** [lab_in100/coherence_report.json](../../tracks/03_unlearning_v2/results/lab_in100/coherence_report.json)
  - `orig_lab val_acc = 0.3412`, `gold_oracle val_acc = 0.3288` (5 époques sans augmentation ni `lr` élevé sur `head` sont insuffisantes).
  - Classement toujours confirmé : **`A_npo_mod_aug`** (`forget=0.2755`, `retain=0.7131`, `pred=0.7551`) **>>** **`H_ssd_lora_mod_aug`** (`forget=0.0173`, `retain=0.0390`).
  - *Suite :* passer à `10 époques` avec `head lr=1e-3` / `backbone lr=5e-5` + augmentations pour dépasser `val_acc >= 85%`, puis lancer nos sous-agents par piste.

# Track 3 — Analyse Comparative des Benchmarks (`run_15901661` & `run_15901741`)

- **Benchmark v1 (`run_15901661`) :** [summary.md](../../tracks/03_unlearning_v2/results/run_15901661/summary.md) · [summary.json](../../tracks/03_unlearning_v2/results/run_15901661/summary.json)
- **Benchmark v2 (`run_15901741`) :** [summary.md](../../tracks/03_unlearning_v2/results/run_15901741/summary.md) · [summary.json](../../tracks/03_unlearning_v2/results/run_15901741/summary.json)

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

## 3. Pistes d'amélioration immédiates (avant 1ʳᵉ soumission)
1. Tester **augmentations modérées** (sans `grayscale` ni `rotate15`) sur `A_salun_npo`, `E_lora_r8`, `H_ssd_plus_lora` et `B_bad_teacher` (avec `retain_kl_weight = 4.0` ou LoRA sur B).
2. Sélectionner les **2–3 meilleurs profils distincts** (ex. `A_no_aug` / `mod_aug`, `H_ssd_plus_lora_mod_aug`, `B_bad_teacher_tuned`) pour nos premières soumissions réelles afin de calibrer nos seuils locaux contre le vrai score serveur.

# Inventaire des clusters — Team TRYON (2026-11-22 07h30)

> Testé en direct (ssh BatchMode / sbatch réels / sessions créées). Doctrine Chef : tout GPU libre hors réservation ; ne jamais faire la queue dans `cispahack` ; canal réseau unique via `scripts/jureca_xfer.py` (TOTP + dérive horloge WSL2 −10 s, marge 20 s).

## Flotte opérationnelle

### 1. JURECA — `jureca05.fz-juelich.de` (MOTEUR PRINCIPAL)
- **Accès** : `ssh ouaziz1@jureca05.fz-juelich.de`, clé `infra/juelich_ed25519`, TOTP via `scripts/jureca_xfer.py`. Alias round-robin `jureca.fz-juelich.de` À ÉVITER (nœuds cassés) ; valides : `jureca03/04/05/06/08`.
- **Compte projet** : `training2666` (groupes confirmés). Workspace : `$HOME/tryon_ws/unl` (venv `torch 2.15.0+cu132` + `timm 1.0.31`, `model.pt` + `forget_images.npz` sur place ; quota disque ~20 Go — nettoyé 07h02).
- **Partitions VALIDES (associations vérifiées)** : `dc-gpu` (178 nœuds × 4 GPU), `dc-gpu-devel` (12 nœuds, sature vite), `dc-gpu-large` (178), **`dc-gpu-devel-sw` (jamais essayée — réserve à explorer)**.
- **FERMÉES à notre compte** : `dc-hwai` (Invalid account/partition), `dc-cpu*` (pas dans nos associations — le CPU farm est inaccessible), partitions `eval-*`.
- **Réservation `cispahack`** (120 GPU) : doctrine = ne pas y faire la queue.

### 2. Google Colab Pro (DÉMARRAGE IMMÉDIAT, ZÉRO FILE)
- **3 sessions max simultanées** (règle Chef), solde ~335 CU. Types : A100 (6,73 CU/h), L4 (1,94), T4.
- **Contrainte dure** : upload de `.pt` vers Colab IMPOSSIBLE (erreur 500, limite <1 Mo par fichier API Jupyter) — Colab entraîne depuis HuggingFace et **exporte** ses `.pt` par `colab download` (le retour marche).
- CLI : `/home/chedli/.local/bin/colab` (guide `infra/colab-agent-guide.md` : heartbeat, tarball sans .pt, `colab stop` systématique).

### 3. Oracle Cloud ×3 — joignables mais HORS compute
| Hôte | ssh | Specs | Verdict |
|---|---|---|---|
| `oracle-arm` (141.253.110.210) | ✅ ubuntu | 2 vCPU, 11 Go RAM, pas de GPU, pas de torch | trop faible pour ViT (entraînement : jours/éval : heures) |
| `oracle-x86-01` (158.178.204.123) | ✅ ubuntu | 2 vCPU, pas de GPU, pas de torch | idem |
| `oracle-x86-02` (88.96.62.91) | ✅ ubuntu | 2 vCPU, pas de GPU, pas de torch | idem |

### 4. Inaccessibles
`JUWELS`, `JUDAC`, `DEEP` (aucune config/clé locale), `sangkhavongs1` (pas de clé/MFA locales — lui seul peut se configurer).

## Jobs en vol au 07h30 (répartition flotte)
- **JURECA dc-gpu (file)** : `15905327_` rerun 20 configs (driver md5 d2120b4e : rétention in_window), `15905326` diamant (redo frais), `15905334` Phase3 (10×2-vues), `15905335_` SCRUB+R2 (7 configs).
- **Colab** : `track3_p3` (A100) + `track3_sr2` (A100) créées, tarball v4 uploadé — execs en lancement.
- **En attente de dispatch** : `LabKL` (pack 11 `.pt` en transfert), `HeadSurgery` (2 poids d'entrée en transfert), `SoupLab` (après rerun), sondes A/B (après diff SweepMax).

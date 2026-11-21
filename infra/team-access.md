# Accès Équipe — Cluster GPU Jülich (JURECA)

> **Usage strictement limité au hackathon (repo privé).** Révoquer/rotater la clé et le mot de passe après l'événement.

## 1. Identifiants JuDoor (`https://judoor.fz-juelich.de/login`)
- **Username JSC :** `ouaziz1`
- **Password :** `UV84vqwmdEGfxS`
- **Projet :** `training2666`
- **Secret TOTP (MFA) :** `XYOFKN26KL2RATEUZJ7D7WDPKRE4S2ZH2PB3GZTHDOYRZ2MPPWSMH34POGTKXVQJ74SNAV6QCQUC4WJL`
  - Générer un code TOTP à la volée : `python3 scripts/totp.py`
- **Codes de secours (Reset codes) :**
  `5mijun35c9o4bwyy`, `becomzld64h95sw5`, `zog79ddf1r9neswd`, `5l70ohtt1wxf34po`, `kqf0x0zwsle69ddp`, `0105kfcplajf8cg5`, `ntxy6bvl6l7bq0kz`, `vj06t6j7qnckt1tf`, `hf1do3lmtw8uki9z`, `esf5t9hm24iudk80`

## 2. Clé SSH du projet
- Clé privée : [juelich_ed25519](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/infra/juelich_ed25519)
- Clé publique : [juelich_ed25519.pub](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/infra/juelich_ed25519.pub)

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINgpkENIqGxARS4M8daoImb8MFdslgLd6AwOobB06+HU juelich-hackathon-2026
```

## 3. Connexion SSH (pour chaque teammate)
```bash
chmod 600 infra/juelich_ed25519
python3 scripts/totp.py   # récupérer le code à 6 chiffres
ssh -i infra/juelich_ed25519 -o MACs=hmac-sha2-256-etm@openssh.com ouaziz1@jureca.fz-juelich.de
```

Ou dans `~/.ssh/config` (avec `ControlMaster` pour ne saisir le TOTP qu'une fois) :
```sshconfig
Host juelich
  Hostname jureca.fz-juelich.de
  User ouaziz1
  IdentityFile /chemin/vers/European_AI_Championship/infra/juelich_ed25519
  MACs hmac-sha2-256-etm@openssh.com
  ControlMaster auto
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist 12h
```

> **Note IP :** JuDoor exige une clause `from="..."` sur la clé publique SSH. Si un teammate se connecte depuis une autre IP publique sur le compte `ouaziz1`, ajouter son IP dans la clause `from="..."` sur JuDoor (`Manage SSH-keys` sur JURECA).

---

## 4. Accès individuel des coéquipiers (`sangkhavongs1`, `bassim1`)

Les droits `setfacl` (`rwx` récursif + `default:rwx`) sont déjà activés pour `sangkhavongs1` et `bassim1` sur `/p/scratch/training2666/ouaziz1` et tout le dossier `TRYON`.

Chaque coéquipier depuis **son propre compte JuDoor** (`https://judoor.fz-juelich.de/`) doit :
1. Activer sa **MFA (TOTP)** personnelle sur JuDoor.
2. Signer les **Usage Agreements** (`jureca` et `judac`).
3. Uploader **sa propre clé SSH publique** (`~/.ssh/id_ed25519.pub`) avec la clause `from="..."` proposée par JuDoor pour son IP.
4. Configurer `~/.ssh/config` en local :
   ```sshconfig
   Host juelich
     Hostname jureca.fz-juelich.de
     User <sangkhavongs1 ou bassim1>
     IdentityFile ~/.ssh/<sa_cle_ed25519>
     MACs hmac-sha2-256-etm@openssh.com
     ControlMaster auto
     ControlPath ~/.ssh/cm-%r@%h:%p
     ControlPersist 12h
   ```

---

## 5. Chemins partagés `TRYON` & environnements Python (ne pas utiliser `$USER`)

Tout le workspace est dans le scratch du team lead (`ouaziz1`) :

| Ressource | Chemin absolu sur JURECA |
| :--- | :--- |
| **Racine équipe `TRYON`** | `/p/scratch/training2666/ouaziz1/TRYON` |
| **Track 1 — Watermark** | `/p/scratch/training2666/ouaziz1/TRYON/Hackathon_Watermark_Localisation_Task` |
| **Track 2 — Model Equality** | `/p/scratch/training2666/ouaziz1/TRYON/model_equality` |
| **Track 3 — Unlearning** | `/p/scratch/training2666/ouaziz1/TRYON/unlearning` |

Activation des environnements Python déjà installés (inutile de recréer un `.venv`) :
```bash
# Track 3 (Unlearning)
source /p/scratch/training2666/ouaziz1/TRYON/unlearning/.venv/bin/activate

# Track 2 (Model Equality)
source /p/scratch/training2666/ouaziz1/TRYON/model_equality/.venv/bin/activate
```

---

## 6. Template `sbatch` prêt à l'emploi (`dc-gpu`, A100)

```bash
#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=/p/scratch/training2666/ouaziz1/TRYON/unlearning/output/%u_%j.out

set -euo pipefail

TASK_DIR="/p/scratch/training2666/ouaziz1/TRYON/unlearning"
cd "$TASK_DIR"
source "$TASK_DIR/.venv/bin/activate"

PYTHONPATH="$TASK_DIR" srun python mon_script.py
```

---

## 7. Règles d'équipe & garde-fous

1. **Ne pas écraser les fichiers communs** (`.venv/`, `model.pt`, `forget_images.npz`, `task_template.py`, `submission.py`) — travailler dans son propre sous-dossier (ex. `work_sang/`, `work_bassim/`) et préfixer les logs Slurm avec `%u_%j.out`.
2. **Ne pas exécuter `teammate.sh` officiel tel quel** : il contient le bug `PROJECT_NAME="training2625"` au lieu de `training2666` (cf. [gpu-juelich-setup.md](../docs/infra/gpu-juelich-setup.md)).
3. **Track 2 (`model_equality`)** : ne jamais consommer de requêtes sur `ep-01` à `ep-33` (budget strict de **100 réponses/endpoint**) avant validation sur `ref-01` à `ref-09`.

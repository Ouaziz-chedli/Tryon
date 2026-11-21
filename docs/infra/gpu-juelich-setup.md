# Setup GPU Jülich (JURECA)

Source : [GPU-usage-paris.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/infra/GPU-usage-paris.pdf)

## État actuel — Team `TRYON` (déjà configuré)

> **Tout le setup initial est terminé.** Les teammates partagent le compte `ouaziz1` (voir [infra/team-access.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/infra/team-access.md) et `python3 scripts/totp.py`).

- **Compte Team Lead :** `ouaziz1` · **Coéquipiers :** `sangkhavongs1`, `bassim1` · **Projet :** `training2666` · **Login node testé :** `jrlogin06.jureca`
- **JuDoor :** MFA TOTP activé, Usage Agreements `jureca` et `judac` signés, clé SSH `infra/juelich_ed25519.pub` propagée et active.
- **Dossier partagé sur JURECA :** `/p/scratch/training2666/ouaziz1/TRYON/`
  - `model_equality/` — téléchargé + `.venv/` prêt (`scikit-learn`, `numpy`, `pandas`, `scipy`, `requests`).
  - `unlearning/` — téléchargé (`model.pt`, `forget_images.npz`) + `.venv/` prêt (`torch 2.15.0`, `torchvision 0.29.2`, `timm 1.0.31`, CUDA 13).
  - `Hackathon_Watermark_Localisation_Task/` — téléchargé (`README.txt`, `starter_kit.zip`, `train.zip`, `test.zip`) ; **à faire :** dézipper les archives et créer le `.venv/` si nécessaire.
  - `.uv/` et `.cache/` — caches partagés configurés dans `~/.bashrc`.

---

## 1. Ressources disponibles
- **40 compute nodes** (1 nœud réservé par job) :
  - `4× NVIDIA A100` (40 GB HBM2e chacun)
  - `2× AMD EPYC 7742` (128 cœurs physiques, 2.25 GHz)
  - `512 GB DDR4`
- **12 login nodes** (accès initial et dev) : `2× NVIDIA Quadro RTX8000`, `1024 GB DDR4`.

---

## 2. Pré-requis sur JuDoor (`judoor.fz-juelich.de/login`)
1. Créer / activer son compte sur `judoor.fz-juelich.de/login`.
2. Activer la **MFA (TOTP)** (Google Authenticator / FreeOTP / etc.).
3. Signer le **Usage Agreement** pour `JURECA` (et `JUDAC` si affiché).
4. **Uploader la clé SSH publique (`ed25519` obligatoire, `id_rsa` interdit)** avec une clause `from="<IP_RANGE>"` :
   - Clé locale disponible : `~/.ssh/id_ed25519.pub`
   - IP publique actuelle détectée : `78.112.61.231` (ou plage IP si IP dynamique).
   - Attendre jusqu'à **15 minutes** après l'upload pour propagation.

---

## 3. Configuration SSH locale (`~/.ssh/config`)
```sshconfig
Host juelich
  Hostname jureca.fz-juelich.de
  User <TON_USERNAME_JSC>
  IdentityFile ~/.ssh/id_ed25519
  MACs hmac-sha2-256-etm@openssh.com
```
> À la connexion, un **JSC TOTP Verification code** est demandé (application d'authentification TOTP).

---

## 4. Setup environnement sur JURECA (après connexion)

### Activer le projet & dossier utilisateur
```bash
jutil env activate -p training2614   # ou l'ID projet affiché dans JuDoor (ex: training2666 dans main.sh)
cd $PROJECT
mkdir $USER
chown $USER:$USER $USER
chmod o-r $USER
```

### Installer `uv` et rediriger le cache hors `$HOME`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cat << EOF >> ~/.bashrc
export UV_PYTHON_INSTALL_DIR=$PROJECT/$USER/uv/python
export UV_CACHE_DIR=$PROJECT/$USER/uv/cache
export UV_TOOL_DIR=$PROJECT/$USER/uv/tools
EOF
exec bash
```

### Cloner et lancer le setup officiel du hackathon (`SprintML/hackathon`)

> **Important :** Le **team lead** exécute `hackathon_setup.sh` **en premier** (une seule fois), puis chaque teammate exécute `teammate.sh` depuis son propre compte JURECA.

1. **Team lead (une seule fois) :**
```bash
mkdir -p ~/hackathon-scripts && cd ~/hackathon-scripts
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/hackathon_setup.sh -O hackathon_setup.sh
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh -O teammate.sh
# Éditer TEAM_LEAD, TEAMMATE_1..3 et TEAM_FOLDER dans hackathon_setup.sh
source hackathon_setup.sh
```
Le dossier partagé est créé dans `/p/scratch/training2666/<team-lead>/<team-folder>/` (contient les 3 datasets, `.venv/` par tâche, `.uv/` et `.cache/`).

2. **Teammates (`sangkhavongs1`, `bassim1` — depuis leur propre compte JURECA) :**
```bash
mkdir -p ~/hackathon-scripts && cd ~/hackathon-scripts
wget https://huggingface.co/datasets/SprintML/hackathon/resolve/main/teammate.sh -O teammate.sh
sed -i 's/^TEAM_LEAD=.*/TEAM_LEAD="ouaziz1"/' teammate.sh
sed -i 's/^TEAM_FOLDER=.*/TEAM_FOLDER="TRYON"/' teammate.sh
sed -i 's/PROJECT_NAME="training2625"/PROJECT_NAME="training2666"/' teammate.sh
source teammate.sh
```

3. **À chaque nouvelle connexion :**
```bash
jutil env activate -p training2666
source ~/.bashrc
cd /p/scratch/training2666/<team-lead>/<team-folder>/<dataset-name>
source .venv/bin/activate
```

---

## 5. Lancer un job GPU via Slurm (`main.sh`)
```bash
#!/bin/bash
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=128
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --output=output/%j.out

cd $PROJECT/$USER/Hackathon
source .venv/bin/activate
srun python sample_submission.py
```
Soumission : `sbatch main.sh` puis suivi des logs via `cat output/<job_id>.out`.

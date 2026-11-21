# Setup GPU Jülich (JURECA)

Source : [GPU-usage-paris.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/infra/GPU-usage-paris.pdf)

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

### Cloner le repo de setup du hackathon
```bash
git clone https://huggingface.co/datasets/SprintML/hackathon
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

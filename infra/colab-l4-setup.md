# Guide Google Colab Pro (GPU L4) via CLI

Ce guide permet d'utiliser **Google Colab Pro** sur un GPU **L4 (24 Go VRAM)** directement depuis un terminal Linux/macOS/WSL2 grâce à l'outil officiel `google-colab-cli`, sans passer par le navigateur ni attendre dans la file `PD` de JURECA.

> **Coût & autonomie observés :** une session **L4** consomme **`1.94 CU/h`** (soit **~205 heures** pour `400 Compute Units`). C'est le meilleur compromis prix/vitesse pour le ViT-B/16 (~3–4 Go VRAM utilisés).

---

## 1. Installation (avec `uv`)

```bash
uv tool install --with keyrings.alt google-colab-cli
```
*(Sous WSL2 / Linux sans keyring graphique, `--with keyrings.alt` est indispensable pour que le token OAuth soit sauvegardé.)*

Vérifier que `colab` est accessible :
```bash
export PATH="$HOME/.local/bin:$PATH"
colab version
```

---

## 2. Authentification Google (1 seule fois)

```bash
colab usage
```
1. Cliquer sur le lien OAuth2 affiché, valider avec le compte Google ayant **Colab Pro**.
2. Coller le code d'autorisation retourné dans le terminal.
3. Vérifier l'affichage du solde (`Current balance: ... compute units`).

---

## 3. Cycle de vie d'une session GPU L4

```bash
# Créer une session L4 nommée (ex. track3)
colab new -s track3 --gpu L4

# Vérifier l'état et les fichiers
colab status -s track3
colab ls -s track3 /content

# Envoyer un fichier / archive vers la VM
colab upload -s track3 local_archive.tar.gz /tmp/local_archive.tar.gz

# Exécuter un script Python (avec timeout large pour les runs longs)
colab exec -s track3 --timeout 900 -f mon_script.py

# Récupérer un résultat en local
colab download -s track3 /content/unlearning/output/summary.json ./summary.json

# IMPORTANT : arrêter la VM dès que c'est fini pour stopper la conso (1.94 CU/h)
colab stop -s track3
```

---

## 4. Setup rapide Track 3 sur Colab L4

Depuis la session `track3` (via `colab exec -s track3 --timeout 900`), télécharger les poids/données via `huggingface_hub` (ne pas compter sur `git clone` seul à cause de Git LFS) :

```python
import os, subprocess
from huggingface_hub import hf_hub_download

subprocess.check_call(["bash", "-lc", "git clone https://huggingface.co/datasets/SprintML/unlearning /content/unlearning && pip install -q timm datasets pyarrow"])
for fname in ("model.pt", "forget_images.npz"):
    hf_hub_download(repo_id="SprintML/unlearning", repo_type="dataset", filename=fname, local_dir="/content/unlearning")
```

> **Note importante :** Les sessions Colab sont éphémères. Toujours récupérer les `.pt` et `.json` avec `colab download` (ou les copier vers Google Drive / JURECA) **avant** d'appeler `colab stop`.

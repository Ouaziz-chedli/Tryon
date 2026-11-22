# Guide Agent — Utiliser Google Colab Pro (`colab` CLI) sans pièges

Ce guide est destiné aux **agents IA** travaillant sur ce repo pour exécuter des runs GPU (`A100` / `L4`) via `/home/chedli/.local/bin/colab` sans perdre de Compute Units ni tomber dans les pièges réseau du CLI.

*(Voir aussi le guide d'installation initial : [colab-l4-setup.md](colab-l4-setup.md).)*

---

## 1. Commandes de base & Quotas

- **Binaire exact :** `/home/chedli/.local/bin/colab` (l'OAuth2 est déjà authentifié sur la machine).
- **Vérifier le solde et les sessions actives AVANT toute création :**
  ```bash
  /home/chedli/.local/bin/colab usage
  /home/chedli/.local/bin/colab sessions
  ```
- **Tarifs indicatifs :** `L4` (`~1.94 CU/h`), `A100` (`~6.73 CU/h`).
- **Règle de nommage multi-agents :** choisir un nom de session **unique et dédié** (`-s track3_h`, `-s track3_i`, `-s track3_base`, `-s track1_...`). Ne jamais toucher ni stopper la session d'un autre agent.

---

## 2. Créer une session & Gérer les refus (`503` / Quota)

```bash
/home/chedli/.local/bin/colab new -s mon_run --gpu A100
```
- **Si `Allocation refused (TooManyAssignmentsError)` :** le plafond simultané de GPU sur le compte est atteint. Vérifier `colab sessions` et attendre qu'un autre agent appelle `colab stop`.
- **Si `503 Service Unavailable` sur `A100` :** réessayer une fois ou basculer sur `--gpu L4` (`colab new -s mon_run --gpu L4`), qui suffit largement pour ViT-B/16 (VRAM utilisée `~4 Go` sur `24 Go`).

---

## 3. Envoyer le code proprement (Règle critique `< 1 Mo`)

> **Piège n°1 — `500 Server Error` sur `colab upload` :**
> 1. Ne **jamais** inclure `output/`, `results/`, `*.pt`, `*.npz` ou `__pycache__/` dans l'archive (un `.tar.gz` de `300 Mo` fait planter l'API Jupyter `/api/contents/`).
> 2. Uploader vers `/content/...` (et non `/tmp/...` qui peut échouer selon le montage Jupyter).

```bash
tar --exclude='output' --exclude='results' --exclude='*.pt' --exclude='*.npz' --exclude='__pycache__' \
    -czf /tmp/harness_work.tar.gz -C tracks/03_unlearning_v2 .
/home/chedli/.local/bin/colab upload -s mon_run /tmp/harness_work.tar.gz /content/harness_work.tar.gz
```

Depuis la VM Colab, télécharger les poids/données publics directement depuis HuggingFace (beaucoup plus rapide que d'uploader depuis le PC local) :
```python
from huggingface_hub import hf_hub_download
for fname in ("model.pt", "forget_images.npz"):
    hf_hub_download(repo_id="SprintML/unlearning", repo_type="dataset", filename=fname, local_dir="/content/work")
```

---

## 4. Exécuter un script long sans coupure WebSocket (`Heartbeat` obligatoire)

> **Piège n°2 — `RuntimeError: Connection was lost.` :**
> Si votre script reste silencieux plus de ~60–90 secondes pendant un entraînement GPU, le proxy WebSocket de Colab coupe la connexion de `colab exec`.

**Pattern obligatoire pour tout `colab exec -s mon_run --timeout 1800` :** lancer le script enfant avec `python -u` et imprimer un point `.` toutes les `15 s` via `select.select` :

```python
import os, select, subprocess, sys

os.chdir("/content/work")
env = {**os.environ, "PYTHONPATH": "/content/work:/content/work/harness"}
subprocess.run([sys.executable, "harness/tests/mon_test.py"], env=env, check=True)

proc = subprocess.Popen(
    [sys.executable, "-u", "harness/mon_script.py"],
    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)
while True:
    ready, _, _ = select.select([proc.stdout], [], [], 15.0)
    if ready:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            continue
        print(line, end="", flush=True)
    else:
        print(".", end="", flush=True)
        if proc.poll() is not None:
            break
if proc.wait() != 0:
    raise SystemExit(proc.returncode)
```

---

## 5. Reprise après déconnexion ou redémarrage local

Si votre client local est coupé ou redémarré pendant un run : **la VM Colab continue de calculer à distance !**
Ne jamais lancer `colab stop` ou recréer la session à l'aveugle :
1. Vérifier l'état distant :
   ```bash
   /home/chedli/.local/bin/colab status -s mon_run
   /home/chedli/.local/bin/colab ls -s mon_run /content/work/output
   ```
2. Tant que `Status: BUSY`, attendre (ou poller avec un timer `schedule`, jamais `sleep`) jusqu'à `Status: IDLE` ou l'apparition du `.json` final, puis télécharger normalement.

---

## 6. Rapatriement & Arrêt obligatoire (`0 CU/h`)

Les disques Colab sont **éphémères** : tout est détruit à l'arrêt de la VM.
Toujours exécuter dans cet ordre exact à la fin de chaque run :

```bash
/home/chedli/.local/bin/colab download -s mon_run /content/work/output/results.json ./output/results.json
/home/chedli/.local/bin/colab download -s mon_run /content/work/output/best_model.pt ./output/best_model.pt
/home/chedli/.local/bin/colab stop -s mon_run
/home/chedli/.local/bin/colab sessions
```

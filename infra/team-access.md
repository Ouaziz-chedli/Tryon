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

> **Note IP :** JuDoor exige une clause `from="..."` sur la clé publique SSH. Si un teammate se connecte depuis une autre IP publique, ajouter son IP dans la clause `from="..."` sur JuDoor (`Manage SSH-keys` sur JURECA).

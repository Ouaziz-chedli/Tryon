# AGENTS.md

Point d'entrée pour tout agent travaillant sur ce repository.

1. **Lire d'abord [README.md](README.md)** pour la vue d'ensemble du projet (**Team `TRYON`** — European AI Championship Paris 2026) et les liens vers toute la documentation.
2. **Consulter [docs/README.md](docs/README.md)** pour les fiches détaillées des 3 tracks, des papiers de recherche et du setup cluster GPU Jülich (`JURECA`, projet `training2666`).
3. **Règles du repo :**
   - Ne jamais commiter de poids de modèles (`.pt`, `.pth`, `.npz`), de datasets ou d'environnements (`.venv/`) — déjà couverts par `.gitignore`.
   - Ne pas consommer de requêtes sur les endpoints évalués de la Track 2 (`ep-01` à `ep-33`, budget strict de 100 réponses/endpoint) avant validation sur `ref-01` à `ref-09`.
   - **Soumissions serveur (cooldowns) :** Track 3 Unlearning = **15 minutes** ; Track 1 Watermark = **10 minutes** ; soumission échouée = **2 minutes** (ne soumettre qu'un checkpoint validé localement via le harnais ; `API_KEY` dans `infra/team-access.md`).
   - **Cluster JURECA (`training2666`, Team `TRYON`) :** maximum **4 jobs Slurm concurrents** pour l'équipe (`ouaziz1`, `sangkhavongs1`, `bassim1`). Toujours vérifier `squeue -A training2666` avant de lancer.

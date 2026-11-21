# Analyse de papier — Zero-Shot Machine Unlearning (IEEE TIFS 2023)

- **Titre :** *Zero-Shot Machine Unlearning*
- **Auteurs :** Vikram S. Chundawat, Ayush K. Tarun, Murari Mandal, Mohan Kankanhalli (National University of Singapore)
- **Journal :** IEEE Transactions on Information Forensics and Security, 2023 (`arXiv:2201.05629v3`)
- **PDF local :** [2201.05629v3.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/papers/2201.05629v3.pdf)
- **Pertinence directe :** **Track 3 — Machine Unlearning** ([03-unlearning.md](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/03-unlearning.md), [Unlearning.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/Unlearning.pdf))

---

## 1. Positionnement & Différence structurelle avec Track 3

C'est le seul des 4 papiers cités dans [Unlearning.pdf](file:///home/chedli/Code/20-Casquettes/hackatons/European_AI_Championship/docs/tracks/Unlearning.pdf) qui opère **sans accès au retain set $\mathcal{D}_r$**.
- **Cadre du papier :** *Class-level unlearning* strict ($\mathcal{D}_r = \emptyset, \mathcal{D}_f = \emptyset$, seuls $\theta_{\text{orig}}$ et l'indice de la classe à oublier $C_f$ sont connus ; la Section V-F précise que le *subset/cohort unlearning* n'est pas traité directement par le filtre de classe).
- **Cadre CISPA Track 3 :** *Subset / Selective unlearning* ($\mathcal{D}_f$ de 3 500 images fourni sur 45 classes actives, 55 classes absentes, $\mathcal{D}_r = \emptyset$).
- **Ce que le papier nous apporte directement :** On n'a pas besoin de synthétiser du bruit pour deviner $\mathcal{D}_f$ (puisqu'on possède les vraies images $\mathcal{D}_f$), mais ses deux mécanismes — **Error-Minimizing Impressions** et **Gated Knowledge Transfer (GKT)** — constituent la fondation mathématique pour **ancrer $\mathcal{D}_r$ sans données** (en particulier sur les 55 classes absentes et hors du sous-espace de $\mathcal{D}_f$).

---

## 2. Les 2 Mécanismes Mathématiques du Papier

### 2.1 Error Minimization-Maximization Noise (Section IV-B, Algorithme 1, Éq. 5–6)
1. **Bruit maximisant l'erreur ($\mathcal{N}_f$, anti-samples pour $i \in C_f$) :**
   $$L_N^{\text{max}}(z) = -\mathcal{L}_{\text{CE}}(M(z; \phi), i) + \lambda \|z\|_2$$
2. **Bruit minimisant l'erreur ($\mathcal{N}_r$, impressions synthétiques de rétention pour $j \in C_r$, Éq. 6) :**
   $$L_N^{(j)}(\mathcal{N}_r^{(j)}) = \mathcal{L}_{\text{CE}}\!\left(M(\mathcal{N}_r^{(j)}; \phi), j\right)$$
   optimisé par descente de gradient sur les pixels $\mathcal{N}_r^{(j)}$ (`400` pas, `lr = 0.1`, poids $\phi$ du modèle figés).
3. **Constat critique des auteurs (Section V-C1, Table I, p. 7) :**
   Fine-tuner le modèle sur $(\mathcal{N}_f, \mathcal{N}_r)$ avec une **Cross-Entropy dure** sans vraies données de réparation fait chuter l'accuracy sur $\mathcal{D}_r$ (de 84 % à 20–29 % sur CIFAR-10) car des labels durs sur du bruit synthétique détruisent la structure inter-classes des logits (*dark knowledge*). D'où la nécessité d'utiliser la **distillation KL** de GKT.

### 2.2 Gated Knowledge Transfer — GKT (Section IV-C, Algorithme 2, Éq. 7–10)
Un générateur $G(z)$ produit des pseudo-échantillons $x_p = G(z)$ filtrés par une porte passe-bande (*Band-Pass Filter* $F(x_p)$) avant distillation du Teacher $T = M_{\text{orig}}$ vers le Student $S = M_{\text{sub}}$ :
1. **Filtre Passe-Bande (Éq. 10) :** Avec $t_p = \text{softmax}(T(x_p)/\tau)$ :
   $$F(x_p) = \prod_{i \in C_f} \mathbb{I}\!\left(t_p^{(i)} < \epsilon\right), \qquad \epsilon \le \frac{1}{K}$$
   (Pour $K=100$ classes, $\epsilon \le 0.01$ garantit que l'échantillon filtré n'active pas les classes/directions à oublier).
2. **Perte du Student sur les échantillons filtrés (Éq. 8–9) :**
   $$L_S(\theta) = D_{\text{KL}}\!\left(T(x_p) \parallel S(x_p)\right) + \beta L_{\text{at}}, \qquad L_{\text{at}} = \sum_{l \in N_L} \left\|\frac{f(A_l^{(t)})}{\|f(A_l^{(t)})\|_2} - \frac{f(A_l^{(s)})}{\|f(A_l^{(s)})\|_2}\right\|_2$$
   où $f(A_l)$ est la carte d'activation moyenne au carré du bloc $l$ ($\beta = 250$).
3. **Phénomène de rebond & arrêt anticipé (Fig. 4 & Fig. 7) :**
   Au bout d'un trop grand nombre d'itérations, l'accuracy sur $\mathcal{D}_f$ rebondit brusquement ; l'unlearning doit donc être stoppé dès que la cible sur $\mathcal{D}_f$ est atteinte.

---

## 3. Applications Concrètes pour Track 3 (`ViT-B/16` en `224×224`)

1. **Pourquoi le filtre GKT explique et corrige notre proxy `PatchMix` :**
   - Dans notre pipeline `tracks/03_unlearning_v2/`, `PatchMix` (`grid=4` et `grid=7`) génère des mosaïques `224×224` ayant exactement le spectre haute-fréquence natif du domaine (contrairement à `CIFAR-100` `32×32` upscalé qui créait un raccourci de résolution).
   - En y ajoutant le **filtre passe-bande GKT au niveau échantillon** — c'est-à-dire en rejetant ou sous-pondérant toute mosaïque $x_p$ trop proche d'une image $x_f \in \mathcal{D}_f$ dans l'espace CLS ($\max_{x_f} \cos(h_{\text{orig}}(x_p), h_{\text{orig}}(x_f)) > \tau_{\text{sim}}$) ou trop confiante sur la classe source ($p_{\text{orig}}^{(y_i)}(x_p) > 0.25$) — on obtient à coût zéro un proxy `retain` garanti hors-forget !
2. **Ajout du terme d'alignement d'activation $L_{\text{at}}$ (Éq. 9) en plus du `retain_kl_loss` ($T=2$) :**
   - Coupler `retain_kl_loss` à $T=2$ avec une pénalité d'alignement normalisé du token CLS $\|h_{\text{orig}}(x_p)/\|h_{\text{orig}}(x_p)\|_2 - h_{\text{sub}}(x_p)/\|h_{\text{sub}}(x_p)\|_2\|_2^2$ verrouille la géométrie interne des blocs `8..11` et protège les 55 classes absentes.

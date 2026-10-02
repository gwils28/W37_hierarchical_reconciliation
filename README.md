# W37 — « Cohérent ne veut pas dire calibré »

Veille hebdomadaire, semaine ISO 2026-W37 : la réconciliation hiérarchique, de MinT ponctuel au probabiliste.
Le plan complet est dans [`W37.md`](W37.md). Ce dépôt couvre pour l'instant les **blocs 1 à 4**.

## Démarrage rapide

```bash
uv sync                  # Python 3.12 + dépendances verrouillées (uv.lock)
uv run pytest            # 53 tests
uv run w37 mint          # bloc 2 : MinT(shrink) maison vs HierarchicalForecast
uv run w37 job           # bloc 3 : un run du job de réconciliation avec quality gate
uv run w37 biais         # bloc 4 : MinT propage le biais d'une seule prévision de base
uv run w37 quantiles     # bloc 4 : MinT réconcilie des moyennes, pas des quantiles
uv run jupyter lab
```

Les mêmes commandes existent en raccourci `make` : `make install | test | test-fast | lint | mint | job | biais | quantiles | notebooks | lab`.
Tutoriels pas à pas : [bloc 2](02_mint_from_scratch/TUTORIEL.md) · [bloc 3](03_system_design/TUTORIEL.md) ·
[bloc 4](04_fondamentaux/TUTORIEL.md).

## Organisation

```
.
├── pyproject.toml / uv.lock / .python-version   # projet uv, versions épinglées, Python 3.12
├── Makefile                                     # raccourcis
├── configs/
│   └── reconciliation.yaml                      # contrat du job (bloc 3)
├── src/w37_reconciliation/                      # le package
│   ├── data.py                                  # hiérarchie synthétique, découpage, format large
│   ├── forecasting.py                           # prévisions de base + résidus in-sample
│   ├── mint/                                    # bloc 2 : shrinkage, projection, assertions, expérience
│   ├── pipeline/                                # bloc 3 : contrat, hash de S, choix de méthode, gate, job
│   ├── fundamentals/                            # bloc 4 : biais, quantiles A/B/C, bootstrap, diagnostics
│   ├── viz.py                                   # style des graphiques
│   └── cli.py                                   # commande `w37`
├── tests/
│   ├── unit/                                    # 46 tests rapides, sans entraînement de modèle
│   └── integration/                             # 7 tests de bout en bout (marqueur `slow`)
├── 01_veille/
│   ├── README.md                                # ressources vérifiées, synthèse, « ce qui est surcoté »
│   ├── GUIDE_RECONCILIATION.md                  # guide d'introduction : MinT et lecture des articles
│   └── figures/                                 # 2 schémas SVG du guide
├── 02_mint_from_scratch/
│   ├── README.md                                # résultats et analyse
│   ├── TUTORIEL.md                              # relancer le bloc 2
│   └── notebooks/                               # 4 notebooks pédagogiques, exécutés
├── 03_system_design/
│   ├── README.md                                # décision d'architecture, trade-offs, critères de bascule
│   └── TUTORIEL.md                              # relancer le bloc 3, expériences sur le gate
└── 04_fondamentaux/
    ├── README.md                                # cadre, biais, quantiles, mur numérique, question d'entretien
    ├── TUTORIEL.md                              # relancer le bloc 4, expériences à faire soi-même
    ├── notebooks/                               # 3 notebooks pédagogiques, exécutés
    └── figures/                                 # 7 figures SVG (README et article)
```

Les **dossiers numérotés** contiennent la documentation et les notebooks de chaque bloc. Le **code** vit dans un
seul package, `src/w37_reconciliation`, et les **tests** sont séparés dans `tests/`. Le λ de Schäfer–Strimmer, par
exemple, est écrit une seule fois (`mint/shrinkage.py`) et sert à la fois au bloc 2 et au choix de méthode du
bloc 3.

## Avancement

| # | Bloc | Livrable | État |
|---|---|---|---|
| 1 | Veille | 4 ressources + 2 marque-pages vérifiés ; synthèse en 3 points ; note « surcoté » | ✅ |
| 2 | Ingénierie & code | MinT(shrink) de zéro : 3 assertions vertes, écart à la bibliothèque **1,07e-5** (< 1e-3) ; 4 notebooks | ✅ |
| 3 | System design | Architecture + `reconciliation.yaml` v3 + job et gate exécutables et testés | ✅ |
| 4 | Fondamentaux | Biais propagé (seuil ≈ 4,5), quantiles A/B/C (B : **0,764** au lieu de 0,900), bootstrap joint, Gauss-Markov ; 3 notebooks | ✅ |
| 5–6 | Mini-projet éCO2mix, BayesReconPy | — | à faire |

## Ce qu'il faut retenir des blocs 1 à 4

1. **L'écart à `HierarchicalForecast` ne vient pas de λ, contrairement à ce que dit le plan.** Les deux λ sont égaux
   à 2e-5 près. Environ 99,8 % de l'écart vient de la covariance de base : la bibliothèque centre les résidus et
   divise par T−1, alors que le squelette du plan prend `E'E/T`. Démonstration dans le
   [notebook 04](02_mint_from_scratch/notebooks/04_equivalence_bibliotheque.ipynb).
2. **Le repli « si m > T » du YAML d'origine contredit le bloc 2.** Le shrinkage existe précisément pour le cas
   m > T. Le repli est donc maintenant déclenché par le conditionnement mesuré de `W`.
3. **Même avec T ≫ m, la covariance empirique est mal conditionnée (≈ 3e4).** Le résidu d'un agrégat est presque la
   somme des résidus de ses feuilles. Le shrinkage ramène le conditionnement à ≈ 23
   ([notebook 02](02_mint_from_scratch/notebooks/02_covariance_et_shrinkage.ipynb)).
4. **HierarchicalForecast est passée en 1.5.3** (24/09/2026). Elle donne des résultats identiques au bit près sur ce
   jeu, mais toute version doit être épinglée dans un chiffre publié.
5. **MinT préserve l'absence de biais, il ne la crée pas.** Un biais de +15 sur le seul total en injecte 4,8 dans
   chaque feuille. Au-delà d'un seuil, MinT fait pire que le bottom-up à tous les niveaux
   ([bloc 4, notebook 01](04_fondamentaux/notebooks/01_cadre_et_biais.ipynb)).
6. **Cohérent ne veut pas dire calibré.** Appliquer `P` à des quantiles donne un vecteur cohérent et faux. Projeter
   des échantillons **indépendants** donne 0,764 de couverture au total pour 0,900 visé. Seul un échantillonnage
   **joint** (bootstrap aux mêmes instants) est calibré
   ([notebook 02](04_fondamentaux/notebooks/02_quantiles_et_calibration.ipynb)).
7. **Le λ de Schäfer–Strimmer naïf ne passe pas à l'échelle.** Le tenseur `T × m × m` pèse 31 Go à m = 5 000.
   La version actuelle passe par la Gram `T × T` : 0,6 Go, pour un résultat identique à 1e-10 près
   (`mint/shrinkage.py`).

# W37 — « Cohérent ne veut pas dire calibré »

Veille hebdomadaire, semaine ISO 2026-W37 : la réconciliation hiérarchique, de MinT ponctuel au probabiliste.
Le plan complet est dans [`W37.md`](W37.md). Ce dépôt couvre pour l'instant les **blocs 1 à 3**.

## Organisation

```
.
├── W37.md                         # plan de la semaine (8 blocs)
├── requirements.txt               # versions épinglées (Python 3.12)
├── 01_veille/
│   └── README.md                  # ressources vérifiées, écarts au plan, synthèse, « ce qui est surcoté »
├── 02_mint_from_scratch/
│   ├── mint_from_scratch.py       # MinT(shrink) en NumPy + 3 assertions + diagnostic de l'écart
│   ├── README.md                  # démarche, résultats, pourquoi l'écart n'est pas nul
│   └── outputs/                   # logs d'exécution (hierarchicalforecast 1.5.1 et 1.5.3)
└── 03_system_design/
    ├── README.md                  # décision d'architecture, trade-offs, critères de bascule
    ├── reconciliation.yaml        # contrat du job (v3)
    ├── reconciliation_job.py      # hash_check, choose_method, quality_gate
    ├── test_reconciliation_job.py # 11 tests : chaque règle doit pouvoir échouer
    ├── demo_job.py                # run complet : hash → méthode → réconciliation → gate → artefact
    └── outputs/                   # log du run, artefact publié, registre des hash de S
```

## Installation

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python 02_mint_from_scratch/mint_from_scratch.py
.venv/bin/pytest -q 03_system_design
(cd 03_system_design && ../.venv/bin/python demo_job.py)
```

## Avancement

| # | Bloc | Livrable | État |
|---|---|---|---|
| 1 | Veille | 4 ressources + 2 marque-pages vérifiés ; synthèse en 3 points ; note « surcoté » | ✅ |
| 2 | Ingénierie & code | `mint_from_scratch.py` : 3 assertions vertes, écart à la bibliothèque **1,07e-5** (< 1e-3) | ✅ |
| 3 | System design | Architecture + `reconciliation.yaml` v3 + gate exécutable et testé (11/11) | ✅ |
| 4–8 | Fondamentaux, mini-projet éCO2mix, Chronos-2, BayesReconPy, article | — | à faire |

## Ce qu'il faut retenir des blocs 1 à 3

1. **L'écart à `HierarchicalForecast` ne vient pas de λ, contrairement à ce que dit le plan.** Les deux λ sont égaux
   à 2e-5 près. Environ 99,8 % de l'écart vient de la covariance de base : la bibliothèque centre les résidus et
   divise par T−1, le squelette du plan prend `E'E/T`. Détail dans [`02_mint_from_scratch/README.md`](02_mint_from_scratch/README.md).
2. **Le repli « si m > T » du YAML d'origine contredit le bloc 2.** Le shrinkage existe précisément pour le cas
   m > T. Le repli est donc maintenant déclenché par le conditionnement mesuré de `W`.
3. **HierarchicalForecast est passée en 1.5.3** (24/09/2026). Elle donne des résultats identiques au bit près sur ce
   jeu, mais toute version doit être épinglée dans un chiffre publié.

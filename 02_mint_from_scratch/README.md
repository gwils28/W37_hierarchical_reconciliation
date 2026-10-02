# Bloc 2 — MinT(shrink) de zéro en NumPy

> Objectif : écrire `G = (S'W⁻¹S)⁻¹S'W⁻¹` de ma main, avec un shrinkage Schäfer–Strimmer, et prouver l'équivalence
> numérique avec `HierarchicalForecast` (`MinTrace(method="mint_shrink")`).

## Lancer

```bash
uv sync                  # une fois
uv run w37 mint          # les assertions + le diagnostic, en ≈ 3 s
uv run jupyter lab 02_mint_from_scratch/notebooks
```

Pas à pas complet : **[TUTORIEL.md](TUTORIEL.md)**.

Environnement de référence : Python 3.12, `numpy 2.5.3`, `pandas 2.3.3`, `statsforecast 2.1.1`,
`hierarchicalforecast 1.5.1` (épinglée pour comparer au chiffre du plan), verrouillés dans `uv.lock`.

## Les notebooks

| # | Notebook | Contenu |
|---|---|---|
| 01 | [`01_hierarchie_et_previsions_de_base`](notebooks/01_hierarchie_et_previsions_de_base.ipynb) | matrice `S`, séries par niveau, prévisions de base incohérentes, résidus in-sample |
| 02 | [`02_covariance_et_shrinkage`](notebooks/02_covariance_et_shrinkage.ipynb) | $W_1$, corrélations, λ de Schäfer–Strimmer calculé pas à pas, mur `m > T` |
| 03 | [`03_projection_mint`](notebooks/03_projection_mint.ipynb) | géométrie de la projection oblique, $G$, $P$, BottomUp vs OLS vs MinT |
| 04 | [`04_equivalence_bibliotheque`](notebooks/04_equivalence_bibliotheque.ipynb) | écart à `HierarchicalForecast` et sa décomposition, MASE par niveau |

## Le code

| Brique | Fichier |
|---|---|
| λ de Schäfer–Strimmer, covariance rétrécie | `src/w37_reconciliation/mint/shrinkage.py` |
| $G$, $P$, réconciliation | `src/w37_reconciliation/mint/projection.py` |
| Les assertions | `src/w37_reconciliation/mint/checks.py` |
| Expérience complète, décomposition de l'écart | `src/w37_reconciliation/mint/experiment.py` |
| Tests | `tests/unit/test_shrinkage.py`, `tests/unit/test_projection.py`, `tests/integration/test_mint_vs_library.py` |

## Ce qui a été fait

1. **Hiérarchie** : 2 régions × 2 canaux, 80 trimestres, synthétique (graine 0). 6 séries : 2 agrégats + 4 feuilles.
   Ce sont les données du squelette du plan.
2. **Prévisions de base** : `AutoETS(season_length=4)` et `SeasonalNaive(4)`, `h = 8`, avec `fitted=True`.
3. **Garde-fou anti-fuite n°1** : `assert Y_fit.ds.max() < test.ds.min()`. `W` n'est estimée que sur les résidus
   one-step du train (T = 72).
4. **Shrinkage Schäfer–Strimmer** (`schafer_strimmer_lambda`) écrit à partir de la formule de l'article (2005),
   cible « D » (corrélations ramenées vers 0, variances conservées) :
   `λ* = Σᵢ≠ⱼ Var(rᵢⱼ) / Σᵢ≠ⱼ rᵢⱼ²`, avec `Var(rᵢⱼ) = T/(T−1)³ · Σₖ (wₖᵢⱼ − w̄ᵢⱼ)²`, puis borné à [0, 1].
   Je n'ai pas lu le code de la bibliothèque avant d'avoir figé mon implémentation.
5. **Projection** (`mint_projection`) : `G` est calculé par `solve` plutôt que par une inversion explicite de `S'W⁻¹S`,
   puis `P = S G`.
6. **Les trois assertions** du plan, plus une quatrième : l'écart à la bibliothèque doit être **non nul**
   (un écart de 0 exact voudrait dire que j'ai recopié leur code).
7. **Diagnostic de l'écart résiduel** (voir plus bas), puis **démonstration du piège `m > T`**.

## Résultats (hierarchicalforecast 1.5.1 — identiques au bit près en 1.5.3)

Sortie de `uv run w37 mint` :

```
series (m) = 6, feuilles (nb) = 4, residus (T) = 72
lambda SS  : 0.113325
P^2 == P   : 2.220e-16
SGS == S   : 2.220e-16
incoherence: 5.684e-14
ecart rel. : 1.074e-05   (abs. 1.772e-03 sur des valeurs ~165)
  [OK] SGS == S (< 1e-12)
  [OK] incoherence (< 1e-9)
  [OK] ecart lib (< 1e-3)
  [OK] ecart lib non nul

decomposition de l'ecart a la bibliotheque :
  covariance de base λ estimé sur résidus         λ  écart absolu
        E'E/T (plan)           non centré 1.133e-01     1.772e-03
np.cov, ddof=1 (lib)           non centré 1.133e-01     3.263e-06
np.cov, ddof=1 (lib)               centré 1.165e-01     4.544e-04

piege m > T : {"T": 4, "m": 6, "rank_W1": 4, "cond_W1": 1.67e+17, "lambda": 0.907, "cond_W": 10.06}
```

| Critère « c'est fini quand… » | Seuil | Mesuré | |
|---|---|---|---|
| `SGS − S` nul | < 1e-12 | 2,2e-16 | ✅ |
| Incohérence des prévisions réconciliées | précision machine | 5,7e-14 (valeurs ~165) | ✅ |
| Écart relatif à la bibliothèque | < 1e-3 | **1,07e-5** (le plan annonçait ~1,3e-5) | ✅ |
| Écart non nul | > 0 | 1,8e-3 en absolu | ✅ |

## Pourquoi l'écart n'est pas nul : la vraie raison

Le plan attribue l'écart à *« l'estimateur de λ qui diffère d'une implémentation à l'autre »*. **Ce n'est vrai qu'à
0,2 % près.** Une fois mon implémentation figée, j'ai récupéré le `W` produit par la bibliothèque et je l'ai comparé à
plusieurs estimateurs candidats :

| Hypothèse sur la covariance de base | max \|diag(C) − diag(W_lib)\| |
|---|---|
| `E'E/T` (non centrée, squelette du plan) | 3,8e-2 |
| `np.cov(E.T)` (**centrée, ddof = 1**) | **8,9e-16** |
| centrée, ddof = 0 | 5,6e-2 |
| non centrée, `/(T−1)` | 6,9e-2 |

Le rapport des corrélations hors diagonale donne un λ de bibliothèque de **0,113303**, contre **0,113325** chez moi.
Les deux λ sont donc quasiment identiques (vérification dans le notebook 04). La décomposition de l'écart absolu (1,77e-3) montre :

- **≈ 99,8 %** de l'écart vient de la **covariance de base** : la bibliothèque centre les résidus et divise par
  `T−1` (`np.cov`), alors que le squelette du plan prend `E'E/T`. Avec la même base, l'écart tombe à **3,3e-6**.
- **≈ 0,2 %** vient de l'estimateur de λ (différence de 2,2e-5). C'est probablement un facteur de normalisation
  différent dans `Var(rᵢⱼ)`.
- Le λ de la bibliothèque est calculé sur des **résidus non centrés**. Si on recalcule λ sur résidus centrés
  (0,1165), l'écart remonte à 4,5e-4.

**Ce que j'en retiens pour une mission.** Un même nom de méthode (`mint_shrink`) recouvre au moins trois choix
d'implémentation qui changent le chiffre publié : le centrage, le `ddof` et la variante de λ. Sur un comité S&OP qui
compare deux runs, ça se voit. Il faut donc épingler la version **et** documenter l'estimateur.

## Les pièges, côté code

| Piège | Ce que fait le code |
|---|---|
| **Fuite n°1** : `W` estimée sur des résidus du test, ou `Y_df=Y_df` passé à `reconcile()` | `reconcile(..., Y_df=Y_fit)`, et `run_mint_experiment` lève une erreur si `Y_fit.ds.max() >= test.ds.min()`. |
| **Fuite n°2** : `TopDown(average_proportions)` sur l'historique complet | Pas de TopDown ici. À traiter dans la quality gate du bloc 3 : les proportions doivent être calculées sur une fenêtre strictement antérieure. |
| **Mur `m > T`** | Avec T' = 4 < m = 6 : `W1` est de rang 4/6 (cond ≈ 1,7e17, donc inutilisable). Le shrinkage choisit tout seul λ = 0,907 et ramène le conditionnement à ≈ 10. **Le shrinkage est une condition d'existence, pas une commodité.** |

# Tutoriel — relancer le bloc 2 (MinT(shrink) de zéro)

Ce tutoriel part d'une machine vierge et va jusqu'aux quatre notebooks exécutés. Comptez environ 5 minutes, dont
la majeure partie pour l'installation.

## 0. Prérequis

| Outil | Version | Vérifier |
|---|---|---|
| Linux / macOS (ou WSL) | — | — |
| [uv](https://docs.astral.sh/uv/) | ≥ 0.5 | `uv --version` |
| git | — | `git --version` |

Python n'a **pas** besoin d'être installé : uv télécharge le Python 3.12 indiqué dans `.python-version`.

Pour installer uv sans droits administrateur :

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh        # installe dans ~/.local/bin
```

## 1. Récupérer le dépôt et installer

```bash
git clone git@github.com:gwils28/W37_hierarchical_reconciliation.git
cd W37_hierarchical_reconciliation
uv sync                 # crée .venv/ et installe les versions exactes de uv.lock
```

`uv sync` installe :

- le package du projet `w37_reconciliation`, en mode éditable (une modification dans `src/` est prise en compte
  immédiatement) ;
- les dépendances épinglées : `hierarchicalforecast==1.5.1`, `statsforecast==2.1.1`, numpy, pandas, matplotlib ;
- les groupes `dev` (pytest, ruff) et `notebooks` (JupyterLab, ipykernel).

> Pourquoi épingler `hierarchicalforecast==1.5.1` alors que la 1.5.3 existe ? Les chiffres publiés dans le README
> (écart de 1,07e-5) ont été mesurés avec cette version. Ils sont identiques au bit près en 1.5.3, mais un chiffre
> publié doit toujours citer sa version.

## 2. Vérification rapide en ligne de commande

```bash
uv run w37 mint          # ou : make mint
```

Sortie attendue (environ 3 s) :

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

La commande renvoie le code **0** si les quatre critères sont `[OK]`, et **1** sinon. Elle peut donc servir de
vérification en CI.

Options : `uv run w37 mint --n 120 --h 12 --seed 3` change la taille de l'historique, l'horizon et la graine.

## 3. Les notebooks, pas à pas

```bash
uv run jupyter lab 02_mint_from_scratch/notebooks      # ou : make lab
```

Dans JupyterLab, choisissez le noyau **Python 3** : c'est celui du `.venv` du projet. Lisez les notebooks dans
l'ordre, chacun reprenant là où le précédent s'arrête :

| # | Notebook | Ce qu'on y apprend | Graphiques / matrices |
|---|---|---|---|
| 01 | `01_hierarchie_et_previsions_de_base.ipynb` | `y = S b`, prévisions par série, incohérence, résidus | matrice `S`, séries par niveau, incohérence par horizon, résidus |
| 02 | `02_covariance_et_shrinkage.ipynb` | $W_1$, corrélations, λ de Schäfer–Strimmer **calculé à la main**, le mur `m > T` | heatmaps $W_1$ / $R$ / $W$, signal vs bruit par paire, conditionnement en fonction de T′ |
| 03 | `03_projection_mint.ipynb` | projection orthogonale vs oblique, $G$, $P$, $P^2 = P$, $SGS = S$ | géométrie 2D, heatmaps de $G$ et $P$, BottomUp vs OLS vs MinT, avant / après |
| 04 | `04_equivalence_bibliotheque.ipynb` | comparaison à `HierarchicalForecast`, **pourquoi l'écart n'est pas nul**, MASE par niveau | carte des écarts, décomposition de l'écart, écart de MASE vs BottomUp |

Les notebooks sont versionnés **avec leurs sorties** : les graphiques s'affichent directement sur GitHub, sans
rien exécuter. Pour les régénérer après une modification du code :

```bash
make notebooks           # ré-exécute les 4 notebooks et réécrit leurs sorties
```

## 4. Les tests du bloc 2

```bash
uv run pytest tests/unit/test_shrinkage.py tests/unit/test_projection.py      # < 1 s
uv run pytest tests/integration/test_mint_vs_library.py                       # ≈ 2 s, entraîne AutoETS
```

| Test | Ce qu'il garantit |
|---|---|
| `test_lambda_is_bounded`, `test_lambda_high_when_correlations_are_noise`, `test_lambda_low_when_correlations_are_real` | λ se comporte comme attendu aux deux extrêmes |
| `test_shrinkage_makes_singular_covariance_invertible` | le piège `m > T` est bien évité |
| `test_projection_is_idempotent_and_unbiased`, `test_ols_is_orthogonal_projection`, `test_trusting_leaves_approaches_bottom_up` | les propriétés de $P$ et les cas limites (OLS, BottomUp) |
| `test_close_to_library_but_not_identical` | le critère du plan : 0 < écart < 1e-3 |
| `test_gap_comes_from_covariance_base_not_lambda` | le diagnostic de l'écart reste vrai si le code évolue |

## 5. Où est le code

```
src/w37_reconciliation/
├── data.py             # make_region_channel_data, build_hierarchy, train_test_split, to_wide
├── forecasting.py      # fit_base_forecasts (AutoETS + SeasonalNaive, fitted=True), residual_matrix
├── mint/
│   ├── shrinkage.py    # schafer_strimmer_lambda, shrunk_covariance  ← le λ écrit de zéro
│   ├── projection.py   # mint_projection (G, P), reconcile
│   ├── checks.py       # unbiasedness_error, incoherence, relative_gap
│   └── experiment.py   # run_mint_experiment, gap_decomposition, m_greater_than_T_demo
├── viz.py              # palette et heatmaps des notebooks
└── cli.py              # `w37 mint`
```

Pour utiliser les briques dans votre propre code :

```python
from w37_reconciliation.mint import shrunk_covariance, mint_projection, reconcile
cov = shrunk_covariance(E)              # E : résidus in-sample (T × m), colonnes dans l'ordre de S
G, P = mint_projection(S, cov.W)
y_tilde = reconcile(y_hat, P)           # y_hat : (h × m)
```

## 6. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `uv: command not found` | uv n'est pas dans le PATH | `export PATH="$HOME/.local/bin:$PATH"` |
| `ModuleNotFoundError: w37_reconciliation` dans Jupyter | mauvais noyau | lancer `uv run jupyter lab` depuis la racine du dépôt et choisir le noyau Python 3 |
| `ecart rel.` différent de 1.074e-05 | versions différentes de `uv.lock` | `uv sync --locked` ; vérifier avec `uv pip list \| grep -E "hierarchical\|statsforecast\|numpy"` |
| Avertissements `FutureWarning` de statsforecast | bruit des bibliothèques | sans effet sur les résultats ; ils sont filtrés dans la CLI et les notebooks |

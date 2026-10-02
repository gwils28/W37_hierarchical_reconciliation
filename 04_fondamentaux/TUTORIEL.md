# Tutoriel — relancer le bloc 4 (fondamentaux)

Ce tutoriel suppose le dépôt installé (`uv sync`, voir le [tutoriel du bloc 2](../02_mint_from_scratch/TUTORIEL.md)
§ 0–1). Comptez environ 10 minutes pour tout relancer et lire les sorties.

## 1. Vérification rapide en ligne de commande

### Où ça casse n°1 : le biais

```bash
uv run w37 biais          # ou : make biais
```

Sortie attendue (< 1 s) :

```
Contre-exemple : biais de +15 sur le seul total, W = diag(1, 9, 9, 9)

           vérité  base  bottom-up    MinT  erreur bottom-up  erreur MinT
Total        60.0  75.0       60.0  74.464               0.0       14.464
Feuille_1    10.0  10.0       10.0  14.821               0.0        4.821
...
Monte-Carlo : MAE par niveau selon le biais du total (total σ=1, feuilles σ=3)
...
```

Ce qu'il faut regarder : la colonne `erreur MinT` est non nulle sur **toutes** les feuilles, alors qu'aucune
feuille n'était biaisée. Dans le Monte-Carlo, la colonne `MinT` passe au-dessus de `bottom-up` quand le biais
grandit.

### Où ça casse n°2 : les quantiles

```bash
uv run w37 quantiles              # ρ = 0,6 par défaut ; ou : make quantiles
uv run w37 quantiles --rho 0.9    # séries plus corrélées
```

Sortie attendue (≈ 5 s, extrait) :

```
          q_exact  A: P@quantiles  cov A  B: éch. indép.  cov B  C: éch. dépendants  cov C
Total     230.864         234.273  0.922         217.383  0.764             230.955  0.900
```

Ce qu'il faut regarder : les trois colonnes `cov`. Seule la colonne C vaut 0,900 partout.

> 💡 **Expérience à faire.** Relancez avec `--rho 0.0` puis `--rho 0.9`. La couverture B au total **baisse** quand
> ρ monte (0,816 → 0,746), alors que l'écart « somme des quantiles » se resserre. Les deux erreurs vont en sens
> opposé.

## 2. Les notebooks, pas à pas

```bash
uv run jupyter lab 04_fondamentaux/notebooks
```

Choisissez le noyau **Python 3** (celui du `.venv`), puis lisez les notebooks dans l'ordre :

| # | Notebook | Ce qu'on y apprend | Durée de lecture |
|---|---|---|---|
| 01 | `01_cadre_et_biais.ipynb` | le cadre `SGS = S`, la projection, **comment un biais se répand** et à partir de quand il coûte | 15 min |
| 02 | `02_quantiles_et_calibration.ipynb` | pourquoi des quantiles ne se réconcilient pas, méthodes A / B / C, **bootstrap joint** | 20 min |
| 03 | `03_diagnostic_et_entretien.ipynb` | le mur `m > T`, les trois hypothèses de la question d'entretien, Gauss-Markov, **la réponse rédigée** | 15 min |

Chaque notebook suit le même code d'encadrés : 💡 intuition · 🧪 exemple chiffré · ⚠️ point d'attention ·
📌 à retenir. Pour une révision express, lisez seulement les encadrés 📌.

Les notebooks sont versionnés **avec leurs sorties**. Ils exportent leurs figures en SVG dans
`04_fondamentaux/figures/`, et c'est là que le README et l'article vont les chercher. Pour tout régénérer :

```bash
make notebooks            # ré-exécute les notebooks des blocs 2 et 4
```

> ⚠️ **Point d'attention — mémoire.** La cellule « mur numérique » du notebook 03 estime une covariance pour
> m = 5 000 séries. Elle demande environ 0,6 Go et 25 s. Une ancienne version du calcul de λ allouait un tenseur
> de 31 Go, qui faisait tuer le noyau par l'OOM killer. Pour vous protéger d'une régression, vous pouvez plafonner
> la mémoire du noyau : une allocation excessive lève alors une `MemoryError` lisible au lieu d'un kill silencieux.
>
> ```bash
> (ulimit -v 16777216; make notebooks)     # 16 Go de mémoire virtuelle au plus
> ```

## 3. Expériences à faire soi-même

| Question | Où | Quoi changer |
|---|---|---|
| Le biais contamine-t-il autant si on fait **moins** confiance au total ? | notebook 01, § 2 | `w_diag=(9, 9, 9, 9)` dans `bias_counterexample` |
| Où est le seuil si les feuilles sont plus précises ? | notebook 01, § 3 | `bias_sweep(..., sd_leaf=1.5)` |
| Un bootstrap joint tient-il avec peu d'historique ? | notebook 02, § 6 | `bootstrap_coverage(T=40)` |
| Gauss-Markov tient-il avec une `W` **fausse** ? | notebook 03, § 5 | appeler `weighted_trace_check` avec une `W` différente de celle qui génère les erreurs |

## 4. Les tests du bloc 4

```bash
uv run pytest tests/unit/test_fundamentals.py tests/unit/test_shrinkage.py    # < 1 s
```

| Test | Ce qu'il garantit |
|---|---|
| `test_counterexample_matches_plan` | les chiffres du contre-exemple du plan |
| `test_bias_spreads_along_column_of_P` | biais réconcilié = colonne de `P` × biais de base |
| `test_mint_beats_bottom_up_without_bias_and_loses_with_large_bias` | l'existence du seuil |
| `test_residual_bias_table_flags_only_biased_series` | le test de l'hypothèse 1 ne fait pas de faux positif |
| `test_method_A_is_wrong_in_both_directions`, `test_method_B_undercovers_total`, `test_method_C_is_calibrated_everywhere` | les trois résultats du § 3 du README |
| `test_rho_sweep_errors_go_in_opposite_directions` | le fait contre-intuitif du balayage |
| `test_joint_bootstrap_recovers_calibration` | le bootstrap joint fait aussi bien que la vraie covariance |
| `test_random_G_is_unbiased`, `test_no_level_weighting_beats_mint_when_W_is_true` | Gauss-Markov, quelle que soit la pondération |
| `test_lambda_matches_dense_tensor_formula` | le λ sans tenseur `T × m × m` est identique à la formule directe |

## 5. Utiliser les briques dans son code

```python
from w37_reconciliation.fundamentals import bias_allocation, toy_hierarchy
from w37_reconciliation.fundamentals.diagnostics import residual_bias_table

# Hypothèse 1 : un biais de base ? E = résidus in-sample (T × m), colonnes dans l'ordre de S
residual_bias_table(E, names)            # colonne « biais suspect » : |t| > 2

# Où partirait un biais de +1 sur la série 0, avec ma W ?
bias_allocation(S, W, series=0)
```

Pour le probabiliste en production, n'écrivez pas votre propre bootstrap : passez `level=[90]` et
`intervals_method="bootstrap"` ou `"permbu"` à `HierarchicalReconciliation.reconcile()`. Vérifiez ensuite la couverture **par niveau** en backtest. Une
couverture correcte aux feuilles ne dit rien de la couverture au total.

## 6. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| Noyau Jupyter tué sans message pendant le notebook 03 | ancienne version de `shrinkage.py` (tenseur de 31 Go) | `git pull`, puis relancer avec `ulimit -v` comme au § 2 pour obtenir une `MemoryError` lisible |
| `findfont: Failed to find font weight semibold` | DejaVu Sans n'a pas de graisse semibold | sans effet : matplotlib prend le gras |
| Couvertures légèrement différentes du README | autre graine ou autre nombre de tirages | les chiffres du README sont obtenus avec la graine 37 et N = 400 000 |

# Tutoriel — relancer le bloc 6 (BayesReconPy)

Ce tutoriel suppose le dépôt installé (voir le [tutoriel du bloc 2](../02_mint_from_scratch/TUTORIEL.md) § 0–1).
Il faut un accès internet pour le premier téléchargement (76 Mo).

## 1. Installer

```bash
uv sync                                  # le groupe « bayesrecon » est installé par défaut
uv run python -c "import bayesreconpy.reconc_td_cond; print('ok')"
```

Le groupe est déclaré dans `pyproject.toml` :

```toml
bayesrecon = ["bayesreconpy==0.5.0", "pulp>=2.9,<4"]
```

⚠️ **Pourquoi `pulp<4`.** BayesReconPy 0.5.0 demande `PuLP>=2.9` sans borne supérieure. PuLP 4.0.0 (25/09/2026)
ne fournit plus `PULP_CBC_CMD`, que le paquet importe : sans la borne, `import bayesreconpy.reconc_td_cond`
échoue avec `ImportError: cannot import name 'PULP_CBC_CMD'`.

Hors de ce dépôt, l'équivalent pip est :

```bash
pip install "bayesreconpy==0.5.0" "pulp>=2.9,<4"      # Python ≥ 3.12 obligatoire
```

## 2. La commande

```bash
uv run w37 bayesrecon download           # télécharge l'exemple M5 au commit du tag v0.5.0, vérifie le sha256
uv run w37 bayesrecon run                # 4 méthodes, scores, skill scores, quality gate (≈ 10 s)
```

Ou en une fois : `make bayesrecon`. La sortie est reproduite dans le [README](README.md) § 3. Ce qu'il faut
regarder :

- la ligne **`articles · moyennes < 0`** : 11 pour MinT gaussien, 0 ailleurs ;
- la ligne **`articles · bornes basses < 0`** : 2 992 pour MinT gaussien ;
- la **quality gate** : seul MinT gaussien est `[BLOQUÉ]`.

## 3. Les notebooks

```bash
uv run jupyter lab 06_bayesreconpy/notebooks
```

| Ordre | Notebook | Durée de lecture | Calcul |
|---|---|---|---|
| 1 | `01_conditionnement_a_la_main.ipynb` | 15 min | < 5 s, sans données |
| 2 | `02_m5_negatifs_et_precision.ipynb` | 20 min | ≈ 30 s (télécharge M5 si absent) |

## 4. Utiliser BayesReconPy dans son code

Les enveloppes de `w37_reconciliation.bayesrecon.methods` montrent l'appel minimal et protègent des pièges :

```python
from bayesreconpy.reconc_td_cond import reconc_td_cond

fc_bottom = {name: pmf.copy() for name, pmf in pmfs.items()}   # COPIES : reconc_td_cond modifie ses entrées
fc_upper = {"mu": mu_upper, "Sigma": Sigma_upper}             # gaussienne multivariée des agrégats
rec = reconc_td_cond(A, fc_bottom, fc_upper, bottom_in_type="pmf",
                     num_samples=10_000, return_type="pmf", seed=37)
rec["bottom_reconciled"]["pmf"]                               # une pmf réconciliée par article
```

| Piège | Symptôme | Parade |
|---|---|---|
| entrées modifiées en place par `reconc_td_cond` | `ValueError: probabilities do not sum to 1` au calcul suivant | passer des copies |
| hiérarchie non équilibrée | `It is impossible to find the lowest upper level … should be duplicated` | dupliquer les articles pour que chacun soit dans un seul groupe du niveau le plus bas |
| `fc_bottom` passé en liste | `fc_bottom must be a dictionary with the names of the bottom TS as keys` | un dictionnaire nom → pmf |
| `fc_upper` en liste de `{"mean", "sd"}` (le format de `reconc_buis`) | `TypeError: list indices must be integers or slices, not str` | `{"mu": vecteur, "Sigma": matrice}` pour TD-cond et MixCond |

## 5. Les tests du bloc 6

```bash
uv run pytest tests/unit/test_bayesrecon.py      # 13 tests, < 1 s, sans réseau
```

| Test | Ce qu'il garantit |
|---|---|
| `test_bayesreconpy_buis_recovers_the_exact_conditional_distribution` | la bibliothèque calcule bien le conditionnement (comparé à l'énumération exacte) |
| `test_buis_by_hand_matches_the_exact_conditional_distribution` | l'algorithme expliqué dans le notebook 01 est le bon |
| `test_conditioning_methods_do_not_mutate_their_inputs` | la parade contre l'effet de bord de `reconc_td_cond` fonctionne |
| `test_every_method_is_coherent_and_only_the_gaussian_one_goes_negative` | cohérence de toutes les méthodes, négatifs seulement pour la gaussienne |
| `test_rps_*`, `test_interval_score_*`, `test_skill_score_*` | les scores ont les valeurs attendues sur des cas calculables à la main |

## 6. Expériences à faire soi-même

| Question | Où | Quoi changer |
|---|---|---|
| Les négatifs dépendent-ils du total prévu ? | notebook 01 | `Toy(mu_total=0.2)` ou `Toy(mu_total=2.0)` |
| Combien de tirages faut-il ? | `configs/bayesrecon.yaml` | `num_samples: 2000` puis `50000`, et comparer les skill scores |
| Le résultat dépend-il de la graine ? | `configs/bayesrecon.yaml` | `seed`, et mesurer l'écart des skill scores (le bruit Monte-Carlo du § 6 du notebook 02) |
| MinT avec un plancher à zéro ? | notebook 02 | tronquer les moyennes MinT à 0 et mesurer la perte de cohérence |

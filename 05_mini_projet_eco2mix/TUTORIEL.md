# Tutoriel : relancer le bloc 5 (mini-projet éCO2mix)

Ce tutoriel suppose le dépôt installé (`uv sync`, voir le [tutoriel du bloc 2](../02_mint_from_scratch/TUTORIEL.md)
§ 0–1). Il faut un accès internet pour le premier téléchargement (environ 90 Mo).

## 1. Le pipeline en quatre commandes

Chaque étape lit les sorties de la précédente sur disque, dans `data/eco2mix/` (non versionné) :

```
data/eco2mix/raw/        13 CSV bruts + manifest.json (URL, date, sha256)   <- download
data/eco2mix/interim/    horaire propre, qualité, national, additivité      <- prepare
data/eco2mix/results/    prévisions, échelles MASE, diagnostic de W         <- backtest
```

```bash
uv run w37 eco2mix download                     # ≈ 5 min la première fois, ensuite instantané
uv run w37 eco2mix prepare                      # ≈ 4 s
uv run w37 eco2mix backtest                     # protocole principal, 4 origines, ≈ 20 s
uv run w37 eco2mix report                       # tableau, assertion de cohérence, chiffre clé
```

Ou en une fois : `make eco2mix`.

Sortie attendue de `prepare` :

```
  12 régions × 105,191 heures (2013-01-01 → 2024-12-31 22:00 UTC)
  heures imputées : 144 ; doublons retirés : 288
  additivité national − Σ régions : médiane +0.00 MW ; max |écart| par année ≤ 6 MW sauf {2013: 14.0, 2016: 215.0}
```

Ce qu'il faut regarder : **144 heures imputées** et **288 doublons**, soit exactement 1 et 2 par an et par
région. C'est le défaut de changement d'heure de la source. Un autre nombre signalerait un changement de la
source.

`report` renvoie le code **0** si l'assertion de cohérence passe, sinon il lève une erreur : il peut servir de
vérification en CI.

### L'étude de robustesse

```bash
make eco2mix-robustness                         # 119 origines de 2024, ≈ 40 min
```

Elle est indispensable pour conclure (notebook 04) : les 4 origines du protocole principal ne suffisent pas à
classer les méthodes.

### Les études de confirmation et le benchmark

```bash
make eco2mix-confirmation                       # 2023 (≈ 9 min), puis AutoETS sur 2024 (≈ 17 min)
uv run w37 eco2mix benchmark --which robustness # baselines naïves + RTE J−1 + gains, sans réentraîner
```

Chaque étude est déclarée sous `confirmation:` dans `configs/eco2mix.yaml` et ne change qu'une chose (l'année,
ou le modèle de base) ; `--which` accepte son nom. `benchmark` ajoute à n'importe quelle étude les 3 baselines
naïves et la prévision RTE (France seule), calculées à la volée, et imprime le MASE et le gain agrégé contre
SN 168 h, le modèle de base et le bottom-up.

## 2. Les notebooks

```bash
uv run jupyter lab 05_mini_projet_eco2mix
```

| Ordre | Notebook | Durée de lecture | Prérequis |
|---|---|---|---|
| 1 | `notebooks/01_donnees_et_hierarchie.ipynb` | 15 min | `download` + `prepare` |
| 2 | `notebooks/02_modeles_de_base_et_hypotheses.ipynb` | 20 min | idem (réentraîne une origine, ≈ 10 s) |
| 3 | `notebooks/03_reconciliation_protocole_principal.ipynb` | 15 min | `backtest` |
| 4 | `notebooks/04_robustesse_et_tests.ipynb` | 25 min | `backtest --which robustness` |
| 5 | `notebooks/05_benchmark_et_confirmations.ipynb` | 30 min | `robustness`, `annee_2023`, `ets_2024` (réentraîne 4 origines, ≈ 30 s) |
| - | `reconciliation_eco2mix.ipynb` | - | rien : il lance tout lui-même |

Pour une révision express, lisez les encadrés « À retenir » et « Test d'hypothèse ». Les notebooks sont versionnés **avec leurs sorties**
et exportent leurs figures en SVG dans `05_mini_projet_eco2mix/figures/`.

Pour tout régénérer :

```bash
(ulimit -v 16777216; make notebooks)            # plafond de 16 Go : une MemoryError plutôt qu'un kill
```

## 3. Expériences à faire soi-même

| Question | Où | Quoi changer |
|---|---|---|
| Le classement tient-il sur 2022 ? | `configs/eco2mix.yaml` | sous `confirmation:`, `annee_2022: {start: "2022-01-11", end: "2022-12-30", every_days: 3}`, puis `backtest --which annee_2022` |
| Et avec un autre modèle de base ? | `eco2mix/backtest.py` (`build_models`) + `configs/eco2mix.yaml` | ajouter le modèle à la fabrique, puis une étude avec `models` et `reconcile_model` |
| Et à un horizon plus long ? | `configs/eco2mix.yaml` | `h: 48` |
| MinT avec une fenêtre plus courte ? | `configs/eco2mix.yaml` | `train_weeks: 26` |
| Combien d'origines faut-il pour que le test devienne net ? | notebook 04, § 2 | sous-échantillonner les origines et regarder les p-valeurs |
| Le défaut d'heure existe-t-il dans le jeu national ? | notebook 01 | appliquer `clean_half_hourly` à `read_national` |

**Point d'attention.** Modifier le protocole **après** avoir vu les résultats, puis garder la version qui
arrange, c'est du *p-hacking*. Toute modification doit être consignée dans le README (§ « Ce que ce bloc corrige
ou précise ») avec sa raison.

## 4. Les tests du bloc 5

```bash
uv run pytest tests/unit/test_eco2mix.py              # 18 tests, < 2 s, sans réseau
uv run pytest tests/unit/test_eco2mix_benchmark.py    # 12 tests, < 2 s : baselines, RTE, gains, études
uv run pytest tests/integration/test_eco2mix_backtest.py   # 4 tests, ≈ 3 s, entraîne MSTL
```

| Test | Ce qu'il garantit |
|---|---|
| `test_october_dst_gap_is_reproduced_and_imputed`, `test_march_dst_duplicates_are_counted_and_removed` | le défaut de changement d'heure est corrigé et compté |
| `test_hourly_resampling_is_a_mean_not_a_sum` | 30 min → 1 h ne change pas l'unité |
| `test_long_gaps_are_refused_not_interpolated` | aucune donnée n'est fabriquée en silence |
| `test_a_missing_row_silently_breaks_the_total_and_is_caught` | une ligne absente est détectée |
| `test_split_is_strictly_before_origin_and_exactly_h_long` | pas de fuite dans le backtest |
| `test_mase_is_reported_per_level_never_globally`, `test_mase_ignores_scales_of_origins_without_forecasts` | la métrique est calculée par niveau, sur les bonnes origines |
| `test_incoherence_detects_a_non_summing_forecast` | l'assertion de cohérence peut échouer |
| `test_naive_t_overrejects_on_autocorrelated_noise_but_hac_does_not` | le test de biais garde sa taille sur des données autocorrélées |
| `test_past_error_reconciliation_only_uses_finished_origins` | l'expérience sur W n'utilise que le passé |
| `test_every_reconciliation_is_coherent` (intégration) | le pipeline complet livre des prévisions cohérentes |
| `test_naive_baselines_read_the_right_past_values`, `test_naive_baselines_are_coherent_by_construction` | les baselines naïves lisent le bon passé et somment exactement |
| `test_rte_debiasing_only_uses_errors_before_the_origin` | la correction de biais de RTE ne voit pas le futur |
| `test_aggregate_gain_is_the_ratio_of_mean_mase`, `test_convergence_ends_on_the_aggregate_gain` | le gain du benchmark est celui qu'on relit dans le tableau |
| `test_rank_probabilities_count_ties_as_shared_first_place` | BU et MSTL, identiques en région, sont classés ex aequo |
| `test_a_confirmation_study_changes_only_what_it_declares` | une étude de confirmation ne change que ce qu'elle déclare |
| `test_past_error_window_keeps_only_the_last_finished_origins` | la fenêtre de l'expérience sur W oublie bien le passé lointain |

## 5. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `fichiers bruts modifiés depuis le téléchargement` | un CSV de `data/eco2mix/raw/` a été modifié | `uv run w37 eco2mix download --force` |
| `requests.exceptions.ReadTimeout` pendant `download` | API ODRÉ lente | relancer : les fichiers déjà téléchargés sont conservés |
| `trou de N h … > max_gap_hours` | la source a changé (nouveau trou) | examiner la période citée avant d'élargir `max_gap_hours` |
| `historique insuffisant` dans `backtest` | origine trop proche de 2013 pour 52 semaines | déplacer l'origine ou réduire `train_weeks` |
| Processus tué sans message | mémoire saturée | relancer sous `ulimit -v 16777216` pour obtenir une `MemoryError` lisible |
| Chiffres légèrement différents du README | données révisées par RTE depuis le téléchargement | comparer les sha256 de `manifest.json` |

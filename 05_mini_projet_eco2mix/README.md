# Bloc 5 · Mini-projet éCO2mix : réconcilier les prévisions de consommation des régions et de la France

> **Objectif métier.** Supprimer les arbitrages manuels du comité de pilotage en livrant un jeu unique de
> prévisions de consommation électrique dont **la somme des régions reproduit exactement le total France**,
> sans retouche.

**Réponse courte.** L'objectif est atteint : toute méthode réconciliée livre des prévisions qui somment
exactement, et c'est vérifié par assertion à chaque origine. Sur cette hiérarchie, la meilleure méthode est le
**bottom-up**. MinT shrink, la méthode de référence, fait **moins bien** sur l'année 2024. La cause est
identifiée et démontrée par une expérience contrôlée : sa matrice W est estimée sur des résidus in-sample qui ne
ressemblent pas aux vraies erreurs de prévision.

**Mis à l'épreuve** (§ 6) : un benchmark complet face à 4 baselines naïves, au modèle statistique seul et à la
prévision J−1 de RTE, puis trois études de confirmation (une autre année, un autre modèle de base, la cause
testée deux fois de plus). Les conclusions tiennent, avec deux nuances : les écarts entre réconciliations sont
3 fois plus petits en 2023, et le classement dépend de l'heure prévue.

## Lancer

```bash
uv sync
make eco2mix                      # télécharge (1re fois, ≈ 5 min), prépare, backtest 4 origines, rapport
make eco2mix-robustness           # étude de robustesse : 119 origines de 2024 (≈ 40 min)
make eco2mix-confirmation         # études de confirmation : 2023, puis AutoETS sur 2024 (≈ 25 min)
uv run w37 eco2mix benchmark --which robustness   # tableau du benchmark (baselines naïves, RTE, gains)
uv run jupyter lab 05_mini_projet_eco2mix
```

Pas à pas complet : **[TUTORIEL.md](TUTORIEL.md)**. La *Definition of Done* du plan tient dans un notebook qui
tourne de bout en bout sans intervention : **[`reconciliation_eco2mix.ipynb`](reconciliation_eco2mix.ipynb)**.

## Comment lire ce bloc

| Notebook | Ce qu'on y apprend |
|---|---|
| [`01_donnees_et_hierarchie`](notebooks/01_donnees_et_hierarchie.ipynb) | provenance, défaut de changement d'heure de la source, test d'additivité, saisonnalités, corrélation des régions |
| [`02_modeles_de_base_et_hypotheses`](notebooks/02_modeles_de_base_et_hypotheses.ipynb) | SeasonalNaive et MSTL ; biais, bruit blanc, structure de W ; l'hypothèse cachée de `mint_shrink` |
| [`03_reconciliation_protocole_principal`](notebooks/03_reconciliation_protocole_principal.ipynb) | le protocole, ce que fait chaque méthode, le tableau de la Definition of Done, le cas du 12 novembre |
| [`04_robustesse_et_tests`](notebooks/04_robustesse_et_tests.ipynb) | 119 origines, tests appariés avec correction de Holm, expérience contrôlée sur W, conclusion |
| [`05_benchmark_et_confirmations`](notebooks/05_benchmark_et_confirmations.ipynb) | benchmark complet en courbes (horizon, mois, métrique, convergence), confirmations sur 2023 et avec AutoETS, bilan |
| [`reconciliation_eco2mix`](reconciliation_eco2mix.ipynb) | la Definition of Done, exécutable de bout en bout |

Les notebooks suivent le code d'encadrés du dépôt : intuition · exemple · point d'attention ·
à retenir, plus **test d'hypothèse** (hypothèse formulée, testée, puis acceptée ou rejetée).

---

## 1. Les données

**Source** : [éCO2mix régional, données consolidées et définitives](https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-cons-def/information/)
et [éCO2mix national](https://odre.opendatasoft.com/explore/dataset/eco2mix-national-cons-def/information/),
**Open Data Réseaux Énergies (ODRÉ) : RTE, Licence Ouverte v2.0**. Elles sont téléchargées par l'API
(`w37 eco2mix download`), et chaque fichier brut est tracé dans un manifeste (URL, date, taille, sha256).

**Hiérarchie** : 12 régions métropolitaines → France. Le total est construit comme la somme des régions.

![Les 12 régions](figures/regions_2024.svg)

### Trois découvertes faites en vérifiant les données

| Découverte | Conséquence | Traitement |
|---|---|---|
| **La Nouvelle-Aquitaine s'arrête au 31/12/2024** ; les 11 autres régions vont jusqu'au 30/06/2026 | depuis 2025, la somme des régions n'est plus le total France | protocole limité aux **données définitives 2013 → 2024** |
| **Défaut de changement d'heure** : la source range 48 demi-heures par jour local. Chaque année et dans chaque région : **2 doublons** fin mars, **1 heure manquante** fin octobre | MSTL refuse les trous ; une ligne absente fausse le total sans aucune erreur | doublons fusionnés, heure interpolée et marquée (144 heures sur 12 ans × 12 régions) ; tout trou de plus d'une heure lève une erreur |
| **Le total officiel n'est pas toujours la somme des régions** : +7 à +15 MW toute l'année 2013, jusqu'à 222 MW les 7-8 octobre 2016 | réconcilier vers un total incohérent n'aurait pas de sens | test année par année : écart ≤ 6 MW (l'arrondi) de 2014 à 2024 hors 2016, donc **sur toute la période étudiée** |

![Le défaut de changement d'heure](figures/changement_heure.svg)

![Test d'additivité](figures/additivite.svg)

## 2. Le protocole, fixé avant les résultats

Tout est dans [`configs/eco2mix.yaml`](../configs/eco2mix.yaml) :

| Élément | Choix | Pourquoi |
|---|---|---|
| Pas de temps | 1 h, **moyenne** des deux demi-heures, en UTC | des MW restent des MW ; pas de jour de 23 ou 25 h |
| Modèles de base | **SeasonalNaive(168)** (baseline n°1) et **MSTL([24, 168])** (réconcilié) | double saisonnalité jour / semaine |
| Fenêtre d'entraînement | 52 semaines glissantes, **strictement antérieures** à l'origine | ni fuite, ni W estimée sur le test |
| Horizon | h = 24 h, prévision du jour J à J 00:00 UTC | |
| Réconciliations | BU, OLS, WLS struct, WLS var, MinT shrink (HierarchicalForecast 1.5.1) | de W = rien à W pleine |
| Métrique | **MASE par niveau** (France ; moyenne des 12 régions), échelle = naïf saisonnier sur le train | jamais une moyenne globale |
| Protocole principal | **4 mardis de novembre 2024** (Definition of Done) | |
| Étude de robustesse | **119 origines**, une tous les 3 jours de janvier à décembre 2024 | tous les jours de la semaine, toutes les saisons |
| Études de confirmation (ajoutées après les résultats, § 6) | **2023** (118 origines) ; **AutoETS** au lieu de MSTL sur 2024 | changer une seule chose à la fois |

![Le protocole](figures/protocole.svg)

## 3. Les hypothèses de MinT, testées

| | Hypothèse | Test | Résultat |
|---|---|---|---|
| Additivité | Le total France = Σ régions | écart national − somme, année par année | **acceptée** sur 2014-2024 hors 2016 (≤ 6 MW) |
| H1 | Prévisions de base sans biais | t HAC + Holm, sur les **erreurs hors échantillon** (119 origines) | **acceptée** : aucun biais significatif |
| H2 | Résidus in-sample = bruit blanc | Ljung-Box (24 h, 168 h) | **rejetée** partout ; autocorrélation négative à 3-6 h et 48 h : signature d'un **lissage bilatéral** |
| H3 | W in-sample ∝ covariance des vraies erreurs | corrélation moyenne entre régions, IC bootstrap sur les origines | **rejetée** : 0,11 in-sample contre **0,56 [0,45 ; 0,65]** hors échantillon |

**Le test de biais sur résidus in-sample ne prouve rien avec MSTL** : ses résidus sont les *restes* d'une
décomposition, centrés par construction (notebook 02, § 4). Le test utile est hors échantillon.

![Corrélations : ce que voit W, et la réalité](figures/correlations_in_vs_oos.svg)

## 4. Résultats

### Protocole principal (Definition of Done) : 4 mardis de novembre 2024

Sortie de `uv run w37 eco2mix report` :

```
MASE par niveau (main, 4 origines) : moyenne et écart-type

             France mean  France std  Régions mean  Régions std
SN                 0.817       0.329         0.825        0.152
base               0.467       0.520         0.580        0.464
BU                 0.500       0.607         0.580        0.464
OLS                0.469       0.527         0.544        0.388
WLS struct         0.481       0.564         0.560        0.423
WLS var            0.485       0.573         0.567        0.438
MinT shrink        0.463       0.507         0.542        0.383

Incohérence max (MW) après réconciliation : 0.00e+00   [OK] < 1e-06
Incohérence max (MW) des prévisions de base MSTL : 671.9
MinT shrink vs BU, Régions : -2.25% ± 9.01% (meilleur à 2/4 origines)
MinT shrink vs BU, France  : +0.81% ± 20.99% (meilleur à 2/4 origines)
```

![Figure clé, protocole principal](figures/figure_cle.svg)

Une origine domine tout : le **12 novembre 2024**, où la consommation saute de 3,7 GW par rapport au mardi
précédent. Un contrefactuel montre que le lundi férié qui termine la fenêtre d'entraînement n'en explique que
10 %. La prévision J−1 de RTE, qui utilise la météo, fait 3 à 4 fois moins d'erreur ce jour-là : un choc
météorologique est l'explication la plus plausible, **non vérifiée** faute de températures dans le jeu.

![Le 12 novembre](figures/douze_novembre.svg)

**Chiffre clé du protocole principal** : MinT shrink fait varier le MASE régional de **−2,2 %** par rapport au
bottom-up, mais n'est meilleur qu'à 2 origines sur 4, et l'écart n'est **pas significatif** (p = 0,43, IC de la
différence très large).

### Étude de robustesse : 119 origines de 2024

```
MASE par niveau (robustness, 119 origines) : moyenne et écart-type

             France mean  France std  Régions mean  Régions std
SN                 0.947       0.996         0.939        0.812
base               0.368       0.387         0.427        0.291
BU                 0.352       0.356         0.427        0.291
OLS                0.366       0.385         0.439        0.316
WLS struct         0.358       0.371         0.432        0.304
WLS var            0.356       0.368         0.430        0.299
MinT shrink        0.370       0.388         0.439        0.315

MinT shrink vs BU, Régions : +2.25% ± 7.70% (meilleur à 48/119 origines)
MinT shrink vs BU, France  : +5.31% ± 20.12% (meilleur à 53/119 origines)
```

![Figure clé, robustesse](figures/figure_cle_robustesse.svg)

- **Le bottom-up est la meilleure réconciliation**, aux deux niveaux. Au niveau France, sommer les 12 prévisions
  régionales fait mieux que la prévision MSTL directe du total.
- **MinT shrink fait moins bien que le bottom-up sur les 13 séries**, sans exception. Après correction de Holm
  sur 8 comparaisons, l'écart reste significatif au test t (p ≈ 0,05), à la limite pour Wilcoxon et HAC : la
  preuve est **modérée**, le signe est constant.
- Les 4 mardis de novembre étaient tombés du côté favorable d'une distribution très étalée.

![Distribution des écarts](figures/distribution_ecarts.svg)

### L'expérience qui identifie la cause

On garde tout identique, sauf W : au lieu des résidus in-sample, on l'estime sur les **erreurs hors échantillon
des origines passées**, strictement antérieures, donc sans fuite (`reconcile_with_past_errors`, testé).

| 99 origines | MASE régional vs BU | p (Wilcoxon) |
|---|---|---|
| MinT shrink (W in-sample) | **+2,6 %** | 0,017 |
| MinT, W hors échantillon | **−0,1 %** | 0,83 |

![L'expérience sur W](figures/experience_w.svg)

Corriger W **supprime le handicap** de MinT, mais ne lui permet pas de battre le bottom-up. Plausiblement, les
erreurs régionales sont fortement corrélées et de même signe : il n'y a pas d'information croisée à exploiter.

## 5. Le chiffre clé

> **Sur 119 origines de 2024, MinT shrink augmente le MASE régional de 2,3 % par rapport au bottom-up**
> (+5,3 % au niveau France), et fait moins bien sur les 13 séries. Sur les 4 origines du protocole principal,
> il le baissait de 2,2 %, sans significativité. La cause est une hypothèse cachée de `mint_shrink` : W est
> estimée sur des résidus in-sample qui ne ressemblent pas aux erreurs de prévision.

## 6. Benchmark et études de confirmation

Notebook : [`05_benchmark_et_confirmations`](notebooks/05_benchmark_et_confirmations.ipynb). Les études de
confirmation ont été ajoutées **après** les résultats ci-dessus ; elles changent une seule chose à la fois et
gardent tout le reste du protocole.

### Les concurrents

| Famille | Solutions | Cohérentes ? |
|---|---|---|
| naïves | **Naïf** (dernière valeur), **Moy. 4 sem.**, **SN 168 h**, **SN 24 h** (la veille) | oui, par construction |
| statistique | **MSTL** seul (et **AutoETS** dans une étude de confirmation) | non (jusqu'à 2,2 GW d'écart) |
| réconciliées | **BU, OLS, WLS struct, WLS var, MinT shrink** | oui |
| externe | **RTE J−1** (faite la veille, avec la météo), et sa version **débiaisée** sans fuite | sans objet |

**Point d'attention.** La prévision RTE publiée sous-estime la consommation d'environ 1,1 GW en permanence
(différence de périmètre probable, **non vérifiée**). La version débiaisée lui retire son erreur moyenne des
52 semaines précédant chaque origine. Ce n'est pas un concurrent à information égale, c'est un repère.

### L'escalier du benchmark (2024, 119 origines)

Sortie de `uv run w37 eco2mix benchmark --which robustness` :

```
                   France mean  France std  Régions mean  Régions std
Naïf                     1.448       0.454         1.276        0.347
Moy. 4 sem.              1.102       1.087         1.072        0.871
SN 168 h                 0.947       0.996         0.939        0.812
SN 24 h                  0.736       0.554         0.758        0.462
MSTL (base)              0.368       0.387         0.427        0.291
BU                       0.352       0.356         0.427        0.291
OLS                      0.366       0.385         0.439        0.316
WLS struct               0.358       0.371         0.432        0.304
WLS var                  0.356       0.368         0.430        0.299
MinT shrink              0.370       0.388         0.439        0.315
RTE J-1                  0.410       0.182           NaN          NaN
RTE J-1 débiaisée        0.304       0.125           NaN          NaN
```

![L'escalier du benchmark](figures/benchmark_escalier.svg)

Le modèle statistique fait **61 %** de gain sur le naïf saisonnier au niveau France ; les réconciliations
forment ensuite un plateau de quelques pourcents, dont le bottom-up est le point bas. Seule la prévision RTE
débiaisée, qui a la météo, passe dessous (**13,6 %** de mieux que le bottom-up).

**Point d'attention.** Le benchmark mesure un **gain agrégé**, 1 − MASE moyen / MASE moyen de la référence,
qui se relit dans le tableau. La moyenne des gains par origine (le chiffre clé du § 5) explose entre méthodes
très différentes : elle donne RTE débiaisée à −52 % du bottom-up alors que son MASE moyen est meilleur, parce que
les jours calmes produisent des rapports énormes.

### Ce que montrent les courbes

| Question | Réponse |
|---|---|
| Le gain dépend-il de l'heure prévue ? | **oui** : au niveau France, MSTL direct et MinT battent le bottom-up de 7 à 9 % sur les heures 1-6, puis perdent 6 à 7 % sur les heures 7-24 |
| Et de la saison ? | MinT shrink ne bat le bottom-up en région qu'un mois sur 12 ; le bottom-up bat MSTL direct 10 mois sur 12 au niveau France |
| Et de la métrique ? | **non** : même classement en MASE, MAE, RMSE et MAPE |
| Combien d'origines faut-il ? | avec 4, le gain de MinT sur BU va de −10,8 % à +4,3 % selon le tirage ; il faut 22 origines en 2024, 96 en 2023, pour que tous les tirages donnent le même signe |

![Gain sur le bottom-up, heure par heure](figures/benchmark_gain_horizon.svg)

![Gain sur le bottom-up, mois par mois](figures/benchmark_gain_mois.svg)

![Classement selon la métrique](figures/benchmark_metriques.svg)

![Convergence du gain avec le nombre d'origines](figures/benchmark_convergence.svg)

Autres courbes du notebook : MASE heure par heure de toutes les solutions (`benchmark_horizon.svg`), gain de
chaque étage sur le naïf (`benchmark_gain_naif_horizon.svg`), gain sur le naïf saisonnier mois par mois
(`benchmark_gain_sn_mois.svg`).

### Trois études de confirmation

| Étude | Ce qui change | Résultat |
|---|---|---|
| **Une autre année** : 2023, 118 origines | l'année | même escalier ; le bottom-up reste au plateau bas, MinT shrink perd encore (−0,8 % en région, −1,5 % au niveau France), mais **rien n'est significatif** (p = 1 après Holm) |
| **Un autre modèle de base** : AutoETS sur 2024 | MSTL → AutoETS, dont les résidus in-sample sont de vraies erreurs à un pas | modèle bien plus faible (MASE 0,64) ; **le handicap de MinT disparaît** (−0,0 % en région) ; toutes les réconciliations à ±2 % du bottom-up, rien de significatif |
| **La cause, W** : W estimée sur les erreurs passées, sur 2023 et selon la fenêtre | W in-sample → W hors échantillon | le handicap de MinT s'efface aussi en 2023 (−1,0 % → −0,3 % en région) ; de 3 à 99 origines passées, MinT reste à ±3 % du bottom-up |

La mesure directe du mécanisme :

| Modèle de base | Corrélation entre régions vue par W (in-sample) | Corrélation réelle (hors échantillon) | MinT shrink vs BU (régions) |
|---|---|---|---|
| MSTL | 0,11 à 0,13 | 0,56 | −2,8 % |
| AutoETS | 0,49 à 0,57 | 0,66 | −0,0 % |

![Synthèse des études](figures/confirmation_synthese.svg)

### Bilan

| Conclusion | Verdict |
|---|---|
| Toute réconciliation livre des chiffres cohérents | **confirmé** dans toutes les études |
| Le levier principal est le modèle de base | **confirmé** : 31 à 66 % de gain du modèle sur le naïf saisonnier, ≤ 5 % pour la réconciliation, et la météo (RTE débiaisée) fait 12 à 14 % de mieux que la meilleure réconciliation les deux années |
| Le bottom-up est la meilleure réconciliation | **nuancé** : premier en 2024, premier ou à égalité en 2023, à égalité avec AutoETS ; jamais battu de façon significative |
| MinT shrink fait moins bien que le bottom-up | **confirmé avec MSTL** (2024 et 2023, plus faiblement en 2023), absent avec AutoETS |
| La cause est W estimée sur des résidus in-sample | **confirmé** par trois expériences (W hors échantillon en 2024 et 2023, AutoETS) |
| 4 origines ne suffisent pas à classer | **confirmé** sur les deux années |
| *Nouveau* : le classement dépend de l'horizon | avec MSTL, MinT gagne les 6 premières heures au niveau France ; profil inverse avec AutoETS |

## 7. Recommandation

1. **Livrer le bottom-up.** Il remplit l'objectif (cohérence exacte), il est le plus précis ici, et il est
   le plus facile à défendre : « le national est la somme des régions ».
2. **Ne pas utiliser `mint_shrink` avec les résidus in-sample de MSTL** (ni d'un modèle à lissage bilatéral).
   Si MinT est nécessaire, estimer W sur des **erreurs de backtest glissant**.
3. **Investir dans les modèles de base** (température, jours fériés) plutôt que dans la réconciliation : les
   plus grosses erreurs sont communes à toutes les régions, et aucune réconciliation ne les rattrape. La prévision
   RTE débiaisée, qui a la météo, fait 12 à 14 % de mieux que la meilleure réconciliation, en 2023 comme en 2024.
4. **Refaire le benchmark avant d'étendre la recommandation à un horizon infra-journalier** : avec MSTL, la
   prévision directe du total et MinT battent le bottom-up sur les 6 premières heures.

Les études de confirmation (§ 6) **renforcent** le point 1 (le bottom-up n'est jamais battu de façon
significative, sur 356 origines et deux modèles de base) et le point 2 (le handicap de MinT se reproduit en 2023
et disparaît dès que W voit de vraies erreurs). Elles **nuancent** l'ampleur : en 2023, choisir entre BU, WLS et
MinT ne change presque rien.

## Ce que ce bloc corrige ou précise par rapport au plan

1. **Le jeu ne va pas jusqu'en janvier 2026 pour toutes les régions** : la Nouvelle-Aquitaine s'arrête fin 2024.
2. **« Vérifier qu'aucune région n'a de trou »** : il y en a dans toutes les régions, chaque année (défaut de
   changement d'heure). Et une ligne absente est plus dangereuse qu'un NaN : elle ne lève aucune erreur.
3. **4 origines ne permettent pas de classer les méthodes** : le classement s'inverse sur 119 origines.
4. **Le test de biais sur résidus in-sample est sans valeur avec MSTL** ; le faire hors échantillon.
5. **Études ajoutées après les résultats** (§ 6) : l'année 2023, AutoETS, des baselines naïves et RTE J−1.
   Elles ne remplacent pas le protocole, elles l'éprouvent ; aucune n'a servi à choisir une méthode.
6. **La règle de Newey-West (1994) sous-dimensionne le HAC** sur des séries très persistantes (taille réelle
   27 % au lieu de 5 % en simulation) : on utilise ⌈1,3 √n⌉ retards (Lazarus, Lewis, Stock et Watson, 2018).

## Limites

| Limite | Piste |
|---|---|
| Ni météo ni calendrier dans les modèles de base | ajouter température et jours fériés |
| Hiérarchie plate (1 total, 12 feuilles) | hiérarchie plus profonde (régions × secteurs) |
| Deux années d'évaluation (2023, 2024), h = 24 h | d'autres horizons, notamment infra-journaliers (le classement en dépend, § 6) |
| Deux modèles de base seulement (MSTL, AutoETS) | un modèle avec météo et calendrier, comparé à RTE à information égale |
| Prévisions ponctuelles | réconciliation probabiliste (bloc 6, BayesReconPy) |
| La prévision J−1 de RTE n'est qu'un repère : elle est faite la veille, avec la météo | comparaison à information égale |

## Le code

| Brique | Fichier |
|---|---|
| Téléchargement, manifeste, sha256 | `src/w37_reconciliation/eco2mix/source.py` |
| Contrôles qualité, changement d'heure, 30 min → 1 h, hiérarchie | `src/w37_reconciliation/eco2mix/prepare.py` |
| Backtest à origines glissantes, réconciliation, expérience sur W | `src/w37_reconciliation/eco2mix/backtest.py` |
| Benchmark : baselines naïves, RTE J−1, gains, courbes, convergence, tests par étude | `src/w37_reconciliation/eco2mix/benchmark.py` |
| MASE par niveau, incohérence, chiffre clé | `src/w37_reconciliation/eco2mix/metrics.py` |
| Tests d'hypothèses (HAC, Ljung-Box, Holm, tests appariés, Diebold-Mariano) | `src/w37_reconciliation/eco2mix/inference.py` |
| Orchestration des étapes et des fichiers | `src/w37_reconciliation/eco2mix/run.py` |
| Figure clé, courbes du benchmark | `src/w37_reconciliation/eco2mix/figures.py` |
| Protocole | `configs/eco2mix.yaml` |
| Tests | `tests/unit/test_eco2mix.py`, `tests/unit/test_eco2mix_benchmark.py` (données synthétiques, < 2 s), `tests/integration/test_eco2mix_backtest.py` |

Environnement de référence : Python 3.12, `hierarchicalforecast 1.5.1`, `statsforecast 2.1.1`,
`statsmodels 0.15.0`, verrouillés dans `uv.lock`. Données téléchargées le 02/10/2026 à 22 h UTC (dates exactes
dans `data/eco2mix/raw/manifest.json`).

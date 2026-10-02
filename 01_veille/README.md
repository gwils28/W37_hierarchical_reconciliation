# Bloc 1 — Veille technologique & scientifique

> Thème W37 : *« Cohérent ne veut pas dire calibré »* — réconciliation hiérarchique, de MinT ponctuel au probabiliste.
> Budget : 20 min · Vérification des ressources réalisée le **2026-10-02** (le plan les annonçait vérifiées au 06/09/2026).

## 1. Ce qui a été fait

1. Chaque lien du plan a été ouvert (pages arXiv `abs`, page publication de R. Hyndman, PyPI, GitHub).
2. Pour chaque ressource : titre exact, auteurs, date de soumission et affirmation-clé du plan **confrontés au résumé** officiel.
3. Les écarts entre le plan et la source sont consignés dans la colonne « Écart » et en §4.
4. Synthèse et note « surcoté » reformulées à partir de ce qui a effectivement été vérifié.

> ⚠️ Vérification limitée aux **résumés** (abstracts) et pages d'accueil — pas de lecture intégrale des PDF dans le
> budget de 20 min. Les affirmations qui dépassent le résumé sont marquées *« non vérifié »*.

## 2. Les quatre ressources de l'année

| # | Ressource | Vérifié | Ce que dit le résumé | Pourquoi ça compte pour moi | Écart avec le plan |
|---|---|---|---|---|---|
| 1 | **A Forecast Combination Framework for Hierarchical and Grouped Time Series Reconciliation** — Xixi Li, Zijia Chen, James W. Taylor, Xiaojie Mao — [arXiv 2608.13886](https://arxiv.org/abs/2608.13886) — v1 14/08/2026, v2 17/08/2026 | ✅ | Construit des prévisions candidates structurées (linéairement indépendantes) pour les feuilles ; les *« mean-squared-error optimal combination weights exactly recover the widely used Minimum Trace (MinT) reconciliation »*. Cadre pénalisé modulaire : shrinkage de covariance, pénalisation des poids, estimation série par série. | MinT = combinaison optimale ⇒ toute la boîte à outils « forecast combination » (shrinkage, positivité, pénalités) devient réutilisable. Argument d'architecture : la réconciliation est un **estimateur**, pas un prorata. | Aucun. |
| 2 | **Online forecast reconciliation using linear models** — Tobias Rønlev-Knudsen, Henrik Madsen, Jan Kloppenborg Møller — [arXiv 2606.23326](https://arxiv.org/abs/2606.23326) — 22/06/2026 | ✅ | Réconciliation par régression linéaire ; schéma d'inférence **récursif** inspiré des moindres carrés récursifs (implémenté dans `PyOnlineForecast`) ; liens explicites **ridge ↔ bayésien ↔ shrinkage** ; formalisation par graphe + loi normale matricielle. Cas d'étude : **charge de réseau de chaleur** avec **hiérarchie temporelle**. | Option D du bloc 3 (mise à jour en ligne de `W`). Pont direct vers le mini-projet énergie (bloc 5). | L'affiliation DTU Compute n'apparaît pas sur la page arXiv (*non vérifié*, plausible vu les auteurs). Le résumé parle de hiérarchie **temporelle**, pas transversale : à garder en tête avant de le transposer tel quel à 12 régions. |
| 3 | **Hierarchical Bayes meets hierarchical forecasting: A flexible framework for level-focused forecasts** — Arwen Nugteren, Mahdi Abolghasemi, Kerrie Mengersen, Christopher Drovandi — [arXiv 2606.23009](https://arxiv.org/abs/2606.23009) — 22/06/2026 | ✅ | Bayésien hiérarchique qui partage l'information entre niveaux pendant l'estimation ; peut **pénaliser l'incohérence de façon souple** et **concentrer le modèle sur les niveaux pertinents pour la décision**. | Réponse formelle au « mon appro se décide au SKU » ; c'est la justification du `level_weights` du YAML (bloc 3). | La notation « λᵢ par niveau » n'est pas dans le résumé (*non vérifié* dans le texte). Nuance : la cohérence y est **souple** (pénalisée), pas exacte — incompatible avec une quality gate `coherence_tol_abs: 1e-6` sans projection finale. |
| 4 | **Billions-Scale Forecast Reconciliation** — Tianyu Wang, Matthew C. Johnson, Steven Klee, Matthew L. Malloy — [arXiv 2602.05030](https://arxiv.org/abs/2602.05030) — v1 04/02/2026, v2 06/02/2026 | ✅ | Réconciliation comme optimisation sous contraintes d'égalité/additivité ; efficace *« when the dimension of the problem exceeds four billion forecasted values »* ; pour une classe restreinte de problèmes et une perte pondérée adéquatement, **moindres carrés ≡ réconciliation par parts**, étendu aux hiérarchies chevauchantes. | Tue l'objection « trop lourd pour nos volumes ». | Le plan ne citait pas les auteurs — ajoutés. |

## 3. Marque-pages pour les blocs 2 et 4

| Ressource | Vérifié | Points confirmés |
|---|---|---|
| Panagiotelis, Gamakumara, Athanasopoulos, Hyndman — *Probabilistic forecast reconciliation: properties, evaluation and score optimisation*, **EJOR 306(2), 2023** — [page](https://robjhyndman.com/publications/coherentprob/) | ✅ | Définition de la réconciliation d'une **densité** + génération d'échantillons réconciliés ; poids optimisés sur **energy score / variogram score** (par descente de gradient stochastique) ; le **log score est impropre** pour comparer réconcilié vs non réconcilié ; dans le cas elliptique, la vraie loi prédictive est récupérable par réconciliation. |
| *Nonlinear Probabilistic Forecast Reconciliation* — Anubhab Biswas, Lorenzo Zambon, Lorenzo Nespoli, Giorgio Corani — [arXiv 2604.26668](https://arxiv.org/abs/2604.26668) — 29/04/2026 | ✅ | Deux approches : projection des échantillons sur une **variété cohérente non linéaire**, et conditionnement de la loi jointe via un algorithme fondé sur l'**Unscented Kalman Filter** (meilleur et plus rapide). Auteurs = équipe IDSIA/SUPSI de `bayesRecon` / BayesReconPy (lien direct avec le bloc 7). |
| [awesome-forecast-reconciliation](https://github.com/danigiro/awesome-forecast-reconciliation) | ✅ | CC-BY-SA-4.0, maintenu par D. Girolimetto & Y. F. Yang ; ~130 papiers, packages R (`FoReco`, `hts`, `bayesRecon`, `fabletools`…) et Python (`HierarchicalForecast`, `pyhts`, GluonTS), thèses. |
| [HierarchicalForecast — PyPI](https://pypi.org/project/hierarchicalforecast/) | ⚠️ | 1.5.1 publiée le 04/03/2026 ✅, Apache-2.0 ✅, Python 3.10 → 3.14 ✅. **Mais la dernière version est désormais 1.5.3 (24/09/2026)**. Le bloc 2 épingle volontairement **1.5.1** pour comparer au chiffre de référence du plan (voir `02_mint_from_scratch/README.md`). |

## 4. Écarts relevés (à retenir pour l'article)

- **HierarchicalForecast 1.5.3** est sortie après la rédaction du plan : épingler la version dans tout chiffre publié.
- **2606.23326** travaille sur une hiérarchie **temporelle** (et non région × canal). L'argument « estimation récursive » reste valable, mais l'exemple n'est pas un cas transversal.
- **2606.23009** : cohérence **souple**. Utile pour pondérer les niveaux, pas pour garantir contractuellement que la somme tombe juste.

## 5. Synthèse — ce qui bouge, ce que ça change pour moi

1. **La réconciliation est un estimateur, plus un post-traitement.** Combinaison de prévisions (2608.13886) ou régression multivariée régularisée (2606.23326) : elle devient régularisable, testable et défendable devant un comité d'architecture.
2. **Les deux objections commerciales tombent.** « Trop lourd » → 4 milliards de valeurs (2602.05030). « Tout recalculer chaque nuit » → estimation récursive (2606.23326).
3. **Le trou restant : le probabiliste et le non-linéaire.** Cohérent en moyenne ≠ calibré en distribution (EJOR 2023, bloc 4) ; la contrainte cesse d'être linéaire dès qu'on touche au prix (2604.26668). C'est l'intersection *forecasting × pricing* où se positionner.

## 6. Ce qui est surcoté

> **La réconciliation vendue comme un gain de précision.** MinT *préserve* l'absence de biais (`SGS = S`), il ne la crée
> pas : une seule prévision de base biaisée contamine toutes les feuilles. L'argument de vente est la **cohérence**
> (fin des arbitrages manuels), pas la précision — fréquente, jamais garantie. Ne jamais la promettre par écrit.

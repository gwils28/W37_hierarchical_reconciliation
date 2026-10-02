# Tutoriel — relancer le bloc 3 (job de réconciliation avec quality gate)

Prérequis et installation : identiques au bloc 2, voir [`../02_mint_from_scratch/TUTORIEL.md`](../02_mint_from_scratch/TUTORIEL.md),
sections 0 et 1. En résumé : `uv sync` à la racine du dépôt.

## 1. Lancer un run

```bash
uv run w37 job           # ou : make job
```

Le run enchaîne les cinq étapes de l'option A (batch matérialisé) :

```
hash_check ──▶ choose_method ──▶ reconcile ──▶ quality_gate ──▶ publication immuable
(S a-t-il      (mint_shrink ou    (Hierarchical-   (cohérence,       ou blocage
 changé ?)      repli wls_struct)  Forecast)        négatifs, MASE)
```

Sortie attendue : un JSON `lineage` + `gate`, puis une ligne `PUBLIE -> 03_system_design/outputs/...`. Le code
retour vaut **0** si l'artefact est publié et **2** si le gate le bloque, ce qui permet de chaîner la commande
dans un orchestrateur.

Extrait du gate sur la hiérarchie de démonstration :

| Niveau | MASE MinT(shrink) | MASE BottomUp | Δ |
|---|---|---|---|
| région | 1,1373 | 1,1368 | +0,05 % |
| feuille | 1,1718 | 1,1794 | −0,64 % |

## 2. Ce que le run produit

```
03_system_design/outputs/                       (ignoré par git : régénérable)
├── s_hash_registry.json                        source@version -> sha256(S + libellés)
└── forecast_reconciled/
    └── run_id=20261002T194904Z-d4717d/         une partition par run, jamais réécrite
        ├── part-0.csv                          unique_id, ds, y_reconciled, forecast_date
        ├── _lineage.json                       run_id, s_hash, w_version, méthode + raison, versions des libs
        └── _gate.json                          tous les contrôles et leurs valeurs
```

Options :

```bash
uv run w37 job --output-dir /tmp/w37            # écrire ailleurs
uv run w37 job --config mon_contrat.yaml        # autre contrat
uv run w37 job --source demo_region_canal@v2    # déclarer une nouvelle version de la hiérarchie
```

## 3. Trois expériences pour voir le gate travailler

**a. Refus d'une hiérarchie modifiée sans bump de version.** Lancez un run, renommez un canal dans
`make_region_channel_data` (`src/w37_reconciliation/data.py`, par exemple `"RHD"` → `"CHR"`), puis relancez :

```bash
uv run w37 job                                  # RuntimeError: S a changé pour demo_region_canal@v1 sans bump…
uv run w37 job --source demo_region_canal@v2    # accepté : nouvelle version déclarée
```

**b. Dégradation à un niveau.** Dans `configs/reconciliation.yaml`, passez `max_mase_degradation` à `0.0`. Le
niveau `region` (+0,05 %) suffit alors à bloquer la publication : `BLOQUE (block_publish) : ['MASE region dégradé
de +0.1% …']`, avec le code retour 2.

**c. Repli de méthode.** Passez `fallback_when.min_residuals` à `100` : avec seulement 72 résidus, le job bascule
sur `wls_struct`, et `_lineage.json` indique `"reason": "T=72 < min_residuals"`.

Remettez le YAML en état après chaque expérience (`git checkout configs/reconciliation.yaml`).

## 4. Les tests du bloc 3

```bash
uv run pytest tests/unit/test_quality_gate.py tests/unit/test_pipeline_guards.py   # < 1 s
uv run pytest tests/integration/test_job.py                                        # ≈ 1 s
```

Le principe est que **chaque règle du contrat a au moins un test où elle échoue**. Une règle qu'on n'a jamais vue
échouer ne protège rien.

| Règle du contrat | Test |
|---|---|
| `coherence_tol_abs` / `coherence_tol_rel` | `test_blocks_incoherent_forecasts`, `test_tolerates_float_noise_on_large_values` |
| valeurs non finies | `test_blocks_nan` |
| `max_negative_share` | `test_blocks_negative_share` |
| `compare_to_baseline` + `max_mase_degradation` | `test_blocks_mase_degradation_at_any_level`, `test_accepts_degradation_within_tolerance` |
| `hash_check` | `test_hash_check_refuses_changed_structure_without_bump`, `test_changed_hierarchy_without_version_bump_is_refused` |
| `fallback_when` | `test_keeps_mint_shrink_when_m_exceeds_T`, `test_falls_back_on_short_history`, `test_falls_back_on_degenerate_residuals` |
| `immutable` + `lineage` | `test_run_publishes_artifact_with_lineage` |

## 5. Où est le code

```
configs/reconciliation.yaml            # le contrat (v3) — seule source des seuils
src/w37_reconciliation/pipeline/
├── contract.py          # load_contract : lit le YAML, échoue tôt si une section manque
├── hierarchy.py         # s_hash, hash_check
├── method_selection.py  # choose_method : réutilise mint.shrinkage (même λ que le bloc 2)
├── quality_gate.py      # mase, quality_gate, GateReport
└── job.py               # run_job : enchaîne le tout et publie l'artefact
```

## 6. Adapter à une autre hiérarchie

`run_job` prend n'importe quelle `Hierarchy`. Seul le dictionnaire `levels` doit correspondre aux clés de
`level_weights` du contrat :

```python
from w37_reconciliation.data import build_hierarchy
from w37_reconciliation.pipeline import load_contract
from w37_reconciliation.pipeline.job import run_job

# mon_df : colonnes ds, y, pays, famille, sku (une ligne par sku et par date)
hier = build_hierarchy(mon_df, spec=[["pays"], ["pays", "famille"], ["pays", "famille", "sku"]])
res = run_job(hier, load_contract(), "outputs", source="dim_produit@v7",
              levels={"national": "pays", "famille": "pays/famille", "feuille": "pays/famille/sku"},
              h=13, season=52, freq="W-MON")
print(res.report.passed, res.published_to)
```

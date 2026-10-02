"""Bloc 5 — mini-projet éCO2mix : réconcilier les prévisions de consommation des 12 régions et de la France.

Une étape par module, chacune lisant les sorties de la précédente sur disque (`run.py`) :
`source` (téléchargement + manifeste) -> `prepare` (contrôles qualité, 30 min -> 1 h, hiérarchie)
-> `backtest` (origines glissantes, réconciliation) -> `metrics` (MASE par niveau, cohérence)
-> `inference` (tests d'hypothèses) -> `figures`.
"""

.PHONY: install test test-fast lint mint job biais quantiles eco2mix eco2mix-robustness bayesrecon notebooks lab clean

install:        ## crée .venv et installe le projet + groupes dev/notebooks (uv.lock)
	uv sync

test:           ## tous les tests (unitaires + intégration)
	uv run pytest

test-fast:      ## tests unitaires seuls (< 1 s)
	uv run pytest -m "not slow"

lint:
	uv run ruff check src tests

mint:           ## bloc 2 : MinT maison vs HierarchicalForecast
	uv run w37 mint

job:            ## bloc 3 : un run du job de réconciliation
	uv run w37 job

biais:          ## bloc 4 : MinT propage le biais d'une seule prévision de base
	uv run w37 biais

quantiles:      ## bloc 4 : MinT réconcilie des moyennes, pas des quantiles
	uv run w37 quantiles

eco2mix:        ## bloc 5 : télécharger (1re fois), préparer, backtester (4 origines), rapport
	uv run w37 eco2mix download
	uv run w37 eco2mix prepare
	uv run w37 eco2mix backtest
	uv run w37 eco2mix report

eco2mix-robustness:  ## bloc 5 : étude de robustesse, 119 origines de 2024 (≈ 40 min)
	uv run w37 eco2mix backtest --which robustness
	uv run w37 eco2mix report --which robustness

bayesrecon:     ## bloc 6 : MinT face à BayesReconPy sur M5 (télécharge l'exemple la 1re fois)
	uv run w37 bayesrecon run

notebooks:      ## ré-exécute les notebooks des blocs 2, 4, 5 et 6 et enregistre leurs sorties
	for nb in 02_mint_from_scratch/notebooks/*.ipynb 04_fondamentaux/notebooks/*.ipynb \
	          05_mini_projet_eco2mix/notebooks/*.ipynb 05_mini_projet_eco2mix/reconciliation_eco2mix.ipynb \
	          06_bayesreconpy/notebooks/*.ipynb; do \
		uv run jupyter nbconvert --to notebook --execute --inplace "$$nb"; \
	done

lab:            ## ouvre JupyterLab à la racine (notebooks des blocs 2, 4, 5 et 6)
	uv run jupyter lab

clean:
	rm -rf 03_system_design/outputs .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

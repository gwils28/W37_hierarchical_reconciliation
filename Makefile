.PHONY: install test test-fast lint mint job biais quantiles notebooks lab clean

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

notebooks:      ## ré-exécute les notebooks des blocs 2 et 4 et enregistre leurs sorties
	for nb in 02_mint_from_scratch/notebooks/*.ipynb 04_fondamentaux/notebooks/*.ipynb; do \
		uv run jupyter nbconvert --to notebook --execute --inplace "$$nb"; \
	done

lab:            ## ouvre JupyterLab à la racine (notebooks des blocs 2 et 4)
	uv run jupyter lab

clean:
	rm -rf 03_system_design/outputs .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

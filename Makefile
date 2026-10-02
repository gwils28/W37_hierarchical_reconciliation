.PHONY: install test test-fast lint mint job notebooks lab clean

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

notebooks:      ## ré-exécute les notebooks du bloc 2 et enregistre leurs sorties
	for nb in 02_mint_from_scratch/notebooks/*.ipynb; do \
		uv run jupyter nbconvert --to notebook --execute --inplace "$$nb"; \
	done

lab:            ## ouvre JupyterLab sur les notebooks du bloc 2
	uv run jupyter lab 02_mint_from_scratch/notebooks

clean:
	rm -rf 03_system_design/outputs .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

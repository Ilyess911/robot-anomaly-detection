.PHONY: setup test lint format benchmark notebooks verify clean

VENV := .venv
PY   := $(VENV)/bin/python

setup:  ## Crée l'environnement et installe les versions exactes du benchmark
	python3 -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements-lock.txt
	$(PY) -m ipykernel install --user --name robot-anomaly \
		--display-name "Python (robot-anomaly)"

test:  ## Les tests de données et de protocole
	$(PY) -m pytest tests/ -q

lint:  ## Style et erreurs statiques
	$(PY) -m ruff check src scripts tests
	$(PY) -m ruff format --check src scripts tests

format:  ## Reformate
	$(PY) -m ruff format src scripts tests
	$(PY) -m ruff check --fix src scripts tests

benchmark:  ## Rejoue le comparatif et réécrit reports/benchmark.json
	$(PY) scripts/benchmark.py --output reports/benchmark.json

notebooks:  ## Ouvre les notebooks, à exécuter dans l'ordre 01 à 05
	$(VENV)/bin/jupyter notebook notebooks/

# La cible de la CI. L'ordre n'est pas arbitraire : le lint échoue en une
# seconde, les tests en dix, le benchmark en plusieurs minutes. On perd le
# moins de temps possible avant de savoir que quelque chose ne va pas.
verify: lint test benchmark  ## Tout, dans l'ordre du plus rapide au plus lent

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

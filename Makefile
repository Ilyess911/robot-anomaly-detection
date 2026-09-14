.PHONY: setup test lint format benchmark experiments deployment figures reproduce detect notebooks demo verify clean

VENV := .venv
PY   := $(VENV)/bin/python

setup:  ## Create the environment with the exact versions behind the published numbers
	python3 -m venv $(VENV)
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements-lock.txt
	$(PY) -m ipykernel install --user --name robot-anomaly \
		--display-name "Python (robot-anomaly)"

test:  ## Dataset, protocol, duplicate and detector tests
	$(PY) -m pytest tests/ -q

lint:  ## Style and static errors
	$(PY) -m ruff check src scripts tests app
	$(PY) -m ruff format --check src scripts tests app

format:  ## Reformat in place
	$(PY) -m ruff format src scripts tests app
	$(PY) -m ruff check --fix src scripts tests app

benchmark:  ## Baselines and supervised models under both split protocols
	$(PY) scripts/benchmark.py --output reports/benchmark.json

experiments:  ## One-class detection under repeated CV, calibration, transfer
	$(PY) scripts/experiments.py --output reports/experiments.json

deployment:  ## Cost-sensitive operating point and inference cost
	$(PY) scripts/deployment.py --output reports/deployment.json

figures:  ## Redraw every image the README publishes, from reports/
	$(PY) scripts/figures.py --reports reports --output assets

reproduce: benchmark experiments deployment figures  ## Every published number and figure, from scratch

detect:  ## Score a subset with a fitted detector, see scripts/detect.py --help
	$(PY) scripts/detect.py --subset LP1

demo:  ## Local Streamlit demo, needs requirements-demo.txt
	$(VENV)/bin/streamlit run app/streamlit_app.py

notebooks:  ## Open notebooks/, to run in order 01 to 05
	$(VENV)/bin/jupyter notebook notebooks/

# The CI target. The order is not arbitrary: lint fails in a second, the tests
# in twenty, the benchmark in minutes. As little time as possible is spent
# before finding out that something is wrong.
verify: lint test benchmark experiments deployment  ## Everything, fastest to slowest

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

PY ?= .venv/bin/python
.PHONY: setup run test smoke clean
setup:            ## venv + deps
	python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements.txt
run:              ## needs data/cs-training.csv (Kaggle Give Me Some Credit) -> results/validation_report.html
	$(PY) run.py
smoke:            ## synthetic data, no download
	$(PY) run.py --synthetic --out results_smoke
test:
	$(PY) -m pytest -q
clean:
	rm -rf results_smoke .pytest_cache

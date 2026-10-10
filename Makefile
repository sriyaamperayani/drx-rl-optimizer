.PHONY: install test lint experiment ablations

install:
	pip install -e ".[dev]"

test:
	pytest --cov=drxgym --cov-report=term-missing

lint:
	ruff check src tests scripts

experiment:
	drxgym experiment --out runs/fr2_lambda2

ablations:
	drxgym experiment --out runs/fr2_lambda5 --lambda-latency 5
	drxgym experiment --out runs/fr1_lambda2 --beam-search-ms 0

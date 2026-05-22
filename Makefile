.PHONY: install test lint fmt run clean

install:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check src tests

fmt:
	ruff format src tests

run:
	debater

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ .coverage htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +

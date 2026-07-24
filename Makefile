# Development convenience targets. Use PYTHON=... to point at a specific
# interpreter (defaults to python3).
PYTHON ?= python3

.DEFAULT_GOAL := help

.PHONY: help install lint format format-check typecheck test cov check clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev dependencies (editable)
	$(PYTHON) -m pip install -e ".[dev]"

lint: ## Run ruff lint checks
	ruff check .

format: ## Auto-format the codebase with ruff
	ruff format .

format-check: ## Verify formatting without modifying files
	ruff format --check .

typecheck: ## Run mypy (advisory)
	mypy

test: ## Run the offline test suite
	pytest

cov: ## Run tests with a coverage report
	pytest --cov=onnx_graph_diff --cov-report=term-missing

check: lint format-check test ## Run the full pre-PR gate (lint + format + tests)

clean: ## Remove caches and build artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

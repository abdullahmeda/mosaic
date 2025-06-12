.PHONY: help install install-dev test lint lint-check format type-check check build clean publish-test publish

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package
	pip install -e .

install-dev:  ## Install the package with development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest tests/ -v

lint:  ## Run linting and auto-fix issues
	ruff check . --fix --exit-non-zero-on-fix --unsafe-fixes

lint-check:  ## Run linting without auto-fixing (for CI)
	ruff check .

format:  ## Format code
	ruff format .

type-check:  ## Run type checking
	mypy .

check:  ## Run all checks (linting, formatting, type checking, tests)
	@echo "Running linting..."
	ruff check .
	@echo "Checking code formatting..."
	ruff format --check .
	@echo "Running type checking..."
	mypy .
	@echo "Running tests..."
	pytest tests/ -v
	@echo "All checks passed!"

build:  ## Build the package
	python -m build

clean:  ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .ruff_cache/
	rm -rf .mypy_cache/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

publish-test:  ## Publish to test PyPI
	python -m twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	python -m twine upload dist/*
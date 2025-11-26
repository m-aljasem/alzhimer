# Makefile for alzhimer

.PHONY: help install install-dev train test clean lint format setup-data docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make install-dev   - Install development dependencies"
	@echo "  make setup-data    - Download dataset from Kaggle"
	@echo "  make train         - Train the model"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean generated files"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

setup-data:
	python ../../setup_data.py --project alzhimer

train:
	python src/train.py

test:
	pytest tests/ -v

lint:
	flake8 src/ tests/
	pylint src/

format:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf dist/
	rm -rf build/

docker-build:
	docker build -t alzhimer:latest .

docker-run:
	docker run -p 8501:8501 alzhimer:latest

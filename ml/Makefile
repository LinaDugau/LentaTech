.PHONY: help build up down logs clean test

help:
	@echo "Available commands:"
	@echo "  make build    - Build Docker images"
	@echo "  make up       - Start all services"
	@echo "  make down     - Stop all services"
	@echo "  make logs     - View logs"
	@echo "  make clean    - Clean temporary files"
	@echo "  make test     - Run tests"

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "Services started!"
	@echo "UI: http://localhost:8501"
	@echo "API: http://localhost:8000"
	@echo "API Docs: http://localhost:8000/docs"

down:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	rm -rf tmp/jobs/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

test:
	@echo "Running tests..."
	python -m pytest tests/ -v

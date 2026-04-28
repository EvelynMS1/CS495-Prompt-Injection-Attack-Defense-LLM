# Makefile for setting up Python 3.13, Poetry, and venv
# Compatible with macOS and Windows (Git Bash / WSL / MSYS2)

PYTHON_VERSION := 3.13
VENV_DIR := .venv

# Detect OS
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    PYTHON := python
    VENV_PYTHON := $(VENV_DIR)/Scripts/python
    VENV_ACTIVATE := $(VENV_DIR)/Scripts/activate
    POETRY := $(VENV_DIR)/Scripts/poetry
else
    DETECTED_OS := $(shell uname -s)
    PYTHON := python3
    VENV_PYTHON := $(VENV_DIR)/bin/python
    VENV_ACTIVATE := $(VENV_DIR)/bin/activate
    POETRY := $(VENV_DIR)/bin/poetry
endif

.PHONY: all setup install-python install-poetry venv install clean help

all: setup ## Run full setup

help: ## Show this help
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

setup: install-python install-poetry venv install ## Full setup: Python, Poetry, venv, and dependencies

install-python: ## Install Python 3.13
ifeq ($(DETECTED_OS),Darwin)
	@echo "==> Installing Python $(PYTHON_VERSION) on macOS..."
	@if command -v brew >/dev/null 2>&1; then \
		brew install python@$(PYTHON_VERSION) || brew upgrade python@$(PYTHON_VERSION); \
	else \
		echo "Error: Homebrew is required on macOS. Install it from https://brew.sh"; \
		exit 1; \
	fi
else ifeq ($(DETECTED_OS),Windows)
	@echo "==> Installing Python $(PYTHON_VERSION) on Windows..."
	@if command -v winget >/dev/null 2>&1; then \
		winget install Python.Python.$(PYTHON_VERSION) --accept-source-agreements --accept-package-agreements; \
	elif command -v choco >/dev/null 2>&1; then \
		choco install python --version=$(PYTHON_VERSION) -y; \
	else \
		echo "Error: winget or Chocolatey is required on Windows."; \
		echo "  Install winget: https://aka.ms/getwinget"; \
		echo "  Install Chocolatey: https://chocolatey.org/install"; \
		exit 1; \
	fi
else
	@echo "==> Installing Python $(PYTHON_VERSION) on Linux..."
	@if command -v apt-get >/dev/null 2>&1; then \
		sudo add-apt-repository -y ppa:deadsnakes/ppa && \
		sudo apt-get update && \
		sudo apt-get install -y python$(PYTHON_VERSION) python$(PYTHON_VERSION)-venv; \
	elif command -v dnf >/dev/null 2>&1; then \
		sudo dnf install -y python$(PYTHON_VERSION); \
	else \
		echo "Error: Unsupported Linux distribution. Install Python $(PYTHON_VERSION) manually."; \
		exit 1; \
	fi
endif
	@echo "==> Python installed:"
	@$(PYTHON) --version

install-poetry: ## Install Poetry
	@echo "==> Installing Poetry..."
ifeq ($(DETECTED_OS),Windows)
	@if not exist "$(POETRY)" ( \
		pip install poetry \
	)
else
	@if ! command -v poetry >/dev/null 2>&1; then \
		curl -sSL https://install.python-poetry.org | $(PYTHON) -; \
	fi
endif
	@echo "==> Poetry installed:"
	@poetry --version

venv: ## Create virtual environment
	@echo "==> Creating virtual environment in $(VENV_DIR)..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo "==> Virtual environment created at $(VENV_DIR)"
	@echo "    Activate with: source $(VENV_ACTIVATE)"

install: venv ## Install project dependencies with Poetry inside venv
	@echo "==> Installing dependencies with Poetry..."
	@. $(VENV_ACTIVATE) && poetry install --no-interaction 2>/dev/null || \
		(echo "==> No pyproject.toml found. Initializing Poetry project..." && \
		 . $(VENV_ACTIVATE) && poetry init --no-interaction --python="^$(PYTHON_VERSION)" && \
		 poetry install --no-interaction)
	@echo "==> Dependencies installed."

clean: ## Remove virtual environment
	@echo "==> Removing virtual environment..."
	rm -rf $(VENV_DIR)
	@echo "==> Clean complete."

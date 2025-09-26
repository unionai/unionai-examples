# Makefile for unionai-examples testing framework
# For local development - GitHub Actions handles CI/CD

.PHONY: test test-quick test-dry-run clean help setup-venv install-deps check-venv

# Default subdirectory
DIR ?= v2
PYTHON ?= python
VENV_PATH ?= $(HOME)/.venv

# Default target
help:
	@echo "Local Development Commands:"
	@echo "  setup-venv           - Create virtual environment with uv and install flyte"
	@echo "  test [DIR=v2]        - Run all example tests in subdirectory"
	@echo "  test-quick [DIR=v2]  - Run tests with shorter timeout (60s)"
	@echo "  test-dry-run [DIR=v2] - Show what would be run without executing"
	@echo "  install-deps         - Install additional testing dependencies"
	@echo "  clean                - Clean test logs and reports"
	@echo "  check-venv           - Check if virtual environment is active"
	@echo "  help                 - Show this help"
	@echo ""
	@echo "Examples:"
	@echo "  make setup-venv                    # First time setup"
	@echo "  make test DIR=tutorials            # Test tutorials directory"
	@echo "  make test-quick DIR=v2 FILTER=hello  # Quick test with filter"
	@echo "  make test-dry-run DIR=integrations # Preview what would run"
	@echo ""
	@echo "Environment Variables:"
	@echo "  DIR       - Subdirectory to test (default: v2)"
	@echo "  FILTER    - Pattern to filter tests"
	@echo "  PYTHON    - Python executable (default: python)"
	@echo "  VENV_PATH - Virtual environment path (default: ~/.venv)"

# Create virtual environment using uv (matches your flyte-venv function)
setup-venv:
	@echo "🐍 Setting up virtual environment with uv..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "❌ uv is not installed. Please install it first:"; \
		echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi
	uv venv --clear --python cpython@3.13 $(VENV_PATH)
	@echo "📦 Installing flyte with prerelease packages..."
	$(VENV_PATH)/bin/python -m pip install --no-cache --prerelease=allow --upgrade flyte
	@echo "✅ Virtual environment ready!"
	@echo "   To activate: source $(VENV_PATH)/bin/activate"
	@echo "   Or use: make test (will check activation automatically)"

# Check if virtual environment is active
check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "⚠️  Virtual environment not active."; \
		echo "   Run: source $(VENV_PATH)/bin/activate"; \
		echo "   Or run: make setup-venv"; \
		exit 1; \
	else \
		echo "✅ Virtual environment active: $$VIRTUAL_ENV"; \
	fi

# Install additional common dependencies
install-deps: check-venv
	@echo "📦 Installing additional testing dependencies..."
	$(PYTHON) -m pip install pandas numpy requests pydantic

# Run all tests (with venv check)
test: check-venv
	$(PYTHON) tests/test_runner.py $(DIR) --config tests/config.json $(if $(FILTER),--filter "$(FILTER)")

# Run tests with shorter timeout for quick feedback
test-quick: check-venv
	$(PYTHON) tests/test_runner.py $(DIR) --config tests/config.json --timeout 60 $(if $(FILTER),--filter "$(FILTER)")

# Show what would be run without executing
test-dry-run: check-venv
	$(PYTHON) tests/test_runner.py $(DIR) --config tests/config.json --dry-run $(if $(FILTER),--filter "$(FILTER)")

# Clean logs and reports
clean:
	rm -rf tests/logs/*
	@echo "Cleaned test logs and reports"

# Development helpers
dev-setup: install-deps
	@echo "Development environment setup complete"

# Check if test framework works
check:
	$(PYTHON) tests/test_runner.py --help
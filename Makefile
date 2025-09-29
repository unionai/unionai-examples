# Makefile for unionai-examples testing framework

.PHONY: test test-dry-run clean help setup-venv update-flyte check-venv

# Default target
help:
	@echo "Testing Framework Commands:"
	@echo "  test [FILE=path] [FILTER=pattern] - Run tests (cloud execution)"
	@echo "  test-dry-run [FILE=path] [FILTER=pattern] - Preview tests (local validation)"
	@echo "  clean                    - Clean test logs and reports"
	@echo "  setup-venv              - Create virtual environment with uv"
	@echo "  update-flyte            - Update to latest flyte version"
	@echo ""
	@echo "Examples:"
	@echo "  make test                                    # Run all tests in cloud"
	@echo "  make test-dry-run                           # Preview all tests locally"
	@echo "  make test FILE=v2/user-guide/getting-started/hello.py  # Test specific file in cloud"
	@echo "  make test-dry-run FILE=v2/tutorials/trading_agents/main.py  # Preview specific file"
	@echo "  make test FILTER=hello                      # Test files matching 'hello'"
	@echo "  make test-dry-run FILTER=user-guide         # Preview user-guide examples"
	@echo ""
	@echo "Parameters:"
	@echo "  FILE    - Specific file path to test (takes precedence over FILTER)"
	@echo "  FILTER  - Pattern to match against file paths"

# Environment variables
PYTHON ?= python
VENV_PATH ?= $(HOME)/.venv

# Create virtual environment using uv
setup-venv:
	@echo "🐍 Setting up virtual environment with uv..."
	@if ! command -v uv >/dev/null 2>&1; then \
		echo "❌ uv not found. Please install uv first:"; \
		echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi
	uv venv $(VENV_PATH) --python 3.12
	@echo "✅ Virtual environment created at $(VENV_PATH)"
	@echo "� Activate with: source $(VENV_PATH)/bin/activate"

# Check if virtual environment is active
check-venv:
	@if [ -z "$$VIRTUAL_ENV" ]; then \
		echo "❌ Virtual environment not active"; \
		echo "💡 Activate with: source $(VENV_PATH)/bin/activate"; \
		echo "💡 Or create new one with: make setup-venv"; \
		exit 1; \
	fi
	@echo "✅ Virtual environment active: $$VIRTUAL_ENV"

# Update flyte to latest version
update-flyte: check-venv
	@echo "🚀 Updating flyte to latest version..."
	uv pip install --no-cache --prerelease=allow --upgrade --force-reinstall flyte
	@echo "✅ Flyte updated! Current version:"
	@$(PYTHON) -c "import flyte; print(f'   flyte {flyte.__version__}')"

# Run tests with cloud execution
test: check-venv
	@if [ -n "$(FILE)" ]; then \
		echo "🎯 Testing specific file: $(FILE)"; \
		$(PYTHON) test/test_runner.py --production --file "$(FILE)"; \
	elif [ -n "$(FILTER)" ]; then \
		echo "🔍 Testing files matching: $(FILTER)"; \
		$(PYTHON) test/test_runner.py --production --filter "$(FILTER)"; \
	else \
		echo "🚀 Running all production tests..."; \
		$(PYTHON) test/test_runner.py --production; \
	fi

# Preview tests with local validation
test-dry-run: check-venv
	@if [ -n "$(FILE)" ]; then \
		echo "🎯 Previewing specific file: $(FILE)"; \
		$(PYTHON) test/test_runner.py --production --dry-run --file "$(FILE)"; \
	elif [ -n "$(FILTER)" ]; then \
		echo "🔍 Previewing files matching: $(FILTER)"; \
		$(PYTHON) test/test_runner.py --production --dry-run --filter "$(FILTER)"; \
	else \
		echo "👀 Previewing all production tests..."; \
		$(PYTHON) test/test_runner.py --production --dry-run; \
	fi

# Clean logs and reports
clean:
	rm -rf test/logs/*
	@echo "🧹 Cleaned test logs and reports"
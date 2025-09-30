# Makefile for unionai-examples testing framework

.PHONY: test test-preview test-local clean help setup-venv update-flyte check-venv

# Default target
help:
	@echo "Testing Framework Commands:"
	@echo "  test [FILE=path] [FILTER=pattern] - Run tests (cloud execution)"
	@echo "  test-local [FILE=path] [FILTER=pattern] - Run tests (local execution with flyte run --local)"
	@echo "  test-preview [FILE=path] [FILTER=pattern] - Preview tests (show what would run)"
	@echo "  clean                    - Clean test logs and reports"
	@echo "  setup-venv              - Create virtual environment with uv"
	@echo "  update-flyte            - Update to latest flyte version"
	@echo ""
	@echo "Examples:"
	@echo "  make test                                    # Run all tests in cloud"
	@echo "  make test-local                             # Run all tests locally"
	@echo "  make test-preview                           # Preview all tests"
	@echo "  make test FILE=v2/user-guide/getting-started/hello.py  # Test specific file in cloud"
	@echo "  make test-local FILE=v2/tutorials/trading_agents/main.py  # Test specific file locally"
	@echo "  make test-preview FILE=v2/tutorials/trading_agents/main.py  # Preview specific file"
	@echo "  make test FILTER=hello                      # Test files matching 'hello'"
	@echo "  make test-local FILTER=user-guide          # Test user-guide examples locally"
	@echo "  make test-preview FILTER=user-guide         # Preview user-guide examples"
	@echo "  make test-local VERBOSE=vv FILE=hello.py    # Test with medium verbosity"
	@echo "  make test-local VERBOSE=3 FILTER=user-guide # Test with maximum verbosity"
	@echo ""
	@echo "Parameters:"
	@echo "  FILE    - Specific file path to test (takes precedence over FILTER)"
	@echo "  FILTER  - Pattern to match against file paths"
	@echo "  VERBOSE - Verbosity level: 0/'' (quiet), 1/v (-v), 2/vv (-vv), 3/vvv (-vvv)"

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

# Run tests with local execution using flyte run --local
test-local: check-venv
	@if [ -n "$(FILE)" ]; then \
		echo "🎯 Testing specific file locally: $(FILE)"; \
		$(PYTHON) test/test_runner.py --production --local --file "$(FILE)" $(if $(VERBOSE),--verbose "$(VERBOSE)"); \
	elif [ -n "$(FILTER)" ]; then \
		echo "🔍 Testing files matching locally: $(FILTER)"; \
		$(PYTHON) test/test_runner.py --production --local --filter "$(FILTER)" $(if $(VERBOSE),--verbose "$(VERBOSE)"); \
	else \
		echo "🖥️  Running all production tests locally..."; \
		$(PYTHON) test/test_runner.py --production --local $(if $(VERBOSE),--verbose "$(VERBOSE)"); \
	fi

# Preview tests
test-preview: check-venv
	@if [ -n "$(FILE)" ]; then \
		echo "🎯 Previewing specific file: $(FILE)"; \
		$(PYTHON) test/test_runner.py --production --preview --file "$(FILE)"; \
	elif [ -n "$(FILTER)" ]; then \
		echo "🔍 Previewing files matching: $(FILTER)"; \
		$(PYTHON) test/test_runner.py --production --preview --filter "$(FILTER)"; \
	else \
		echo "👀 Previewing all production tests..."; \
		$(PYTHON) test/test_runner.py --production --preview; \
	fi

# Clean logs and reports
clean:
	rm -rf test/logs/*
	@echo "🧹 Cleaned test logs and reports"
#!/bin/bash
# Setup script for unionai-examples testing environment
# This matches the flyte-venv() function in your .zshrc

set -e

VENV_PATH="${HOME}/.venv"
PYTHON_VERSION="cpython@3.13"

echo "🐍 Setting up Flyte virtual environment..."

# Check if uv is available
if ! command -v uv >/dev/null 2>&1; then
    echo "❌ uv is not installed. Please install it first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "🗑️  Clearing existing virtual environment..."
uv venv --clear --python ${PYTHON_VERSION} ${VENV_PATH}

echo "📦 Installing flyte with prerelease packages..."
${VENV_PATH}/bin/python -m pip install --no-cache --prerelease=allow --upgrade flyte

echo "✅ Virtual environment setup complete!"
echo ""
echo "To activate the environment:"
echo "   source ${VENV_PATH}/bin/activate"
echo ""
echo "Or add this to your shell profile:"
echo "   alias flyte-activate='source ${VENV_PATH}/bin/activate'"
echo ""
echo "To run tests:"
echo "   source ${VENV_PATH}/bin/activate"
echo "   make test DIR=v2"
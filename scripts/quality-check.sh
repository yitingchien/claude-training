#!/bin/bash

set -e

echo "🧹 Running code quality checks..."
echo

echo "📝 Running black (code formatting)..."
uv run black --check --diff .

echo
echo "📋 Running isort (import sorting)..."
uv run isort --check-only --diff .

echo
echo "🔍 Running flake8 (linting)..."
uv run flake8 .

echo
echo "🔎 Running mypy (type checking)..."
uv run mypy backend/ main.py

echo
echo "✅ All quality checks passed!"
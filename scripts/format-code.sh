#!/bin/bash

set -e

echo "🎨 Formatting code..."
echo

echo "📝 Running black (code formatting)..."
uv run black .

echo
echo "📋 Running isort (import sorting)..."
uv run isort .

echo
echo "✅ Code formatting complete!"
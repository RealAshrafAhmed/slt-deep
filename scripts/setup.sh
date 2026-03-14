#!/bin/bash
# Complete workspace setup script
# This script installs all dependencies and builds workspace packages

set -e  # Exit on any error

echo "🚀 Setting up workspace..."

if ! command -v mise > /dev/null 2>&1; then
    echo "❌ mise is required for this repository." >&2
    echo "   Install it first: https://mise.jdx.dev/getting-started.html" >&2
    exit 1
fi

# Install all dependencies (including workspace packages)
echo "📦 Installing all dependencies and building workspace packages..."
uv sync --extra dev

# Install mise-managed tools.
echo "🧰 Installing mise-managed tools..."
mise install
echo "✅ mise tools installed"

# Trust all notebooks to enable widgets and interactive elements
echo "🔐 Trusting notebooks to enable widgets..."
if find . -name "*.ipynb" -type f | head -1 > /dev/null 2>&1; then
    uv run jupyter trust **/*.ipynb 2>/dev/null || echo "ℹ️  Some notebooks may need manual trusting"
    echo "✅ Notebooks trusted"
else
    echo "ℹ️  No notebooks found to trust"
fi

# Clean up any existing cache/checkpoint files
echo "🧹 Cleaning up temporary files..."
./scripts/cleanup.sh > /dev/null 2>&1 || true

echo "✅ Setup complete!"
echo ""
echo "🎉 Your research monorepo is ready! Next steps:"
echo "  • Run 'mise tasks' to inspect available monorepo tasks"
echo "  • Run './scripts/lab.sh' to start Jupyter Lab"
echo "  • Run 'uv run pre-commit install' to set up pre-commit hooks"
echo "  • Check README.md for more commands and customization tips"
echo "  • Start coding in notebooks/ or create your first project in projects/"

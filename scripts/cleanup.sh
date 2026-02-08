#!/bin/bash
# Comprehensive cleanup script for Jupyter, Python, and build artifacts

echo "🧹 Starting comprehensive workspace cleanup..."

# Remove all .ipynb_checkpoints directories
echo "Cleaning Jupyter checkpoints..."
CHECKPOINTS_FOUND=$(find . -name ".ipynb_checkpoints" -type d 2>/dev/null | wc -l)
find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null || true
echo "✅ Removed $CHECKPOINTS_FOUND .ipynb_checkpoints directories"

# Remove Python cache files
echo "Cleaning Python cache..."
CACHE_FOUND=$(find . -name "__pycache__" -type d 2>/dev/null | wc -l)
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

PYC_FOUND=$(find . -name "*.pyc" 2>/dev/null | wc -l)
find . -name "*.pyc" -delete 2>/dev/null || true
echo "✅ Removed $CACHE_FOUND __pycache__ directories and $PYC_FOUND .pyc files"

# Remove build artifacts
echo "Cleaning build artifacts..."
rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage 2>/dev/null || true
echo "✅ Removed build/dist/egg-info directories"

# Remove IDE files (optional - uncomment if needed)
# echo "Cleaning IDE files..."
# rm -rf .vscode/settings.json .idea/ *.swp *.swo 2>/dev/null || true
# echo "✅ Removed IDE temporary files"

# Remove Jupyter runtime files (optional - uncomment if needed)
# echo "Cleaning Jupyter runtime cache..."
# rm -rf ~/.local/share/jupyter/runtime/* 2>/dev/null || true
# echo "✅ Cleaned Jupyter runtime cache"

# Show disk space saved (rough estimate)
echo ""
echo "🎉 Cleanup complete!"
echo "💾 Freed up space from cached files and temporary artifacts"
echo ""
echo "To prevent future clutter:"
echo "  • Run this script periodically: ./scripts/cleanup.sh"
echo "  • Use .gitignore to exclude temp files (already configured)"
echo "  • Consider enabling pre-commit hooks: uv run pre-commit install"

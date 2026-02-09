#!/bin/zsh
# Register the current Python environment as a Jupyter kernel and start Jupyter Lab

# Register kernel
uv run python -m ipykernel install --user --name=slt-deep

# Trust notebooks to enable widgets (in case new ones were added)
if find . -name "*.ipynb" -type f | head -1 > /dev/null 2>&1; then
    echo "🔐 Trusting notebooks..."
    uv run jupyter trust **/*.ipynb 2>/dev/null || true
fi

# Start Jupyter Lab
uv run jupyter lab

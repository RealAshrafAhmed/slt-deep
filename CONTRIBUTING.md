# Contributing Guide

## Development Setup

1. Clone the repository and navigate to it
2. Install dependencies:
   ```bash
   uv sync --extra dev
   ```
3. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Workflow

### Before Committing

Pre-commit hooks will automatically run, but you can manually check:

```bash
uv run ruff format .     # Format code
uv run ruff check .      # Lint code
uv run pytest           # Run tests
```

### Adding a New Package

```bash
# Create the package
uv init packages/mypackage --lib

# Sync to register it
uv sync --extra dev

# Add dependencies to it
uv add numpy --package mypackage
```

### Adding a New Project

```bash
# Create the project
uv init projects/myproject --package

# Sync to register it
uv sync --extra dev

# Add local package dependencies
cd projects/myproject
# Edit pyproject.toml to add: dependencies = ["mypackage"]
uv sync --extra dev
```

### Writing Tests

- Place tests in `tests/` directory within each package
- Name files `test_*.py` or `*_test.py`
- Run with `uv run pytest`

### Jupyter Notebooks

- Shared notebooks go in `notebooks/`
- Project-specific notebooks go in `projects/yourproject/`
- Always restart kernel after updating packages
- Keep notebooks clean (use `nbqa-ruff` for linting)

### Code Style

- Line length: 100 characters
- Formatting: Handled by `ruff format`
- Linting: Handled by `ruff check`
- Type hints: Checked by `pyright`

### Git Workflow

1. Create a branch for your changes
2. Make your changes
3. Commit (pre-commit hooks will run automatically)
4. Push and create a PR

## Questions?

Check the main [README.md](README.md) for more details.

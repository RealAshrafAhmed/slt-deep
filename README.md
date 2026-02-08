# Python Research Monorepo Template
*A production-ready template for managing multiple research projects and shared libraries in a single repository using modern Python tooling.*

## Why You Need This

**Tired of research code chaos?** This template solves the problems every researcher faces:

- 🔄 **Stop copying utility functions** between notebooks and projects
- 🚫 **End "it works on my machine"** dependency conflicts
- 🤝 **Share code with collaborators** without publishing to PyPI
- 📚 **Keep notebooks clean** while building reusable libraries
- 📝 **Write papers seamlessly** - LaTeX integration with your analysis
- ⚡ **Fast setup** - from clone to working in under 2 minutes

Transform scattered research code into organized, professional projects that others can actually use.

## Cheatsheet

```bash
# Install all dependencies (main + dev + workspace packages)
uv sync --extra dev

# Install only main dependencies (+ workspace packages)
uv sync

# Start Jupyter Lab (with auto-kernel registration)
./scripts/lab.sh

# Or start Jupyter Lab directly
uv run jupyter-lab

# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Run tests
uv run pytest

# Clean workspace (removes 100s of cache files!)
./scripts/cleanup.sh

# Compile LaTeX papers
./scripts/publish.sh compile example_paper

# Convert notebooks to PDF
./scripts/publish.sh notebook projects/slt/quasi-singular-models/notebooks/fit_dataset.ipynb

# Set up pre-commit hooks (after installing dev dependencies)
uv run pre-commit install
```

---

# For those who want to read...

## Why This Structure?

**Problem:** Research codebases often become messy: scattered notebooks, duplicated utility code, inconsistent dependencies, and no testing infrastructure.

**Solution:** This monorepo structure provides:

- **📦 Shared Packages**: Write utilities once, use everywhere. Packages can be imported in any notebook or project.
- **🔬 Isolated Projects**: Each research project gets its own directory with dependencies, while sharing common code.
- **📓 Notebook-Friendly**: Jupyter integration that actually works - imports from local packages seamlessly.
- **🔄 Reproducibility**: Lockfile ensures everyone uses the same dependency versions. Pre-commit hooks enforce code quality.
- **⚡ Fast Iteration**: `uv` makes dependency management fast and ergonomic.
- **🧪 Testing Ready**: Example test structure so you can validate critical code paths.
- **📄 Publication Ready**: LaTeX integration - compile notebooks to PDFs or write papers that reference your analysis.

**When to use this:**
- Working on multiple related research projects
- Building reusable tools for experiments (MCMC utilities, plotting helpers, data loaders)
- Collaborating with others who need your code
- Writing papers that reference computational analysis
- Publishing research with reproducible code

**When NOT to use this:**
- Single standalone Jupyter notebook analysis
- Quick exploratory data analysis (use a simple directory instead)

## How to Use This Template

### 1. Initial Setup

```bash
# Clone or copy this template
git clone <this-repo-url> my-research
cd my-research

# Complete setup (installs all deps + builds workspace packages)
./scripts/setup.sh

# OR do it manually:
# Install all dependencies (main + dev + workspace packages)
uv sync --extra dev

# Set up pre-commit hooks (optional but recommended)
uv run pre-commit install

# Start Jupyter Lab (with auto-kernel registration)
./scripts/lab.sh
```


## Dependency Structure

This project uses a structured approach to dependencies:

- **Production dependencies**: Core packages required for running simulations, including `papermill` for programmatic notebook execution.
- **Dev dependencies**: Development tools including Jupyter Lab, formatters, linters, test runners, and interactive notebook tools.
- **Publishing dependencies**: LaTeX and document generation tools including Quarto and nbconvert for creating papers.

To install different dependency sets:

```bash
# Install only main dependencies (production)
uv sync

# Install main + dev dependencies (development)
uv sync --extra dev

# Install all dependency groups (including publishing tools)
uv sync --all-groups
```
## Common Commands

### Packages

```bash
# Sync workspace packages and dependencies (do this after adding new packages)
uv sync --extra dev

# Add a dependency to a specific package
cd packages/my_package && uv add numpy

# Add a dev dependency to root
uv add --dev pytest

# Add a main dependency to root
uv add papermill

# Force full reinstall (nuclear option)
rm uv.lock && uv sync --extra dev
```

### Jupyter

```bash
# Start Jupyter Lab with automatic kernel registration (recommended)
./scripts/lab.sh

# Or start Jupyter Lab directly
uv run jupyter-lab

# Register kernel manually (if kernel issues)
uv run python -m ipykernel install --user --name=research-monorepo

# Trust notebooks (enables widgets like PyMC progress bars)
uv run jupyter trust **/*.ipynb

# Run notebooks programmatically (production)
uv run papermill input.ipynb output.ipynb -p param_name param_value
```

**Note:** The `./scripts/lab.sh` script automatically registers the current environment as a Jupyter kernel and trusts notebooks (enabling widgets like PyMC progress bars) before starting Jupyter Lab. Always restart the Jupyter kernel after installing/updating packages or syncing workspace packages.

### Development

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Run tests
uv run pytest
```

### Cleanup

```bash
# Full cleanup (checkpoints + Python cache + temp files) - Recommended!
./scripts/cleanup.sh

# Quick cleanup - just Jupyter checkpoints
find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null || true

# Clean Python cache only
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Clean all build artifacts
uv run ruff clean  # (if using ruff cache)
rm -rf build/ dist/ *.egg-info/ .pytest_cache/
```

**💡 Pro tip:** Run `./scripts/cleanup.sh` regularly to keep your workspace clean and fast. It can free up significant disk space!

### LaTeX & Publishing

```bash
# Set up LaTeX environment (one-time setup)
./scripts/publish.sh setup

# Compile LaTeX documents (organized by project)
./scripts/publish.sh compile slt-quasi-singular/example_paper

# Convert notebooks to PDF with Quarto
./scripts/publish.sh quarto slt-quasi-singular/notebook_to_paper

# Direct notebook → PDF conversion
./scripts/publish.sh notebook projects/slt/quasi-singular-models/notebooks/fit_dataset.ipynb

# Watch and auto-compile LaTeX
./scripts/publish.sh watch slt-quasi-singular/example_paper

# Clean LaTeX build artifacts
./scripts/publish.sh clean slt-quasi-singular/example_paper
```

**📄 Paper workflow:** Write LaTeX or Quarto documents directly in `projects/*/` directories alongside your notebooks and data. This keeps all project assets together and makes it easy to reference figures and results.

## Customizing the Template

### Remove Example Content

The template includes example content you should remove or replace:

```bash
# Remove example package (if you don't need PyMC extensions)
rm -rf packages/pymc_extensions

# Remove example project (if starting fresh)
rm -rf projects/slt

# Remove example notebook
rm notebooks/example_usage.ipynb

# Then sync to update workspace
uv sync --extra dev
```

### Customize for Your Research

1. **Update metadata** in root `pyproject.toml`:
   ```toml
   [project]
   name = "your-research-name"
   description = "Your research description"
   ```

2. **Add your common dependencies** to root `pyproject.toml`:
   ```bash
   # These are available everywhere
   uv add torch transformers datasets

   # Add dev/jupyter tools
   uv add --dev jupyterlab ipykernel
   ```

3. **Create your first shared package**:
   ```bash
   uv init packages/myutils --lib
   cd packages/myutils
   # Edit pyproject.toml to add dependencies
   # Add your code in src/myutils/
   ```

4. **Create your first research area and projects**:
   ```bash
   # Create a research area directory
   mkdir -p projects/slt

   # Create shared libraries for this research area
   uv init projects/slt/packages/kl --lib

   # Create individual experiments (simple directories, not packages)
   mkdir -p projects/slt/quasi-singular-models/{notebooks,data,paper}
   mkdir -p projects/slt/inverse-temperature/{notebooks,data,paper}

   # Experiments import from libraries, never from each other
   # If you need to share code between experiments, create a new library
   ```

5. **Adjust linting rules** in `pyproject.toml` if you find them too strict:
   ```toml
   [tool.ruff.lint]
   ignore = [
     "E501",  # line too long
     # Add more rules to ignore as needed
   ]
   ```

### Typical Research Workflow

```
1. Start with exploratory analysis in notebooks/
2. Extract useful functions into packages/ as they stabilize (global libraries)
3. Create a research area: projects/slt/
4. Create domain-specific libraries in projects/slt/packages/kl/
5. Create individual experiments as simple directories:
   mkdir -p projects/slt/quasi-singular-models/{notebooks,data,paper}
6. Use clean imports in notebooks/experiments:
   - Global: from scipy_extensions import ...
   - Domain: from kl import ...
   - Never: from other_experiment import ... (extract to library instead!)
7. When experiments need to share code: extract it to a library first
8. Write tests for libraries (not experiments)
9. Write papers in projects/slt/quasi-singular-models/paper/ alongside analysis
10. Use papermill to parameterize and batch-run notebooks
11. Convert notebooks to PDFs or create LaTeX papers for publication
```

### Troubleshooting

**"ModuleNotFoundError" for workspace packages in notebooks:**
```bash
# 1. Ensure workspace packages are built and installed
uv sync --extra dev

# 2. Restart the Jupyter kernel
# In Jupyter: Kernel → Restart Kernel

# 3. If still failing, check package __init__.py files exist and export modules
```

**Notebook widgets not working (progress bars, dropdowns):**
```bash
# 1. Trust the notebook (done automatically by setup.sh and lab.sh)
uv run jupyter trust **/*.ipynb

# 2. Restart Jupyter kernel: Kernel → Restart Kernel
# 3. For PyMC progress bars, use progress_bar="autonotebook"
```

**Workspace packages not updating:**
```bash
# Force rebuild of workspace packages
uv sync --extra dev --reinstall
```

**Clean up Jupyter checkpoint clutter:**
```bash
# Quick cleanup script (removes checkpoints + Python cache + build artifacts)
./scripts/cleanup.sh

# Or manually remove just checkpoints
find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null || true
```

**Workspace feels sluggish or taking too much space:**
```bash
# Full cleanup - removes all temporary files and caches
./scripts/cleanup.sh

# Check what's taking space
du -sh ./* | sort -hr | head -10
```

**Quick setup script (Recommended):**
```bash
# One command to set up everything
./scripts/setup.sh
```

### Adding Experiment Tracking

For Weights & Biases or MLflow:

```bash
# Add to main dependencies (available everywhere)
uv add wandb
# or
uv add mlflow

# Then sync
uv sync --extra dev
```

### Adding GPU Dependencies

```bash
# Add PyTorch with CUDA support
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or add to pyproject.toml dependencies and sync
uv sync --extra dev
```

## Philosophy & Design Decisions

### Why uv instead of Poetry/conda?

- **Speed**: 10-100x faster than pip, resolves dependencies in seconds
- **Compatibility**: Drop-in replacement for pip, works with standard pyproject.toml
- **Workspace support**: Native monorepo support with `[tool.uv.workspace]`
- **No lock-in**: If uv disappears, you still have standard Python packaging

### Why monorepo instead of multiple repos?

- **Shared code**: Extract common utilities into packages/ without publishing to PyPI
- **Atomic changes**: Update a utility function and all projects using it in one commit
- **Easier onboarding**: Clone once, get everything
- **Better for research**: Most research projects share 80% of their tooling

### Why packages/ vs projects/?

This template supports **two-tier code sharing** with clean architectural boundaries:

- **`packages/`**: **Global libraries** - Reusable libraries used across ALL research areas (scipy extensions, plotting utilities, data loaders). Should have tests.
- **`projects/research-area/packages/`**: **Domain-specific libraries** - Shared code within a research area (e.g., `projects/slt/packages/kl/` used by all SLT experiments).
- **`projects/research-area/experiment/`**: **Individual experiments** - Simple directories containing notebooks, data, and papers. Import from libraries but never from each other.

**Example structure:**
```
packages/scipy_extensions/           # Global library - everyone uses
projects/
  slt/                               # Singular Learning Theory research
    packages/kl/                     # Domain library - only SLT experiments use
    quasi-singular-models/           # Experiment directory
      notebooks/, data/, paper/      # Just research artifacts
    inverse-temperature/             # Experiment directory
      notebooks/, data/, paper/      # Just research artifacts
  deep-learning/
    packages/neural-nets/            # Domain library
    transformer-analysis/            # Experiment directory
```

**Key principle:** Experiments never import from each other. All shared code must be extracted to libraries first. This prevents tight coupling and forces good abstractions.

### Dependency Philosophy

- **Main dependencies**: Only packages needed for production/simulation runs (notably `papermill` for programmatic notebook execution)
- **Dev dependencies**: All interactive and development tools (Jupyter Lab, formatters, linters, etc.)
- **Clean separation**: Keeps production environments lean while full dev environment has everything needed

### Alternatives Considered

- **Separate repos per project**: Too much overhead, code duplication
- **Single directory with utils.py**: Breaks at scale, no proper imports
- **Poetry monorepo**: Slower, less ergonomic workspace support
- **Conda environments**: Slow, large, not reproducible enough

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow and guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

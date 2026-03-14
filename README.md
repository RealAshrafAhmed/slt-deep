# SLT Deep: Singular Learning Theory Analysis of Deep Learning Models
*Research repo applying Singular Learning Theory tools to understand deep learning models, starting with deep linear networks.*

> **Note:** This repository is based on the [skeletons-monorepo-for-research](https://github.com/RealAshrafAhmed/skeletons-monorepo-for-research) template and maintains upstream sync for template improvements.

## Research Focus

**Deep Linear Networks through SLT Lens:** Starting with concrete posterior sampling analysis:

- 🎯 **DLN Posterior Sampling** - SGLD, HMC, and hybrid samplers for Bayesian DLN analysis
- 📊 **Sampler Comparison** - Visualizing and validating different MCMC approaches
- 🗺️ **MAP Exploration** - Local posterior geometry around maximum a posteriori estimates
- 📈 **Known Quantities** - Computing observables to validate sampler accuracy
- 🧮 **Toy Models** - Regular (non-singular) DLN for controlled experiments

## Active Investigation: DLN Posterior Sampling

**Goal:** Implement and compare MCMC samplers on toy Deep Linear Networks, building foundational tools for broader SLT analysis.
### Iterative Development Plan

**Phase 1: Core Sampling Infrastructure** 📦
- Create `packages/samplers/` with SGLD, HMC implementations
- Implement 2-phase sampler (SGD → Langevin)
- Basic toy DLN model (regular/non-singular)

**Phase 2: Sampler Validation** ✅
- Compare samplers on known DLN posterior
- Compute theoretical vs sampled quantities
- Visualize sample paths and convergence

**Phase 3: MAP Exploration** 🗺️
- Local posterior visualization around MAP
- Sampler behavior analysis near modes
- Effective sample size comparison

**Phase 4: Documentation & Extension** 📝
- Comprehensive notebooks with theory
- Performance benchmarks
- Foundation for LLC computation

### Project Structure
```
projects/dln/
├── packages/
│   └── samplers/              # Core sampling algorithms
│       ├── sgld.py           # Stochastic Gradient Langevin Dynamics
│       ├── hmc.py            # Hamiltonian Monte Carlo
│       ├── hybrid.py         # 2-phase SGD → Langevin
│       └── toy_dln.py        # Regular DLN test models
└── experiments/
    └── posterior-sampling/    # Sampler comparison study
        ├── notebooks/
        │   ├── 01_sampler_comparison.ipynb
        │   ├── 02_map_exploration.ipynb
        │   └── 03_validation.ipynb
        ├── data/             # Generated samples and results
        └── paper/            # Analysis writeup
```

## Template Benefits

This repo inherits the clean research architecture:

- 🔄 **Shared SLT utilities** - MCMC samplers, KL divergence tools, LLC computation
- 🚫 **Reproducible experiments** - Consistent environments across all analysis
- 🤝 **Collaborative research** - Clean code sharing without PyPI publishing
- 📚 **Theory + Implementation** - Jupyter notebooks + reusable libraries
- 📝 **Publication ready** - LaTeX integration for papers
- ⚡ **Fast iteration** - Modern Python tooling (uv, ruff, pre-commit)

**Next Steps:**
1. Create DLN sampling infrastructure (`uv init projects/dln/packages/samplers --lib`)
2. Implement toy regular DLN model for testing
3. Build SGLD sampler with proper gradient noise scaling
4. Add HMC with automatic step size tuning
5. Create hybrid SGD→Langevin 2-phase approach
6. Validate on analytical posterior (if available) or known moments

**Success Metrics:**
- Samplers converge to same posterior distribution
- Computed observables match theoretical values
- Clear visualization of local posterior geometry
- Reproducible benchmarking framework

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

**Research Topics Covered:**
- Deep Linear Network analysis with Local Learning Coefficient computation
- MCMC sampling techniques (HMC, SGLD) for Bayesian deep learning
- KL divergence profiling and model comparison
- Phase transition analysis in learning dynamics
- SLT observable estimation and effective dimension computation

**Useful for:**
- SLT researchers studying deep learning models
- Theoretical machine learning investigations
- Bayesian analysis of neural networks
- Understanding learning dynamics through singular learning theory

## Getting Started with SLT Research

### 1. Setup Environment

```bash
# Clone this research repo
git clone https://github.com/RealAshrafAhmed/slt-deep.git
cd slt-deep

# Install mise (required): https://mise.jdx.dev/getting-started.html
# Verify it is available
mise --version

# Complete setup (installs all deps + builds SLT libraries)
./scripts/setup.sh

# Install configured tools and inspect tasks
mise install
mise tasks

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

### Orchestration and Execution Model

- **Task orchestration**: `mise`
- **Python environment/runtime**: `uv`
- **Notebook execution**: `papermill`

Use `mise` tasks as project entrypoints, and keep notebook execution parameterized through `uv run papermill`.

### Project Execution Examples

```bash
# List available orchestrated tasks
mise tasks

# Markov chain learning pipeline (single regime, full variant)
mise run mcl:pipeline -- --regime single --variant full --epochs 5

# Run only training stage for a variant
mise run mcl:training -- --regime single --variant no_token --epochs 25

# Run full matrix for markov-chain-learning
mise run mcl:matrix

# Restricted matrix
mise run mcl:matrix -- --regimes single,two_far --variants full,no_pos --epochs 20
```

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

## Setting Up SLT Research Projects

### Starting Your Investigation

This repo comes with the Deep Linear Network (DLN) project ready to go:

```bash
# The project structure spans multiple deep learning model families
projects/
├── dln/                          # Deep Linear Networks (active)
│   ├── packages/
│   │   ├── samplers/            # HMC and SGLD implementations
│   │   └── llc/                 # Local Learning Coefficient computation
│   └── experiments/
│       ├── basic-dln/           # Single hidden layer analysis
│       └── multi-layer/         # Deep network investigation
├── llms/                         # Large Language Models (planned)
│   ├── packages/
│   │   ├── transformers/        # Transformer-specific SLT tools
│   │   └── scaling/             # Scaling law analysis
│   └── experiments/
└── other-models/                 # Additional architectures (planned)
    ├── packages/
    └── experiments/
```

### Adding New SLT Libraries

```bash
# Create domain-specific SLT tools for different model families
uv init projects/dln/packages/samplers --lib
uv init projects/dln/packages/llc --lib

# Future: LLM-specific tools
uv init projects/llms/packages/transformers --lib
uv init projects/llms/packages/scaling --lib

# Add global SLT utilities shared across all model types
uv init packages/slt-utils --lib

# Sync workspace after adding packages
uv sync --extra dev
```

### Syncing with Template Updates

This repo maintains sync with the upstream skeleton template:

```bash
# Pull latest template improvements
git fetch upstream
git merge upstream/main

# Resolve any conflicts (usually in project-specific files)
# Then update your research-specific content as needed
```

### Research Project Structure

1. **Update project metadata** in root `pyproject.toml` (already done):
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

This research repo uses **two-tier code sharing** optimized for SLT investigations:

- **`packages/`**: **Global SLT tools** - Utilities shared across all research areas (KL divergence, model utilities, plotting). Should have tests.
- **`projects/research-area/packages/`**: **Domain-specific libraries** - Specialized code for each investigation (e.g., `projects/dln/packages/llc/` for Local Learning Coefficient computation).
- **`projects/research-area/experiment/`**: **Individual experiments** - Simple directories containing notebooks, data, and papers. Import from libraries but never from each other.

**Current SLT research structure:**
```
packages/
  scipy_extensions/                  # Global utilities for all projects
projects/
  dln/                               # Deep Linear Network investigations (active)
    packages/
      samplers/                      # HMC and SGLD implementations
      llc/                          # Local Learning Coefficient computation
    experiments/
      basic-dln/                     # Single layer analysis
        notebooks/, data/, paper/    # Research artifacts only
      multi-layer/                   # Deep network analysis
        notebooks/, data/, paper/    # Research artifacts only
  llms/                              # Large Language Model investigations (planned)
    packages/
      transformers/                  # Transformer-specific SLT tools
      scaling/                      # Scaling law analysis
    experiments/
      gpt-analysis/                  # GPT family SLT investigation
        notebooks/, data/, paper/
  other-models/                      # Additional deep learning models (planned)
    packages/
      cnns/                         # CNN-specific SLT analysis
    experiments/
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

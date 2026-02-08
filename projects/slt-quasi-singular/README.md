# SLT Quasi-Singular Project

This project demonstrates Bayesian mixture model fitting using PyMC. It generates synthetic data from normal mixtures and fits posteriors with different sample sizes to study concentration behavior.

## Project Structure

```
slt-quasi-singular/
├── src/                # Package source code
│   └── slt_quasi_singular/
│       └── __init__.py
├── notebooks/          # Jupyter notebooks for analysis
│   ├── create_dataset.ipynb
│   ├── fit_dataset.ipynb
│   └── analyzee_dataset.ipynb
├── paper/              # Paper and documentation
│   ├── slt-quasi-singular.tex
│   ├── references.bib
│   └── notebook_to_paper.qmd
├── data/               # Generated datasets
├── figures/            # Generated plots and figures
└── README.md          # This file
```

## Notebooks Overview

- **`notebooks/create_dataset.ipynb`** - Generates experiment runs with mixture data at multiple sample sizes
- **`notebooks/fit_dataset.ipynb`** - Fits Bayesian posteriors and visualizes concentration across sample sizes
- **`notebooks/analyzee_dataset.ipynb`** - Additional analysis of fitted models

## How to Run the Notebooks

1. **Install all dependencies (from the monorepo root):**
   ```bash
   uv sync --extra dev

This project demonstrates Bayesian mixture model fitting using PyMC. It generates synthetic data from normal mixtures and fits posteriors with different sample sizes to study concentration behavior.


## Notebooks Overview

- **`create_dataset.ipynb`** - Generates experiment runs with mixture data at multiple sample sizes
- **`fit_dataset.ipynb`** - Fits Bayesian posteriors and visualizes concentration across sample sizes
- **`analyzee_dataset.ipynb`** - Additional analysis of fitted models

## How to Run the Notebooks

1. **Install all dependencies (from the monorepo root):**
   ```bash
   uv sync --all-packages
   ```

2. **Start Jupyter Lab (from the monorepo root):**
   ```bash
   ./scripts/lab.sh
   ```
   Then navigate to `projects/slt-quasi-singular/notebooks/` and open the notebooks in order.

## How to Run with Papermill

**Run notebooks programmatically:**
```bash
# Generate new dataset
uv run papermill projects/slt-quasi-singular/notebooks/create_dataset.ipynb projects/slt-quasi-singular/notebooks/create_dataset_output.ipynb

# Fit posteriors to existing data
uv run papermill projects/slt-quasi-singular/notebooks/fit_dataset.ipynb projects/slt-quasi-singular/notebooks/fit_dataset_output.ipynb
```

## How to Compile the Paper

**Compile the LaTeX paper (from the monorepo root):**
```bash
./scripts/publish.sh compile slt-quasi-singular/paper/slt-quasi-singular
```

**Render the Quarto notebook-to-paper document:**
```bash
./scripts/publish.sh quarto slt-quasi-singular/paper/notebook_to_paper
```

The paper source files are located in the `paper/` directory.


## Why Split into Multiple Notebooks?
Splitting the workflow into separate notebooks allows for modular, reproducible analysis and enables the use of tools like **Papermill** to automate and parameterize long-running computations. This is especially useful for multi-hour simulations or when running on a cluster.

**Tip:** You can run each notebook non-interactively with Papermill, chaining outputs as inputs for the next stage.
   uv run jupyter lab
   ```
   Then navigate to `projects/slt-quasi-singular/` and open the notebooks in order.

## How to Run with Papermill

**Run notebooks programmatically:**
```bash
# Generate new dataset
uv run papermill projects/slt-quasi-singular/create_dataset.ipynb projects/slt-quasi-singular/create_dataset_output.ipynb

2. **Run the notebook non-interactively:**
   ```bash
   uv run papermill projects/slt-quasi-singular/normal_samples.ipynb projects/slt-quasi-singular/normal_samples_output.ipynb
   ```

## Why Split into Multiple Notebooks?
Splitting the workflow into separate notebooks allows for modular, reproducible analysis and enables the use of tools like **Papermill** to automate and parameterize long-running computations. This is especially useful for multi-hour simulations or when running on a cluster.

**Tip:** You can run each notebook non-interactively with Papermill, chaining outputs as inputs for the next stage.

## Notes
- All dependencies are managed with `uv`.
- For troubleshooting, see the monorepo README.

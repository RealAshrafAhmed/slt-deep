# Inverse Temperature Analysis

*An example experiment demonstrating the clean architecture for SLT research.*

## Overview

This experiment studies the role of inverse temperature (β = 1/T) in singular learning theory, particularly in phase transitions and model complexity.

## Structure

- `notebooks/` - Jupyter notebooks for exploratory analysis
- `data/` - Experimental data and results
- `paper/` - LaTeX source for publication

**Note**: This is a simple experiment directory, not a Python package. It imports from libraries but doesn't export any code.

## Clean Architecture

This experiment follows the clean architecture principle:

```python
# ✅ GOOD: Import from libraries
from scipy_extensions import normal_mixture  # Global library
from kl import kl_divergence_gaussian        # Domain-specific library

# ❌ BAD: Never import from other experiments
# from quasi_singular_models import ...       # Would create tight coupling

# ✅ If you need shared code: extract it to a library first!
# Then both experiments can import from the library
```

## Usage Example

```python
# notebooks/phase_transition_analysis.ipynb
import numpy as np
from kl import kl_divergence_gaussian, symmetrized_kl
from scipy_extensions import normal_mixture

# Analyze phase transitions using shared libraries
beta_range = np.linspace(0.1, 10.0, 100)
kl_values = []

for beta in beta_range:
    # Use global utility
    posterior = normal_mixture.fit_model(data, inverse_temp=beta)

    # Use domain-specific utility
    kl_div = kl_divergence_gaussian(
        mu1=posterior.mean, sigma1=posterior.std,
        mu2=true_mu, sigma2=true_sigma
    )
    kl_values.append(kl_div)

# Analysis continues...
```

## Getting Started

1. Add your analysis notebooks to `notebooks/`
2. Store experimental data in `data/`
3. Write your paper in `paper/`
4. If you develop reusable functions, extract them to `projects/slt/packages/` first
5. **Never** import code directly from `../quasi-singular-models/` - use libraries instead!

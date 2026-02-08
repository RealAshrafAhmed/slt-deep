# scipy_extensions

Extensions and samplers for scientific computing, including normal mixture samplers.

## Features

- **Normal mixture sampling** - Generate samples from multivariate normal mixtures with specified weights, means, and covariances
- **Component tracking** - Return both samples and component assignments for analysis
- **Flexible interface** - Compatible with NumPy random generators for reproducible research

## Usage

```python
from scipy_extensions import normal_mixture
import numpy as np

# Define mixture parameters
weights = [0.3, 0.7]
means = [0, 2]
stds = [1, 0.5]

# Generate samples
rng = np.random.default_rng(42)
samples, components = normal_mixture.sample(
    n=1000, weights=weights, means=means, stds=stds, random_state=rng
)
```

This package demonstrates how to structure reusable utilities in a research monorepo.

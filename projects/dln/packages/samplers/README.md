# Samplers: MCMC for Deep Linear Networks

Core sampling algorithms for Bayesian analysis of Deep Linear Networks (DLNs) in SLT research.

## Algorithms Implemented

- **SGLD** (`sgld.py`) - Stochastic Gradient Langevin Dynamics with proper noise scaling
- **HMC** (`hmc.py`) - Hamiltonian Monte Carlo via blackjax integration
- **Hybrid** (`hybrid.py`) - 2-phase SGD→Langevin for efficient MAP exploration
- **Toy DLN** (`toy_dln.py`) - Regular (non-singular) DLN models for testing

## Usage

```python
from samplers.sgld import SGLDSampler
from samplers.toy_dln import RegularDLN

# Create toy model
model = RegularDLN(input_dim=2, hidden_dim=3, output_dim=1)

# Sample posterior
sampler = SGLDSampler(learning_rate=0.01, temperature=1.0)
samples = sampler.sample(model, data, n_samples=1000)
```

## Design Principles

- **JAX-first**: All implementations use JAX for automatic differentiation
- **Composable**: Samplers work with any differentiable log-posterior
- **Validated**: Each sampler tested against known analytical results
- **Extensible**: Easy to add new sampling algorithms or model types

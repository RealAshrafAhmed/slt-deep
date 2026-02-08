# Posterior Sampling Experiment

**Objective**: Compare SGLD, HMC, and Hybrid MCMC samplers on Deep Linear Network posterior distributions.

## Structure

```
posterior-sampling/
├── notebooks/
│   ├── 01_sampler_comparison.ipynb    # Main comparison notebook
│   └── 01_sampler_comparison.md       # Overview document
├── data/                              # Generated datasets and results
└── paper/                            # Analysis writeup (future)
```

## Key Components

### Samplers Tested
- **SGLD**: Stochastic Gradient Langevin Dynamics with proper noise scaling
- **HMC**: Hamiltonian Monte Carlo via blackjax integration
- **Hybrid**: Two-phase SGD→Langevin for efficient MAP + sampling

### Model Configuration
- **Regular DLN**: 1D → 2D → 1D Deep Linear Network
- **Non-singular**: Avoids degeneracies for controlled testing
- **Gaussian priors**: Independent normal priors on all parameters
- **Additive noise**: Gaussian observation noise

### Analysis Dimensions
- **Convergence**: Trace plots, running averages, burn-in assessment
- **Efficiency**: Effective sample size, autocorrelation times
- **Accuracy**: Posterior means vs true parameters, credible interval coverage
- **Local Behavior**: Exploration patterns around MAP estimate

## Running the Experiment

1. **Install dependencies**: Ensure `samplers` package is available
   ```bash
   cd projects/dln/packages/samplers
   uv sync
   ```

2. **Run notebook**: Execute `01_sampler_comparison.ipynb`
   - Generates synthetic data from known DLN
   - Runs all three samplers with comparable settings
   - Produces diagnostic plots and performance metrics

3. **Interpret results**:
   - Compare effective sample sizes across methods
   - Validate posterior coverage against true parameters
   - Analyze local sampling behavior around MAP

## Expected Outcomes

- **Validation**: All samplers should converge to similar posterior distributions
- **Performance ranking**: HMC typically most efficient, SGLD simplest, Hybrid balanced
- **Coverage verification**: 95% credible intervals should contain ~95% of true parameters
- **Local insights**: Different exploration patterns around posterior mode

## Success Metrics

✅ **Convergence**: All samplers reach stationary distribution
✅ **Consistency**: Similar posterior means across methods (within Monte Carlo error)
✅ **Coverage**: Appropriate credible interval coverage rates
✅ **Efficiency**: Reasonable effective sample sizes (>10% of raw samples)

## Future Extensions

- Test on singular (truly degenerate) DLN configurations
- Higher-dimensional parameter spaces and data
- Temperature schedules for improved exploration
- Integration with Local Learning Coefficient computation

---

*Part of the DLN hyper-project investigating SLT applications to deep learning.*

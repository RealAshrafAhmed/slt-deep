# Feature: Reference Chain for Singular Posterior

## Goal
Produce a trustworthy reference MCMC chain for the MarkovTransformer posterior
(d=910, ~600 degenerate directions) that we can verify via multi-chain diagnostics.

## Two Approaches

### Approach A — Stan-style NUTS with online mass matrix adaptation
**Idea:** Extend NUTS to support full mass matrices + add Stan-style windowed
sample-covariance estimation during warmup. No Hessian needed.

**Steps:**
1. [x] Extend `nuts()` backend to support full (2-D) mass matrices
   - Reuse the eigendecomposition + `sample_momentum / M_inv / kinetic` pattern
     already in `hmc()` backend
   - Update `_leapfrog()` to accept callable M_inv instead of diagonal tensor
2. [x] Add Stan-style windowed mass matrix adaptation to `nuts()`
   - Expanding windows: 25, 50, 100, 200, ...
   - Three phases: (I) fast ε adapt, (II) ε + M adapt, (III) final ε adapt
   - Estimate M = sample covariance from warmup draws within each window
3. [ ] Diagnostic notebook cells:
   - Run 2+ chains, compare adapted M eigenspectra across chains
   - Plot ε convergence trace per chain
   - Plot warmup M eigenvalue spectrum vs Hessian (if available)
   - Compute R-hat across chains, ESS per eigendirection
   - ACF heatmap for production samples

**Diagnostics to check warmup is sufficient:**
- ε trace has plateaued before end of warmup
- M eigenspectra agree across chains
- Split-R-hat < 1.01 on production samples
- No divergences (|ΔH| > 1000)

### Approach B — Pre-computed M from pooled short chains + stability grid
**Idea:** Run many short chains with identity M, pool samples to estimate
a single shared M = sample covariance, then use stability grid to pick (ε, L)
and run production chains.

**Steps:**
1. [ ] Run K short chains (e.g. 4 chains × 500 samples) with NUTS + identity M
2. [ ] Pool all samples → compute sample covariance Σ̂
3. [ ] Eigendecompose Σ̂, compare spectrum to Hessian
4. [ ] Stability grid with M = Σ̂ to find (ε, L) frontier
5. [ ] Production run: HMC with fixed (ε, L, M), 4+ chains from dispersed inits
6. [ ] R-hat + ESS diagnostics

**Advantage:** Saves warmup time per chain (shared M, no per-chain adaptation).
**Risk:** If short chains don't mix well with identity M, the covariance estimate
is biased toward the starting region.

## Current Status
- Approach A: NUTS extended with full matrix + windowed adaptation (implemented)
- Approach B: Not started

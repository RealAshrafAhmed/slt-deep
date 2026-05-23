# Sampling Singular Models: Beyond NUTS

## Problem Statement

NUTS fails for singular statistical models because:
1. The U-turn criterion assumes quadratic-like level sets (bounded orbits)
2. Degenerate directions have no curvature → trajectories go to infinity without turning
3. Step size ε must be tiny to avoid divergences in stiff non-quadratic regions
4. Result: L×ε ≪ π, no decorrelation, max depth always hit

## Core Ideas

### Idea 1: Learned Metric Network

**Concept**: Train a neural network f: θ → M(θ) that predicts the local mass matrix
(Riemannian metric) at any point in parameter space.

**Why this is different from existing work**:
- NeuTra/L2HMC assume the target is approximately Gaussian (regular) — wrong for singular models
- SoftAbs/RHMC compute the Hessian at every point — O(d³) per leapfrog step
- Our approach: amortize the cost by learning the metric from (θ, H(θ)) training pairs

**Key questions**:
- Q1: What is the training data? Pairs (θ_i, H(θ_i)) at random points near the singular set?
- Q2: How to ensure SPD output? Cholesky factor parameterization? Or accept degeneracy?
- Q3: For singular models, should the net predict a DEGENERATE metric (not SPD)?
      The true Fisher metric IS degenerate at singularities.
- Q4: How many training points are needed? The metric varies smoothly away from the
      singular set but has algebraic structure near it.
- Q5: Can we exploit the algebraic structure? Near a normal crossing divisor,
      the metric has a known form: g_ii ~ |w_i|^{2(k_i - 1)}. The net only needs
      to learn the coordinate change (resolution map), not the metric from scratch.
- Q6: Generalization: does the net need to work across the full parameter space,
      or only in the typical set (ball of radius ~β^{-1/(2λ)} around W₀)?

**Architecture considerations**:
- Input: θ ∈ ℝ^d (flat parameter vector)
- Output: L ∈ ℝ^{d×d} lower triangular (Cholesky of M), or eigenvalues + rotations
- For d=2000, full output is 2M parameters — likely need low-rank structure
- Alternative: output only eigenvalue corrections in the Hessian eigenbasis
  (keep eigenvectors from MLE Hessian, learn eigenvalue map θ → λ(θ))

**Training procedure**:
1. Sample θ_i from prior or from preliminary MCMC run
2. Compute H(θ_i) via approx_hessian (expensive but parallelizable)
3. Train net to minimize ||f(θ_i) - H(θ_i)||_F or a spectral loss
4. Use trained f(θ) as position-dependent mass matrix in HMC

**Computational budget**:
- Training data generation: K Hessian evaluations at ~10s each = K×10s
- Net training: standard supervised learning
- Sampling cost: one forward pass of the net per leapfrog step (cheap if net is small)
- Break-even: amortized cost < K_samples × d³ (direct Hessian at each step)


### Idea 2: Symmetry-Augmented HMC (Group Teleportation)

**Concept**: After each HMC/NUTS step, apply a random element of the model's symmetry
group to teleport along the singular set. This gives O(1) mixing in degenerate directions
without needing the integrator to traverse them.

**Why this works**:
- Neural networks have discrete symmetries: permutation of neurons, sign flips
- The loss (and hence posterior) is INVARIANT under these symmetries
- Applying a symmetry σ: θ → σ(θ) is a valid MCMC move with acceptance probability 1
  (deterministic, measure-preserving, leaves target invariant)
- The symmetry orbits lie ALONG the singular set W₀
- This directly addresses NUTS's inability to traverse flat directions

**Key questions**:
- Q1: What are the symmetries of a transformer?
      - Permutation of attention heads (if multi-head)
      - Permutation of neurons within feed-forward layers
      - Sign flips in certain weight matrices (if activation is odd)
      - For MarkovTransformer specifically: ?
- Q2: How to enumerate the symmetry group? For a layer with k neurons,
      the permutation group has k! elements — too large to enumerate.
      Need to sample random group elements.
- Q3: How to implement the group action efficiently?
      - Permutation: just reorder rows/columns of weight matrices
      - Sign flip: negate certain rows and corresponding columns
- Q4: Does applying ALL symmetries at once give better mixing than one at a time?
- Q5: How does this interact with the mass matrix?
      After teleportation, the mass matrix (learned from the old position) may be wrong.
      Need to either: (a) use a symmetry-invariant mass matrix, or
      (b) recompute/re-predict the metric at the new position.
- Q6: Is the singular set exactly the symmetry orbit, or does it have additional structure?
      For overparameterized models, W₀ may be larger than the symmetry orbit.
      Symmetry only helps with the symmetry-induced degeneracy, not all degeneracy.
- Q7: Detailed balance: since symmetries are deterministic and measure-preserving,
      the combined kernel (HMC step + random symmetry) satisfies detailed balance
      w.r.t. the posterior. Verify this formally.

**Implementation plan for MarkovTransformer**:
1. Identify symmetries of MarkovTransformer architecture
2. Implement group action: given a permutation π, transform weight matrices
3. After each NUTS step, apply a random permutation with probability p_teleport
4. Measure: does ACF decrease faster in degenerate directions?

**Expected impact**:
- Non-degenerate directions: no change (HMC already handles these)
- Degenerate directions corresponding to symmetries: instant decorrelation
- Degenerate directions NOT from symmetry: no improvement (need Idea 1 or other)


## Implementation Roadmap

### Phase 0: Baseline (DONE)
- [x] NUTS with Hessian-based mass matrix + shrinkage adaptation
- [x] ACF diagnostics per eigendirection
- [x] Identify degenerate vs non-degenerate directions

### Phase 1: Symmetry Teleportation (simpler, more immediate)
- [ ] Catalog symmetries of MarkovTransformer
- [ ] Implement permutation action on flat parameter vector
- [ ] Add post-step symmetry move to NUTS loop
- [ ] Compare ACF with/without teleportation
- [ ] Measure: what fraction of degeneracy is explained by symmetry?

### Phase 2: Algebraic Order Estimation (builds toward Idea 1)
- [ ] For each degenerate eigendirection v_i, do line search K(w₀ + t·v_i)
- [ ] Fit algebraic order k_i from log K vs log |t|
- [ ] Use k_i to set mass eigenvalue: M_i = n^{(k_i-1)/k_i}
- [ ] Compare sampling efficiency with algebraic-aware mass vs uniform 1/σ²

### Phase 3: Learned Metric Network
- [ ] Generate training data: (θ_j, H(θ_j)) pairs at ~100 points
- [ ] Design net architecture (low-rank output in Hessian eigenbasis)
- [ ] Train metric predictor
- [ ] Integrate into HMC: position-dependent mass matrix
- [ ] Benchmark: sampling efficiency vs fixed mass matrix

### Phase 4: Combined System
- [ ] HMC with learned metric (Phase 3) + symmetry teleportation (Phase 1)
- [ ] Adaptive termination criterion for singular directions
- [ ] Full comparison against: Stan, fixed-mass NUTS, diagonal-adapted NUTS


## Connections to Existing Literature

- Girolami & Calderhead (2011): Riemannian manifold HMC
- Betancourt (2013): SoftAbs metric for RMHMC
- Zhang & Sutton (2011): Quasi-Newton MCMC
- Hoffman et al. (2019): NeuTra-HMC (normalizing flows)
- Levy, Hoffman & Ermon (2018): L2HMC (learned integrator)
- Nishimura & Dunson (2020): Recycling HMC with permutation symmetries
- Watanabe (2009): Algebraic geometry of singular models (SLT foundation)
- Lau, Drton & Bühlmann (2023): MCMC for singular models


## What Makes This a Paper

1. **Novel diagnosis**: NUTS U-turn criterion provably fails for singular models
   (we can prove this: dot product never becomes negative in null space)
2. **Novel solution 1**: Symmetry teleportation — first application of neural network
   symmetry groups as MCMC mixing moves for singular learning theory
3. **Novel solution 2**: Learned metric that respects algebraic structure of singularities
4. **Empirical demonstration**: On a concrete singular model (transformer learning
   Markov chains) showing the failure mode and improvement
5. **Theoretical contribution**: Connecting SLT algebraic geometry (resolution of
   singularities, RLCT) to practical MCMC algorithm design

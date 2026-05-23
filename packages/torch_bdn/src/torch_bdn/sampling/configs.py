"""
Typed sampler configuration dataclasses.

Each config carries the algorithm identity and its parameters.
No ``thin`` parameter — NUTS/HMC don't need it, and for SG methods
the user should increase ``n_samples`` directly.

Usage::

    from torch_bdn.sampling import Sampler, NUTS, SGLD

    result = sampler.sample(NUTS(n_warmup=500), n_samples=2000)
    result = sampler.sample(SGLD(lr=0.5, n_warmup=1000), n_samples=2000)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Exact (full-batch) samplers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NUTS:
    """NUTS (No-U-Turn Sampler) — Hoffman & Gelman 2014.

    Adaptively selects trajectory length via U-turn criterion.
    Step-size is automatically adapted during warmup via dual averaging.

    Supports identity, diagonal (1-D), and full (2-D) mass matrices.
    When ``adapt_mass_matrix=True``, uses Stan-style windowed warmup
    to estimate the mass matrix from sample covariance.
    """

    n_warmup: int = 500
    step_size: float = 0.01
    max_tree_depth: int = 10
    target_accept: float = 0.80
    mass_matrix: torch.Tensor | None = None
    adapt_mass_matrix: bool = False


@dataclass(frozen=True)
class HMC:
    """Standard Hamiltonian Monte Carlo with fixed trajectory length.

    Requires manual tuning of ``n_leapfrog``.  Prefer :class:`NUTS` unless
    you have a specific reason to fix trajectory length.

    When *mass_matrix* is a full (2-D) matrix, directions with eigenvalues
    below ``λ_max × 1e-3`` are treated as degenerate.  By default their
    mass is set to 1 (identity).  Set *null_space_mass* to override this:

    - **scalar** → constant mass in all degenerate directions (e.g.
      ``1/σ²`` for a Gaussian prior ``N(μ, σ²I)``).
    - **1-D tensor** ``(d,)`` → per-parameter mass floor; the backend
      projects it onto the degenerate eigendirections.
    - **2-D tensor** ``(d, d)`` → full matrix (e.g. the prior precision);
      the backend projects it onto the degenerate subspace.
    """

    n_warmup: int = 500
    step_size: float = 0.01
    n_leapfrog: int = 25
    target_accept: float = 0.65
    adapt_step_size: bool = False
    mass_matrix: torch.Tensor | None = None
    null_space_mass: torch.Tensor | float | None = None
    degen_threshold: float = 1e-3


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic-gradient samplers
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SGLD:
    """Stochastic Gradient Langevin Dynamics.

    Approximate sampler using mini-batch gradients.  Suitable for large
    datasets where full-batch HMC / NUTS is too expensive.

    If *preconditioner* is provided it should be the Hessian of the
    negative log-likelihood at the MAP (or nearby).  The backend will
    invert it (with Tikhonov damping) to build a preconditioner
    ``M = (H + δI)⁻¹`` so that:

        θ_{t+1} = θ_t + (ε/2) M ∇log p  +  sqrt(ε M) ξ,   ξ ~ N(0, I)

    This rescales each direction by the inverse curvature, giving
    isotropic effective step size across the spectrum.
    """

    n_warmup: int = 1000
    lr: float = 0.01
    batch_size: int | None = None
    noise_scale: float | None = None
    preconditioner: torch.Tensor | None = None


@dataclass(frozen=True)
class SGHMC:
    """Stochastic Gradient Hamiltonian Monte Carlo — Chen et al. 2014.

    Like SGLD but with momentum: the chain accumulates velocity along
    low-gradient directions, traversing flat regions in O(L) steps
    instead of the O(L²) random-walk of SGLD.  A friction term (α)
    dissipates energy to compensate for stochastic-gradient noise.

    Update equations (per step)::

        r  ←  r  +  ε ∇̃ log p(θ)  −  α r  +  N(0, 2αε)
        θ  ←  θ  +  ε r

    **Tuning guide**

    ``lr`` (step size ε):
        Controls the discretisation resolution.  Start around 0.01;
        reduce if the chain diverges or energy drifts, increase if
        mixing is too slow.  With a preconditioner, lr can often be
        larger because the effective curvature is normalised.

    ``alpha`` (friction):
        Damps the momentum.  Must exceed the stochastic-gradient noise
        variance β̂ for stability (we set β̂ = 0, so any α > 0 works).
        Higher α → faster energy dissipation, more diffusive (SGLD-
        like).  Lower α → longer momentum coherence, better traversal
        of flat directions but risk of oscillation.  Typical range
        0.01–0.5; default 0.1 is a safe starting point.

    ``batch_size``:
        Larger batches reduce gradient noise, allowing smaller α.
        For N ≲ 5k, full-batch is feasible and keeps things simple.
        Auto-selected if ``None``.

    ``resample_momentum``:
        Periodically resampling momentum from N(0, ε I) prevents slow
        energy drift.  Set to 0 to rely on friction alone.  Default 50
        is reasonable; increase for long-correlation targets.

    ``preconditioner``:
        The Hessian H of the negative log-likelihood at the MAP.
        The backend builds M = H⁺ + I_null (pseudoinverse on the
        column space, identity on the null space) and applies it to
        both the gradient and the noise, equalising step sizes across
        the eigenspectrum.  Strongly recommended when the Hessian
        condition number is large or the model has near-degenerate
        directions (common in SLT / singular models).

    ``n_warmup``:
        Burn-in iterations discarded before collecting samples.
        Should be long enough for the chain to reach the typical set.
        Monitor trace plots; 2×–4× the mixing time is a rough rule.
    """

    n_warmup: int = 1000
    lr: float = 0.01
    alpha: float = 0.1
    batch_size: int | None = None
    resample_momentum: int = 50
    preconditioner: torch.Tensor | None = None


@dataclass(frozen=True)
class RMHMC:
    """Riemannian Manifold HMC with SoftAbs metric — Girolami & Calderhead 2011.

    Recomputes the Hessian at every leapfrog position to adapt the mass
    matrix to local curvature.  Uses the SoftAbs regularisation
    (Betancourt 2013) so that degenerate eigenvalues are smoothly mapped
    to 1/α instead of zero, avoiding singular metrics.

    This is the "simplified" variant: the metric-derivative terms
    (∂G/∂q) are omitted.  The MH correction ensures the correct
    stationary distribution.

    **Very expensive** — each leapfrog step requires a full Hessian
    computation (~10 s for 910 params).  Use small ``n_leapfrog`` and
    few samples when exploring.
    """

    n_warmup: int = 50
    step_size: float = 0.01
    n_leapfrog: int = 5
    target_accept: float = 0.65
    adapt_step_size: bool = True
    alpha: float = 1000.0
    chunk_size: int = 10240
    n_fixpoint: int = 4


# Union type for all configs
SamplerConfig = NUTS | HMC | SGLD | SGHMC | RMHMC


# ─────────────────────────────────────────────────────────────────────────────
# Chain initialisation strategies
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Perturb:
    """Initialise extra chains by perturbing the current parameters.

    Each chain starts at  θ₀ + ε  where  ε ~ N(0, σ²I)  and
    σ = ``scale`` × 0.01 × max(‖θ₀‖, 1).
    """

    scale: float = 1.0


@dataclass(frozen=True)
class Prior:
    """Initialise extra chains by drawing from the prior (standard normal)."""


InitStrategy = Perturb | Prior

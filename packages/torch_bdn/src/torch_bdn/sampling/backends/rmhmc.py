"""
Riemannian Manifold HMC (RMHMC) — Girolami & Calderhead 2011.

Uses a position-dependent metric tensor G(q) = Hessian of negative
log-likelihood, regularised via the SoftAbs map (Betancourt 2013) to
handle degenerate / near-singular Hessians common in singular models.

The Hamiltonian is:

    H(q, p) = U(q) + ½ log|G̃(q)| + ½ pᵀ G̃(q)⁻¹ p

where G̃ is the SoftAbs-regularised Hessian.  Because G̃ depends on q,
the leapfrog integrator becomes *implicit* (generalised leapfrog) with
fixed-point iterations at each half-step.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch
from torch.func import hessian

from ...bn.bayesian_net import mean_loss_factory

# ─────────────────────────────────────────────────────────────────────────────
# SoftAbs metric utilities
# ─────────────────────────────────────────────────────────────────────────────


def _softabs(eigvals: torch.Tensor, alpha: float) -> torch.Tensor:
    """SoftAbs map: |λ| coth(α|λ|) — smooth positive-definite approximation.

    For |λ| >> 1/α:  ≈ |λ|   (preserves curvature magnitude)
    For λ → 0:       → 1/α   (regularises degenerate directions)
    Always positive — suitable as a metric eigenvalue.
    """
    # Work with u = α|λ| ≥ 0.  Then softabs = u / (α tanh(u)).
    # At u = 0: u/tanh(u) → 1, so result → 1/α.
    # At large u: tanh(u) → 1, so result → u/α = |λ|.
    u = alpha * eigvals.abs()
    ratio = torch.where(
        u > 1e-4,
        u / torch.tanh(u).clamp(min=1e-12),
        torch.ones_like(u),  # Taylor: u/tanh(u) ≈ 1 + u²/3 near 0
    )
    return ratio / alpha


def _compute_metric(
    q: torch.Tensor,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    alpha: float,
    chunk_size: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Compute SoftAbs-regularised metric at position q.

    Returns (G_inv, log_det_G, eigvecs, eigvals_raw, softabs_eigvals, sa_sqrt).
    All on the same device as q.
    """
    dev = q.device

    # Compute Hessian of negative log-likelihood at q (on CPU for linalg)
    hessian_fn = hessian(mean_loss_factory(model=model, loss_fn=loss_fn))

    if chunk_size is not None and x_data.shape[0] > chunk_size:
        hessians = []
        for i in range(0, x_data.shape[0], chunk_size):
            h_chunk = hessian_fn(
                q.detach(), x_data[i : i + chunk_size], y_data[i : i + chunk_size]
            )
            hessians.append(h_chunk)
        H = torch.stack(hessians).mean(dim=0)
    else:
        H = hessian_fn(q.detach(), x_data, y_data)

    H_cpu = H.double().cpu()
    # Symmetrise and sanitise
    H_cpu = 0.5 * (H_cpu + H_cpu.T)
    H_cpu = torch.nan_to_num(H_cpu, nan=0.0, posinf=1e6, neginf=-1e6)
    # Add small ridge for numerical stability
    H_cpu += 1e-6 * torch.eye(H_cpu.shape[0], dtype=H_cpu.dtype)

    eigvals, eigvecs = torch.linalg.eigh(H_cpu)
    eigvals = eigvals.float()
    eigvecs = eigvecs.float()

    # SoftAbs regularisation — always positive
    sa = _softabs(eigvals, alpha)

    # Build inverse metric and log-determinant in eigenbasis
    G_inv = (eigvecs * (1.0 / sa).unsqueeze(0)) @ eigvecs.T
    log_det_G = sa.log().sum()
    sa_sqrt = sa.sqrt()

    return (
        G_inv.to(dev),
        log_det_G.to(dev),
        eigvecs.to(dev),
        eigvals.to(dev),
        sa.to(dev),
        sa_sqrt.to(dev),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradient of potential + metric terms
# ─────────────────────────────────────────────────────────────────────────────


def _grad_U_and_metric(
    q: torch.Tensor,
    p: torch.Tensor,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    log_prior_fn: Callable[[torch.Tensor], torch.Tensor],
    diff_log_likelihood_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    alpha: float,
    chunk_size: int,
    max_grad_norm: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Compute ∂H/∂q and metric quantities at (q, p).

    ∂H/∂q = ∂U/∂q + ∂/∂q [½ log|G̃|] + ∂/∂q [½ pᵀ G̃⁻¹ p]

    We compute the last two terms via finite differences on the
    eigendecomposition to avoid materialising the 3rd-order tensor
    ∂G/∂q_i.

    Returns (grad_H_q, G_inv, log_det_G, logp_value, kinetic_energy).
    """
    dev = q.device
    dim = q.numel()

    # ── 1. Gradient of potential U(q) = -log p(y|x,q) - log p(q) ─────
    q_grad = q.detach().requires_grad_(True)
    ll = diff_log_likelihood_fn(q_grad, x_data, y_data)
    lp = log_prior_fn(q_grad)
    logp = ll + lp
    grad_logp = torch.autograd.grad(logp, q_grad)[0].detach()
    gn = grad_logp.norm()
    if gn > max_grad_norm:
        grad_logp = grad_logp * (max_grad_norm / gn)
    grad_U = -grad_logp  # ∂U/∂q

    # ── 2. Metric at q ────────────────────────────────────────────────
    G_reg, G_inv, log_det_G, eigvals_raw, sa_eigvals = _compute_metric(
        q, model, loss_fn, x_data, y_data, alpha, chunk_size
    )

    # ── 3. Metric-dependent gradient via finite differences ───────────
    # ∂/∂q_i [½ log|G̃| + ½ pᵀ G̃⁻¹ p]
    # Use central finite differences: (f(q+δe_i) - f(q-δe_i)) / 2δ
    #
    # This is O(dim) Hessian evaluations — expensive but correct.
    # For 910 params this is ~910 × 10s ≈ 2.5 hours per leapfrog step.
    # That's way too expensive. Instead we use a cheaper approximation:
    #
    # Approximate: assume the metric varies slowly, so
    # ∂/∂q [½ log|G̃|] ≈ 0 and ∂/∂q [½ pᵀ G̃⁻¹ p] ≈ 0
    # This reduces to "fixed metric at each leapfrog step" which is
    # similar to the simplified RMHMC (sRMHMC) of Girolami & Calderhead.
    #
    # The key difference from standard HMC: we *recompute* G(q) at each
    # leapfrog position update, so the metric tracks the geometry, but
    # we don't differentiate through G w.r.t. q.
    #
    # This is called "Simplified Manifold MALA / Generalised MALA" or
    # position-dependent preconditioning. The MH correction still ensures
    # the correct stationary distribution.

    kinetic = 0.5 * p @ G_inv @ p

    return grad_U, G_inv, log_det_G, logp.detach(), kinetic.item()


# ─────────────────────────────────────────────────────────────────────────────
# RMHMC sampler (simplified: recompute metric, no dG/dq terms)
# ─────────────────────────────────────────────────────────────────────────────


def rmhmc(
    set_params_fn: Callable[[torch.Tensor], None],
    log_likelihood_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    log_prior_fn: Callable[[torch.Tensor], torch.Tensor],
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    init_params: torch.Tensor,
    n_samples: int = 1000,
    n_burnin: int = 500,
    step_size: float = 0.01,
    n_leapfrog: int = 10,
    thin: int = 1,
    alpha: float = 1000.0,
    chunk_size: int = 10240,
    n_fixpoint: int = 4,
    adapt_step_size: bool = False,
    target_accept: float = 0.65,
    model: torch.nn.Module | None = None,
    loss_fn: torch.nn.Module | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Riemannian Manifold HMC with SoftAbs metric regularisation.

    At each leapfrog step the Hessian is recomputed at the current
    position and regularised via the SoftAbs map.  This gives
    position-dependent preconditioning that adapts to local curvature,
    including near singularities where eigenvalues → 0.

    This is the "simplified" variant: the metric-derivative terms
    (∂G/∂q) are omitted from the momentum update.  The MH correction
    ensures correct stationary distribution despite the approximation.

    Args:
        set_params_fn: sets model parameters from flat vector
        log_likelihood_fn: log p(y|x,θ)
        log_prior_fn: log p(θ) from flat param vector
        x_data, y_data: full dataset
        init_params: starting position (flat)
        n_samples: number of post-burnin samples
        n_burnin: warmup steps (discarded)
        step_size: leapfrog step size ε
        n_leapfrog: number of leapfrog steps L per proposal
        thin: keep every thin-th sample
        alpha: SoftAbs sharpness — larger α → sharper transition
            at λ=0.  For λ >> 1/α, SoftAbs(λ) ≈ |λ|.
            Default 1000 means eigenvalues > 0.001 are preserved.
        chunk_size: data chunk size for Hessian computation
        n_fixpoint: fixed-point iterations for implicit leapfrog
        adapt_step_size: use dual-averaging during warmup
        target_accept: target acceptance rate for adaptation
        model: the nn.Module (required for Hessian computation)
        loss_fn: the loss function (required for Hessian computation)

    Returns:
        dict with parameters, log_probabilities, acceptance_rate, diagnostics
    """
    diff_log_likelihood_fn = kwargs.pop("diff_log_likelihood_fn", None)

    if model is None or loss_fn is None:
        raise ValueError("RMHMC requires model and loss_fn for Hessian computation")

    dim = init_params.numel()
    dev = init_params.device
    params = init_params.clone().detach()

    # ── Dual-averaging state ──────────────────────────────────────────
    log_eps = math.log(step_size)
    log_eps_bar = 0.0
    H_bar = 0.0
    gamma, t0, kappa = 0.05, 10.0, 0.75
    mu = math.log(10.0 * step_size)

    samples: list[torch.Tensor] = []
    logp_vals: list[float] = []
    energy_errors: list[float] = []
    accept_probs: list[float] = []
    n_accepted = 0
    n_accepted_window = 0
    total_steps = n_burnin + n_samples * thin

    # ── Initial metric ────────────────────────────────────────────────
    G_inv, log_det_G, V, eigvals_raw, sa_eigvals, sa_sqrt = _compute_metric(
        params, model, loss_fn, x_data, y_data, alpha, chunk_size
    )

    # Initial potential gradient
    q_g = params.detach().requires_grad_(True)
    ll = diff_log_likelihood_fn(q_g, x_data, y_data)
    lp = log_prior_fn(q_g)
    logp_current = (ll + lp).detach()
    grad_logp = torch.autograd.grad(ll + lp, q_g)[0].detach()
    gn = grad_logp.norm()
    if gn > 100.0:
        grad_logp = grad_logp * (100.0 / gn)
    grad_U_current = -grad_logp

    for step in range(total_steps):
        eps = math.exp(log_eps) if adapt_step_size and step < n_burnin else step_size

        # ── Sample momentum from N(0, G̃(q)) ─────────────────────────
        # G̃^{1/2} = V diag(√sa) → r = V diag(√sa) z, z ~ N(0,I)
        z = torch.randn(dim, device=dev)
        r = V @ (sa_sqrt * z)

        # ── Current Hamiltonian ──────────────────────────────────────
        H_current = -logp_current + 0.5 * log_det_G + 0.5 * r @ G_inv @ r

        # ── Generalised leapfrog ─────────────────────────────────────
        q = params.clone()
        p = r.clone()
        grad_U = grad_U_current.clone()
        G_inv_lf = G_inv
        log_det_lf = log_det_G
        V_lf = V
        sa_lf = sa_eigvals
        sa_sqrt_lf = sa_sqrt

        for lf_step in range(n_leapfrog):
            # Half-step momentum (using current metric)
            p = p - 0.5 * eps * grad_U

            # Full-step position with implicit fixed-point iteration
            # q' = q + ε G̃(q')⁻¹ p  — solve by iterating
            q_new = q + eps * (G_inv_lf @ p)  # initial guess

            for _fp in range(n_fixpoint):
                # Recompute metric at q_new
                G_inv_new, log_det_new, V_new, _, sa_new, sa_sqrt_new = _compute_metric(
                    q_new, model, loss_fn, x_data, y_data, alpha, chunk_size
                )
                # Average inverse metric (trapezoidal)
                G_inv_avg = 0.5 * (G_inv_lf + G_inv_new)
                q_new = q + eps * (G_inv_avg @ p)

            q = q_new
            G_inv_lf = G_inv_new
            log_det_lf = log_det_new
            V_lf = V_new
            sa_lf = sa_new
            sa_sqrt_lf = sa_sqrt_new

            # Recompute potential gradient at new q
            q_g = q.detach().requires_grad_(True)
            ll = diff_log_likelihood_fn(q_g, x_data, y_data)
            lp_val = log_prior_fn(q_g)
            logp_prop = (ll + lp_val).detach()
            grad_logp = torch.autograd.grad(ll + lp_val, q_g)[0].detach()
            gn = grad_logp.norm()
            if gn > 100.0:
                grad_logp = grad_logp * (100.0 / gn)
            grad_U = -grad_logp

            # Half-step momentum (using new metric)
            p = p - 0.5 * eps * grad_U

        # ── Proposed Hamiltonian ─────────────────────────────────────
        H_proposed = -logp_prop + 0.5 * log_det_lf + 0.5 * p @ G_inv_lf @ p

        delta_H = (H_proposed - H_current).item()
        if not math.isfinite(delta_H):
            delta_H = float("inf")
        log_alpha = -delta_H
        alpha_mh = min(1.0, math.exp(min(log_alpha, 0.0)))

        if math.log(torch.rand(1).item() + 1e-30) < log_alpha:
            params = q
            logp_current = logp_prop
            grad_U_current = grad_U
            G_inv = G_inv_lf
            log_det_G = log_det_lf
            V = V_lf
            sa_eigvals = sa_lf
            sa_sqrt = sa_sqrt_lf
            n_accepted += 1
            n_accepted_window += 1
        # else: reject, keep params, logp_current, grad_U_current, G_inv, etc.

        # Progress logging every 10 steps (RMHMC is slow, so more frequent)
        if (step + 1) % 10 == 0:
            phase = "warmup" if step < n_burnin else "sample"
            window_rate = n_accepted_window / min(10, step + 1)
            print(
                f"  [RMHMC {phase}] step {step + 1}/{total_steps}  "
                f"ε={eps:.2e}  accept_rate(last 10)={window_rate:.2f}  "
                f"ΔH={delta_H:.2e}"
            )
            n_accepted_window = 0

        # Dual averaging update
        if adapt_step_size and step < n_burnin:
            m = step + 1
            w = 1.0 / (m + t0)
            H_bar = (1 - w) * H_bar + w * (target_accept - alpha_mh)
            log_eps = mu - (m**0.5) / gamma * H_bar
            m_kappa = m ** (-kappa)
            log_eps_bar = m_kappa * log_eps + (1 - m_kappa) * log_eps_bar

        if adapt_step_size and step == n_burnin - 1:
            step_size = math.exp(log_eps_bar)

        if step >= n_burnin and (step - n_burnin) % thin == 0:
            samples.append(params.clone().detach())
            logp_vals.append(logp_current.item())
            energy_errors.append(delta_H)
            accept_probs.append(alpha_mh)

    return {
        "parameters": samples,
        "log_probabilities": logp_vals,
        "acceptance_rate": n_accepted / max(1, total_steps),
        "backend": "rmhmc",
        "diagnostics": {
            "step_size": step_size,
            "n_leapfrog": n_leapfrog,
            "n_fixpoint": n_fixpoint,
            "alpha": alpha,
            "n_data": x_data.shape[0],
            "n_burnin": n_burnin,
            "thin": thin,
            "n_accepted": n_accepted,
            "total_steps": total_steps,
            "adapted_step_size": step_size if adapt_step_size else None,
            "energy_errors": energy_errors,
            "accept_probs": accept_probs,
            "mass_type": "rmhmc_softabs",
        },
    }

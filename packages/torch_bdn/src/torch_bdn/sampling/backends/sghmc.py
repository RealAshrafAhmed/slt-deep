"""
SGHMC (Stochastic Gradient Hamiltonian Monte Carlo) sampler.

Chen, Fox & Guestrin (2014) "Stochastic Gradient Hamiltonian Monte Carlo".

Like HMC but uses mini-batch gradients with a friction term to compensate
for the noise introduced by stochastic gradients.  No Metropolis correction
(which would require full-batch evaluation), so this is an approximate
sampler — but it scales to large datasets.

Update equations (per step):
    r  ←  r  +  ε ∇̃ log p(θ)  −  α r  +  η,   η ~ N(0, 2(α − β̂) ε I)
    θ  ←  θ  +  ε r

where
    ∇̃ log p(θ)  =  (n/B) ∇ log p(y_batch | x_batch, θ)  +  ∇ log p(θ)
    α   = friction coefficient (user-specified, default 0.1)
    β̂   = estimated noise from stochastic gradients (set to 0 for simplicity;
           the friction alone stabilises the chain)
"""

from collections.abc import Callable
from typing import Any

import torch


def sghmc(
    set_params_fn: Callable[[torch.Tensor], None],
    log_likelihood_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    log_prior_fn: Callable[[torch.Tensor], torch.Tensor],
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    init_params: torch.Tensor,
    n_samples: int = 1000,
    batch_size: int | None = None,
    lr: float = 0.01,
    alpha: float = 0.1,
    n_burnin: int = 1000,
    thin: int = 1,
    resample_momentum: int = 50,
    diff_log_likelihood_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ]
    | None = None,
    preconditioner: torch.Tensor | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Stochastic Gradient Hamiltonian Monte Carlo.

    Args:
        set_params_fn: sets model parameters from flat tensor
        log_likelihood_fn: (x_batch, y_batch) → log p(y|x, θ)  (legacy path)
        log_prior_fn: (flat_params) → log p(θ)
        x_data, y_data: full dataset tensors
        init_params: starting parameter vector
        n_samples: posterior samples to collect
        batch_size: mini-batch size (auto-selected if None)
        lr: step size ε — controls discretisation; start ~0.01
        alpha: friction coefficient (momentum decay).  Higher = more
            diffusive (SGLD-like); lower = longer momentum coherence.
            Typical range 0.01–0.5.
        n_burnin: warmup iterations (discarded)
        thin: keep every thin-th sample
        resample_momentum: resample momentum from N(0, ε I) every this
            many steps (0 = never).  Prevents slow energy drift.
        diff_log_likelihood_fn: (flat_params, x, y) → log p(y|x,θ).
            Differentiable w.r.t. flat_params (uses functional_call).
            When provided, replaces the broken set_params + log_likelihood
            path whose gradient is severed by vector_to_parameters.
        preconditioner: Hessian H of the negative log-likelihood at the
            MAP.  When provided, the sampler builds M = H⁺ + I_null
            (pseudoinverse on column space, identity on null space) and
            preconditions both gradient and noise.

    Returns:
        dict with keys: parameters, log_probabilities, acceptance_rate,
        backend, diagnostics
    """
    device = init_params.device
    dim = init_params.numel()
    n_data = x_data.shape[0]

    # Auto batch size
    if batch_size is None:
        if n_data < 1000:
            batch_size = min(n_data, 64)
        elif n_data < 10000:
            batch_size = 128
        else:
            batch_size = 256

    # ── Build preconditioner from Hessian ────────────────────────────────
    # Same pseudoinverse + identity-on-null-space construction as SGLD.
    #   m_i = 1/λ_i   if λ_i > δ   (column space)
    #   m_i = 1        if λ_i ≤ δ   (null space)
    # Normalised so median(m) ≈ 1.
    if preconditioner is not None:
        H = preconditioner.float().to("cpu")
        eigvals_h, eigvecs_h = torch.linalg.eigh(H)
        lambda_max = float(eigvals_h.max().clamp(min=1e-8))
        delta = lambda_max * 1e-3
        informative = eigvals_h > delta
        m_eig = torch.ones_like(eigvals_h)
        m_eig[informative] = 1.0 / eigvals_h[informative]
        m_median = m_eig.median()
        if m_median > 0:
            m_eig = m_eig / m_median
        n_degen = int((~informative).sum())
        _V = eigvecs_h.to(device)
        _m = m_eig.to(device)
        _sqrt_m = m_eig.sqrt().to(device)
        _preconditioned = True
    else:
        _preconditioned = False
        n_degen = 0

    params = init_params.clone().detach()
    momentum = torch.randn(dim, device=device) * (lr**0.5)

    # Noise variance: 2 (α - β̂) ε  with β̂ = 0
    noise_std = (2.0 * alpha * lr) ** 0.5

    samples: list[torch.Tensor] = []
    warmup_trace: list[torch.Tensor] = []
    logp_vals: list[float] = []
    warmup_logp: list[float] = []
    total_steps = n_burnin + n_samples * thin
    n_batches = (n_data + batch_size - 1) // batch_size

    indices = torch.randperm(n_data, device=device)

    for step in range(total_steps):
        # Resample momentum periodically to avoid energy drift
        if resample_momentum > 0 and step % resample_momentum == 0:
            momentum = torch.randn(dim, device=device) * (lr**0.5)

        # --- mini-batch ---
        batch_idx = step % n_batches
        if batch_idx == 0:
            indices = torch.randperm(n_data, device=device)
        start = batch_idx * batch_size
        end = min(start + batch_size, n_data)
        bi = indices[start:end]
        xb, yb = x_data[bi], y_data[bi]
        actual_batch = xb.shape[0]

        # --- stochastic gradient of log joint ---
        p = params.clone().detach().requires_grad_(True)

        if diff_log_likelihood_fn is not None:
            ll = diff_log_likelihood_fn(p, xb, yb)
        else:
            set_params_fn(p)
            ll = log_likelihood_fn(xb, yb)

        lp = log_prior_fn(p)

        grad_ll = torch.autograd.grad(
            ll,
            p,
            retain_graph=True,
            allow_unused=True,
        )[0]
        grad_lp = torch.autograd.grad(
            lp,
            p,
            allow_unused=True,
        )[0]
        if grad_ll is None:
            grad_ll = torch.zeros(dim, device=device)
        if grad_lp is None:
            grad_lp = torch.zeros(dim, device=device)

        # Scale likelihood gradient for unbiased estimate
        grad_logp = (n_data / actual_batch) * grad_ll + grad_lp

        # Clip for safety
        gn = grad_logp.norm()
        if gn > 100.0:
            grad_logp = grad_logp * (100.0 / gn)

        logp_approx = (ll + lp).detach()

        # --- SGHMC update (optionally preconditioned) ---
        with torch.no_grad():
            if _preconditioned:
                # M g = V diag(m) V^T g
                g_eig = _V.T @ grad_logp
                Mg = _V @ (_m * g_eig)
                # noise: sqrt(2 α ε) * V diag(sqrt(m)) ξ
                xi = torch.randn(dim, device=device)
                noise = (2.0 * alpha * lr) ** 0.5 * (_V @ (_sqrt_m * xi))
                momentum = momentum + lr * Mg - alpha * momentum + noise
                params = params + lr * momentum
            else:
                noise = noise_std * torch.randn(dim, device=device)
                momentum = momentum + lr * grad_logp - alpha * momentum + noise
                params = params + lr * momentum
            params.clamp_(-50.0, 50.0)

        # --- collect ---
        if step < n_burnin:
            # Save warmup trace (thin to ~1000 points max for memory)
            warmup_thin = max(1, n_burnin // 1000)
            if step % warmup_thin == 0:
                warmup_trace.append(params.clone().detach())
                warmup_logp.append(logp_approx.item())
        elif (step - n_burnin) % thin == 0:
            samples.append(params.clone().detach())
            logp_vals.append(logp_approx.item())

    return {
        "parameters": samples,
        "log_probabilities": logp_vals,
        "acceptance_rate": 1.0,  # no MH step
        "backend": "sghmc",
        "diagnostics": {
            "lr": lr,
            "alpha": alpha,
            "batch_size": batch_size,
            "noise_std": noise_std,
            "resample_momentum": resample_momentum,
            "n_data": n_data,
            "n_burnin": n_burnin,
            "thin": thin,
            "total_steps": total_steps,
            "preconditioned": _preconditioned,
            "n_degenerate_directions": n_degen,
            "n_total_directions": dim,
            "warmup_trace": warmup_trace,
            "warmup_logp": warmup_logp,
        },
    }

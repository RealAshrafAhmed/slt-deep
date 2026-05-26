"""
HMC / NUTS sampler implementation.

Provides both:
  - Standard HMC with fixed leapfrog trajectory length
  - NUTS (No-U-Turn Sampler, Hoffman & Gelman 2014) which adaptively
    selects trajectory length via the U-turn criterion

Both use full-dataset gradients and are exact (up to leapfrog
discretisation) with Metropolis / multinomial correction.

NUTS also includes dual-averaging step-size adaptation during warmup.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────


def _grad_log_joint(
    params: torch.Tensor,
    set_params_fn: Callable[[torch.Tensor], None],
    log_likelihood_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    log_prior_fn: Callable[[torch.Tensor], torch.Tensor],
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    diff_log_likelihood_fn: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ]
    | None = None,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (grad_log_joint, log_joint_value) at *params*.

    When *diff_log_likelihood_fn* is provided it is used instead of the
    legacy ``set_params_fn`` → ``log_likelihood_fn`` path, which breaks
    the autograd graph because ``vector_to_parameters`` severs the
    connection between the flat vector and the model parameters.

    The ``beta`` parameter is the inverse temperature: the tempered target
    is  π_β(θ) ∝ p(D|θ)^β · p(θ),  i.e. the likelihood is raised to the
    power β while the prior is untempered.  β=1 recovers the true posterior.
    """
    p = params.detach().requires_grad_(True)

    if diff_log_likelihood_fn is not None:
        ll = diff_log_likelihood_fn(p, x_data, y_data)
    else:
        # Legacy path (gradient only flows through the prior).
        set_params_fn(p)
        ll = log_likelihood_fn(x_data, y_data)

    lp = log_prior_fn(p)
    logp = beta * ll + lp
    grad = torch.autograd.grad(logp, p)[0]

    return grad.detach(), logp.detach()


def _leapfrog(
    q: torch.Tensor,
    r: torch.Tensor,
    grad_U: torch.Tensor,
    step_size: float,
    M_inv_diag: torch.Tensor,
    grad_U_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One leapfrog step. Returns (q_new, r_new, grad_U_new, logp_new)."""
    r_half = r - 0.5 * step_size * grad_U
    q_new = q + step_size * M_inv_diag * r_half
    grad_U_new, logp_new = grad_U_fn(q_new)
    r_new = r_half - 0.5 * step_size * grad_U_new
    return q_new, r_new, grad_U_new, logp_new


def _leapfrog_general(
    q: torch.Tensor,
    r: torch.Tensor,
    grad_U: torch.Tensor,
    step_size: float,
    M_inv_fn: Callable[[torch.Tensor], torch.Tensor],
    grad_U_fn: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One leapfrog step with callable M_inv. Returns (q_new, r_new, grad_U_new, logp_new)."""
    r_half = r - 0.5 * step_size * grad_U
    q_new = q + step_size * M_inv_fn(r_half)
    grad_U_new, logp_new = grad_U_fn(q_new)
    r_new = r_half - 0.5 * step_size * grad_U_new
    return q_new, r_new, grad_U_new, logp_new


def _kinetic(r: torch.Tensor, M_inv_diag: torch.Tensor) -> torch.Tensor:
    return 0.5 * (r * M_inv_diag * r).sum()


def _build_mass_ops(
    mass_matrix: torch.Tensor | None,
    dim: int,
    dev: torch.device,
) -> tuple[
    Callable[[], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    str,
]:
    """Build (sample_momentum, M_inv, kinetic, mass_type) from a mass matrix.

    Supports None (identity), 1-D (diagonal), and 2-D (full) mass matrices.
    """
    if mass_matrix is None:

        def sample_momentum():
            return torch.randn(dim, device=dev)

        def M_inv(r: torch.Tensor) -> torch.Tensor:
            return r

        def kinetic(r: torch.Tensor) -> torch.Tensor:
            return 0.5 * r.dot(r)

        return sample_momentum, M_inv, kinetic, "identity"

    elif mass_matrix.dim() == 1:
        M_diag = mass_matrix.detach().clone().float().to(dev)
        M_inv_d = 1.0 / M_diag
        M_sqrt_d = M_diag.sqrt()

        def sample_momentum():
            return torch.randn(dim, device=dev) * M_sqrt_d

        def M_inv(r: torch.Tensor) -> torch.Tensor:
            return M_inv_d * r

        def kinetic(r: torch.Tensor) -> torch.Tensor:
            return 0.5 * (r * M_inv_d * r).sum()

        return sample_momentum, M_inv, kinetic, "diagonal"

    elif mass_matrix.dim() == 2:
        H_cpu = mass_matrix.detach().float().cpu()
        eigvals_h, eigvecs_h = torch.linalg.eigh(H_cpu)
        # For degenerate directions (eigenvalue < λ_max × 1e-3), use
        # identity mass (1.0) instead of the tiny estimated eigenvalue.
        # This avoids M⁻¹ blowing up when the sample covariance is
        # rank-deficient (n_draws < d).
        lambda_max = float(eigvals_h.max().clamp(min=1e-8))
        threshold = lambda_max * 1e-3
        degenerate = eigvals_h < threshold
        eigvals_h = eigvals_h.clone()
        eigvals_h[degenerate] = 1.0  # identity mass in null space

        V = eigvecs_h.to(dev)
        _m_eig = eigvals_h.to(dev)
        _m_inv_eig = (1.0 / eigvals_h).to(dev)
        _m_sqrt_eig = eigvals_h.sqrt().to(dev)

        def sample_momentum():
            z = torch.randn(dim, device=dev)
            return V @ (_m_sqrt_eig * z)

        def M_inv(r: torch.Tensor) -> torch.Tensor:
            return V @ (_m_inv_eig * (V.T @ r))

        def kinetic(r: torch.Tensor) -> torch.Tensor:
            Vtr = V.T @ r
            return 0.5 * (_m_inv_eig * Vtr * Vtr).sum()

        return sample_momentum, M_inv, kinetic, "full"

    else:
        raise ValueError(
            f"mass_matrix must be 1-D (diagonal) or 2-D (full), got {mass_matrix.dim()}-D"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standard HMC
# ─────────────────────────────────────────────────────────────────────────────


def hmc(
    set_params_fn: Callable[[torch.Tensor], None],
    log_likelihood_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    log_prior_fn: Callable[[torch.Tensor], torch.Tensor],
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    init_params: torch.Tensor,
    n_samples: int = 1000,
    step_size: float = 0.01,
    n_leapfrog: int = 25,
    n_burnin: int = 500,
    thin: int = 1,
    mass_matrix: torch.Tensor | None = None,
    null_space_mass: torch.Tensor | float | None = None,
    adapt_step_size: bool = False,
    target_accept: float = 0.65,
    degen_threshold: float = 1e-3,
    beta: float = 1.0,
    **kwargs,
) -> dict[str, Any]:
    """
    Standard Hamiltonian Monte Carlo with fixed trajectory length.

    For automatic trajectory-length selection use ``nuts()`` instead.

    Args:
        set_params_fn: sets model parameters from flat tensor
        log_likelihood_fn: (x, y) → log p(y|x, θ)
        log_prior_fn: (flat_params) → log p(θ)
        x_data, y_data: full dataset tensors
        init_params: starting parameter vector
        n_samples: posterior samples to collect
        step_size: leapfrog step size ε
        n_leapfrog: number of leapfrog steps L per proposal
        n_burnin: warmup iterations (discarded)
        thin: keep every thin-th sample
        mass_matrix: Mass matrix for HMC.
            - 1-D tensor → diagonal mass (element-wise).
            - 2-D tensor → full mass.  Decomposed via eigendecomposition
              with the same pseudoinverse + identity-on-null-space
              construction as the SGLD/SGHMC preconditioner.
            - None → identity mass.
        null_space_mass: Mass to use for degenerate directions of the
            mass matrix (eigenvalues below λ_max × 1e-3).  Defaults to 1
            (identity).  Typical usage: pass the prior precision so that
            degenerate directions get mass matching the prior curvature.
            - scalar → constant mass in all degenerate directions.
            - 1-D tensor (d,) → per-parameter mass floor; projected onto
              the degenerate eigendirections via Vᵀ diag(floor) V.
            - 2-D tensor (d, d) → full matrix (e.g. prior Hessian);
              projected onto the degenerate subspace.
        adapt_step_size: use dual-averaging during warmup to tune ε
        target_accept: target Metropolis acceptance rate (for adaptation)

    Returns:
        dict with parameters, log_probabilities, acceptance_rate, diagnostics
    """
    diff_log_likelihood_fn = kwargs.pop("diff_log_likelihood_fn", None)

    dim = init_params.numel()
    dev = init_params.device
    params = init_params.clone().detach()

    # ── Build mass matrix operations ──────────────────────────────────────
    # We need three callables:
    #   sample_momentum() → r ~ N(0, M)
    #   M_inv(r)          → M⁻¹ r
    #   kinetic(r)        → ½ rᵀ M⁻¹ r
    n_degen = 0

    if mass_matrix is None:
        # Identity mass
        def sample_momentum():
            return torch.randn(dim, device=dev)

        def M_inv(r: torch.Tensor) -> torch.Tensor:
            return r

        def kinetic(r: torch.Tensor) -> torch.Tensor:
            return 0.5 * r.dot(r)

        mass_type = "identity"

    elif mass_matrix.dim() == 1:
        # Diagonal mass
        M_diag = mass_matrix.detach().clone().float().to(dev)
        M_inv_d = 1.0 / M_diag

        def sample_momentum():
            return torch.randn(dim, device=dev) * M_diag.sqrt()

        def M_inv(r: torch.Tensor) -> torch.Tensor:
            return M_inv_d * r

        def kinetic(r: torch.Tensor) -> torch.Tensor:
            return 0.5 * (r * M_inv_d * r).sum()

        mass_type = "diagonal"

    elif mass_matrix.dim() == 2:
        # Full mass — eigendecomposition with pseudoinverse.
        # Degenerate directions (eigenvalue < λ_max × degen_threshold) get mass from
        # null_space_mass if provided, otherwise identity.
        H_cpu = mass_matrix.detach().float().cpu()
        eigvals_h, eigvecs_h = torch.linalg.eigh(H_cpu)
        lambda_max = float(eigvals_h.max().clamp(min=1e-8))
        delta = lambda_max * degen_threshold
        informative = eigvals_h > delta
        n_degen = int((~informative).sum())

        # M eigenvalues: λ_i if informative, else from null_space_mass
        m_eig = torch.ones_like(eigvals_h)
        m_eig[informative] = eigvals_h[informative]

        if null_space_mass is not None and n_degen > 0:
            degen_mask = ~informative
            if isinstance(null_space_mass, int | float):
                # Scalar → constant mass in all degenerate directions
                m_eig[degen_mask] = float(null_space_mass)
            elif isinstance(null_space_mass, torch.Tensor):
                nsm = null_space_mass.detach().float().cpu()
                if nsm.dim() == 0:
                    # Scalar tensor
                    m_eig[degen_mask] = nsm.item()
                elif nsm.dim() == 1:
                    # Per-parameter mass floor → project onto eigenbasis:
                    # m_j = vⱼᵀ diag(floor) vⱼ for each degenerate direction j
                    V_degen = eigvecs_h[:, degen_mask]  # (d, n_degen)
                    m_eig[degen_mask] = (
                        (V_degen * nsm.unsqueeze(1)).sum(0).clamp(min=1e-10)
                    )
                elif nsm.dim() == 2:
                    # Full matrix → project: m_j = vⱼᵀ P vⱼ for degenerate j
                    V_degen = eigvecs_h[:, degen_mask]  # (d, n_degen)
                    PV = nsm @ V_degen  # (d, n_degen)
                    m_eig[degen_mask] = (V_degen * PV).sum(0).clamp(min=1e-10)
                else:
                    raise ValueError(
                        f"null_space_mass must be scalar, 1-D, or 2-D, got {nsm.dim()}-D"
                    )

        # M⁻¹ eigenvalues
        m_inv_eig = 1.0 / m_eig

        V = eigvecs_h.to(dev)
        _m_eig = m_eig.to(dev)
        _m_inv_eig = m_inv_eig.to(dev)
        _m_sqrt_eig = m_eig.sqrt().to(dev)
        _m_inv_sqrt_eig = m_inv_eig.sqrt().to(dev)

        def sample_momentum():
            # r = V diag(√m) z,  z ~ N(0,I)  →  r ~ N(0, M)
            z = torch.randn(dim, device=dev)
            return V @ (_m_sqrt_eig * z)

        def M_inv(r: torch.Tensor) -> torch.Tensor:
            # M⁻¹ r = V diag(m_inv) Vᵀ r
            return V @ (_m_inv_eig * (V.T @ r))

        def kinetic(r: torch.Tensor) -> torch.Tensor:
            # ½ rᵀ M⁻¹ r = ½ ||diag(√m_inv) Vᵀ r||²
            Vtr = V.T @ r
            return 0.5 * (_m_inv_eig * Vtr * Vtr).sum()

        mass_type = "full"
    else:
        raise ValueError(
            f"mass_matrix must be 1-D (diagonal) or 2-D (full), got {mass_matrix.dim()}-D"
        )

    def _grad_U(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g, lp = _grad_log_joint(
            q,
            set_params_fn,
            log_likelihood_fn,
            log_prior_fn,
            x_data,
            y_data,
            diff_log_likelihood_fn=diff_log_likelihood_fn,
            beta=beta,
        )
        return -g, lp

    # Dual-averaging state (Nesterov 2009 / NUTS paper §3.2)
    log_eps = math.log(step_size)
    log_eps_bar = 0.0
    H_bar = 0.0
    gamma, t0, kappa = 0.05, 10.0, 0.75
    mu = math.log(10.0 * step_size)

    samples: list[torch.Tensor] = []
    logp_vals: list[float] = []
    energy_errors: list[float] = []  # ΔH per post-warmup step
    accept_probs: list[float] = []  # α per post-warmup step
    warmup_eps_trace: list[float] = []  # ε at each warmup step
    n_accepted = 0
    n_accepted_window = 0  # accepted in current 100-step window
    total_steps = n_burnin + n_samples * thin

    _, logp_current = _grad_U(params)

    for step in range(total_steps):
        eps = math.exp(log_eps) if adapt_step_size and step < n_burnin else step_size

        r = sample_momentum()
        H_current = -logp_current + kinetic(r)

        q = params.clone()
        r_lf = r.clone()
        gU, _ = _grad_U(q)

        # Leapfrog integration
        for _ in range(n_leapfrog):
            r_lf = r_lf - 0.5 * eps * gU
            q = q + eps * M_inv(r_lf)
            gU, logp_prop = _grad_U(q)
            r_lf = r_lf - 0.5 * eps * gU

        H_proposed = -logp_prop + kinetic(-r_lf)
        delta_H = (H_proposed - H_current).item()
        # Guard against NaN/inf from divergent leapfrog — treat as rejection
        if not math.isfinite(delta_H):
            delta_H = float("inf")
        log_alpha = -delta_H  # = H_current - H_proposed
        alpha = min(1.0, math.exp(min(log_alpha, 0.0)))

        if math.log(torch.rand(1).item() + 1e-30) < log_alpha:
            params = q
            logp_current = logp_prop
            n_accepted += 1
            n_accepted_window += 1

        # Progress logging every 100 steps
        if (step + 1) % 100 == 0:
            phase = "warmup" if step < n_burnin else "sample"
            window_rate = n_accepted_window / 100
            print(
                f"  [HMC {phase}] step {step + 1}/{total_steps}  "
                f"ε={eps:.2e}  accept_rate(last 100)={window_rate:.2f}"
            )
            n_accepted_window = 0

        # Dual averaging update
        if adapt_step_size and step < n_burnin:
            m = step + 1
            w = 1.0 / (m + t0)
            H_bar = (1 - w) * H_bar + w * (target_accept - alpha)
            log_eps = mu - (m**0.5) / gamma * H_bar
            m_kappa = m ** (-kappa)
            log_eps_bar = m_kappa * log_eps + (1 - m_kappa) * log_eps_bar
            warmup_eps_trace.append(math.exp(log_eps_bar))

        if adapt_step_size and step == n_burnin - 1:
            step_size = math.exp(log_eps_bar)

        if step >= n_burnin and (step - n_burnin) % thin == 0:
            samples.append(params.clone().detach())
            logp_vals.append(logp_current.item())
            energy_errors.append(delta_H)
            accept_probs.append(alpha)

    return {
        "parameters": samples,
        "log_probabilities": logp_vals,
        "acceptance_rate": n_accepted / max(1, total_steps),
        "backend": "hmc",
        "diagnostics": {
            "step_size": step_size,
            "n_leapfrog": n_leapfrog,
            "n_data": x_data.shape[0],
            "n_burnin": n_burnin,
            "thin": thin,
            "n_accepted": n_accepted,
            "total_steps": total_steps,
            "adapted_step_size": step_size if adapt_step_size else None,
            "warmup_eps_trace": warmup_eps_trace,
            "energy_errors": energy_errors,
            "accept_probs": accept_probs,
            "mass_type": mass_type,
            "n_degenerate_directions": n_degen,
            "n_total_directions": dim,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stan-style windowed warmup schedule
# ─────────────────────────────────────────────────────────────────────────────


def _stan_warmup_schedule(
    n_warmup: int,
    init_buffer: int = 75,
    term_buffer: int = 50,
    base_window: int = 25,
) -> list[tuple[int, int, str]]:
    """Compute Stan-style warmup windows.

    Returns a list of (start, end, phase) tuples where phase is one of:
    - "init": initial fast ε adaptation (no M adaptation)
    - "adapt": ε + M adaptation (expanding windows)
    - "term": terminal ε adaptation (M locked)

    Within "adapt" windows, the mass matrix is re-estimated from the
    sample covariance at the end of each window.
    """
    if n_warmup < init_buffer + term_buffer + base_window:
        # Not enough warmup for windowed adaptation — just do ε adaptation
        return [(0, n_warmup, "init")]

    windows: list[tuple[int, int, str]] = []

    # Phase I: initial ε adaptation
    windows.append((0, init_buffer, "init"))

    # Phase II: expanding windows for M adaptation
    adapt_start = init_buffer
    adapt_end = n_warmup - term_buffer
    cursor = adapt_start
    window_size = base_window

    while cursor < adapt_end:
        end = min(cursor + window_size, adapt_end)
        # If next window wouldn't fit, extend current to fill
        if end + base_window > adapt_end:
            end = adapt_end
        windows.append((cursor, end, "adapt"))
        cursor = end
        window_size *= 2  # double window size each time

    # Phase III: terminal ε adaptation
    windows.append((adapt_end, n_warmup, "term"))

    return windows


# ─────────────────────────────────────────────────────────────────────────────
# NUTS  (No-U-Turn Sampler — Hoffman & Gelman 2014, Algorithm 6)
# ─────────────────────────────────────────────────────────────────────────────


def _uturn(
    q_minus: torch.Tensor,
    q_plus: torch.Tensor,
    r_minus: torch.Tensor,
    r_plus: torch.Tensor,
) -> bool:
    """Check U-turn criterion: trajectory endpoints moving apart?"""
    dq = q_plus - q_minus
    return bool((dq @ r_minus).item() < 0 or (dq @ r_plus).item() < 0)


def nuts(
    set_params_fn: Callable[[torch.Tensor], None],
    log_likelihood_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    log_prior_fn: Callable[[torch.Tensor], torch.Tensor],
    x_data: torch.Tensor,
    y_data: torch.Tensor,
    init_params: torch.Tensor,
    n_samples: int = 1000,
    step_size: float = 0.01,
    n_burnin: int = 500,
    thin: int = 1,
    max_tree_depth: int = 10,
    mass_matrix: torch.Tensor | None = None,
    adapt_step_size: bool = True,
    target_accept: float = 0.80,
    adapt_mass_matrix: bool = False,
    beta: float = 1.0,
    step_offset: int = 0,
    n_total_expected: int | None = None,
    chain_label: str | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    No-U-Turn Sampler (NUTS) — Hoffman & Gelman 2014, Algorithm 6.

    Adaptively selects trajectory length by doubling the tree until a
    U-turn is detected.  Includes dual-averaging step-size adaptation
    during warmup so only ``step_size`` needs a rough initial guess.

    When ``adapt_mass_matrix=True``, uses Stan-style windowed warmup:
    expanding windows estimate the mass matrix from sample covariance
    during warmup.  Supports identity, diagonal (1-D), and full (2-D)
    mass matrices.

    Args:
        set_params_fn: sets model parameters from flat tensor
        log_likelihood_fn: (x, y) → log p(y|x, θ)
        log_prior_fn: (flat_params) → log p(θ)
        x_data, y_data: full dataset
        init_params: starting parameter vector
        n_samples: posterior samples to collect
        step_size: initial leapfrog step size ε (adapted during warmup)
        n_burnin: warmup iterations (discarded; step-size adaptation here)
        thin: keep every thin-th post-warmup sample
        max_tree_depth: maximum binary-tree depth (trajectory ≤ 2^depth steps)
        mass_matrix: mass matrix.
            - None → identity mass.
            - 1-D tensor → diagonal mass.
            - 2-D tensor → full mass (eigendecomposed).
        adapt_step_size: whether to adapt ε via dual averaging during warmup
        target_accept: target acceptance probability for dual averaging
        adapt_mass_matrix: if True, use Stan-style windowed warmup to adapt
            a diagonal mass matrix from marginal variances of warmup draws.
            The initial ``mass_matrix`` is used for the first window;
            subsequent windows re-estimate from warmup samples.
            Diagonal adaptation is the standard for high-dimensional models
            (Hoffman & Gelman 2014, Izmailov et al. 2021).

    Returns:
        dict with parameters, log_probabilities, acceptance_rate, diagnostics
    """
    diff_log_likelihood_fn = kwargs.pop("diff_log_likelihood_fn", None)

    dim = init_params.numel()
    dev = init_params.device
    params = init_params.clone().detach()

    # Store initial mass matrix for shrinkage target during adaptation
    M_initial = (
        mass_matrix.detach().clone().cpu().float() if mass_matrix is not None else None
    )

    # Track the current mass matrix (updated during adaptation)
    current_mass_matrix = (
        mass_matrix.detach().clone().cpu() if mass_matrix is not None else None
    )

    # Build initial mass matrix operations
    sample_momentum, M_inv_fn, kinetic_fn, mass_type = _build_mass_ops(
        mass_matrix, dim, dev
    )

    def _grad_U(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g, lp = _grad_log_joint(
            q,
            set_params_fn,
            log_likelihood_fn,
            log_prior_fn,
            x_data,
            y_data,
            diff_log_likelihood_fn=diff_log_likelihood_fn,
            beta=beta,
        )
        return -g, lp  # grad of U = -grad log p, and logp value

    def _H(q: torch.Tensor, r: torch.Tensor, logp: torch.Tensor) -> float:
        return (-logp + kinetic_fn(r)).item()

    # ── Dual-averaging state ──────────────────────────────────────────────
    log_eps = math.log(step_size)
    log_eps_bar = 0.0
    H_bar = 0.0
    gamma, t0, kappa = 0.05, 10.0, 0.75
    mu = math.log(10.0 * step_size)

    def _reset_dual_avg():
        nonlocal log_eps, log_eps_bar, H_bar, mu
        # Reset dual averaging but keep the current ε as starting point
        current_eps = math.exp(log_eps_bar) if log_eps_bar != 0.0 else math.exp(log_eps)
        log_eps = math.log(current_eps)
        log_eps_bar = math.log(current_eps)
        H_bar = 0.0
        mu = math.log(10.0 * current_eps)

    # ── Windowed warmup schedule ──────────────────────────────────────────
    warmup_schedule = _stan_warmup_schedule(n_burnin) if adapt_mass_matrix else []
    # Pre-compute which window each warmup step belongs to
    warmup_step_to_window: dict[int, int] = {}
    warmup_window_ends: set[int] = set()
    for wi, (ws, we, wp) in enumerate(warmup_schedule):
        for s in range(ws, we):
            warmup_step_to_window[s] = wi
        warmup_window_ends.add(we - 1)  # last step of window

    # Buffer for warmup draws (for M estimation)
    warmup_draw_buffer: list[torch.Tensor] = []

    samples: list[torch.Tensor] = []
    logp_vals: list[float] = []
    tree_depths: list[int] = []
    leapfrog_counts: list[int] = []
    divergences: list[int] = []  # step indices where divergence occurred
    warmup_eps_trace: list[float] = []
    warmup_mass_updates: list[dict[str, Any]] = []  # track M updates
    n_accepted = 0
    total_steps = n_burnin + n_samples * thin
    # Chunk-level accumulators for progress logging
    _chunk_divs = 0
    _chunk_alphas: list[float] = []

    gU_current, logp_current = _grad_U(params)

    for step in range(total_steps):
        eps = math.exp(log_eps) if adapt_step_size and step < n_burnin else step_size

        # Sample momentum
        r0 = sample_momentum()
        H0 = _H(params, r0, logp_current)

        # Slice variable  u ~ Uniform(0, exp(-H0))  ⟹  log u = -H0 - Exp(1)
        log_u = -H0 - torch.distributions.Exponential(1.0).sample().item()

        # Initialise tree
        q_minus = params.clone()
        q_plus = params.clone()
        r_minus = r0.clone()
        r_plus = r0.clone()
        gU_minus = gU_current.clone()
        gU_plus = gU_current.clone()
        logp_minus = logp_current.clone()
        logp_plus = logp_current.clone()

        q_prop = params.clone()
        logp_prop = logp_current.clone()
        gU_prop = gU_current.clone()  # track gradient of accepted proposal
        depth = 0
        n_valid = 1  # number of acceptable states in the tree
        keep_going = True
        n_leapfrog = 0  # total leapfrog steps this iteration
        step_diverged = False

        while keep_going and depth < max_tree_depth:
            # Choose direction: extend tree forward or backward
            direction = 1 if torch.rand(1).item() < 0.5 else -1

            # Take 2^depth leapfrog steps in `direction`
            n_steps = 2**depth
            q_new = None
            logp_new = None
            n_valid_subtree = 0
            subtree_diverged = False

            if direction == 1:
                q_walk = q_plus.clone()
                r_walk = r_plus.clone()
                gU_walk = gU_plus.clone()
            else:
                q_walk = q_minus.clone()
                r_walk = r_minus.clone()
                gU_walk = gU_minus.clone()

            for _i in range(n_steps):
                q_walk, r_walk, gU_walk, logp_walk = _leapfrog_general(
                    q_walk,
                    direction * r_walk,
                    gU_walk,
                    eps,
                    M_inv_fn,
                    _grad_U,
                )
                r_walk = direction * r_walk  # un-flip for storage
                n_leapfrog += 1

                H_walk = _H(q_walk, r_walk, logp_walk)

                # Divergence check
                if H_walk - H0 > 1000.0:
                    subtree_diverged = True
                    break

                # Is this state acceptable? (within the slice)
                if -H_walk >= log_u:
                    n_valid_subtree += 1
                    # Multinomial sampling from the trajectory
                    if torch.rand(1).item() < 1.0 / max(1, n_valid + n_valid_subtree):
                        q_new = q_walk.clone()
                        logp_new = logp_walk.clone()
                        gU_new = gU_walk.clone()

            if subtree_diverged:
                step_diverged = True
                keep_going = False
                break

            # Update tree endpoints
            if direction == 1:
                q_plus = q_walk
                r_plus = r_walk
                gU_plus = gU_walk
                logp_plus = logp_walk
            else:
                q_minus = q_walk
                r_minus = r_walk
                gU_minus = gU_walk
                logp_minus = logp_walk

            # Accept the subtree's proposal with appropriate probability
            if q_new is not None and n_valid_subtree > 0:
                accept_prob = n_valid_subtree / max(1, n_valid + n_valid_subtree)
                if torch.rand(1).item() < accept_prob:
                    q_prop = q_new
                    logp_prop = logp_new
                    gU_prop = gU_new

            n_valid += n_valid_subtree

            # U-turn check on the full tree span
            if _uturn(q_minus, q_plus, r_minus, r_plus):
                keep_going = False

            depth += 1

        # Accept the NUTS proposal (always accepted; selection is built into the tree)
        alpha_nuts = min(1.0, n_valid / max(1, 2**depth))
        # Override: divergent steps signal ε too large → alpha=0 for dual averaging
        if step_diverged:
            alpha_nuts = 0.0
        _chunk_alphas.append(alpha_nuts)
        params = q_prop
        logp_current = logp_prop
        gU_current = gU_prop  # reuse cached gradient — no extra forward/backward
        n_accepted += 1

        # ── Warmup bookkeeping ────────────────────────────────────────────
        if step < n_burnin:
            # Dual-averaging step-size adaptation
            if adapt_step_size:
                # Use step-within-window for dual averaging when doing windowed warmup
                if adapt_mass_matrix and step in warmup_step_to_window:
                    wi = warmup_step_to_window[step]
                    ws, we, wp = warmup_schedule[wi]
                    m = step - ws + 1  # step within window
                else:
                    m = step + 1
                w = 1.0 / (m + t0)
                H_bar = (1 - w) * H_bar + w * (target_accept - alpha_nuts)
                log_eps = mu - (m**0.5) / gamma * H_bar
                m_kappa = m ** (-kappa)
                log_eps_bar = m_kappa * log_eps + (1 - m_kappa) * log_eps_bar
                warmup_eps_trace.append(math.exp(log_eps_bar))

            # Mass matrix adaptation (windowed)
            if adapt_mass_matrix and step in warmup_step_to_window:
                wi = warmup_step_to_window[step]
                ws, we, wp = warmup_schedule[wi]

                if wp == "adapt":
                    warmup_draw_buffer.append(params.clone().detach().cpu())

                    # End of adaptation window → update M
                    if step == we - 1 and len(warmup_draw_buffer) >= 2:
                        draws = torch.stack(warmup_draw_buffer)  # (n, d)
                        n_draws = draws.shape[0]

                        # Shrinkage toward initial mass matrix (Hessian):
                        # M = (1-α)*Σ_sample + α*M_initial
                        # α = k/(n+k) — shrinks toward Hessian when few samples
                        shrink = 5.0 / (n_draws + 5.0)
                        centered = draws - draws.mean(0)
                        cov = (centered.T @ centered) / (n_draws - 1)

                        if M_initial is not None and M_initial.dim() == 2:
                            # Shrink toward the Hessian-based mass matrix
                            new_M = (1 - shrink) * cov + shrink * M_initial
                        elif M_initial is not None and M_initial.dim() == 1:
                            # Diagonal initial M — shrink toward it
                            new_M = (1 - shrink) * cov + shrink * torch.diag(M_initial)
                        else:
                            # No initial M — shrink toward identity
                            new_M = (1 - shrink) * cov + shrink * 1e-3 * torch.eye(dim)

                        adapt_type = "full"

                        sample_momentum, M_inv_fn, kinetic_fn, mass_type = (
                            _build_mass_ops(new_M, dim, dev)
                        )
                        current_mass_matrix = new_M.detach().clone().cpu()
                        warmup_mass_updates.append(
                            {
                                "window": wi,
                                "step": step,
                                "n_draws": n_draws,
                                "type": adapt_type,
                                "shrinkage": shrink,
                            }
                        )
                        warmup_draw_buffer.clear()

                        # Reset dual averaging after M update
                        _reset_dual_avg()

                elif wp == "term" and step == ws:
                    # Start of terminal window — reset dual avg with locked M
                    _reset_dual_avg()

        # Finalise step size at end of warmup
        if adapt_step_size and step == n_burnin - 1:
            step_size = math.exp(log_eps_bar)

        # Progress logging every 10 steps
        if step_diverged:
            _chunk_divs += 1
        if (step + 1) % 10 == 0:
            phase = "warmup" if step < n_burnin else "sample"
            maxd_str = " (hit max)" if depth >= max_tree_depth else ""
            chunk_alpha = sum(_chunk_alphas) / max(1, len(_chunk_alphas))
            global_step = step_offset + step + 1
            global_total = n_total_expected if n_total_expected is not None else (step_offset + total_steps)
            chain_tag = f" {chain_label}" if chain_label else ""
            print(
                f"  [NUTS {phase}{chain_tag}] step {global_step}/{global_total}  "
                f"ε={eps:.2e}  depth={depth}{maxd_str}  L={n_leapfrog}  "
                f"α={chunk_alpha:.2f}  divs={_chunk_divs}/10  "
                f"mass={mass_type}"
            )
            _chunk_divs = 0
            _chunk_alphas = []

        # Track divergences
        if step_diverged:
            divergences.append(step)

        # Collect
        if step >= n_burnin and (step - n_burnin) % thin == 0:
            samples.append(params.clone().detach())
            logp_vals.append(logp_current.item())
            tree_depths.append(depth)
            leapfrog_counts.append(n_leapfrog)

    return {
        "parameters": samples,
        "log_probabilities": logp_vals,
        "acceptance_rate": n_accepted / max(1, total_steps),
        "backend": "nuts",
        "diagnostics": {
            "step_size": step_size,
            "max_tree_depth": max_tree_depth,
            "mean_tree_depth": sum(tree_depths) / max(1, len(tree_depths)),
            "tree_depths": tree_depths,
            "leapfrog_counts": leapfrog_counts,
            "mean_leapfrog": sum(leapfrog_counts) / max(1, len(leapfrog_counts)),
            "n_hit_max_depth": sum(1 for d in tree_depths if d >= max_tree_depth),
            "n_divergences": len(divergences),
            "divergent_steps": divergences,
            "n_data": x_data.shape[0],
            "n_burnin": n_burnin,
            "thin": thin,
            "n_accepted": n_accepted,
            "total_steps": total_steps,
            "adapted_step_size": step_size if adapt_step_size else None,
            "warmup_eps_trace": warmup_eps_trace,
            "warmup_mass_updates": warmup_mass_updates,
            "warmup_schedule": warmup_schedule,
            "mass_type": mass_type,
            "adapted_mass_matrix": current_mass_matrix,
        },
    }

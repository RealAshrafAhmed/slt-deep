"""Constrained fitting: run the SAME training procedure used to find the MLE, but
hold a chosen set of parameter coordinates fixed.

This is the general procedure behind profile-likelihood surfaces. At a grid point
we pin two (or more) raw parameter coordinates at chosen values and re-run training
over the remaining ("free") parameters until it converges the way the original fit
did, then read off the loss/NLL there.

The ONLY change versus ordinary training is in backprop: the gradient entries of
the fixed coordinates are zeroed before each optimizer step (and the values hard
re-pinned afterwards as a numerical safeguard), so the optimizer never moves them
and their optimizer state stays at zero. Optimizer, learning rate, batching, epoch
count, and loss are otherwise identical to the caller's training regime — no custom
optimizer, no regularization added here, no gradient-subspace projection.
"""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


@dataclass
class ConstrainedFitResult:
    """Outcome of a constrained fit.

    Attributes:
        loss: Final loss at the converged constrained point (full-data, the same
            reduction loss_fn uses).
        theta: Fitted flat parameter vector (fixed coords hold their pinned values).
        epochs_run: Number of epochs actually run.
        free_grad_norm: L2 norm of the final full-batch gradient over the FREE
            coordinates (diagnostic; small means a genuine constrained minimum).
    """

    loss: float
    theta: torch.Tensor
    epochs_run: int
    free_grad_norm: float


def _zero_fixed_grads(
    model: nn.Module, fixed_mask_per_param: list[torch.Tensor | None]
):
    """Zero the .grad entries of the fixed coordinates, per parameter tensor."""
    for p, mask in zip(model.parameters(), fixed_mask_per_param):
        if mask is not None and p.grad is not None:
            p.grad[mask] = 0.0


def fit_constrained(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]],
    *,
    fixed_indices: torch.Tensor,
    fixed_values: torch.Tensor | None = None,
    init_theta: torch.Tensor | None = None,
    epochs: int = 100,
    lr: float = 3e-3,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    optimizer_kwargs: dict | None = None,
    full_data: tuple[torch.Tensor, torch.Tensor] | None = None,
    restore: bool = True,
) -> ConstrainedFitResult:
    """Run training with the chosen parameter coordinates held fixed.

    This mirrors a normal training loop. The caller supplies ``batches``, a
    zero-arg callable returning a fresh iterable of (x, y) mini-batches for one
    epoch (e.g. a closure over a DataLoader, or a single full-batch tuple yielded
    once) — so batching/shuffling exactly match the original training regime. The
    optimizer, lr, epoch count, and loss are the caller's; the only modification is
    zeroing the fixed coordinates' gradients each step.

    Args:
        model: Model to train (modified in place; restored on exit iff ``restore``).
        loss_fn: Maps ``(model(x), y) -> scalar loss`` (same reduction as training).
        batches: Zero-arg callable -> iterable of (x, y) mini-batches for one epoch.
        fixed_indices: 1-D long tensor of flat-parameter indices to hold fixed
            (``parameters_to_vector`` order).
        fixed_values: Values to pin those indices at. Defaults to the model's
            current values there.
        init_theta: Optional flat vector to initialize from. Defaults to the model's
            current parameters. Its fixed entries are overridden by ``fixed_values``.
        epochs: Number of training epochs (match the original regime).
        lr: Learning rate (match the original regime).
        optimizer_cls / optimizer_kwargs: Optimizer to use (defaults to Adam). Do
            not add regularization here unless the original training did.
        full_data: Optional (x, y) over the whole sample, used once at the end to
            report the converged loss and free-gradient norm. Defaults to
            concatenating the first epoch's batches.
        restore: Restore the model's entry parameters before returning.

    Returns:
        ConstrainedFitResult with the converged loss, fitted theta, epochs run, and
        free-gradient norm.
    """
    optimizer_kwargs = dict(optimizer_kwargs or {})
    fixed_indices = torch.as_tensor(fixed_indices, dtype=torch.long)

    entry_theta = parameters_to_vector(model.parameters()).detach().clone()
    theta = (
        init_theta.detach().clone() if init_theta is not None else entry_theta.clone()
    )

    if fixed_values is None:
        fixed_values = entry_theta[fixed_indices].clone()
    else:
        fixed_values = torch.as_tensor(fixed_values, dtype=theta.dtype).clone()

    # Pin fixed coords in the starting point, load into the model.
    theta[fixed_indices] = fixed_values
    vector_to_parameters(theta, model.parameters())

    # Per-parameter boolean masks marking which entries are fixed (for grad zeroing).
    fixed_global = {int(k) for k in fixed_indices.tolist()}
    masks: list[torch.Tensor | None] = []
    offset = 0
    for p in model.parameters():
        n = p.numel()
        local = [g - offset for g in fixed_global if offset <= g < offset + n]
        if local:
            m = torch.zeros(n, dtype=torch.bool)
            m[torch.tensor(local, dtype=torch.long)] = True
            masks.append(m.view_as(p))
        else:
            masks.append(None)
        offset += n

    opt = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)

    model.train()
    first_epoch_cache: list[tuple[torch.Tensor, torch.Tensor]] = []
    for ep in range(epochs):
        for xb, yb in batches():
            if ep == 0:
                first_epoch_cache.append((xb, yb))
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            _zero_fixed_grads(model, masks)  # the only change vs normal training
            opt.step()
            # Hard re-pin fixed coords (strip any numerical drift).
            with torch.no_grad():
                cur = parameters_to_vector(model.parameters())
                cur[fixed_indices] = fixed_values
                vector_to_parameters(cur, model.parameters())
    model.eval()

    # Report converged loss + free-grad norm on the whole sample.
    if full_data is not None:
        xf, yf = full_data
    else:
        xf = torch.cat([xb for xb, _ in first_epoch_cache], dim=0)
        yf = torch.cat([yb for _, yb in first_epoch_cache], dim=0)

    for p in model.parameters():
        p.grad = None
    loss_full = loss_fn(model(xf), yf)
    loss_full.backward()
    grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    free_mask = torch.ones_like(grad, dtype=torch.bool)
    free_mask[fixed_indices] = False
    free_grad_norm = grad[free_mask].norm().item()

    with torch.no_grad():
        theta_out = parameters_to_vector(model.parameters()).detach().clone()
        final_loss = loss_fn(model(xf), yf).item()

    if restore:
        vector_to_parameters(entry_theta, model.parameters())

    return ConstrainedFitResult(
        loss=final_loss,
        theta=theta_out,
        epochs_run=epochs,
        free_grad_norm=free_grad_norm,
    )


def top_diagonal_curvature_indices(hessian: torch.Tensor, k: int = 2) -> torch.Tensor:
    """Indices of the ``k`` parameters with the largest Hessian diagonal entries.

    A cheap alternative to an eigendecomposition for choosing profile axes: H_ii is
    the curvature along parameter i alone (vs. the top eigenvector, which is the
    stiffest *combination* of parameters). Returns flat-parameter indices, sorted
    by descending diagonal curvature.

    NOTE: single coordinates are usually PROFILE-flat in an over-parameterized model
    (the other params compensate when you re-minimize), so for a profile surface with
    visible curvature prefer the stiffest eigenvectors via fit_constrained_directions.
    """
    diag = torch.diagonal(hessian)
    return torch.topk(diag, k).indices


def fit_constrained_directions(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batches: Callable[[], Iterable[tuple[torch.Tensor, torch.Tensor]]],
    *,
    basis: torch.Tensor,
    coeffs: torch.Tensor,
    origin: torch.Tensor,
    init_theta: torch.Tensor | None = None,
    epochs: int = 60,
    lr: float = 3e-3,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    optimizer_kwargs: dict | None = None,
    full_data: tuple[torch.Tensor, torch.Tensor] | None = None,
    cosine_decay: bool = True,
    avg_frac: float = 0.0,
    restore: bool = True,
) -> ConstrainedFitResult:
    """Like fit_constrained, but pins the coefficients along a set of DIRECTIONS
    (columns of ``basis``) instead of raw coordinates — the natural choice when the
    profile axes are Hessian eigenvectors (param combinations, not single weights).

    The constraint is  <theta - origin, basis[:, c]> == coeffs[c]  for each column c.
    It is enforced by (1) projecting span(basis) out of the gradient AND the optimizer
    step each iteration, and (2) hard re-pinning theta's components along basis after
    each step. ``basis`` columns must be orthonormal.

    With ``avg_frac > 0`` the reported loss is the mean full-data loss over the last
    ``avg_frac`` fraction of epochs (use for minibatch SGD, which settles to a
    distribution, not a point). ``cosine_decay`` decays lr to ~0 over the run.
    """
    optimizer_kwargs = dict(optimizer_kwargs or {})
    # basis: (D, k) orthonormal columns = the directions whose coefficients we pin.
    coeffs = torch.as_tensor(coeffs, dtype=basis.dtype)

    entry_theta = parameters_to_vector(model.parameters()).detach().clone()
    base = (
        init_theta.detach().clone() if init_theta is not None else entry_theta.clone()
    )

    def _pin(theta):
        # Pin <theta - origin, basis[:,c]> = coeffs[c]: theta += basis (coeffs - basis^T(theta-origin)).
        resid = coeffs - basis.t() @ (theta - origin)
        return theta + basis @ resid

    theta = _pin(base.clone())
    vector_to_parameters(theta, model.parameters())
    opt = optimizer_cls(model.parameters(), lr=lr, **optimizer_kwargs)

    avg_start = int(epochs * (1.0 - avg_frac))
    losses_late: list[float] = []
    first_epoch_cache: list[tuple[torch.Tensor, torch.Tensor]] = []

    model.train()
    for ep in range(epochs):
        if cosine_decay:
            for grp in opt.param_groups:
                grp["lr"] = 0.5 * lr * (1.0 + math.cos(math.pi * ep / max(epochs, 1)))
        for xb, yb in batches():
            if ep == 0:
                first_epoch_cache.append((xb, yb))
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            g = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
            g = g - basis @ (
                basis.t() @ g
            )  # project gradient onto complement of span(basis)
            off = 0
            for p in model.parameters():
                n = p.numel()
                p.grad = g[off : off + n].view_as(p)
                off += n
            opt.step()
            with torch.no_grad():  # re-pin components along basis
                cur = _pin(parameters_to_vector(model.parameters()))
                vector_to_parameters(cur, model.parameters())
        if avg_frac > 0 and ep >= avg_start and full_data is not None:
            with torch.no_grad():
                losses_late.append(loss_fn(model(full_data[0]), full_data[1]).item())
    model.eval()

    if full_data is not None:
        xf, yf = full_data
    else:
        xf = torch.cat([xb for xb, _ in first_epoch_cache], dim=0)
        yf = torch.cat([yb for _, yb in first_epoch_cache], dim=0)

    for p in model.parameters():
        p.grad = None
    loss_fn(model(xf), yf).backward()
    grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    free_grad_norm = (grad - basis @ (basis.t() @ grad)).norm().item()

    with torch.no_grad():
        theta_out = parameters_to_vector(model.parameters()).detach().clone()
        final_loss = (
            float(sum(losses_late) / len(losses_late))
            if losses_late
            else loss_fn(model(xf), yf).item()
        )

    if restore:
        vector_to_parameters(entry_theta, model.parameters())

    return ConstrainedFitResult(
        loss=final_loss,
        theta=theta_out,
        epochs_run=epochs,
        free_grad_norm=free_grad_norm,
    )

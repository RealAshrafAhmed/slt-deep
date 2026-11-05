"""
Bayesian Neural Network implementation for PyTorch.

Provides PyMC-style interface for Bayesian inference on PyTorch models
using pluggable MCMC sampling backends.
"""

from collections.abc import Callable
from enum import Enum
from typing import overload

import torch
import torch.nn as nn
from torch.func import functional_call, hessian
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class LikelihoodType(Enum):
    """Enumeration of supported likelihood distribution types."""

    GAUSSIAN = (nn.MSELoss,)
    LAPLACE = (nn.L1Loss,)
    CATEGORICAL = (nn.CrossEntropyLoss,)
    BERNOULLI = (nn.BCELoss,)

    def __init__(self, loss_class):
        self.loss_class = loss_class

    @classmethod
    def from_loss(cls, loss_fn: nn.Module) -> "LikelihoodType":
        """Map PyTorch loss function to corresponding likelihood type.

        Args:
            loss_fn: PyTorch loss function instance

        Returns:
            LikelihoodType enum value for the loss function

        Raises:
            ValueError: If loss function is not supported
        """
        loss_type = type(loss_fn)
        for likelihood in cls:
            if likelihood.loss_class == loss_type:
                return likelihood

        supported_losses = [lik.loss_class.__name__ for lik in cls]
        raise ValueError(
            f"Unsupported loss function: {loss_type.__name__}. "
            f"Supported loss functions: {supported_losses}"
        )


def unflatten(named_params, params_vec):
    param_dict = dict(named_params)
    param_names = list(param_dict.keys())
    param_shapes = [p.shape for p in param_dict.values()]
    param_numels = [p.numel() for p in param_dict.values()]

    out, offset = {}, 0
    for name, shape, n in zip(param_names, param_shapes, param_numels):
        out[name] = params_vec[offset : offset + n].view(shape)
        offset += n
    return out


def make_fast_unflatten(model: nn.Module):
    """Cache parameter metadata for fast repeated unflattening."""
    names = []
    shapes = []
    numels = []
    offsets = [0]
    for name, p in model.named_parameters():
        names.append(name)
        shapes.append(p.shape)
        numels.append(p.numel())
        offsets.append(offsets[-1] + p.numel())

    def fast_unflatten(params_vec):
        return {
            n: params_vec[offsets[i] : offsets[i + 1]].view(s)
            for i, (n, s) in enumerate(zip(names, shapes))
        }

    return fast_unflatten


def mean_loss_factory(model, loss_fn):
    def fun(params_vec, x, y):
        params = unflatten(named_params=model.named_parameters(), params_vec=params_vec)
        pred = functional_call(model, params, x)
        return loss_fn(pred, y)  # make sure this uses mean reduction, not sum

    return fun


def approx_hessian(
    model, loss_fn, x: torch.Tensor, y: torch.Tensor, chunk_size=32
) -> torch.Tensor:
    """
    Compute Hessian of negative log-posterior with respect to model parameters.

    This is useful for second-order MCMC methods like Riemannian HMC or Laplace approximation.

    Args:
        model: PyTorch model
        loss_fn: Loss function
        x: Input data tensor
        y: Target data tensor
        chunk_size: Size of data chunks for memory-efficient computation

    Returns:
        Hessian matrix (tensor of shape [num_params, num_params])
    """

    params_vec = parameters_to_vector(model.parameters()).detach()

    if chunk_size is not None and x.shape[0] > chunk_size:
        # Compute Hessian in chunks to save memory
        hessians = []
        for i in range(0, x.shape[0], chunk_size):
            x_chunk = x[i : i + chunk_size]
            y_chunk = y[i : i + chunk_size]

            hessian_fn = hessian(mean_loss_factory(model=model, loss_fn=loss_fn))
            h_chunk = hessian_fn(params_vec, x_chunk, y_chunk)
            hessians.append(h_chunk)

        result = torch.stack(hessians).mean(dim=0)
    else:
        # Compute on full data
        hessian_fn = hessian(mean_loss_factory(model=model, loss_fn=loss_fn))
        result = hessian_fn(params_vec, x, y)

    assert isinstance(result, torch.Tensor)
    return result


def map_hessian(
    model,
    loss_fn,
    prior_logp,
    x: torch.Tensor,
    y: torch.Tensor,
    likelihood_scale: float | None = None,
    chunk_size: int | None = 32,
) -> torch.Tensor:
    """Hessian of the negative joint log-density (the MAP objective).

    Whereas :func:`approx_hessian` differentiates only the (mean) likelihood loss,
    this differentiates the full posterior objective that :meth:`BayesianNet.fit`
    minimizes::

        scale · loss_fn(model(x), y) − log_prior(θ)

    so the returned matrix is ``H_likelihood + H_prior`` — the curvature *at the
    MAP*. For a Gaussian prior ``−0.5·λ·‖θ‖²`` the prior contributes exactly
    ``+λ·I``, lifting the flat/degenerate likelihood directions off zero and giving
    a well-posed (PSD near the MAP) Hessian whose remaining near-zero spectrum is
    the genuine singular geometry rather than non-stationarity artifacts.

    Args:
        model: PyTorch model (its current parameters are the expansion point).
        loss_fn: Mean likelihood loss, e.g. ``CrossEntropyLoss`` (same object used
            to define the likelihood and to fit).
        prior_logp: ``(flat_params) -> log p(θ)`` scalar (this net's prior).
        x, y: Data the likelihood is evaluated on.
        likelihood_scale: Weight on the mean loss; defaults to ``y.numel()`` so the
            mean loss is rescaled to the summed log-likelihood (matches ``fit``).
        chunk_size: Chunk the likelihood Hessian over the data to save memory; the
            prior Hessian (data-independent) is added once.

    Returns:
        Hessian matrix of shape ``[num_params, num_params]``.
    """
    scale = float(likelihood_scale) if likelihood_scale is not None else float(y.numel())

    # Likelihood Hessian on the summed scale = scale × mean-loss Hessian.
    h_lik = scale * approx_hessian(model, loss_fn, x, y, chunk_size=chunk_size)

    # Prior Hessian: −∇²log p(θ) is added (objective has −log_prior). Computed by
    # autograd so it is correct for any prior_logp, not just Gaussian.
    params_vec = parameters_to_vector(model.parameters()).detach()
    neg_log_prior = lambda p: -prior_logp(p)  # noqa: E731
    h_prior = hessian(neg_log_prior)(params_vec)
    assert isinstance(h_prior, torch.Tensor)

    return h_lik + h_prior


class BayesianNet:
    """
    Bayesian wrapper for PyTorch models.

    Enables posterior sampling from PyTorch models using various MCMC backends.
    Automatically maps PyTorch loss functions to appropriate likelihood distributions:

    - MSELoss → Gaussian likelihood
    - L1Loss → Laplace likelihood
    - CrossEntropyLoss → Categorical likelihood
    - BCELoss → Bernoulli likelihood

    Example:
        >>> import torch
        >>> model = nn.Sequential(nn.Linear(2, 1))
        >>> loss_fn = nn.MSELoss()
        >>> def prior_fn(params): return torch.distributions.Normal(0, 1).log_prob(params).sum()
        >>> bayes_net = BayesianNet(model, loss_fn, prior_fn)
        >>> from ..sampling.sampler import Sampler
        >>> sampler = Sampler(bayes_net)
        >>> samples = sampler.sample(
        ...     x_data, y_data,
        ...     num_samples=1000,
        ...     backend='sgld'
        )
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        prior_logp: Callable[[torch.Tensor], torch.Tensor],
        temperature: float = 1.0,
        compile: bool = False,
    ):
        """
        Initialize Bayesian network.

        Args:
            model: PyTorch model (e.g., nn.Sequential, custom nn.Module)
            loss_fn: Loss function (determines likelihood type)
            prior_logp: Callable that takes flattened parameter tensor and returns log probability
            temperature: Temperature for tempered posteriors (1.0 = standard)
            compile: If True, use torch.compile on the forward pass for faster
                     repeated evaluations (e.g. during MCMC sampling).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.prior_logp = prior_logp
        self.temperature = temperature
        self._fast_unflatten = make_fast_unflatten(model)

        # Compile the forward+loss kernel for repeated calls (MCMC hot path)
        def _forward_and_loss(params_dict, x, y):
            y_pred = functional_call(model, params_dict, x)
            return -loss_fn(y_pred, y)

        if compile:
            self._forward_and_loss = torch.compile(_forward_and_loss)
        else:
            self._forward_and_loss = _forward_and_loss

    # ── pickle / deepcopy support (torch.compile objects aren't picklable) ─

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_forward_and_loss", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        model = self.model
        loss_fn = self.loss_fn

        def _forward_and_loss(params_dict, x, y):
            y_pred = functional_call(model, params_dict, x)
            return -loss_fn(y_pred, y)

        self._forward_and_loss = _forward_and_loss

    # ── device helpers ────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        """Device of the first model parameter (all should be on same device)."""
        return next(self.model.parameters()).device

    def to(self, device: torch.device | str) -> "BayesianNet":
        """Move model to *device* and return self for chaining."""
        self.model = self.model.to(device)
        return self

    def logp(self, x_data: torch.Tensor, y_data: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability: log p(θ, y|x) = log p(y|x, θ) + log p(θ)

        Args:
            x_data: Input data
            y_data: Target data

        Returns:
            Log probability (tensor, supports batching)
        """
        loglik = self.log_likelihood(x_data, y_data)
        logprior = self.log_prior_from_params(self.get_parameters(flat=True))
        return (loglik + logprior) / self.temperature

    def log_likelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood: log p(y|x, θ)

        Similar to PyMC's likelihood evaluation, this computes the probability
        of observing the data given the current model parameters.

        Args:
            x: Input data tensor
            y: Target/observed data tensor

        Returns:
            Log-likelihood (tensor, supports batching)
        """
        y_pred = self.model(x)
        loss_value = self.loss_fn(y_pred, y)

        # Convert loss to log-likelihood (negative log-likelihood)
        return -loss_value

    def log_likelihood_from_flat(
        self, flat_params: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """
        Differentiable log-likelihood that preserves the autograd graph
        from *flat_params* through to the returned scalar.

        Uses ``torch.func.functional_call`` so that gradients flow
        back through ``flat_params`` — unlike ``set_parameters`` followed
        by ``log_likelihood``, which severs the graph.
        """
        params_dict = self._fast_unflatten(flat_params)
        return self._forward_and_loss(params_dict, x, y)

    def log_prior_from_params(self, flat_params: torch.Tensor) -> torch.Tensor:
        """
        Compute log-prior: log p(θ) from given parameter tensor.

        This version allows computing the prior directly from a parameter tensor
        without breaking gradient connections, useful for MCMC sampling.

        Args:
            flat_params: Flattened parameter tensor

        Returns:
            Log-prior probability (tensor scalar)
        """
        return self.prior_logp(flat_params)

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        epochs: int = 300,
        lr: float = 3e-3,
        batch_size: int | None = None,
        likelihood_scale: float | None = None,
        init: torch.Tensor | None = None,
        optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: dict | None = None,
        cosine_decay: bool = True,
        lbfgs_polish: bool = True,
        lbfgs_max_iter: int = 200,
        shuffle_generator: torch.Generator | None = None,
        restore: bool = False,
        progress: Callable[[int, float], None] | None = None,
    ) -> torch.Tensor:
        """Fit the model to its MAP estimate: argmin over θ of the negative joint
        log-density. This is *the* point estimate for a Bayesian model — there is no
        separate "MLE fit", since fitting a ``BayesianNet`` means optimizing its
        posterior, with the prior an explicit term (not an implicit Adam
        ``weight_decay``). Minimizes the negative log-posterior built from this net's
        own pieces::

            −log p(θ, y | x) = −[ log p(y | x, θ) + log p(θ) ]
                             =   likelihood_scale · loss_fn(model(x), y) − log_prior(θ)

        so the prior enters the objective directly (and identically flows into any
        Hessian taken of the same objective). ``log p(θ)`` is ``self.prior_logp``;
        the likelihood term reuses ``self.loss_fn`` (the same object that defines
        the likelihood, via :class:`LikelihoodType`).

        Scale convention. ``loss_fn`` is a *mean* loss (mean per-token NLL), i.e.
        ``loss_fn = −(1 / n_tok) · log p(y | x, θ)``. The prior ``log p(θ)`` is a
        *total* log-density over all parameters. ``likelihood_scale`` (default
        ``y.numel()`` ≈ ``n_tok``) sets their relative weight so the joint matches
        the SGLD/HMC backends' ``log_lik + log_prior``. Pass an explicit value to
        override (e.g. the true non-PAD token count). The argmin is invariant to an
        overall rescale, so to stay well-conditioned the optimizer actually minimizes
        the *mean-scale* objective ``mean_loss − (1/scale)·log_prior`` (identical
        minimizer; gradients ~O(1) instead of ~O(scale), so a normal ``lr`` works).

        Convergence. Plain Adam plateaus before stationarity on the stiff, highly
        degenerate small-N landscapes, leaving spurious negative Hessian curvature.
        ``cosine_decay`` anneals the lr to ~0 over ``epochs`` and ``lbfgs_polish``
        runs a final strong-Wolfe L-BFGS pass; together they drive ‖∇‖ down to
        ~1e-2..1e-4 so the Hessian at the returned point is the genuine MAP curvature
        (PSD up to the prior floor). Both default on; turn off for a plain fit.

        Args:
            x, y: Training inputs and targets (the full sample; mini-batched
                internally when ``batch_size`` is set).
            epochs: Number of passes over the data.
            lr: Learning rate.
            batch_size: Mini-batch size. ``None`` = full-batch gradient descent.
            likelihood_scale: Weight on the per-sample mean loss in the objective.
                Defaults to ``y.numel()`` so ``scale · mean_NLL`` ≈ summed NLL.
            init: Optional flat parameter vector to start from. Defaults to the
                model's current parameters.
            optimizer_cls / optimizer_kwargs: Optimizer (default Adam). Do NOT pass
                ``weight_decay`` here — the regularizer is the explicit prior above;
                an Adam decay would add a *second*, implicit prior that would not
                appear in a Hessian of this objective.
            shuffle_generator: RNG for mini-batch shuffling (reproducibility).
            restore: If True, restore the model's entry parameters before returning
                (the returned theta still holds the MAP). If False (default), the
                model is left at the MAP — convenient for an immediate Hessian.
            progress: Optional callback ``(epoch, objective_value)`` per epoch.

        Returns:
            Flat MAP parameter vector (``parameters_to_vector`` order). The model is
            left holding these parameters unless ``restore=True``.
        """
        optimizer_kwargs = dict(optimizer_kwargs or {})
        if "weight_decay" in optimizer_kwargs and optimizer_kwargs["weight_decay"]:
            raise ValueError(
                "fit regularizes via the explicit prior (prior_logp), not the "
                "optimizer. Remove weight_decay from optimizer_kwargs — it would add "
                "a second implicit prior absent from the objective's Hessian."
            )

        scale = float(likelihood_scale) if likelihood_scale is not None else float(
            y.numel()
        )
        inv_scale = 1.0 / scale  # mean-scale prior weight (= λ/n_tok for a Gaussian)

        entry_theta = parameters_to_vector(self.model.parameters()).detach().clone()
        if init is not None:
            self.set_parameters(init.detach().clone())

        opt = optimizer_cls(self.model.parameters(), lr=lr, **optimizer_kwargs)
        sched = (
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
            if cosine_decay
            else None
        )
        n = x.shape[0]
        bs = batch_size or n

        def mean_objective(xb, yb, frac):
            """Mean-scale negative log-posterior on a batch: mean_loss −
            (frac/scale)·log_prior. Summed over an epoch's batches (frac = batch/n)
            this is mean_loss − (1/scale)·log_prior, the full mean-scale objective."""
            mean_loss = self.loss_fn(self.model(xb), yb)
            theta = parameters_to_vector(self.model.parameters())
            return mean_loss - frac * inv_scale * self.log_prior_from_params(theta)

        self.model.train()
        for ep in range(epochs):
            if bs < n:
                perm = torch.randperm(n, generator=shuffle_generator)
            else:
                perm = torch.arange(n)

            last_obj = 0.0
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                xb, yb = x[idx], y[idx]
                frac = xb.shape[0] / n
                opt.zero_grad()
                objective = mean_objective(xb, yb, frac)
                objective.backward()
                opt.step()
                last_obj = float(objective.detach())

            if sched is not None:
                sched.step()
            if progress is not None:
                progress(ep, last_obj)

        # L-BFGS polish (full-batch) to nail stationarity on the degenerate landscape.
        if lbfgs_polish:
            lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                lr=1.0,
                max_iter=lbfgs_max_iter,
                history_size=50,
                tolerance_grad=1e-12,
                tolerance_change=1e-16,
                line_search_fn="strong_wolfe",
            )

            def closure():
                lbfgs.zero_grad()
                loss = mean_objective(x, y, 1.0)
                loss.backward()
                return loss

            lbfgs.step(closure)
        self.model.eval()

        theta_map = parameters_to_vector(self.model.parameters()).detach().clone()
        if restore:
            self.set_parameters(entry_theta)
        return theta_map

    def map_objective(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        likelihood_scale: float | None = None,
    ) -> torch.Tensor:
        """Negative joint log-density ``scale·loss_fn − log_prior`` at current θ.

        The exact objective :meth:`fit` minimizes. Differentiating this with
        ``approx_hessian`` (passing a matching ``loss_fn``) gives the MAP Hessian
        ``H_NLL + H_prior`` — the posterior curvature, consistent with the fit.
        """
        scale = float(likelihood_scale) if likelihood_scale is not None else float(
            y.numel()
        )
        theta = parameters_to_vector(self.model.parameters())
        return scale * self.loss_fn(self.model(x), y) - self.log_prior_from_params(theta)

    def get_params_dict(self) -> dict[str, torch.Tensor]:
        """Extract model parameters as dictionary."""
        return {name: param.clone() for name, param in self.model.named_parameters()}

    def set_params_dict(self, params_dict: dict[str, torch.Tensor]):
        """Load parameters from dictionary into model."""
        for name, param in self.model.named_parameters():
            param.data = params_dict[name]

    def set_parameters(self, flattened_params: torch.Tensor) -> None:
        """
        Set model parameters from flattened parameter vector.

        Uses ``torch.nn.utils.vector_to_parameters`` for a single
        vectorised C++ call instead of a Python-level loop.

        Args:
            flattened_params: Flattened tensor containing all model parameters
        """
        vector_to_parameters(flattened_params, self.model.parameters())

    @overload
    def get_parameters(self, flat: bool = True) -> torch.Tensor: ...

    @overload
    def get_parameters(self, flat: bool = False) -> dict[str, torch.Tensor]: ...

    def get_parameters(
        self, flat: bool = True
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Get current model parameters.

        Args:
            flat: If True, return flattened tensor; if False, return parameter dict

        Returns:
            Flattened tensor containing all model parameters (if flat=True)
            or dictionary mapping parameter names to tensors (if flat=False)
        """
        if flat:
            return torch.cat([p.flatten() for p in self.model.parameters()])
        else:
            return {
                name: param.clone() for name, param in self.model.named_parameters()
            }

"""
MCMC Sampler implementation for BayesianNet models.

Supports multi-chain sampling with optional DEO (Deterministic Even-Odd)
chain permutation for improved mixing across modes.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Any

import torch
from loky import get_reusable_executor

from ..bn.bayesian_net import BayesianNet
from .configs import (
    HMC,
    NUTS,
    RMHMC,
    SGHMC,
    SGLD,
    InitStrategy,
    Perturb,
    Prior,
    SamplerConfig,
)

# ─────────────────────────────────────────────────────────────────────────────
# Multi-chain result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ChainResult:
    """Result from a single MCMC chain."""

    parameters: list[torch.Tensor]
    log_probabilities: list[float]
    acceptance_rate: float
    backend: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiChainResult:
    """Result from multi-chain MCMC sampling.

    Attributes:
        chains: per-chain results (length ``n_chains``)
        n_swaps_proposed: number of chain-swap proposals (DEO permutation)
        n_swaps_accepted: number accepted
        swap_history: detailed log of swap proposals and outcomes
    """

    chains: list[ChainResult]
    n_swaps_proposed: int = 0
    n_swaps_accepted: int = 0
    swap_history: list[dict[str, Any]] = field(default_factory=list)

    # ── convenience accessors ─────────────────────────────────────────────

    @property
    def n_chains(self) -> int:
        return len(self.chains)

    @property
    def swap_rate(self) -> float:
        if self.n_swaps_proposed == 0:
            return 0.0
        return self.n_swaps_accepted / self.n_swaps_proposed

    def swap_acceptance_rate(self) -> float:
        """Alias for :pyattr:`swap_rate` (callable form for API symmetry)."""
        return self.swap_rate

    def flat_parameters(self) -> list[torch.Tensor]:
        """Concatenate all chains' samples into a single list."""
        out: list[torch.Tensor] = []
        for c in self.chains:
            out.extend(c.parameters)
        return out

    def stacked(self) -> torch.Tensor:
        """Stack all chains into shape ``(n_chains, n_samples, dim)``."""
        return torch.stack([torch.stack(c.parameters) for c in self.chains])

    def rhat(self, max_params: int = 100) -> torch.Tensor:
        """Split-R-hat diagnostic (Gelman & Rubin).

        Returns a 1-D tensor of R-hat values for the first ``max_params``
        parameters.  Values near 1.0 indicate convergence.
        """
        stk = self.stacked()  # (C, S, D)
        C, S, D = stk.shape
        d = min(D, max_params)
        stk = stk[:, :, :d].float()

        # Split each chain in half → 2C half-chains
        half = S // 2
        halves = torch.cat(
            [stk[:, :half], stk[:, half : 2 * half]], dim=0
        )  # (2C, half, d)
        m = halves.shape[0]

        chain_means = halves.mean(dim=1)  # (m, d)
        grand_mean = chain_means.mean(dim=0)  # (d,)

        # Between-chain variance B/n
        B_over_n = ((chain_means - grand_mean) ** 2).sum(dim=0) / (m - 1)

        # Within-chain variance W
        chain_vars = halves.var(dim=1)  # (m, d)
        W = chain_vars.mean(dim=0)  # (d,)

        # Var+ estimate and R-hat
        var_plus = W * (half - 1) / half + B_over_n
        rhat = (var_plus / W.clamp(min=1e-10)).sqrt()
        return rhat


# ─────────────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────────────


class Sampler:
    """MCMC sampler for BayesianNet models.

    Supports single- and multi-chain sampling with optional DEO chain
    permutation to improve mixing across modes.
    """

    def __init__(
        self,
        bayes_net: BayesianNet,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        self.bayes_net = bayes_net
        # Ensure data lives on the same device as the model
        dev = bayes_net.device
        self.x = x.to(dev)
        self.y = y.to(dev)

    # ── public API ────────────────────────────────────────────────────────

    def sample(
        self,
        config: SamplerConfig,
        n_samples: int = 1000,
        n_chains: int = 1,
        init_strategy: InitStrategy | None = None,
        swap_every: int = 0,
        n_cores: int | None = None,
        betas: list[float] | None = None,
        per_chain_mass_matrix: list[Any] | None = None,
        per_chain_step_size: list[float] | None = None,
    ) -> MultiChainResult:
        """
        Sample from the posterior distribution.

        Args:
            config: Backend configuration (``NUTS(...)``, ``SGLD(...)``, etc.)
            n_samples: Posterior samples to collect *per chain*.
            n_chains: Number of independent chains to run.
            init_strategy: How to initialise extra chains.  Defaults to
                ``Perturb(scale=1.0)``.
            swap_every: Propose DEO chain swaps every this many samples.
                0 (default) disables swaps entirely.
            n_cores: Number of threads for parallel chain execution.
                ``None`` (default) auto-detects available CPU cores.
                Set to ``1`` to force sequential execution.
            betas: Inverse temperatures for parallel tempering.
                Must have length ``n_chains``.  Chain 0 should have
                ``beta=1.0`` (cold chain targeting the true posterior).
                Remaining chains sample tempered distributions
                π_β(θ) ∝ p(D|θ)^β · p(θ) with β < 1.
                When provided, ``swap_every`` must be > 0 and the swap
                criterion uses the replica-exchange Metropolis rule.
            per_chain_mass_matrix: Optional list of per-chain mass matrix
                seeds (length ``n_chains``).  Overrides ``config.mass_matrix``
                for each chain at warmup start.  Useful for resuming iterative
                refinement loops (e.g. H3-1) where each chain carries its own
                adapted M across rounds.  Only supported with multi-chain
                swap sampling.
            per_chain_step_size: Optional list of per-chain step-size seeds
                (length ``n_chains``).  Same semantics as
                ``per_chain_mass_matrix`` but for the leapfrog step size.

        Returns:
            :class:`MultiChainResult` containing per-chain results and
            swap diagnostics.

        Examples::

            # Single chain (backward compatible)
            result = sampler.sample(NUTS(n_warmup=500), n_samples=2000)

            # 4 chains with DEO swaps every 50 samples
            result = sampler.sample(
                NUTS(n_warmup=500),
                n_samples=2000,
                n_chains=4,
                init_strategy=Perturb(scale=2.0),
                swap_every=50,
            )
            print(result.rhat())

            # Parallel tempering with 4 temperatures
            result = sampler.sample(
                NUTS(n_warmup=500),
                n_samples=2000,
                n_chains=4,
                init_strategy=Perturb(scale=2.0),
                swap_every=20,
                betas=[1.0, 0.5, 0.25, 0.1],
            )
        """
        if n_chains < 1:
            raise ValueError(f"n_chains must be >= 1, got {n_chains}")
        if init_strategy is None:
            init_strategy = Perturb()
        if betas is not None:
            if len(betas) != n_chains:
                raise ValueError(
                    f"len(betas)={len(betas)} must match n_chains={n_chains}"
                )
            if swap_every <= 0:
                raise ValueError(
                    "swap_every must be > 0 when using parallel tempering (betas)"
                )
        if per_chain_mass_matrix is not None and len(per_chain_mass_matrix) != n_chains:
            raise ValueError(
                f"len(per_chain_mass_matrix)={len(per_chain_mass_matrix)} "
                f"must match n_chains={n_chains}"
            )
        if per_chain_step_size is not None and len(per_chain_step_size) != n_chains:
            raise ValueError(
                f"len(per_chain_step_size)={len(per_chain_step_size)} "
                f"must match n_chains={n_chains}"
            )
        if (per_chain_mass_matrix is not None or per_chain_step_size is not None) and (
            swap_every <= 0 or n_chains < 2
        ):
            raise ValueError(
                "per_chain_mass_matrix / per_chain_step_size are only supported "
                "with multi-chain swap sampling (swap_every > 0, n_chains >= 2)"
            )

        effective_cores = _effective_cores(n_cores, n_chains)

        # ── initialise starting points ────────────────────────────────────
        map_params = self.bayes_net.get_parameters(flat=True)
        inits = self._make_inits(map_params, n_chains, init_strategy)

        if swap_every <= 0 or n_chains < 2:
            # Simple case: run chains independently, no swaps
            results = self._run_batch(config, n_samples, inits, effective_cores)
            return MultiChainResult(chains=[_raw_to_chain(r) for r in results])

        # ── interleaved sampling with DEO swaps ───────────────────────────
        return self._run_with_swaps(
            config,
            n_samples,
            n_chains,
            inits,
            swap_every,
            n_cores=effective_cores,
            betas=betas,
            per_chain_mass_matrix=per_chain_mass_matrix,
            per_chain_step_size=per_chain_step_size,
        )

    # ── internal: parallel batch execution ─────────────────────────────────

    def _run_batch(
        self,
        config: SamplerConfig,
        n_samples: int,
        inits: list[torch.Tensor],
        n_cores: int,
    ) -> list[dict[str, Any]]:
        """Run multiple chains, in parallel when *n_cores* > 1.

        Uses ``loky`` process pool (spawn-based, cloudpickle
        serialisation) so closures, lambdas, and PyTorch models are
        shipped to worker processes without fork+autograd issues.
        """
        if n_cores <= 1 or len(inits) <= 1:
            # Sequential fallback
            results: list[dict[str, Any]] = []
            for init in inits:
                self.bayes_net.set_parameters(init)
                results.append(self._run_single(config, n_samples))
            return results

        n_workers = min(len(inits), n_cores)
        return self._run_batch_loky(config, n_samples, inits, n_workers)

    def _run_batch_loky(
        self,
        config: SamplerConfig,
        n_samples: int,
        inits: list[torch.Tensor],
        n_workers: int,
    ) -> list[dict[str, Any]]:
        """Process-based parallel chains via loky + cloudpickle."""
        print(
            f"  [MCMC] Running {len(inits)} chains in parallel ({n_workers} processes)"
        )

        dev = self.bayes_net.device

        # Build per-chain work items: each is a self-contained closure
        # that cloudpickle can serialise (captures model, data, config).
        sampler_ref = self

        def _run_one_chain(
            rank_init: tuple[int, torch.Tensor],
        ) -> tuple[int, dict[str, Any]]:
            rank, init = rank_init
            # Re-seed per chain for statistical independence
            torch.manual_seed(torch.initial_seed() + rank + 1)

            bn_copy = copy.deepcopy(sampler_ref.bayes_net)
            bn_copy.set_parameters(init)
            result = sampler_ref._run_single(config, n_samples, bayes_net=bn_copy)
            # CPU-ify tensors for safe IPC transfer
            result["parameters"] = [t.cpu() for t in result["parameters"]]
            return (rank, result)

        executor = get_reusable_executor(max_workers=n_workers)
        futures = list(executor.map(_run_one_chain, list(enumerate(inits))))

        all_results: list[dict[str, Any] | None] = [None] * len(inits)
        for rank, result in futures:
            all_results[rank] = result

        # Move tensors back to original device if needed
        if dev.type != "cpu":
            for r in all_results:
                assert r is not None
                r["parameters"] = [t.to(dev) for t in r["parameters"]]

        return all_results  # type: ignore[return-value]

    # ── internal: single-chain dispatch ───────────────────────────────────

    def _run_single(
        self,
        config: SamplerConfig,
        n_samples: int,
        bayes_net: BayesianNet | None = None,
        beta: float = 1.0,
        step_offset: int = 0,
        n_total_expected: int | None = None,
        chain_label: str | None = None,
    ) -> dict[str, Any]:
        bn = bayes_net if bayes_net is not None else self.bayes_net
        init_params = bn.get_parameters(flat=True)

        common = dict(
            set_params_fn=bn.set_parameters,
            log_likelihood_fn=bn.log_likelihood,
            log_prior_fn=bn.log_prior_from_params,
            x_data=self.x,
            y_data=self.y,
            init_params=init_params,
            n_samples=n_samples,
            diff_log_likelihood_fn=bn.log_likelihood_from_flat,
            beta=beta,
        )

        if isinstance(config, NUTS):
            from .backends.hmc import nuts

            return nuts(
                **common,
                step_size=config.step_size,
                n_burnin=config.n_warmup,
                thin=1,
                max_tree_depth=config.max_tree_depth,
                mass_matrix=config.mass_matrix,
                adapt_step_size=True,
                target_accept=config.target_accept,
                adapt_mass_matrix=config.adapt_mass_matrix,
                step_offset=step_offset,
                n_total_expected=n_total_expected,
                chain_label=chain_label,
            )

        if isinstance(config, HMC):
            from .backends.hmc import hmc

            return hmc(
                **common,
                step_size=config.step_size,
                n_leapfrog=config.n_leapfrog,
                n_burnin=config.n_warmup,
                thin=1,
                mass_matrix=config.mass_matrix,
                null_space_mass=config.null_space_mass,
                adapt_step_size=config.adapt_step_size,
                target_accept=config.target_accept,
                degen_threshold=config.degen_threshold,
            )

        if isinstance(config, SGLD):
            from .backends.sgld import sgld

            return sgld(
                **common,
                lr=config.lr,
                batch_size=config.batch_size,
                noise_scale=config.noise_scale,
                n_burnin=config.n_warmup,
                thin=1,
                preconditioner=config.preconditioner,
            )

        if isinstance(config, SGHMC):
            from .backends.sghmc import sghmc

            return sghmc(
                **common,
                lr=config.lr,
                alpha=config.alpha,
                batch_size=config.batch_size,
                n_burnin=config.n_warmup,
                thin=1,
                resample_momentum=config.resample_momentum,
                preconditioner=config.preconditioner,
            )

        if isinstance(config, RMHMC):
            from .backends.rmhmc import rmhmc

            return rmhmc(
                **common,
                step_size=config.step_size,
                n_leapfrog=config.n_leapfrog,
                n_burnin=config.n_warmup,
                thin=1,
                alpha=config.alpha,
                chunk_size=config.chunk_size,
                n_fixpoint=config.n_fixpoint,
                adapt_step_size=config.adapt_step_size,
                target_accept=config.target_accept,
                model=bn.model,
                loss_fn=bn.loss_fn,
            )

        raise TypeError(
            f"Unknown config type {type(config).__name__}. "
            f"Expected one of: NUTS, HMC, SGLD, SGHMC, RMHMC."
        )

    # ── internal: DEO chain-swap interleaving ─────────────────────────────

    def _run_with_swaps(
        self,
        config: SamplerConfig,
        n_samples: int,
        n_chains: int,
        inits: list[torch.Tensor],
        swap_every: int,
        n_cores: int = 1,
        betas: list[float] | None = None,
        per_chain_mass_matrix: list[Any] | None = None,
        per_chain_step_size: list[float] | None = None,
    ) -> MultiChainResult:
        """Run chains in segments, proposing DEO swaps between segments.

        DEO (Deterministic Even-Odd) permutation:
        - On even swap rounds: propose swaps (0,1), (2,3), (4,5), ...
        - On odd swap rounds:  propose swaps (1,2), (3,4), (5,6), ...

        When *betas* is provided, uses replica-exchange (parallel tempering)
        swap criterion:  log α = (β_j - β_k) · (U_k - U_j)
        where U = -log_likelihood (the untempered potential).

        Otherwise, each swap is Metropolis-accepted based on log-posterior
        values (same-temperature DEO).
        """
        use_pt = betas is not None
        if use_pt:
            chain_betas = list(betas)
        else:
            chain_betas = [1.0] * n_chains

        # Each chain gets its own BayesianNet copy (independent adaptation)
        print(f"  [PT] Initialising {n_chains} chain replicas...")
        chain_nets = [copy.deepcopy(self.bayes_net) for _ in range(n_chains)]

        # Current position of each chain
        positions = [p.clone() for p in inits]
        # Accumulated samples per chain
        chain_samples: list[list[torch.Tensor]] = [[] for _ in range(n_chains)]
        chain_logps: list[list[float]] = [[] for _ in range(n_chains)]
        chain_accept: list[float] = [0.0] * n_chains
        chain_diagnostics: list[dict[str, Any]] = [{} for _ in range(n_chains)]

        n_swaps_proposed = 0
        n_swaps_accepted = 0
        swap_history: list[dict[str, Any]] = []

        # ── WARMUP ────────────────────────────────────────────────────────
        # Build a warmup config per chain so per-chain mass matrix / step size
        # seeds (when provided) override the shared config values.
        n_warmup = config.n_warmup
        warmup_configs: list[SamplerConfig] = []
        for i in range(n_chains):
            wc = _with_warmup(config, n_warmup)
            seed_eps = (
                per_chain_step_size[i]
                if per_chain_step_size is not None
                else config.step_size
            )
            seed_M = (
                per_chain_mass_matrix[i]
                if per_chain_mass_matrix is not None
                else config.mass_matrix
            )
            warmup_configs.append(
                _with_step_size(wc, float(seed_eps), seed_M)
            )
        print(
            f"  [PT] ═══ WARMUP ({n_warmup} steps × {n_chains} chains) ═══"
        )

        for i in range(n_chains):
            chain_nets[i].set_parameters(positions[i])
            r = self._run_single(
                warmup_configs[i], 1, bayes_net=chain_nets[i],
                beta=chain_betas[i],
                chain_label=f"c{i}|β={chain_betas[i]:.2f}",
            )
            if r["parameters"]:
                positions[i] = r["parameters"][-1].clone()
            chain_diagnostics[i] = r.get("diagnostics", {})
            chain_diagnostics[i]["_acceptance_rate"] = r.get("acceptance_rate")

        # Log warmup summary
        print(f"  [PT] ─── Warmup complete ───")
        for i in range(n_chains):
            diag = chain_diagnostics[i]
            eps = diag.get("adapted_step_size", config.step_size)
            acc = diag.get("_acceptance_rate")
            acc_str = f"{acc:.2f}" if isinstance(acc, (int, float)) else "?"
            print(
                f"        chain {i} (β={chain_betas[i]:.2f}): "
                f"ε={float(eps):.2e}, accept={acc_str}"
            )

        # Build per-chain configs with adapted step sizes from warmup.
        # Fall back to per-chain seeds (or the shared config) when adaptation
        # didn't run (e.g. n_warmup=0 with pre-adapted M/ε passed in).
        chain_configs: list[SamplerConfig] = []
        for i in range(n_chains):
            diag = chain_diagnostics[i]
            seed_eps_i = (
                per_chain_step_size[i]
                if per_chain_step_size is not None
                else config.step_size
            )
            seed_M_i = (
                per_chain_mass_matrix[i]
                if per_chain_mass_matrix is not None
                else config.mass_matrix
            )
            adapted_eps = diag.get("adapted_step_size", seed_eps_i)
            adapted_M = diag.get("adapted_mass_matrix", seed_M_i)
            chain_configs.append(
                _with_warmup(
                    _with_step_size(config, float(adapted_eps), adapted_M),
                    0,
                )
            )

        # ── SAMPLING WITH SWAPS ──────────────────────────────────────────
        n_segments = (n_samples + swap_every - 1) // swap_every
        print(
            f"  [PT] ═══ SAMPLING ({n_samples} steps, "
            f"{n_segments} segments × {swap_every} steps) ═══"
        )

        remaining = n_samples
        swap_round = 0
        samples_done = 0

        # Cold-chain (β=1) per-step diagnostics accumulators
        cold_depths: list[int] = []
        cold_leapfrogs: list[int] = []
        cold_divergences: int = 0
        cold_alphas: list[float] = []
        _prev_cold_divs: int = 0

        while remaining > 0:
            seg_size = min(swap_every, remaining)

            # Run all chains for this segment
            seg_results: list[dict[str, Any]] = []
            for i in range(n_chains):
                chain_nets[i].set_parameters(positions[i])
                r = self._run_single(
                    chain_configs[i], seg_size, bayes_net=chain_nets[i],
                    beta=chain_betas[i],
                    step_offset=samples_done, n_total_expected=n_samples,
                    chain_label=f"c{i}|β={chain_betas[i]:.2f}",
                )
                seg_results.append(r)

            samples_done += seg_size

            for i in range(n_chains):
                raw = seg_results[i]
                chain_samples[i].extend(raw["parameters"])
                chain_logps[i].extend(raw["log_probabilities"])
                chain_accept[i] = raw.get("acceptance_rate", 1.0)
                if raw["parameters"]:
                    positions[i] = raw["parameters"][-1].clone()

            # Track cold-chain diagnostics
            cold_diag = seg_results[0].get("diagnostics", {})
            cold_depths.extend(cold_diag.get("tree_depths", []))
            cold_leapfrogs.extend(cold_diag.get("leapfrog_counts", []))
            cold_divergences += cold_diag.get("n_divergences", 0)
            # acceptance_rate from the segment gives α for this step
            cold_alphas.append(seg_results[0].get("acceptance_rate", 1.0))

            remaining -= seg_size

            # ── DEO swap proposal ─────────────────────────────────────────
            if remaining > 0 and n_chains >= 2:
                start = swap_round % 2  # even=0, odd=1

                for j in range(start, n_chains - 1, 2):
                    k = j + 1
                    n_swaps_proposed += 1

                    if use_pt:
                        # Replica exchange: need untempered log-likelihoods
                        # log α = (β_j - β_k) · (ll_k - ll_j)
                        ll_j = self._log_likelihood(positions[j])
                        ll_k = self._log_likelihood(positions[k])
                        log_alpha = (chain_betas[j] - chain_betas[k]) * (ll_k - ll_j)
                    else:
                        # Same-temperature DEO: full log-joint comparison
                        lp_j = self._log_joint(positions[j])
                        lp_k = self._log_joint(positions[k])
                        lp_j_swap = self._log_joint(positions[k])
                        lp_k_swap = self._log_joint(positions[j])
                        log_alpha = (lp_j_swap + lp_k_swap) - (lp_j + lp_k)

                    accepted = torch.log(torch.rand(1)).item() < log_alpha
                    if accepted:
                        positions[j], positions[k] = positions[k], positions[j]
                        n_swaps_accepted += 1

                    swap_history.append(
                        {
                            "round": swap_round,
                            "pair": (j, k),
                            "accepted": accepted,
                            "log_alpha": log_alpha,
                        }
                    )
                swap_round += 1

                # Log progress every 10 swap rounds (with cold-chain diagnostics)
                if swap_round % 10 == 0:
                    rate_so_far = n_swaps_accepted / max(1, n_swaps_proposed)
                    # Cold-chain stats over last 10 steps
                    recent_depths = cold_depths[-10:]
                    recent_L = cold_leapfrogs[-10:]
                    recent_alpha = cold_alphas[-10:]
                    chunk_max_depth = max(recent_depths) if recent_depths else 0
                    chunk_max_L = max(recent_L) if recent_L else 0
                    avg_alpha = sum(recent_alpha) / max(1, len(recent_alpha))
                    chunk_divs = cold_divergences - _prev_cold_divs
                    _prev_cold_divs = cold_divergences
                    print(
                        f"  [PT] step {samples_done}/{n_samples} │ "
                        f"swap round {swap_round} │ "
                        f"swaps: {n_swaps_accepted}/{n_swaps_proposed} "
                        f"({rate_so_far:.0%}) │ "
                        f"cold: max_depth={chunk_max_depth} max_L={chunk_max_L} "
                        f"α={avg_alpha:.2f} divs={chunk_divs}/10"
                    )

        # ── Final summary ─────────────────────────────────────────────────
        swap_rate = n_swaps_accepted / max(1, n_swaps_proposed)
        max_depth_all = max(cold_depths) if cold_depths else 0
        max_L_all = max(cold_leapfrogs) if cold_leapfrogs else 0
        avg_alpha_all = sum(cold_alphas) / max(1, len(cold_alphas))
        print(
            f"  [PT] ═══ DONE ═══  "
            f"swaps: {n_swaps_accepted}/{n_swaps_proposed} "
            f"({swap_rate:.1%})"
        )
        print(
            f"        cold chain: max_depth={max_depth_all}, "
            f"max_L={max_L_all}, α={avg_alpha_all:.2f}, "
            f"divergences={cold_divergences}/{n_samples}"
        )
        for i in range(n_chains):
            print(
                f"        chain {i} (β={chain_betas[i]:.2f}): "
                f"{len(chain_samples[i])} samples, "
                f"accept={chain_accept[i]:.3f}"
            )

        # Assemble results
        backend_name = type(config).__name__.lower()
        chains = [
            ChainResult(
                parameters=chain_samples[i],
                log_probabilities=chain_logps[i],
                acceptance_rate=chain_accept[i],
                backend=backend_name,
                diagnostics={
                    **chain_diagnostics[i],
                    "beta": chain_betas[i],
                },
            )
            for i in range(n_chains)
        ]
        return MultiChainResult(
            chains=chains,
            n_swaps_proposed=n_swaps_proposed,
            n_swaps_accepted=n_swaps_accepted,
            swap_history=swap_history,
        )

    # ── helpers ───────────────────────────────────────────────────────────

    def _log_joint(self, params: torch.Tensor) -> float:
        """Evaluate log p(θ|data) ∝ log p(data|θ) + log p(θ)."""
        self.bayes_net.set_parameters(params)
        ll = self.bayes_net.log_likelihood(self.x, self.y)
        lp = self.bayes_net.log_prior_from_params(params)
        return (ll + lp).item()

    def _log_likelihood(self, params: torch.Tensor) -> float:
        """Evaluate untempered log p(data|θ) for replica-exchange swaps."""
        self.bayes_net.set_parameters(params)
        ll = self.bayes_net.log_likelihood(self.x, self.y)
        return ll.item()

    @staticmethod
    def _make_inits(
        map_params: torch.Tensor,
        n_chains: int,
        strategy: InitStrategy,
    ) -> list[torch.Tensor]:
        """Generate n_chains starting points."""
        inits = [map_params.clone()]
        for _ in range(n_chains - 1):
            if isinstance(strategy, Perturb):
                norm = map_params.norm().item()
                sigma = strategy.scale * 0.01 * max(norm, 1.0)
                inits.append(map_params + torch.randn_like(map_params) * sigma)
            elif isinstance(strategy, Prior):
                inits.append(torch.randn_like(map_params))
            else:
                raise TypeError(
                    f"Unknown init_strategy type {type(strategy).__name__}. "
                    f"Expected Perturb or Prior."
                )
        return inits


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────


def _raw_to_chain(raw: dict[str, Any]) -> ChainResult:
    return ChainResult(
        parameters=raw["parameters"],
        log_probabilities=raw["log_probabilities"],
        acceptance_rate=raw.get("acceptance_rate", 1.0),
        backend=raw["backend"],
        diagnostics=raw.get("diagnostics", {}),
    )


def _with_warmup(config: SamplerConfig, n_warmup: int) -> SamplerConfig:
    """Return a copy of *config* with ``n_warmup`` overridden."""
    from dataclasses import replace

    return replace(config, n_warmup=n_warmup)  # type: ignore[arg-type]


def _with_step_size(
    config: SamplerConfig,
    step_size: float,
    mass_matrix: Any = None,
) -> SamplerConfig:
    """Return a copy of *config* with adapted step_size and mass_matrix."""
    from dataclasses import replace

    kwargs: dict[str, Any] = {"step_size": step_size}
    if mass_matrix is not None:
        kwargs["mass_matrix"] = mass_matrix
    return replace(config, **kwargs)  # type: ignore[arg-type]


def _effective_cores(n_cores: int | None, n_chains: int) -> int:
    """Resolve the number of worker threads to use."""
    if n_cores is not None:
        return max(1, min(n_cores, n_chains))
    return max(1, min(os.cpu_count() or 1, n_chains))

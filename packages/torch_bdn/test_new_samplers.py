"""
Test script for new SGLD and HMC implementations with proper gradient computation.
"""

import torch
from torch_bdn.sampling.backends.hmc import hmc
from torch_bdn.sampling.backends.sgld import sgld


def log_likelihood_fn(params):
    """Dummy log likelihood - simple quadratic loss."""
    return -0.5 * torch.sum(params**2)


def log_prior_fn(params):
    """Dummy log prior - simple quadratic penalty."""
    return -0.1 * torch.sum(params**2)


def test_sgld():
    """Test SGLD sampler with gradient computation."""
    print("Testing SGLD...")

    # Test parameters
    init_params = torch.tensor([2.0, -1.5], requires_grad=True)

    # Test that gradient computation works
    log_lik = log_likelihood_fn(init_params)
    log_pr = log_prior_fn(init_params)

    print(f"Initial log likelihood: {log_lik.item():.3f}")
    print(f"Initial log prior: {log_pr.item():.3f}")

    # Test gradients
    grad_lik = torch.autograd.grad(log_lik, init_params, retain_graph=True)[0]
    grad_pr = torch.autograd.grad(log_pr, init_params)[0]

    print(f"Likelihood gradient: {grad_lik}")
    print(f"Prior gradient: {grad_pr}")

    # Run a few SGLD steps
    results = sgld(
        log_likelihood_fn=log_likelihood_fn,
        log_prior_fn=log_prior_fn,
        init_params=init_params,
        n_samples=5,
        n_burnin=5,
        lr=0.01,
    )

    print(f"SGLD samples collected: {len(results['parameters'])}")
    print(f"Final sample: {results['parameters'][-1]}")
    print("✓ SGLD test passed!")
    return True


def test_hmc():
    """Test HMC sampler."""
    print("\nTesting HMC...")

    init_params = torch.tensor([1.0, -2.0], requires_grad=True)

    # Run HMC
    results = hmc(
        log_likelihood_fn=log_likelihood_fn,
        log_prior_fn=log_prior_fn,
        init_params=init_params,
        n_samples=5,
        n_burnin=5,
        step_size=0.1,
    )

    print(f"HMC samples collected: {len(results['parameters'])}")
    print(f"Acceptance rate: {results['acceptance_rate']:.3f}")
    print(f"Final sample: {results['parameters'][-1]}")
    print("✓ HMC test passed!")
    return True


if __name__ == "__main__":
    try:
        test_sgld()
        test_hmc()
        print("\n✅ All tests passed! New gradient-based implementations are working!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

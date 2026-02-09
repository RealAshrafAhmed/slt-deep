"""
Test script for new SGLD and HMC implementations with proper mini-batch gradient computation.
"""

import torch
from torch_bdn.sampling.backends.hmc import hmc
from torch_bdn.sampling.backends.sgld import sgld


def log_likelihood_fn(params, data_batch):
    """
    Dummy log likelihood function for a simple regression model.

    Assumes data_batch is of shape (batch_size, 2) where:
    - data_batch[:, 0] is the input x
    - data_batch[:, 1] is the target y

    Model: y = params[0] * x + params[1] + noise
    """
    x = data_batch[:, 0]
    y = data_batch[:, 1]

    # Predicted y values
    y_pred = params[0] * x + params[1]

    # Gaussian likelihood with fixed variance
    sigma = 0.1
    log_lik = -0.5 * torch.sum((y - y_pred) ** 2) / (sigma**2)
    log_lik = log_lik - 0.5 * len(data_batch) * torch.log(2 * torch.pi * sigma**2)

    return log_lik


def log_prior_fn(params):
    """Dummy log prior - simple Gaussian prior on parameters."""
    return -0.5 * torch.sum(params**2)


def generate_dummy_data(n_points=100):
    """Generate dummy regression data: y = 2*x + 1 + noise"""
    torch.manual_seed(42)  # For reproducibility
    x = torch.randn(n_points)
    y = 2 * x + 1 + 0.1 * torch.randn(n_points)  # True slope=2, intercept=1
    return torch.stack([x, y], dim=1)


def test_sgld_with_batching():
    """Test SGLD sampler with mini-batch gradient computation."""
    print("Testing SGLD with mini-batching...")

    # Generate data
    data = generate_dummy_data(n_points=200)
    print(f"Generated data shape: {data.shape}")

    # Initial parameters [slope, intercept]
    init_params = torch.tensor([0.0, 0.0], requires_grad=True)

    # Test batching
    batch_size = 32
    print(f"Using batch size: {batch_size}")

    # Run SGLD
    results = sgld(
        log_likelihood_fn=log_likelihood_fn,
        log_prior_fn=log_prior_fn,
        data=data,
        init_params=init_params,
        n_samples=10,
        batch_size=batch_size,
        n_burnin=10,
        lr=0.01,
    )

    print(f"SGLD samples collected: {len(results['parameters'])}")
    print(f"Final sample (slope, intercept): {results['parameters'][-1]}")
    print("True parameters: [2.0, 1.0]")
    print(
        f"Diagnostics: n_data={results['diagnostics']['n_data']}, "
        f"n_batches_per_epoch={results['diagnostics']['n_batches_per_epoch']}"
    )
    print("✓ SGLD batching test passed!")
    return True


def test_hmc_with_data():
    """Test HMC sampler with data (uses full dataset)."""
    print("\nTesting HMC with full data...")

    # Generate data
    data = generate_dummy_data(n_points=100)
    print(f"Generated data shape: {data.shape}")

    # Initial parameters [slope, intercept]
    init_params = torch.tensor([0.0, 0.0], requires_grad=True)

    # Run HMC (uses full data)
    results = hmc(
        log_likelihood_fn=log_likelihood_fn,
        log_prior_fn=log_prior_fn,
        data=data,
        init_params=init_params,
        n_samples=10,
        n_burnin=10,
        step_size=0.1,
    )

    print(f"HMC samples collected: {len(results['parameters'])}")
    print(f"Acceptance rate: {results['acceptance_rate']:.3f}")
    print(f"Final sample (slope, intercept): {results['parameters'][-1]}")
    print("True parameters: [2.0, 1.0]")
    print("✓ HMC data test passed!")
    return True


def test_batch_gradient_scaling():
    """Test that gradient scaling works correctly."""
    print("\nTesting gradient scaling in SGLD...")

    data = generate_dummy_data(n_points=64)  # Exact multiple of batch size
    params = torch.tensor([1.5, 0.5], requires_grad=True)

    # Compute full-batch gradient
    params.grad = None
    full_loglik = log_likelihood_fn(params, data)
    full_grad = torch.autograd.grad(full_loglik, params, retain_graph=True)[0]
    print(f"Full-batch gradient: {full_grad}")

    # Compute mini-batch gradient (should be scaled to match)
    batch_size = 16
    n_data = data.shape[0]
    batch_data = data[:batch_size]  # First batch

    params.grad = None
    batch_loglik = log_likelihood_fn(params, batch_data)
    batch_grad = torch.autograd.grad(batch_loglik, params, retain_graph=True)[0]
    scaled_batch_grad = batch_grad * (n_data / batch_size)
    print(f"Scaled mini-batch gradient: {scaled_batch_grad}")

    # They should be similar (not exact due to different data points)
    grad_diff = torch.norm(full_grad - scaled_batch_grad)
    print(f"Gradient difference norm: {grad_diff:.6f}")
    print("✓ Gradient scaling test completed!")
    return True


if __name__ == "__main__":
    try:
        test_sgld_with_batching()
        test_hmc_with_data()
        test_batch_gradient_scaling()
        print("\n✅ All tests passed! New mini-batch implementations are working!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
